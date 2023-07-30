import os
import time
import torch
import faiss
import numpy as np
from fairseq import utils
from typing import Tuple


class KNN_Dstore(object):
    def __init__(self, args):
        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        self.negative_distance = None
        self.index = self.setup_faiss(args)

    def setup_faiss(self, args):
        if not args.dstore_filename:
            raise ValueError('Cannot build a datastore without the data.')

        start = time.time()
        index = faiss.read_index(args.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        print('Reading datastore took {} s'.format(time.time() - start))
        index.nprobe = args.probe

        if args.dstore_fp16:
            print('Keys are fp16 and vals are int16')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int16, mode='r', shape=(self.dstore_size, 1))
        else:
            print('Keys are fp32 and vals are int64')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int64, mode='r', shape=(self.dstore_size, 1))

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
                self.keys = np.zeros((self.dstore_size, self.dimension), dtype=np.float16 if args.dstore_fp16 else np.float32)
                self.keys = self.keys_from_memmap[:]
                self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

            del self.vals
            self.vals_from_memmap = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int64, mode='r', shape=(self.dstore_size, 1))
            self.vals = np.zeros((self.dstore_size, 1), dtype=np.int16 if args.dstore_fp16 else np.int64)
            self.vals = self.vals_from_memmap[:]
            self.vals = self.vals.astype(np.int16 if args.dstore_fp16 else np.int64)
            print('Loading to memory took {} s'.format(time.time() - start))

        return index

    def get_knns(self, queries):
        start = time.time()
        dists, knns = self.index.search(queries.detach().cpu().float().numpy(), self.k)
        return dists, knns

    def get_knn_log_prob(self, queries, tgt, pad_idx):
        def dist_func(d, k, q, function=None):
            if not function:
                # Default behavior for L2 metric is to recompute distances.
                # Default behavior for IP metric is to return faiss distances.
                qsize = q.shape
                if self.metric_type == 'l2':
                    start = time.time()
                    knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
                    if self.half:
                        knns_vecs = knns_vecs.half()
                    query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                    l2 = torch.sum((query_vecs - knns_vecs.detach())**2, dim=2)
                    return -1 * l2
                return d

            if function == 'dot':
                qsize = q.shape
                return (torch.from_numpy(self.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

            if function == 'do_not_recomp_l2':
                return -1 * d

            raise ValueError("Invalid knn similarity function!")

        # queries  are TxBxC
        # reshape: (TxB)xC
        qshape = queries.shape
        queries = queries.view(-1, qshape[-1])
        tgt = tgt.contiguous().view(-1)
        dists, knns = self.get_knns(queries[tgt != pad_idx])
        # (T_reducedxB)xK
        dists = torch.from_numpy(dists).cuda()
        start = time.time()
        dists = dist_func(dists, knns, queries[tgt != pad_idx, :], function=self.sim_func)
        self.negative_distance = dists.detach()  # save for lambda tuning
        probs = utils.log_softmax(dists, dim=-1)

        index_mask = torch.eq(torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1), tgt[tgt != pad_idx].unsqueeze(-1)).float()
        index_mask[index_mask == 0] = -10000 # for stability
        index_mask[index_mask == 1] = 0

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone()
        full_yhat_knn_prob = torch.full([qshape[0]*qshape[1]], -10000, dtype=torch.float32).cuda()
        full_yhat_knn_prob[tgt != pad_idx] = yhat_knn_prob

        # TxBx1
        return full_yhat_knn_prob.view(qshape[0], qshape[1], 1)

    def get_negative_distance(self) -> torch.Tensor:
        return self.negative_distance


class In_Memory_KNN_Dstore(KNN_Dstore):
    def __init__(self, args):
        super().__init__(args)
        self.keys = None
        self.values = None
        self.memory_strengths = None
        self.memory_life = None
        self.last_nearest_neighbors = None

        self.adaptive_mem_log_prob_thresh = args.adaptive_mem_log_prob_thresh
        self.prune_memory_strength_thresh = args.prune_memory_strength_thresh
        self.memory_decay_factor = args.memory_decay_factor
        self.lmbda = args.lmbda
        self.use_token_perplexity = args.use_perplexity_mem_strength  # uses target token probability otherwise

        self.only_correct_label_strength_update = True  # memory strength is only updated for nearest neighbors with the correct label
        self.use_cuda = False
        self.use_half_prec = False  # FAISS doesn't support half precision keys
        self.index = None
        self.index_nprobe = args.probe
        self.use_gpu_index = True
        self.use_tensors_for_faiss = False  # Latest FAISS version will directly support PyTorch tensors

        if self.use_gpu_index:
            self.use_cuda = self.use_tensors_for_faiss
        print("!! Using GPU FAISS index?", self.use_gpu_index)
        self.reset_cache()

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_cuda else "cpu")
        print("!! Keystore device:", self.device)

    def reset_cache(self):
        self.temporary_cache = {"k": [], "v": [], "strength": []}

    def setup_faiss(self, args):
        print("!! Skipping FAISS setup for in-memory KNN DStore...")
        return  # don't do anything

    def add_item_to_store(self, k: torch.Tensor, v: torch.Tensor, token_log_probs: torch.Tensor = None) -> None:
        if isinstance(k, np.ndarray):
            k = torch.from_numpy(k)
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)

        # Discard memories that are not important to be saved
        token_importance = None
        if token_log_probs is not None:
            if isinstance(token_log_probs, np.ndarray):
                token_log_probs = torch.from_numpy(token_log_probs)

            if self.adaptive_mem_log_prob_thresh is not None:
                prev_size = len(k)
                important_mems = (token_log_probs < self.adaptive_mem_log_prob_thresh).to(self.device)  # the log-prob is lower than sigma
                k = k[important_mems]
                v = v[important_mems]
                token_log_probs = token_log_probs[important_mems].float()
                if self.use_token_perplexity:
                    token_log_probs = -token_log_probs  # perplexity is negative exponential of probs
                token_importance = torch.exp(token_log_probs)
                if not self.use_token_perplexity:  # token importance is the inverse of the probability assigned to the correct label
                    token_importance = 1. - token_importance
                print(f"\t !! Using sigma={self.adaptive_mem_log_prob_thresh} / Available memories: {prev_size} / Retained memories: {len(k)}")

        if self.use_half_prec:
            k = k.half()  # only keys are required to be converted
        else:
            k = k.to(torch.float32)  # avoid double precision

        k = k.to(self.device)
        v = v.to(self.device)
        if token_importance is not None:
            token_importance = token_importance.float().to(self.device)

        self.temporary_cache["k"].append(k)
        self.temporary_cache["v"].append(v)
        if token_importance is not None:
            self.temporary_cache["strength"].append(token_importance)
        print(f"!! New item added in temporary cache / temporary cache size: {len(self.temporary_cache['k'])} / keys in store: {self.keys.shape if self.keys is not None else 'none'}")

    def prune_weak_memories(self) -> None:
        if self.prune_memory_strength_thresh is None:
            return

        if self.memory_strengths is None:
            if self.keys is None:  # empty datastore
                return
            raise RuntimeError("Weak memory pruning function called without memory strenght initialization")

        retained_mem_mask = self.memory_strengths >= self.prune_memory_strength_thresh
        total_mem = len(self.keys)
        retained_mem = int(torch.sum(retained_mem_mask))
        pruned_mem = total_mem - retained_mem
        if pruned_mem > 0:
            self.keys = self.keys[retained_mem_mask]
            self.values = self.values[retained_mem_mask]
            avg_pruned_memory_life = None
            if self.memory_strengths is not None:
                self.memory_strengths = self.memory_strengths[retained_mem_mask]
                avg_pruned_memory_life = float(self.memory_life[torch.logical_not(retained_mem_mask)].float().mean())
                self.memory_life = self.memory_life[retained_mem_mask]
            print(f"\t !! Memory prune theshold: {self.prune_memory_strength_thresh} / # total memories (old: {total_mem} / new: {len(self.keys)}) / # memories pruned: {pruned_mem} / average pruned memory life: {avg_pruned_memory_life:.4f}")

        # Increment the memory life counter
        self.memory_life += 1

    def update_memory_strengths(self, token_log_probs: torch.Tensor) -> None:
        # Update here the strengths for the nearest neighbors based on their contribution in reducing the perplexity
        if self.last_nearest_neighbors is None:
            return
        assert len(token_log_probs.shape) == 1, token_log_probs.shape
        assert len(self.last_nearest_neighbors) == len(token_log_probs), f"{self.last_nearest_neighbors.shape} != {token_log_probs.shape}"
        assert len(self.last_nearest_neighbors.shape) == 2, self.last_nearest_neighbors.shape
        assert self.last_nearest_neighbor_probs.shape == self.last_nearest_neighbors.shape, f"{self.last_nearest_neighbor_probs.shape} != {self.last_nearest_neighbors.shape}"

        token_log_probs = token_log_probs.float().to(self.device)
        if self.use_token_perplexity:
            token_log_probs = -token_log_probs
        token_importance = torch.exp(token_log_probs)
        if not self.use_token_perplexity:  # token importance is the inverse of the probability assigned to the correct label
            token_importance = 1. - token_importance
        for i in range(len(token_log_probs)):
            tgt_label_contrib = self.lmbda * token_importance[i] * self.last_nearest_neighbor_probs[i, :]  # weight the token importance by the nearest neighbor contribution as well as the lambda val
            self.memory_strengths[self.last_nearest_neighbors[i, :]] += tgt_label_contrib  # upweight memory strength

    def decay_memory_strengths(self) -> None:
        if self.memory_decay_factor is None:
            return
        assert self.memory_strengths is not None, "Memory strength not logged as memory prune threshold is none"

        # Decay the memory strengths
        self.memory_strengths *= self.memory_decay_factor

    def update_datastore(self, model_dim: int = 1024) -> None:
        # Prune old memories
        self.prune_weak_memories()

        # Add the new entries in the cache to the datastore
        old_keys_shape = self.keys.shape if self.keys is not None else None
        self.keys = torch.cat(([self.keys] if self.keys is not None else []) + self.temporary_cache['k'], dim=0)
        self.values = torch.cat(([self.values] if self.values is not None else []) + self.temporary_cache['v'], dim=0)
        if self.prune_memory_strength_thresh is not None:
            num_new_memories = len(self.keys) - (old_keys_shape[0] if old_keys_shape is not None else 0)
            assert len(self.temporary_cache['strength']) == len(self.temporary_cache['k']), self.temporary_cache['strength']
            self.memory_strengths = torch.cat(([self.memory_strengths] if self.memory_strengths is not None else []) +
                                              self.temporary_cache['strength'], dim=0)
            init_counters = torch.zeros((num_new_memories,), dtype=torch.int64).to(self.device)  # initialize memory life counter to zero
            self.memory_life = torch.cat(([self.memory_life] if self.memory_life is not None else []) + [init_counters], dim=0)
        print(f"!! Cache elements integrated into datastore / previous keys shape: {old_keys_shape} / new keys shape: {self.keys.shape}")

        # Validate that the length of the values matches
        assert len(self.keys) == len(self.values), f"{self.keys.shape} != {self.values.shape}"
        if self.prune_memory_strength_thresh is not None:
            assert len(self.keys) == len(self.memory_strengths) == len(self.memory_life), f"{self.keys.shape} != {self.values.memory_strengths} != {self.values.memory_life}"

        # Reset temporary cache
        self.reset_cache()

        # Update index
        self.update_index(model_dim)

        # Update memory strengths due to decay
        self.decay_memory_strengths()

    def update_index(self, model_dim: int = 1024, use_ivf: bool = False) -> None:
        # Supports FAISS-GPU index (https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU)
        # Rebuild the index
        print(f"!! Rebuilding {'GPU' if self.use_gpu_index else 'CPU'}-FAISS index")
        start = time.time()
        keys = self.keys if self.use_tensors_for_faiss else self.keys.cpu().numpy()  # cast to numpy for faiss

        if self.use_gpu_index:
            # https://github.com/facebookresearch/faiss/blob/main/faiss/gpu/test/torch_test_contrib_gpu.py
            res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexFlatL2(res, model_dim)    # build the index
        else:
            self.index = faiss.IndexFlatL2(model_dim)       # build the index

        if use_ivf:
            print("!! Training IVF index")
            nlist = 128  # the number of cells
            self.index = faiss.IndexIVFFlat(self.index, model_dim, nlist)
            self.index.train(keys)  # train IVF index

        self.index.add(keys)  # add elements to index
        print(f"!! Index creation took {time.time() - start} seconds")
        self.index.nprobe = self.index_nprobe  # nprobe is the number of cells (out of nlist) that are visited to perform a search

    def get_knns(self, queries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched version is disabled by default as it is super expensive in terms of memory"""
        assert len(queries.shape) == 2, queries.shape
        self.last_nearest_neighbors = None

        if self.use_half_prec:
            queries = queries.half()  # only keys are in half precision
        else:
            queries = queries.to(torch.float32)  # avoid double precision

        if not self.use_tensors_for_faiss:
            queries = queries.detach().cpu().numpy()  # convert to numpy arrays
        k = min(self.k, len(self.keys))
        selected_dists, nearest_neighbors = self.index.search(queries, k)
        nearest_neighbors = torch.from_numpy(nearest_neighbors).to(self.device)
        selected_dists = torch.from_numpy(selected_dists).to(self.device)
        selected_vals = torch.stack([torch.gather(self.values[:, 0], dim=0, index=nearest_neighbors[i, :])
                                    for i in range(nearest_neighbors.shape[0])], dim=0)  # values are tensor of size [N' x 1]

        return nearest_neighbors, selected_dists, selected_vals

    def get_knn_log_prob(self, queries: torch.Tensor, tgt: torch.Tensor, pad_idx: int) -> torch.Tensor:
        with torch.no_grad():
            # queries  are TxBxC
            # reshape: (TxB)xC
            qshape = queries.shape
            if self.keys is None:  # no prob
                full_yhat_knn_prob = torch.zeros((qshape[0], qshape[1], 1), dtype=torch.float32).cuda()
                return full_yhat_knn_prob

            full_yhat_knn_prob = torch.full([qshape[0]*qshape[1]], -10000, dtype=torch.float32).cuda()
            queries = queries.view(-1, qshape[-1])
            tgt = tgt.contiguous().view(-1)
            nearest_neighbors, dists, vals = self.get_knns(queries[tgt != pad_idx])

            # (T_reducedxB)xK
            sim = -dists.cuda()  # negative of the distance can be considered as similarity which softmax needs
            self.negative_distance = sim.detach()  # save for lambda tuning
            probs = utils.log_softmax(sim, dim=-1)

            # Remove contribution of indices where the prediction disagrees with the target (only computing perplexity of the target sequence)
            index_mask = torch.eq(vals.long().cuda().squeeze(-1), tgt[tgt != pad_idx].unsqueeze(-1)).float()
            index_mask[index_mask == 0] = -10000 # for stability
            index_mask[index_mask == 1] = 0

            # Cache the nn idx and nn probs for memory strength updates
            self.last_nearest_neighbors = nearest_neighbors.detach().to(self.device)
            self.last_nearest_neighbor_probs = torch.nn.functional.softmax(sim.float(), dim=-1).detach().to(self.device)
            if self.only_correct_label_strength_update:
                self.last_nearest_neighbor_probs[index_mask == 0] = 0.  # set update for nearest neighbors with incorrect label to zero

            # (T_reducedxB)
            yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone()  # sum the probablity from the different nearest neighbors with the correct label
            full_yhat_knn_prob[tgt != pad_idx] = yhat_knn_prob

            # TxBx1
            return full_yhat_knn_prob.view(qshape[0], qshape[1], 1)

    def print_datastore_stats(self) -> None:
        print(f"=== [DATASTORE STATS] ===")
        print(f"\t ~ Keys: {self.keys.shape}")
        print(f"\t ~ Values: {self.values.shape}")
        if self.memory_strengths is not None:
            assert self.memory_life is not None
            print(f"\t ~ Memory strengths: {self.memory_strengths.shape}")
            print(f"\t ~ Memory life: {self.memory_life.shape} | min: {int(self.memory_life.min())} / mean: {float(self.memory_life.float().mean()):.4f} / max: {int(torch.max(self.memory_life))}")
        print(f"=========================")

    def save_datastore(self, datastore_path: str) -> None:
        print(f"!! Saving datastore to file: {datastore_path}")
        output_dict = {"keys": self.keys, "values": self.values, "memory_strengths": self.memory_strengths, "memory_life": self.memory_life}
        torch.save(output_dict, datastore_path)
        print(f"!! Datastore successfully saved!")
        self.print_datastore_stats()

    def load_datastore(self, datastore_path: str, freeze_loaded_memories: bool = False) -> None:
        print(f"!! Loading datastore from file: {datastore_path}")
        assert os.path.exists(datastore_path), datastore_path
        output_dict = torch.load(datastore_path)
        self.keys = output_dict["keys"]
        self.values = output_dict["values"]
        self.memory_strengths = output_dict["memory_strengths"]
        if freeze_loaded_memories:
            max_val = torch.finfo(self.memory_strengths.dtype).max
            print(f"!! Freezing memories by setting memory strength to be max val of {self.memory_strengths.dtype}: {max_val}")
            self.memory_strengths = torch.full_like(self.memory_strengths, max_val)
        self.memory_life = output_dict["memory_life"]
        print(f"!! Datastore successfully loaded!")
        self.print_datastore_stats()

        # Rebuild index to ensure that the datastore is usable
        self.update_index()
