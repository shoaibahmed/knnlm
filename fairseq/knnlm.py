import torch
import faiss
import math
import numpy as np
from fairseq import utils
import time
from fairseq.data import Dictionary

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)


class KNN_Dstore(object):
    def __init__(self, args):
        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
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
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))

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
            self.vals_from_memmap = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))
            self.vals = np.zeros((self.dstore_size, 1), dtype=np.int16 if args.dstore_fp16 else np.int)
            self.vals = self.vals_from_memmap[:]
            self.vals = self.vals.astype(np.int16 if args.dstore_fp16 else np.int)
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


class In_Memory_KNN_Dstore(KNN_Dstore):
    def __init__(self, args):
        super().__init__(args)
        self.keys = None
        self.values = None
        self.dist_metric = "squared_l2"
        self.use_vector_db = False
        self.device = None
        self.use_half_prec = True
        self.iterator = 0  # counts the number of items
        self.index_update_steps = 5
        self.insertion_steps = 0  # counts the number of insertions to identify index update
        self.temporary_cache = None
        self.use_temporary_cache = True
        if self.use_vector_db:
            self.setup_vector_db(args)
        else:
            self.use_cuda = True
            torch.device("cuda" if torch.cuda.is_available() and self.use_cuda else "cpu")
            print("!! Keystore device:", self.device)
            if self.use_temporary_cache:
                self.index = None
                self.temporary_cache = {"k": [], "v": []}

    def setup_faiss(self, args):
        print("!! Skipping FAISS setup for in-memory KNN DStore...")
        return  # don't do anything

    def setup_vector_db(self, args):
        print("!! Setting up vector database for in-memory KNN DStore...")
        # https://milvus.io/docs/v2.0.x/example_code.md
        connections.connect(alias="default", host="localhost", port="19530")

        # Create the schema
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="next_token", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields, "Next token database")
        self.vector_db = Collection("next_token_db", schema)

        # Create index
        print("!! Creating index at the start for vector database")
        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
        self.vector_db.create_index("embeddings", index)
        self.vector_db.load()  # load the collection in memory for vector search

    def add_item_to_store(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if isinstance(k, np.ndarray):
            k = torch.from_numpy(k)
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)

        if self.use_half_prec:
            k = k.half()  # only keys are required to be converted

        if self.use_vector_db:
            b = len(k)
            entities = [
                [self.iterator+i for i in range(b)],
                [x.tolist() for x in k.cpu().numpy()],
                [int(x.tolist()[0]) for x in v.cpu().numpy()],  # convert the vector to int
            ]
            self.vector_db.insert(entities)
            self.iterator += b
            print(f"!! New {b} item(s) added to the vector datastore")

            # update the index
            rebuild_index = False
            if rebuild_index:  # not really necessary (https://github.com/milvus-io/milvus/discussions/22280)
                if self.insertion_steps % self.index_update_steps == 0:
                    if self.insertion_steps > 0:
                        # Release collection first to create an index
                        print("!! Releasing vector collection for index creation")
                        self.vector_db.release()

                    # Create an index for the vector DB
                    print("!! Generating index for the vector database")
                    index = {
                        "index_type": "IVF_FLAT",
                        "metric_type": "L2",
                        "params": {"nlist": 128},
                    }
                    self.vector_db.create_index("embeddings", index)

                    print("!! Loading vector collection in memory")
                    self.vector_db.load()  # load the collection in memory for vector search
            self.insertion_steps += 1

        else:
            k = k.to(self.device)
            v = v.to(self.device)

            if self.dist_metric == "cosine":
                k = torch.nn.functional.normalize(k, p=2.0, dim=1)  # l2 normalize the key

            if self.use_temporary_cache:
                self.temporary_cache["k"].append(k)
                self.temporary_cache["v"].append(v)
                print(f"!! New item added in temporary cache / temporary cache size: {len(self.temporary_cache['k'])} / keys in store: {self.keys.shape}")
            else:
                if self.keys is None:
                    assert self.values is None
                    self.keys = k
                    self.values = v
                else:
                    self.keys = torch.cat([self.keys, k], dim=0)
                    self.values = torch.cat([self.values, v], dim=0)
                print(f"!! New item added in memory store / k: {self.keys.shape} / v: {self.values.shape}")

    def update_datastore(self, args, model_dim=1024):
        """Integrates items from temporary cache into the datastore and updates the index"""
        assert self.use_temporary_cache, "Build index assumes that temporary caching is enabled"
        old_keys_shape = self.keys.shape
        self.keys = torch.cat([self.keys] + self.temporary_cache['k'], dim=0)
        self.values = torch.cat([self.values] + self.temporary_cache['v'], dim=0)
        print(f"!! Cache elements integrated into datastore / previous keys shape: {old_keys_shape} / new keys shape: {self.keys.shape}")

        self.update_index(args, model_dim)

    def update_index(self, args, model_dim=1024):
        # TODO: Integrate FAISS-GPU index
        # Rebuild the index
        print("!! Rebuilding FAISS index")
        start = time.time()
        self.index = faiss.IndexFlatL2(model_dim)   # build the index
        self.index.add(self.keys)                  # add vectors to the index
        print(f"!! Index creation took {time.time() - start} seconds")
        self.index.nprobe = args.probe

    def get_knns(self, queries, use_batched_version=False):
        """Batched version is disabled by default as it is super expensive in terms of memory"""
        assert len(queries.shape) == 2, queries.shape
        if self.use_half_prec:
            queries = queries.half()  # only keys are in half precision

        if self.use_vector_db:
            # Search the vector store
            vectors_to_search = [x.tolist() for x in queries.cpu().numpy()]
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }
            result = self.vector_db.search(vectors_to_search, "embeddings", search_params, limit=self.k, output_fields=["next_token"])

            selected_dists = []
            selected_vals = []
            for hits in result:  # iterate over queries
                selected_dists.append([])
                selected_vals.append([])
                for hit in hits:  # iterate over nearest neighbors for each query
                    selected_dists[-1].append(float(hit.distance))
                    selected_vals[-1].append(hit.entity.get('next_token'))
                    # print(f"hit: {hit}, random field: {hit.entity.get('next_token')}")
            selected_dists = torch.tensor(selected_dists)
            selected_vals = torch.tensor(selected_vals)

        elif self.use_temporary_cache:  # temporary caching creating a faiss index
            selected_dists, nearest_neighbors = self.index.search(queries, self.k)
            selected_vals = torch.stack([torch.gather(self.values[:, 0], dim=0, index=nearest_neighbors[i, :])
                                        for i in range(nearest_neighbors.shape[0])], dim=0)  # values are tensor of size [N' x 1]

        else:
            # self.keys shape: B x D
            queries = queries.detach().to(self.device)
            if self.dist_metric in ["l2", "squared_l2"]:
                if use_batched_version:
                    dist = ((queries[:, None, :] - self.keys[None, :, :]) ** 2).sum(dim=2)  # (N' x 1, D) - (1 x N x D).T -> N' x N x D -> N' x N
                else:
                    dist = torch.stack([((queries[i:i+1, :] - self.keys) ** 2).sum(dim=1) for i in range(len(queries))], dim=0)
                assert dist.shape == (len(queries), len(self.keys)), dist.shape
                if self.dist_metric == "l2":
                    dist = torch.sqrt(dist)
            elif self.dist_metric == "cosine":
                queries = torch.nn.functional.normalize(queries, p=2.0, dim=1)  # l2 normalize the query
                if use_batched_version:
                    sim = torch.matmul(queries, self.keys.T)  # (N' x D) x (N x D).T -> N' x N
                else:
                    sim = torch.cat([torch.matmul(queries[i:i+1, :], self.keys.T) for i in range(len(queries))], dim=0)  # (1 x D) x (N x D).T -> [1 x N]
                assert sim.shape == (len(queries), len(self.keys)), sim.shape
                dist = 1.0 - sim  # cosine distance
            else:
                raise RuntimeError(f"Unknown distance metric: {self.dist_metric}")

            # Compute nearest neighbors based on the distance
            dist_idx_sorted = torch.argsort(dist, dim=1, descending=False)
            nearest_neighbors = dist_idx_sorted[:, :self.k]  # as distances are sorted in ascending order

            # Select the nearest neighbors
            selected_dists = torch.gather(dist, dim=1, index=nearest_neighbors)
            selected_vals = torch.stack([torch.gather(self.values[:, 0], dim=0, index=nearest_neighbors[i, :])
                                        for i in range(nearest_neighbors.shape[0])], dim=0)  # values are tensor of size [N' x 1]

        return selected_dists, selected_vals

    def get_knn_log_prob(self, queries, tgt, pad_idx):
        with torch.no_grad():
            # queries  are TxBxC
            # reshape: (TxB)xC
            qshape = queries.shape
            full_yhat_knn_prob = torch.full([qshape[0]*qshape[1]], -10000, dtype=torch.float32).cuda()

            if self.keys is not None or self.iterator > 0:  # iterator takes care of the vector db
                queries = queries.view(-1, qshape[-1])
                tgt = tgt.contiguous().view(-1)
                dists, vals = self.get_knns(queries[tgt != pad_idx])
                # (T_reducedxB)xK
                dists = dists.cuda()
                probs = utils.log_softmax(-dists, dim=-1)

                index_mask = torch.eq(vals.long().cuda().squeeze(-1), tgt[tgt != pad_idx].unsqueeze(-1)).float()
                index_mask[index_mask == 0] = -10000 # for stability
                index_mask[index_mask == 1] = 0

                # (T_reducedxB)
                yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone()
                full_yhat_knn_prob[tgt != pad_idx] = yhat_knn_prob

            # TxBx1
            return full_yhat_knn_prob.view(qshape[0], qshape[1], 1)

    def print_datastore_stats(self) -> None:
        if self.use_vector_db:
            print(f"Milvus vector database / Size: {self.iterator} / Insertion steps: {self.insertion_steps}")
        else:
            print(f"Datastore stats / Keys: {self.keys.shape} / Values: {self.values.shape} / Distance metric: {self.dist_metric}")
