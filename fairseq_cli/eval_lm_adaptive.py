#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""

import logging
import math
import os

import torch
import numpy as np

from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data import LMContextWindowDataset
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer
from fairseq.knnlm import In_Memory_KNN_Dstore

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('fairseq_cli.eval_lm_adaptive')


class WordStat(object):
    def __init__(self, word, is_bpe):
        self.word = word
        self.is_bpe = is_bpe
        self.log_prob = 0
        self.next_word_prob = 0
        self.count = 0
        self.missing_next_words = 0

    def add(self, log_prob, next_word_prob):
        """ increments counters for the sum of log probs of current word and next
            word (given context ending at current word). Since the next word might be at the end of the example,
            or it might be not counted because it is not an ending subword unit,
            also keeps track of how many of those we have seen """
        if next_word_prob is not None:
            self.next_word_prob += next_word_prob
        else:
            self.missing_next_words += 1
        self.log_prob += log_prob
        self.count += 1

    def __str__(self):
        return '{}\t{}\t{}\t{}\t{}\t{}'.format(self.word, self.count, self.log_prob, self.is_bpe,
                                               self.next_word_prob, self.count - self.missing_next_words)


class InMemoryDataStore:
    """
    In memory variant of the datastore introduced in the kNN-LM paper (https://arxiv.org/abs/1911.00172).
    It stores key-value pairs, and returns the log-probability of the next token leveraging both
        the predicted next token probabilities from the LM (parametric memory) as well as the next
        token probability from the nearest neighbors storted in the datastore (non-parametric memory).
    """
    def __init__(self, dist_metric) -> None:
        self.k = None
        self.v = None
        assert dist_metric in ["l2", "squared_l2", "cosine"]
        self.dist_metric = dist_metric
    
    def add_item_to_store(self, k: torch.Tensor, v: torch.Tensor) -> None:
        assert len(k.shape) == len(v.shape), f"{k.shape} != {v.shape}"
        if len(k.shape) == 1:
            assert len(v.shape) == 1, v.shape
            k = k.expand_dim(dim=0)
            v = v.expand_dim(dim=0)

        if isinstance(k, np.ndarray):
            k = torch.from_numpy(k)
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)

        if self.dist_metric == "cosine":
            k = torch.nn.functional.normalize(k, p=2.0, dim=1)  # l2 normalize the key
        if self.k is None:
            assert self.v is None
            self.k = k
            self.v = v
        else:
            self.k = torch.cat([self.k, k], dim=0)
            self.v = torch.cat([self.v, v], dim=0)
        print(f"!! New item added in memory store / k: {self.k.shape} / v: {self.v.shape}")

    def get_knn_log_prob(self, q: torch.Tensor, lm_log_probs: torch.Tensor,
                         num_nn: int=1024) -> torch.Tensor:
        if len(q.shape) == 1:
            q = q.expand_dim(dim=0)

        log_prob_vector = torch.full_like(lm_log_probs, -10000)  # since it is log-prob
        if self.k is not None:  # zero probability for every token otherwise
            # self.k shape: B x D
            if self.dist_metric in ["l2", "squared_l2"]:
                dist = ((self.k - q) ** 2).sum(dim=1)
                if self.dist_metric == "l2":
                    dist = torch.sqrt(dist)
            elif self.dist_metric == "cosine":
                q = torch.nn.functional.normalize(q, p=2.0, dim=1)  # l2 normalize the query
                sim = torch.matmul(q, self.k.T)  # (1 x D) x (N x D).T -> 1 x N
                dist = 1.0 - sim  # cosine distance
            else:
                raise RuntimeError(f"Unknown distance metric: {self.dist_metric}")

            # Compute nearest neighbors based on the distance
            dist_idx_sorted = torch.argsort(dist, dim=1, descending=False)
            nearest_neighbors = dist_idx_sorted[:, :num_nn]

            # Select the nearest neighbors
            selected_dists = torch.gather(dist, dim=1, index=nearest_neighbors)
            selected_vals = torch.gather(self.v, dim=1, index=nearest_neighbors)

            # Compute the normalized log-probs (probability is proportional to the negative distance)
            normalized_log_prob = torch.nn.functional.log_softmax(-selected_dists, dim=-1)

            # Compute aggregation of probabilities for the same word
            for idx in range(normalized_log_prob.shape[1]):
                log_prob_vector.scatter_add_(dim=1, index=selected_vals[:, idx:idx+1], src=normalized_log_prob[:, idx:idx+1])

        # return the final prob vector
        return log_prob_vector

    def get_knn_lm_log_prob(self, lm_log_probs: torch.Tensor, lm_features: torch.Tensor,
                            lambda_val: float, num_nn: int=1024) -> torch.Tensor:
        # Get the log probs from the kNN
        knn_log_probs = self.get_knn_log_prob(lm_features, lm_log_probs, num_nn=num_nn)

        # LogSumExp is used since the two probabities are added, which necessitates exponentiation due to log_probs
        combined_log_probs = torch.logsumexp(torch.stack([lm_log_probs + torch.log(1. - lambda_val),
                                                          knn_log_probs + torch.log(lambda_val)]), dim=0)
        return combined_log_probs

    def print_datastore_stats(self) -> None:
        print(f"Datastore stats / Keys: {self.k.shape} / Values: {self.v.shape} / Distance metric: {self.dist_metric}")


def main_adaptive(parsed_args):
    assert parsed_args.path is not None, '--path required for evaluation!'

    utils.import_user_module(parsed_args)

    logger.info(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu

    task = tasks.setup_task(parsed_args)

    # Load ensemble
    logger.info('loading model(s) from {}'.format(parsed_args.path))
    models, args = checkpoint_utils.load_model_ensemble(
        parsed_args.path.split(os.pathsep),
        arg_overrides=eval(parsed_args.model_overrides),
        task=task,
    )

    for arg in vars(parsed_args).keys():
        if arg not in {
            'self_target', 'future_target', 'past_target', 'tokens_per_sample',
            'output_size_dictionary', 'add_bos_token',
        }:
            setattr(args, arg, getattr(parsed_args, arg))

    # reduce tokens per sample by the required context window size
    assert args.use_adaptive_mem, "This script exclusively implements the adaptive memory"
    args.tokens_per_sample -= args.context_window
    task = tasks.setup_task(args)

    # Load dataset splits
    task.load_dataset(args.gen_subset)
    dataset = task.dataset(args.gen_subset)
    if args.context_window > 0:
        dataset = LMContextWindowDataset(
            dataset=dataset,
            tokens_per_sample=args.tokens_per_sample,
            context_window=args.context_window,
            pad_idx=task.source_dictionary.pad(),
        )
    logger.info('{} {} {} examples'.format(args.data, args.gen_subset, len(dataset)))

    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    for model in models:
        model.make_generation_fast_()
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    assert len(models) > 0

    logger.info('num. model params: {}'.format(sum(p.numel() for p in models[0].parameters())))

    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens or 36000,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(*[
            model.max_positions() for model in models
        ]),
        ignore_invalid_inputs=True,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=args.shuffle_dataset)
    if args.shuffle_dataset:
        print("[INFO] Dataset batch shuffling enabled")

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(task.target_dictionary, args.softmax_batch, args=args)

    score_sum = 0.
    count = 0

    if args.remove_bpe is not None:
        if args.remove_bpe == 'sentencepiece':
            raise NotImplementedError
        else:
            bpe_cont = args.remove_bpe.rstrip()
            bpe_toks = {
                i
                for i in range(len(task.source_dictionary))
                if task.source_dictionary[i].endswith(bpe_cont)
            }
        bpe_len = len(bpe_cont)
    else:
        bpe_toks = None
        bpe_len = 0

    word_stats = dict()

    knn_dstore = In_Memory_KNN_Dstore(args)
    if args.existing_datastore_path is not None:
        knn_dstore.load_datastore(args.existing_datastore_path, args.freeze_loaded_memories)

    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for ex_i, sample in enumerate(t):
            if 'net_input' not in sample:
                continue

            sample = utils.move_to_cuda(sample) if use_cuda else sample

            gen_timer.start()
            hypos = scorer.generate(models, sample, knn_dstore=knn_dstore)
            full_pos_scores = torch.cat([x[0]['positional_scores'] for x in hypos], dim=0)  # flattened
            knn_dstore.update_memory_strengths(full_pos_scores)  # update the strength of the kNN memories
            gen_timer.stop(sample['ntokens'])

            key_dtype = np.float16 if args.dstore_fp16 else np.float32
            val_dtype = np.int16 if args.dstore_fp16 else np.int32

            for i, hypos_i in enumerate(hypos):
                hypo = hypos_i[0]
                pos_scores = hypo['positional_scores'].float()

                key = hypo['dstore_keys'].view(-1, args.decoder_embed_dim).cpu().numpy().astype(key_dtype)
                val = hypo['tokens'].view(-1, 1).cpu().numpy().astype(val_dtype)
                if len(key) != len(pos_scores):  # TODO: Validate this change -- pos scores stops early at tgt length
                    assert len(key) > len(pos_scores), f"{len(key)} < {len(pos_scores)}"
                    print(f"[WARNING] dataset key size ({key.shape}) is larger than the positional scores shape: {pos_scores.shape}. Discarding key tokens...")
                    key = key[:len(pos_scores)]
                    val = val[:len(pos_scores)]
                knn_dstore.add_item_to_store(key, val, pos_scores)  # add item to memory -- only relevant ones

                sample_id = sample['id'][i]

                tokens = hypo['tokens']
                tgt_len = tokens.numel()

                if args.add_bos_token:
                    assert hypo['tokens'][0].item() == task.target_dictionary.bos()
                    tokens = tokens[1:]
                    pos_scores = pos_scores[1:]

                skipped_toks = 0
                if bpe_toks is not None:
                    for i in range(tgt_len - 1):
                        if tokens[i].item() in bpe_toks:
                            skipped_toks += 1
                            pos_scores[i + 1] += pos_scores[i]
                            pos_scores[i] = 0

                score_sum += pos_scores.sum().cpu()
                count += pos_scores.numel() - skipped_toks

                if args.output_word_probs or args.output_word_stats:
                    w = ''
                    word_prob = []
                    is_bpe = False
                    for i in range(len(tokens)):
                        w_ind = tokens[i].item()
                        w += task.source_dictionary[w_ind]
                        if bpe_toks is not None and w_ind in bpe_toks:
                            w = w[:-bpe_len]
                            is_bpe = True
                        else:
                            word_prob.append((w, pos_scores[i].item()))

                            next_prob = None
                            ind = i + 1
                            while ind < len(tokens):
                                if pos_scores[ind].item() != 0:
                                    next_prob = pos_scores[ind]
                                    break
                                ind += 1

                            word_stats.setdefault(w, WordStat(w, is_bpe)).add(pos_scores[i].item(), next_prob)
                            is_bpe = False
                            w = ''
                    if args.output_word_probs:
                        logger.info(
                            str(int(sample_id)) + " "
                            + ('\t'.join('{} [{:2f}]'.format(x[0], x[1]) for x in word_prob))
                        )

            wps_meter.update(sample['ntokens'])
            t.log({'wps': round(wps_meter.avg)})

            if ex_i % args.datastore_update_freq == args.datastore_update_freq - 1:
                debug = False
                if debug:
                    print(f"\t [D] Scores before update: {[x[0]['score'] for x in hypos]} / {[x[0]['positional_scores'].float().sum() for x in hypos]}")
                knn_dstore.update_datastore()
                if debug:
                    new_hypos = scorer.generate(models, sample, knn_dstore=knn_dstore)
                    print(f"\t [D] Scores after update: {[x[0]['score'] for x in new_hypos]} / {[x[0]['positional_scores'].float().sum() for x in new_hypos]}")

    if args.save_knnlm_dstore:
        knn_dstore.save_datastore(args.dstore_mmap)

    avg_nll_loss = -score_sum / count / math.log(2)  # convert to base 2
    logger.info('Evaluated {} tokens in {:.1f}s ({:.2f} tokens/s)'.format(
        gen_timer.n, gen_timer.sum, 1. / gen_timer.avg
    ))
    logger.info('Loss (base 2): {:.4f}, Perplexity: {:.2f}'.format(
        avg_nll_loss, 2**avg_nll_loss
    ))

    if args.output_word_stats:
        for ws in sorted(word_stats.values(), key=lambda x: x.count, reverse=True):
            logger.info(ws)
