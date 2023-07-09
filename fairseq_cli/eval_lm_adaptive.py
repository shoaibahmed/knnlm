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
from fairseq.knnlm import KNN_Dstore
from .eval_lm import WordStat


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('fairseq_cli.eval_lm_adaptive')


class InMemoryRetrievalStore:
    def __init__(self, dist_metric) -> None:
        self.k = None
        self.v = None
        assert dist_metric in ["euclidean", "cosine"]
        self.dist_metric = dist_metric
    
    def add_item_to_store(self, k, v):
        assert len(k.shape) == 1, k.shape
        assert len(v.shape) == 1, v.shape

        if self.dist_metric == "cosine":
            k = torch.nn.functional.normalize(k, p=2.0, dim=0)  # l2 normalize the key
        if self.k is None:
            assert self.v is None
            self.k = k.expand_dim(0)  # 1 x d
            self.v = v.expand_dim(0)  # 1 x d'
        else:
            self.k = torch.concatenate([self.k, k], dim=0)
            self.v = torch.concatenate([self.v, v], dim=0)
        print(f"!! New item added in memory store / k: {k.shape} / v: {v.shape}")

    def get_knn_vals(self, q, k):
        assert isinstance(k, int), k
        assert len(q.shape) == 1, q.shape
        q = q.expand_dim(dim=0)

        if self.k is None:
            return []
        elif self.k.shape < k:
            return self.v  # Return all vals
        else:
            # self.k shape: B x D
            if self.dist_metric == "euclidean":
                dist = torch.sqrt((self.k - q) ** 2).sum(dim=1)
            elif self.dist_metric == "cosine":
                q = torch.nn.functional.normalize(q, p=2.0, dim=1)  # l2 normalize the query
                sim = torch.matmul(q, self.k.T)  # (1 x D) x (N x D).T -> 1 x N
                dist = 1.0 - sim  # cosine distance
            else:
                raise RuntimeError(f"Unknown distance metric: {self.dist_metric}")

            # retrieve values based on the keys
            # TODO: the probability should be aggregated for the same word
            prob = torch.exp(-dist)
            raise NotImplementedError


def main_adaptive(parsed_args):
    assert args.use_adaptive_mem, "This script exclusively implements the adaptive memory"
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
    ).next_epoch_itr(shuffle=False)

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

    if args.knnlm and args.save_knnlm_dstore:
        raise ValueError("Cannot use knnlm while trying to build the datastore!")

    if args.knnlm:
        knn_dstore = KNN_Dstore(args)

    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()

        if args.save_knnlm_dstore:
            print('keytype being saved:', args.knn_keytype)
            if args.dstore_fp16:
                print('Saving fp16')
                dstore_keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float16, mode='w+', shape=(args.dstore_size, args.decoder_embed_dim))
                dstore_vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int16, mode='w+', shape=(args.dstore_size, 1))
            else:
                print('Saving fp32')
                dstore_keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float32, mode='w+', shape=(args.dstore_size, args.decoder_embed_dim))
                dstore_vals = np.memmap(args.dstore_mmap+'_vals.npy', dtype=np.int, mode='w+', shape=(args.dstore_size, 1))

        dstore_idx = 0
        for ex_i, sample in enumerate(t):
            if 'net_input' not in sample:
                continue

            sample = utils.move_to_cuda(sample) if use_cuda else sample

            gen_timer.start()
            if args.knnlm:
                hypos = scorer.generate(models, sample, knn_dstore=knn_dstore)
            else:
                hypos = scorer.generate(models, sample)
            gen_timer.stop(sample['ntokens'])

            key_dtype = np.float16 if args.dstore_fp16 else np.float32
            val_dtype = np.int16 if args.dstore_fp16 else np.int

            for i, hypos_i in enumerate(hypos):
                hypo = hypos_i[0]
                if args.save_knnlm_dstore:
                    shape = hypo['dstore_keys'].shape
                    if shape[0] == args.tokens_per_sample:
                        if args.use_adaptive_mem:
                            raise NotImplementedError
                            # TODO: Decide based on log-prob whether this sequence needs to be stored or not
                            # TODO: One very important detail -- the log prob should take into account the previously generated memory -- continuous updates

                        if dstore_idx + shape[0] > args.dstore_size:
                            shape = [args.dstore_size - dstore_idx]
                            hypo['dstore_keys'] = hypo['dstore_keys'][:shape[0]]

                        dstore_keys[dstore_idx:shape[0]+dstore_idx] = hypo['dstore_keys'].view(-1, args.decoder_embed_dim).cpu().numpy().astype(key_dtype)
                        dstore_vals[dstore_idx:shape[0]+dstore_idx] = hypo['tokens'].view(-1, 1).cpu().numpy().astype(val_dtype)

                        dstore_idx += shape[0]
                    else:
                        print('Skipping this one with shape', shape)

                sample_id = sample['id'][i]

                tokens = hypo['tokens']
                tgt_len = tokens.numel()
                pos_scores = hypo['positional_scores'].float()

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

    if args.save_knnlm_dstore:
        print("dstore_idx", dstore_idx, "final shape", shape)
        print("Keys", dstore_keys.shape, dstore_keys.dtype)
        print("Vals", dstore_vals.shape, dstore_vals.dtype)

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
