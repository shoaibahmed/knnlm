# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sys
import numpy as np
import time

from fairseq import utils
from fairseq.data import Dictionary


class SequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, tgt_dict, softmax_batch=None, compute_alignment=False, args=None):
        self.pad = tgt_dict.pad()
        self.eos = tgt_dict.eos()
        self.softmax_batch = softmax_batch or sys.maxsize
        assert self.softmax_batch > 0
        self.compute_alignment = compute_alignment
        self.args = args
        self.last_lambda_vals = None
        self.steps = 0

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample['net_input']

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            bsz, tsz, dim = first.shape
            if bsz * tsz < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff, mask=None):
            combine_probs = torch.stack([vocab_p, knn_p], dim=0)
            coeffs = torch.ones_like(combine_probs)
            if isinstance(coeff, torch.Tensor):
                assert mask is not None
                coeff = coeff.to(combine_probs.dtype)
                coeffs[0][mask] = torch.log(1 - coeff)
                coeffs[1][mask] = torch.log(coeff)
            else:
                coeffs[0] = np.log(1 - coeff)
                coeffs[1] = np.log(coeff)
            curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

            return curr_prob

        orig_target = sample['target']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            attn = decoder_out[1]
            if type(attn) is dict:
                attn = attn.get('attn', None)

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for i, (bd, tgt, is_single) in enumerate(batched):
                sample['target'] = tgt
                curr_prob = model.get_normalized_probs(bd, log_probs=len(models) == 1, sample=sample).data

                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt)
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample['target'] = orig_target

            # Current probs are of shape: [1, BxT, V] where V is the size of the vocabulary
            # Probs are of shape: [1, BxT, 1] since only the probs for the target label are selected
            # Labels are of size [B, T] -- many of them are padded labels
            # Reshaping it converts it to size (B, T)
            probs = probs.view(sample['target'].shape)

            if 'knn_dstore' in kwargs:
                dstore = kwargs['knn_dstore']
                # TxBxC
                queries = bd[1][self.args.knn_keytype]
                if len(models) != 1:
                    raise ValueError('Only knn *log* probs are supported.')

                # Permute the targets (to match the queries), and permute back the outputs from the kNN-LM
                # queries are: TxBxC / targets are: BxT
                yhat_knn_prob = dstore.get_knn_log_prob(
                        queries,
                        orig_target.permute(1, 0),
                        pad_idx=self.pad)
                yhat_knn_prob = yhat_knn_prob.permute(1, 0, 2).squeeze(-1)
                if self.args.fp16:
                    yhat_knn_prob = yhat_knn_prob.half()
                    probs = probs.half()

                mask = None
                self.last_lambda_vals = None
                learnable_lambda = 'lambda_network' in kwargs and kwargs['lambda_network'] is not None
                with torch.set_grad_enabled(learnable_lambda):
                    if learnable_lambda:
                        negative_distance = dstore.get_negative_distance()  # num_tokens x num nearest neighbors
                        if self.steps > self.args.lambda_network_warmup_steps and negative_distance is not None:
                            mask = orig_target != self.pad  # since nearest neighbor values are only saved for non-padded tokens
                            distance = -negative_distance
                            assert len(distance.shape) == 2, distance.shape
                            distance = distance[:, :10]  # top 10 nearest neighbor distances

                            # Use current probs instead of probs which are target prob (current prob shape: [1, BxT, V])
                            # Queries are of shape: TxBxC, so is misaligned with everything else -- permute first
                            reshaped_probs = curr_prob.reshape(*sample['target'].shape, -1).float()
                            lmbda = kwargs['lambda_network'](queries.permute(1, 0, 2).float(), reshaped_probs,
                                                             distance, mask)

                            if kwargs['lambda_network'].optimize_beta:
                                beta = lmbda
                                print(f"!! Learned beta value: {beta}")
                                weights = torch.exp(beta * negative_distance)  # exponential of negative distance is bounded between 0 and 1
                                if self.args.use_max_weight_lmbda:
                                    lmbda = torch.max(weights, axis=1).values  # num_tokens
                                else:
                                    lmbda = torch.mean(weights, axis=1)  # num_tokens
                                print(f"!! Adaptive lambda value ({'max' if self.args.use_max_weight_lmbda else 'mean'}) with learned beta / min: {torch.min(lmbda):.4f} / mean: {torch.mean(lmbda):.4f} / max: {torch.max(lmbda):.4f}")
                                mask = orig_target != self.pad  # since nearest neighbor values are only saved for non-padded tokens
                            else:
                                use_eps = True  # to avoid numerical instability due to eps close to zero (log becomes inf)
                                if use_eps:
                                    eps = 0.001
                                    lmbda = torch.clamp(lmbda + eps, 0., 1.)
                                print(f"!! Learned lambda value / min: {torch.min(lmbda):.4f} / mean: {torch.mean(lmbda):.4f} / max: {torch.max(lmbda):.4f}")
                            self.last_lambda_vals = lmbda.detach()
                        else:  # none for the first round when the datastore is empty
                            lmbda = self.args.lmbda
                            print(f"!! Learned lambda value set to the default lambda value for the warmup phase: {lmbda}")
                    else:
                        if self.args.use_adaptive_lmbda:
                            negative_distance = dstore.get_negative_distance()  # num_tokens x num nearest neighbors
                            if negative_distance is not None:
                                assert self.args.rbf_beta > 0, f"RBF beta should be greater than zero. Found: {self.args.rbf_beta}"
                                weights = torch.exp(self.args.rbf_beta * negative_distance)  # exponential of negative distance is bounded between 0 and 1
                                if self.args.use_max_weight_lmbda:
                                    lmbda = torch.max(weights, axis=1).values  # num_tokens
                                else:
                                    lmbda = torch.mean(weights, axis=1)  # num_tokens
                                print(f"!! Adaptive lambda value ({'max' if self.args.use_max_weight_lmbda else 'mean'}) (beta={self.args.rbf_beta}) / min: {torch.min(lmbda):.4f} / mean: {torch.mean(lmbda):.4f} / max: {torch.max(lmbda):.4f}")
                                mask = orig_target != self.pad  # since nearest neighbor values are only saved for non-padded tokens
                                self.last_lambda_vals = lmbda.detach()
                            else:  # none for the first round when the datastore is empty
                                lmbda = self.args.lmbda
                                print(f"!! Adaptive lambda value ({'max' if self.args.use_max_weight_lmbda else 'mean'}) set to the default lambda value as distance is none: {lmbda}")
                        else:
                            lmbda = self.args.lmbda

                    # Combine the LM and kNN probs
                    probs = combine_knn_and_vocab_probs(
                                yhat_knn_prob, probs, lmbda, mask=mask)

                    if learnable_lambda and self.last_lambda_vals is not None:  # Update the lambda network
                        if self.steps > self.args.lambda_network_warmup_steps and \
                            self.steps % self.args.lambda_network_update_steps == self.args.lambda_network_update_steps - 1:
                            # TODO: integrate learnable lambda inference mode
                            kwargs['lambda_network'].update_model(probs)
                        probs = probs.detach()

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None and torch.is_tensor(attn):
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample['start_indices'] if 'start_indices' in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = utils.strip_pad(sample['target'][i, start_idxs[i]:], self.pad) \
                if sample['target'] is not None else None
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i]:start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample['net_input']['src_tokens'][i],
                        sample['target'][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            hypos.append([{
                'tokens': ref,
                'score': score_i,
                'attention': avg_attn_i,
                'alignment': alignment,
                'positional_scores': avg_probs_i,
                'dstore_keys': decoder_out[1][self.args.knn_keytype][start_idxs[i]:,i,:] if self.args.save_knnlm_dstore else None,
            }])
        self.steps += 1
        return hypos
