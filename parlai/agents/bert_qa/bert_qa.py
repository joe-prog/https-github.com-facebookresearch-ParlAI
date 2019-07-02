#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.torch_agent import TorchAgent, Output
from .bert_span_dictionary import BertSpanDictionaryAgent
from parlai.core.utils import round_sigfigs
from parlai.zoo.bert.build import download
from parlai.core.utils import neginf
from parlai.agents.transformer.transformer import TransformerRankerAgent
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer.modules import get_n_positions_from_options
from parlai.core.distributed_utils import is_distributed
import os
import torch
import collections

DEBUGMODE=False

_PrelimPrediction = collections.namedtuple(
    "PrelimPrediction", ["start_index", "end_index", "start_logit", "end_logit"]
)

_NbestPrediction = collections.namedtuple(
    "NbestPrediction", ["text", "start_logit", "end_logit"]
)

class BertQaAgent(TorchAgent):
    """ QA based on Bert
    """

    def __init__(self, opt, shared=None):
        init_model, is_finetune = self._get_init_model(opt, shared)
        super().__init__(opt, shared)

        if shared:
            self.model = shared['model']
            self.metrics = shared['metrics']
            states = None
        else:
            # Note: we cannot change the type of metrics ahead of time, so you
            # should correctly initialize to floats or ints here
            self.metrics = {
                'loss': 0.0,
                'examples': 0,
                'rank': 0.0,
                'mrr': 0.0,
                'train_accuracy': 0.0,
            }
            self.build_model()
            if self.fp16:
                self.model = self.model.half()
            if init_model:
                print('Loading existing model parameters from ' + init_model)
                states = self.load(init_model)
            else:
                states = {}
        if self.use_cuda:
            self.model.cuda()

        # set up model and optimizers
        if shared:
            # We don't use get here because hasattr is used on optimizer later.
            if 'optimizer' in shared:
                self.optimizer = shared['optimizer']
        else:
            optim_params = [p for p in self.model.parameters() if p.requires_grad]
            self.init_optim(
                optim_params, states.get('optimizer'), states.get('optimizer_type')
            )
            self.build_lr_scheduler(states, hard_reset=is_finetune)

        if shared is None and is_distributed():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.opt['gpu']], broadcast_buffers=False
            )

    def _split_context_question(self, text):
        split = text.split("\n")
        question = split[-1]
        context = "\n".join(split[0:-1])
        return context, question

    def to_device(self, vec):
        if self.use_cuda:
            return vec.cuda()
        return vec

    def _vectorize_no_device(self, observations):
        """Convert a list of observations into input tensors for the BertForQuestionAnswering model ."""

        b_tokens_ids = []
        b_start_position = []
        b_end_position = []
        b_valid_obs = []
        b_segment_ids = []

        for obs in observations:

            context, question = self._split_context_question(obs["text"])

            question_tokens_id = (
                [self.START_IDX]
                + self.dict.txt2vec(question)
                + [self.END_IDX]
            )

            segment_ids = [0] * len(question_tokens_id)

            if "labels" in obs:
                # train
                context_tokens_ids, context_start_position, context_end_position, valid_obs = self.dict.spantokenize(
                    context, obs["labels"]
                )
            else:
                # valid / test
                valid_obs = True
                context_tokens_ids = [self.START_IDX] + self.dict.txt2vec(context) + [self.END_IDX]
                context_start_position = 0
                context_end_position = len(context_tokens_ids)

            segment_ids += [1] * len(context_tokens_ids)

            start_position = context_start_position + len(question_tokens_id)
            end_position = context_end_position + len(question_tokens_id)
            tokens_ids = question_tokens_id + context_tokens_ids

            if self.text_truncate > 0:
                if len(tokens_ids) > self.text_truncate:
                    original_len = len(tokens_ids)
                    tokens_ids = tokens_ids[-self.text_truncate:]
                    segment_ids = segment_ids[-self.text_truncate:]

                    num_tokens_removed = original_len - len(tokens_ids)

                    if start_position <= num_tokens_removed:
                        # answer truncated - invalid data point
                        valid_obs = False
                        start_position = 0
                        end_position = 0
                    else:
                        start_position -= num_tokens_removed
                        end_position -= num_tokens_removed

            b_tokens_ids.append(tokens_ids)
            b_start_position.append(start_position)
            b_end_position.append(end_position)
            b_valid_obs.append(valid_obs)
            b_segment_ids.append(segment_ids)

        max_tokens_length = max([len(tokens_id) for tokens_id in b_tokens_ids])

        b_input_mask = []

        for tokens_ids, valid_obs, segment_ids in zip(
            b_tokens_ids, b_valid_obs, b_segment_ids
        ):

            # if valid_obs:
            #     # The mask has 1 for real tokens and 0 for padding tokens. Only real
            #     # tokens are attended to.
            input_mask = [1] * len(tokens_ids)
            # else:
            #     # invalid data point
            #     input_mask = [0] * len(tokens_ids)

            # Zero-pad up to the sequence length.
            while len(tokens_ids) < max_tokens_length:
                tokens_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            b_input_mask.append(input_mask)

            assert len(tokens_ids) == max_tokens_length
            assert len(input_mask) == max_tokens_length
            assert len(segment_ids) == max_tokens_length
        return (
            b_tokens_ids,
            b_segment_ids,
            b_input_mask,
            b_start_position,
            b_end_position,
        )

    def _vectorize(self, observations):
        """Convert a list of observations into input tensors for the BertForQuestionAnswering model ."""
        vecs = self._vectorize_no_device(observations)
        return(
            self.to_device(torch.tensor(vecs[0], dtype=torch.long)),
            self.to_device(torch.tensor(vecs[1], dtype=torch.long)),
            self.to_device(torch.tensor(vecs[2], dtype=torch.long)),
            self.to_device(torch.tensor(vecs[3], dtype=torch.long)),
            self.to_device(torch.tensor(vecs[4], dtype=torch.long)),
        )


    def _get_best_indexes(self, logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def _get_prediction(
        self, start_logits, end_logits, input_ids, start_context_position=None
    ):

        start_indexes = self._get_best_indexes(start_logits, self.opt["n_best_size"])
        end_indexes = self._get_best_indexes(end_logits, self.opt["n_best_size"])


        prelim_predictions = []

        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions
                if end_index < start_index:
                    continue
                if start_context_position and start_index < start_context_position:
                    # answer in the question
                    continue
                length = end_index - start_index + 1
                if length > self.text_truncate:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=start_logits[start_index].item(),
                        end_logit=end_logits[end_index].item(),
                    )
                )

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True,
        )

        for pred in prelim_predictions:
            if pred.start_index > 0:  # this is a non-null prediction
                return self.dict.vec2txt(
                    input_ids[pred.start_index:pred.end_index]
                )

        return ""

    def report(self):
        base = super().report()
        m = {}
        examples = self.metrics["examples"]
        if examples > 0:
            m["examples"] = examples
            m["loss"] = self.metrics["loss"]
            m["mean_loss"] = self.metrics["loss"] / examples

        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            base[k] = round_sigfigs(v, 4)
        return base

    def get_output_from_logits(self, b_input_ids, b_start_logits, b_end_logits):
        predictions = []
        b_input_ids = b_input_ids.cpu()
        b_start_logits = b_start_logits.float().cpu()
        b_end_logits = b_end_logits.float().cpu()
        for start_logits, end_logits, input_ids in zip(
            b_start_logits, b_end_logits, b_input_ids
        ):
            prediction = self._get_prediction(start_logits, end_logits, input_ids)
            predictions.append(prediction)

        return Output(predictions)

    def train_step(self, batch):

        self.optimizer.zero_grad()
        self.model.train()

        (b_tokens_ids, b_segment_ids,b_mask,
         b_start_context_position, b_end_context_position) = self._vectorize(batch["observations"])
        b_logits = self.model(
            b_tokens_ids, b_segment_ids
        )
        neg_mask = (b_mask == 0).type_as(b_logits).unsqueeze(2)
        neg_mask *= neginf(b_logits.dtype)
        b_logits = neg_mask + b_logits

        loss_fn = torch.nn.CrossEntropyLoss()
        logits_start = b_logits[:, :, 0]
        logits_end = b_logits[:, :, 1]
        loss_start = loss_fn(logits_start, b_start_context_position)
        loss_end = loss_fn(logits_end, b_end_context_position)
        loss = (loss_start + loss_end) / 2
        self.metrics["examples"] += len(batch["observations"])
        self.metrics["loss"] += loss
        self.backward(loss)
        self.update_params()
        out = self.get_output_from_logits(b_tokens_ids, logits_start, logits_end)
        return out

    def eval_step(self, batch):
        self.model.eval()
        b_tokens_ids, b_segment_ids, _, b_start_context_position, _ = self._vectorize(
            batch["observations"]
        )
        with torch.no_grad():
            b_logits = self.model(
                b_tokens_ids, b_segment_ids
            )
        b_start_logits = b_logits[:, :, 0]
        b_end_logits = b_logits[:, :, 1]
        return self.get_output_from_logits(b_tokens_ids, b_start_logits, b_end_logits)

    def share(self):
        """Share model parameters."""
        shared = super().share()
        shared["model"] = self.model
        shared["metrics"] = self.metrics
        return shared

    def reset_metrics(self):
        """Reset metrics."""
        super().reset_metrics()
        self.metrics["examples"] = 0
        self.metrics["loss"] = 0


    @staticmethod
    def add_cmdline_args(parser):
        TorchAgent.add_cmdline_args(parser)
        TransformerRankerAgent.add_cmdline_args(parser)
        parser = parser.add_argument_group("BERT QA Arguments")
        parser.add_argument(
            "--n-best-size",
            default=16,
            type=int,
            help="The total number of n-best predictions to generate in the nbest_predictions.json "
            "output file.",
        )
        parser.set_defaults(dict_maxexs=0)  # skip building dictionary

    @staticmethod
    def dictionary_class():
        return BertSpanDictionaryAgent

    def build_model(self):
        self.model = BertQAModule(self.opt, self.dict, self.NULL_IDX)


class BertQAModule(torch.nn.Module):
    """ A simple wrapper around the transformer encoder which adds 2 linear
        layers, which can be used as logits for the start and end position.
    """

    def __init__(self, opt, dict, null_idx):
        super(BertQAModule, self).__init__()
        n_positions = get_n_positions_from_options(opt)
        embeddings = torch.nn.Embedding(
            len(dict), opt['embedding_size'], padding_idx=null_idx
        )
        torch.nn.init.normal_(embeddings.weight, 0, opt['embedding_size'] ** -0.5)
        self.encoder = TransformerEncoder(
            n_heads=opt['n_heads'],
            n_layers=opt['n_layers'],
            embedding_size=opt['embedding_size'],
            ffn_size=opt['ffn_size'],
            vocabulary_size=len(dict),
            embedding=embeddings,
            dropout=opt['dropout'],
            attention_dropout=opt['attention_dropout'],
            relu_dropout=opt['relu_dropout'],
            padding_idx=null_idx,
            learn_positional_embeddings=opt['learn_positional_embeddings'],
            embeddings_scale=opt['embeddings_scale'],
            reduction_type='none',
            n_positions=n_positions,
            n_segments=2,
            activation=opt['activation'],
            variant=opt['variant'],
            output_scaling=opt['output_scaling'],
        )
        self.linear_layer = torch.nn.Linear(opt['embedding_size'], 2)

    def forward(self, tokens, segments):
        """ Scores each concatenation text + candidate.
            By default the output of masked elements is
            -infinity
        """
        if DEBUGMODE:
            vec =  tokens.new_full((tokens.size(0), tokens.size(1), 2),
                                   0.5,
                                   dtype=torch.half)
            vec.requires_grad = True
            return vec
        encoded, mask = self.encoder(tokens, None, segments)
        neg_mask = (mask == 0).type_as(encoded).unsqueeze(2)
        neg_mask *= neginf(encoded.dtype)
        res = self.linear_layer(encoded) + neg_mask
        return res
