#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.utils import PaddingUtils
import os
import torch
import collections

try:
    from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
    from pytorch_pretrained_bert.tokenization import BertTokenizer
    from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
    from pytorch_pretrained_bert.optimization import BertAdam
except ImportError:
    raise Exception(("BERT QA needs pytorch-pretrained-BERT installed. \n "
                     "pip install pytorch-pretrained-bert"))


class BertQaAgent(Agent):
    """
    QA based on Hugging Face BERT implementation.
    """

    def __init__(self, opt, shared=None):

        super().__init__(opt, shared)

        if not shared:

            self.tokenizer = BertTokenizer.from_pretrained(
                opt["bert_model"], do_lower_case=opt["do_lower_case"])

            # Prepare model
            self.model = BertForQuestionAnswering.from_pretrained(
                opt["bert_model"],
                cache_dir=os.path.join(
                    str(PYTORCH_PRETRAINED_BERT_CACHE),
                    "distributed_{}".format(opt["local_rank"]),
                ),
            )

        else:
            self.tokenizer = shared["tokenizer"]
            self.model = shared["model"]

        if opt["local_rank"] == -1:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(opt["local_rank"])
            self.device = torch.device("cuda", opt["local_rank"])
            self.n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")

        if opt["fp16"]:
            self.model.half()
        self.model.to(self.device)

        if opt["local_rank"] != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
                )

            self.model = DDP(self.model)
        elif self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Prepare optimizer
        if opt["do_train"]:
            param_optimizer = list(self.model.named_parameters())

            # hack to remove pooler, which is not used
            # thus it produce None grad that break apex
            param_optimizer = [
                n for n in param_optimizer if "pooler" not in n[0]
            ]

            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                    0.01,
                },
                {
                    "params": [
                        p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                    0.0,
                },
            ]

            if opt["fp16"]:
                try:
                    from apex.optimizers import FP16_Optimizer
                    from apex.optimizers import FusedAdam
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
                    )

                self.optimizer = FusedAdam(
                    optimizer_grouped_parameters,
                    lr=opt["learning_rate"],
                    bias_correction=False,
                    max_grad_norm=1.0,
                )
                if opt["loss_scale"] == 0:
                    self.optimizer = FP16_Optimizer(optimizer,
                                                    dynamic_loss_scale=True)
                else:
                    self.optimizer = FP16_Optimizer(
                        optimizer, static_loss_scale=opt["loss_scale"])
                warmup_linear = WarmupLinearSchedule(
                    warmup=opt["warmup_proportion"])
                # t_total=num_train_optimization_steps)
            else:
                self.optimizer = BertAdam(
                    optimizer_grouped_parameters,
                    lr=opt["learning_rate"],
                    warmup=opt["warmup_proportion"],
                )
                # t_total=num_train_optimization_steps)

    def share(self):
        """Share internal states."""
        shared = super().share()
        shared["tokenizer"] = self.tokenizer
        shared["model"] = self.model
        return shared

    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group("BertQaAgent Arguments")
        agent.add_argument(
            "--bert_model",
            default=None,
            type=str,
            required=True,
            help=
            "Bert pre-trained model selected in the list: bert-base-uncased, "
            "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
            "bert-base-multilingual-cased, bert-base-chinese.",
        )
        agent.add_argument(
            "--do_lower_case",
            action="store_true",
            help=
            "Whether to lower case the input text. True for uncased models, False for cased models.",
        )
        agent.add_argument(
            "--local_rank",
            type=int,
            default=-1,
            help="local_rank for distributed training on gpus",
        )
        agent.add_argument(
            "--max_query_length",
            default=256,
            type=int,
            help=
            "The maximum number of tokens for the question. Questions longer than this will "
            "be truncated to this length.",
        )
        agent.add_argument(
            "--n_best_size",
            default=20,
            type=int,
            help=
            "The total number of n-best predictions to generate in the nbest_predictions.json "
            "output file.",
        )

        agent.add_argument(
            "--max_answer_length",
            default=30,
            type=int,
            help=
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another.",
        )

        agent.add_argument(
            "--fp16",
            action="store_true",
            help="Whether to use 16-bit float precision instead of 32-bit",
        )

        agent.add_argument(
            "--learning_rate",
            default=5e-5,
            type=float,
            help="The initial learning rate for Adam.",
        )
        agent.add_argument("--do_train",
                           action="store_true",
                           help="Whether to run training.")
        agent.add_argument(
            "--warmup_proportion",
            default=0.1,
            type=float,
            help=
            "Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
            "of training.",
        )
        agent.add_argument(
            "--loss_scale",
            type=float,
            default=0,
            help=
            "Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
            "0 (default value): dynamic loss scaling.\n"
            "Positive power of 2: static loss scaling value.\n",
        )
        BertQaAgent.dictionary_class().add_cmdline_args(argparser)
        return agent

    def observe(self, observation):
        # each observation is an episode
        self.observation = observation
        self.episode_done = observation["episode_done"]
        return observation

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def batch_act(self, observations):

        batchsize = len(observations)

        # initialize a table of replies with this agent's id
        batch_reply = [{"id": self.getID()} for _ in range(batchsize)]

        # convert the observations into batches of inputs and targets tensors
        input_ids, segment_ids, input_mask, start_positions, end_positions, tokens, valid_inds, is_training = self.vectorize(
            observations)

        if input_ids is None or len(input_ids) == 0:
            # no valid examples, just return empty responses
            return batch_reply

        predictions = self.predict(
            input_ids,
            segment_ids,
            input_mask,
            start_positions,
            end_positions,
            tokens,
            is_training,
        )

        # maps returns predictions back to the right `valid_inds`
        # in the example above, a prediction `world` should reply to `hello`
        c = 0
        for idx, x in enumerate(batch_reply):
            if valid_inds[idx]:
                x["text"] = predictions[c]
                c += 1

        # print(batch_reply)

        return batch_reply

    def _find_answer_index_in_context(self, text, answer):
        return text.find(answer)

    def vectorize(self, observations):
        """Convert a list of observations into input & target tensors."""

        all_input_ids = []
        all_segment_ids = []
        all_input_mask = []
        all_start_positions = []
        all_end_positions = []
        all_tokens = []
        valid_inds = []

        is_training = any(("labels" in obs for obs in observations))

        features = []
        for (index, observation) in enumerate(observations):

            start_position = None
            end_position = None

            if is_training:

                text = observation["text"]

                # get the first label
                label = observation["labels"][0]

                char_start_position = self._find_answer_index_in_context(
                    text, label)

                assert (text[char_start_position:char_start_position +
                             len(label)] == label)

                # 1. Token words before the answer
                query_tokens = self.tokenizer.tokenize(
                    text[:char_start_position])
                start_position = len(query_tokens)

                # 2. Token words for the answer
                answer_tokens = self.tokenizer.tokenize(
                    text[char_start_position:char_start_position + len(label)])
                end_position = start_position + len(answer_tokens)
                query_tokens.extend(answer_tokens)

                # 3. Token words after the answer
                query_tokens.extend(
                    self.tokenizer.tokenize(text[char_start_position +
                                                 len(label):]))

                assert query_tokens[start_position:
                                    end_position] == answer_tokens

                if False:
                    # print debug info
                    print(f"text:{text}")
                    print(f"label:{label}")
                    print(f"char_start_position:{char_start_position}")
                    print(text[char_start_position:char_start_position +
                               len(label)])
                    print(answer_tokens)
                    print(f"start_position:{start_position}")
                    print(f"end_position:{end_position}")
                    print(query_tokens[start_position:end_position])
                    print(query_tokens)

            else:
                query_tokens = self.tokenizer.tokenize(observation["text"])

            if (len(query_tokens) + 2 >
                    self.opt["max_query_length"]):  # 2 = [CLS] and [SEP]
                # exclude example
                valid_inds.append(0)
                continue

            segment_ids = []
            tokens = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < self.opt["max_query_length"]:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            error_msg = ""
            error_msg += f"input_ids: {input_ids}\n"
            error_msg += f"max_query_length: {self.opt['max_query_length']}\n"
            error_msg += f"len(input_ids): {len(input_ids)}\n"

            assert len(input_ids) == self.opt["max_query_length"], error_msg
            assert len(input_mask) == self.opt["max_query_length"], error_msg
            assert len(segment_ids) == self.opt["max_query_length"], error_msg

            all_input_ids.append(input_ids)
            all_segment_ids.append(segment_ids)
            all_input_mask.append(input_mask)
            all_start_positions.append(start_position)
            all_end_positions.append(end_position)
            all_tokens.append(tokens)
            valid_inds.append(1)

        return (
            torch.tensor(all_input_ids, dtype=torch.long),
            torch.tensor(all_segment_ids, dtype=torch.long),
            torch.tensor(all_input_mask, dtype=torch.long),
            torch.tensor(all_start_positions, dtype=torch.long),
            torch.tensor(all_end_positions, dtype=torch.long),
            all_tokens,
            valid_inds,
            is_training,
        )

    def _get_best_indexes(self, logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits),
                                 key=lambda x: x[1],
                                 reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def _get_prediction(self, start_logits, end_logits, tokens):

        start_indexes = self._get_best_indexes(start_logits,
                                               self.opt["n_best_size"])
        end_indexes = self._get_best_indexes(end_logits,
                                             self.opt["n_best_size"])

        _PrelimPrediction = collections.namedtuple(
            "PrelimPrediction",
            ["start_index", "end_index", "start_logit", "end_logit"])
        prelim_predictions = []

        for start_index in start_indexes:
            for end_index in end_indexes:
                # We could hypothetically create invalid predictions
                if end_index < start_index:
                    continue
                length = end_index - start_index + 1
                if length > self.opt["max_answer_length"]:
                    continue
                prelim_predictions.append(
                    _PrelimPrediction(
                        start_index=start_index,
                        end_index=end_index,
                        start_logit=start_logits[start_index],
                        end_logit=end_logits[end_index],
                    ))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True,
        )

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= self.opt["n_best_size"]:
                break

            if pred.start_index > 0:  # this is a non-null prediction
                answer_tokens = tokens[pred.start_index:pred.end_index]
                answer_text = " ".join(answer_tokens)

                # De-tokenize WordPieces that have been split off.
                answer_text = answer_text.replace(" ##", "")
                answer_text = answer_text.replace("##", "")

                # Clean whitespace
                answer_text = answer_text.strip()
                answer_text = " ".join(answer_text.split())

                nbest.append(
                    _NbestPrediction(
                        text=answer_text,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit,
                    ))

                # print(f"{pred.start_logit+pred.end_logit} - {answer_text}")

        # return the best prediciton
        return nbest[0].text

    def predict(
            self,
            input_ids,
            segment_ids,
            input_mask,
            start_positions,
            end_positions,
            list_tokens,
            is_training,
    ):

        if is_training:
            self.model.train()
            loss = self.model(input_ids, segment_ids, input_mask,
                              start_positions, end_positions)

            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if self.opt["fp16"]:
                self.optimizer.backward(loss)
            else:
                loss.backward()

            if self.opt["fp16"]:
                # modify learning rate with special warm up BERT uses
                # if fp16 is False, BertAdam is used and handles this automatically
                lr_this_step = opt["learning_rate"] * warmup_linear.get_lr(
                    global_step, self.opt["warmup_proportion"])
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr_this_step
            self.optimizer.step()
            self.optimizer.zero_grad()


        # predictions
        with torch.no_grad():
            batch_start_logits, batch_end_logits = self.model(
                input_ids, segment_ids, input_mask)

        predictions = []
        for start_logits, end_logits, tokens in zip(batch_start_logits,
                                                    batch_end_logits,
                                                    list_tokens):
            prediction = self._get_prediction(start_logits, end_logits, tokens)
            predictions.append(prediction)

        return predictions
