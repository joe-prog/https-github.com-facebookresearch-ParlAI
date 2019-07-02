#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.dict import DictionaryAgent


class BertSpanDictionaryAgent(DictionaryAgent):

    def __init__(self, opt):
        super().__init__(opt)

    def _find_label_index_in_text(self, text, label):
        return text.find(label)

    def spantokenize(self, text, labels):

        # get the first label
        label = labels[0]

        char_start_position = self._find_label_index_in_text(text, label)

        if char_start_position >= 0:
            # label found in text

            assert text[char_start_position:char_start_position + len(label)] == label

            # 1. Token words before the answer
            tokens = self.tokenize(text[:char_start_position])
            start_position = len(tokens)

            # 2. Token words for the answer
            answer_tokens = self.tokenize(
                text[char_start_position:char_start_position + len(label)]
            )
            end_position = start_position + len(answer_tokens)
            tokens.extend(answer_tokens)

            # 3. Token words after the answer
            tokens.extend(
                self.tokenize(text[char_start_position + len(label):])
            )

            assert tokens[start_position:end_position] == answer_tokens

            tokens_id = [self[tok]
                         for tok in [self.start_token] + tokens + [self.end_token]
                        ]
            start_position += 1
            end_position += 1
            valid = True
        else:
            start_position = 0
            end_position = 0
            tokens_id = [self[self.start_token]] + self.txt2vec(text) + [self[self.end_token]]
            valid = False

        return tokens_id, start_position, end_position, valid
