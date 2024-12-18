import os
from transformers import BertTokenizerFast
from transformers import Qwen2TokenizerFast
from transformers import AddedToken
import torch

TOKEN_TYPE_BERT = 0
TOKEN_TYPE_QWEN = 1

AVAILABLE_TOKEN_TYPES = [TOKEN_TYPE_BERT, TOKEN_TYPE_QWEN]


class Tokenizer:
    def __init__(self, token_type: int = TOKEN_TYPE_QWEN):
        super().__init__()

        assert token_type in AVAILABLE_TOKEN_TYPES, 'token type is unavailable'

        self.eot_text = '[SEP]'
        self.pad_text = '[PAD]'
        self.unk_text = '[UNK]'
        self.user_text = '[USER]'
        self.bot_text = '[BOT]'

        if token_type == TOKEN_TYPE_BERT:
            self.tokenizer = BertTokenizerFast(f"{os.environ['TOKEN_DIR']}bert_vocab.txt")
            self.eot = self.tokenizer.sep_token_id
            self.pad = self.tokenizer.pad_token_id
            self.unk = self.tokenizer.unk_token_id
        elif token_type == TOKEN_TYPE_QWEN:
            self.tokenizer = Qwen2TokenizerFast(
                vocab_file=f"{os.environ['TOKEN_DIR']}qwen_vocab.json",
                merges_file=f"{os.environ['TOKEN_DIR']}qwen_merges.txt",
                unk_token=self.unk_text,
                eos_token=self.eot_text,
                pad_token=self.pad_text)
            self.eot = self.tokenizer.eos_token_id
            self.pad = self.tokenizer.pad_token_id
            self.unk = self.tokenizer.unk_token_id

        self.user_token = AddedToken(self.user_text, lstrip=False, rstrip=False)
        self.bot_token = AddedToken(self.bot_text, lstrip=False, rstrip=False)

        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.user_token, self.bot_token]})
        self.vocab_size = len(self.tokenizer)

    def encode_to_token(self, text: str, unsqueeze=True, covert_tensor=True):
        # [1, 2]
        encoded = self.tokenizer.encode(text, add_special_tokens=False)

        if unsqueeze:
            # tensor: [[1, 2]]
            return torch.tensor(encoded).long().unsqueeze(0)
        else:
            # tensor: [1, 2]
            if covert_tensor:
                return torch.tensor(encoded).long()

            return encoded

    def decode_to_text(self, token: torch.Tensor) -> str:
        return self.tokenizer.decode(token.squeeze(0))


# if __name__ == '__main__':
#    tokenizer = Tokenizer(TOKEN_TYPE_QWEN)
#    print(tokenizer.vocab_size)
#
#    eot = tokenizer.eot
#    pad = tokenizer.pad
#    unk = tokenizer.unk
#
#    print(eot, pad, unk)
#    print(tokenizer.decode_to_text(torch.tensor([eot])),
#          tokenizer.decode_to_text(torch.tensor([pad])),
#          tokenizer.decode_to_text(torch.tensor([unk])))
#
#    print(tokenizer.encode_to_token(' '))
#    print(tokenizer.encode_to_token('[USER]'))
#    print(tokenizer.encode_to_token('[BOT]'))


