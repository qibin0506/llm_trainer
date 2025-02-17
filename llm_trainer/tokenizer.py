import os
from transformers import Qwen2TokenizerFast
from transformers import AutoTokenizer
from transformers import AddedToken
import torch

TOKEN_TYPE_QWEN = 'qwen'
TOKEN_TYPE_YI = "yi"

AVAILABLE_TOKEN_TYPES = [TOKEN_TYPE_QWEN, TOKEN_TYPE_YI]


class Tokenizer:
    def __init__(self, token_type: str = TOKEN_TYPE_YI):
        super().__init__()

        assert token_type in AVAILABLE_TOKEN_TYPES, 'token type is unavailable'

        self.eot_text = '<|endoftext|>'

        self.pad_text = '<|pad|>'
        self.unk_text = '<unk>'

        self.user_text = '<|user|>'
        self.bot_text = '<|assistant|>'

        self.bor_text = '<|beginofreasoning|>'
        self.eor_text = '<|endofreasoning|>'

        additional_special_tokens = []

        if token_type == TOKEN_TYPE_QWEN:
            self.tokenizer = Qwen2TokenizerFast(
                vocab_file=f"{os.environ['TOKEN_DIR']}qwen_vocab.json",
                merges_file=f"{os.environ['TOKEN_DIR']}qwen_merges.txt",
                unk_token=self.unk_text,
                eos_token=self.eot_text,
                pad_token=self.pad_text
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(os.environ['TOKEN_DIR'])
            additional_special_tokens.append(AddedToken(self.pad_text, lstrip=False, rstrip=False))

        additional_special_tokens.append(AddedToken(self.user_text, lstrip=False, rstrip=False))
        additional_special_tokens.append(AddedToken(self.bot_text, lstrip=False, rstrip=False))
        additional_special_tokens.append(AddedToken(self.bor_text, lstrip=False, rstrip=False))
        additional_special_tokens.append(AddedToken(self.eor_text, lstrip=False, rstrip=False))

        self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

        self.eot = self.tokenizer.convert_tokens_to_ids(self.eot_text)

        self.pad = self.tokenizer.convert_tokens_to_ids(self.pad_text)
        self.unk = self.tokenizer.convert_tokens_to_ids(self.unk_text)

        self.user = self.tokenizer.convert_tokens_to_ids(self.user_text)
        self.bot = self.tokenizer.convert_tokens_to_ids(self.bot_text)

        self.bor = self.tokenizer.convert_tokens_to_ids(self.bor_text)
        self.eor = self.tokenizer.convert_tokens_to_ids(self.eor_text)

        self.vocab_size = len(self.tokenizer)

    def encode_to_token(self, text: str, unsqueeze=True, covert_tensor=True):
        # [x,x,x]
        encoded = self.tokenizer.encode(text, add_special_tokens=False)

        if unsqueeze:
            # tensor: [[x,x,x]]
            return torch.tensor(encoded).long().unsqueeze(0)
        else:
            # tensor: # [x,x,x]
            if covert_tensor:
                return torch.tensor(encoded).long()

            return encoded

    def decode_to_text(self, token: torch.Tensor) -> str:
        return self.tokenizer.decode(token.squeeze(0))


# if __name__ == '__main__':
#     tokenizer = Tokenizer(TOKEN_TYPE_YI)
#     print(tokenizer.vocab_size)
#     print(tokenizer.encode_to_token(tokenizer.eot_text), tokenizer.eot)
#     print(tokenizer.encode_to_token(tokenizer.pad_text), tokenizer.pad)
#     print(tokenizer.encode_to_token(tokenizer.unk_text), tokenizer.unk)
#     print(tokenizer.encode_to_token(tokenizer.user_text), tokenizer.user)
#     print(tokenizer.encode_to_token(tokenizer.bot_text), tokenizer.bot)
#     print(tokenizer.encode_to_token(tokenizer.bor_text), tokenizer.bor)
#     print(tokenizer.encode_to_token(tokenizer.eor_text), tokenizer.eor)


