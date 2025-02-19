import os
from transformers import Qwen2TokenizerFast
from transformers import AutoTokenizer
from transformers import AddedToken
from transformers import PreTrainedTokenizerFast
from transformers import LlamaTokenizer
import torch, json

TOKEN_TYPE_QWEN = 'qwen'
TOKEN_TYPE_ZH_LLAMA = "zh_llama"

AVAILABLE_TOKEN_TYPES = [TOKEN_TYPE_QWEN, TOKEN_TYPE_ZH_LLAMA]


class Tokenizer:
    def __init__(self, token_type: str = TOKEN_TYPE_ZH_LLAMA):
        super().__init__()
        assert token_type in AVAILABLE_TOKEN_TYPES, 'token type is unavailable'
        self.token_type = token_type

        self.eot_text = '</s>'

        self.pad_text = '<pad>'
        self.unk_text = '<unk>'

        self.user_text = '<user>'
        self.bot_text = '<assistant>'

        self.bor_text = '<reasoning>'
        self.eor_text = '</reasoning>'

        if token_type == TOKEN_TYPE_QWEN:
            self.tokenizer = Qwen2TokenizerFast(
                vocab_file=f"{os.environ['TOKEN_DIR']}qwen_vocab.json",
                merges_file=f"{os.environ['TOKEN_DIR']}qwen_merges.txt",
                unk_token=self.unk_text,
                eos_token=self.eot_text,
                pad_token=self.pad_text
            )
            additional_special_tokens = [
                AddedToken(self.user_text, lstrip=False, rstrip=False),
                AddedToken(self.bot_text, lstrip=False, rstrip=False),
                AddedToken(self.bor_text, lstrip=False, rstrip=False),
                AddedToken(self.eor_text, lstrip=False, rstrip=False)
            ]

            self.tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
        else:
            self.tokenizer = LlamaTokenizer.from_pretrained(os.environ['TOKEN_DIR'])
            # self.tokenizer = AutoTokenizer.from_pretrained(os.environ['TOKEN_DIR'])
            # self.tokenizer = PreTrainedTokenizerFast.from_pretrained(os.environ['TOKEN_DIR'], trust_remote_code=True)

        self.eot = self.tokenizer.convert_tokens_to_ids(self.eot_text)

        self.pad = self.tokenizer.convert_tokens_to_ids(self.pad_text)
        self.unk = self.tokenizer.convert_tokens_to_ids(self.unk_text)

        self.user = self.tokenizer.convert_tokens_to_ids(self.user_text)
        self.bot = self.tokenizer.convert_tokens_to_ids(self.bot_text)

        self.bor = self.tokenizer.convert_tokens_to_ids(self.bor_text)
        self.eor = self.tokenizer.convert_tokens_to_ids(self.eor_text)

        self.vocab_size = len(self.tokenizer)

    def encode_to_token(self, text: str, unsqueeze=False, covert_tensor=False):
        # [x,x,x]
        encoded = self.tokenizer.encode(text, add_special_tokens=False)

        # if self.token_type == TOKEN_TYPE_MISTRAL:
        #     # 处理MISTRAL每句话前面都会增加一个29473的问题
        #     if encoded[0] == 29473:
        #         encoded = encoded[1:]

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
#     tokenizer = Tokenizer(TOKEN_TYPE_ZH_LLAMA)
#     print(tokenizer.vocab_size)
#     print(tokenizer.tokenizer.vocab_size)
#     print(tokenizer.encode_to_token(tokenizer.eot_text), tokenizer.eot)
#     print(tokenizer.encode_to_token(tokenizer.pad_text), tokenizer.pad)
#     print(tokenizer.encode_to_token(tokenizer.unk_text), tokenizer.unk)
#     print(tokenizer.encode_to_token(tokenizer.user_text), tokenizer.user)
#     print(tokenizer.encode_to_token(tokenizer.bot_text), tokenizer.bot)
#     print(tokenizer.encode_to_token(tokenizer.bor_text), tokenizer.bor)
#     print(tokenizer.encode_to_token(tokenizer.eor_text), tokenizer.eor)
#     print(tokenizer.encode_to_token('什么时候'))
#     print(tokenizer.encode_to_token('你好'))
#     print(tokenizer.decode_to_text(torch.tensor(tokenizer.encode_to_token('什么时候'))))


