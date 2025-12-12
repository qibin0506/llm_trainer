import os
import warnings
from typing import List, Dict, Union
from transformers import AutoTokenizer
import torch


class Tokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(os.environ['TOKEN_DIR'])

        self.text_end = '</s>'

        self.text_pad = '<pad>'
        self.text_unk = '<unk>'

        self.text_user = '<user>'
        self.text_assistant = '<assistant>'

        self.text_think_start = '<think>'
        self.text_think_end = '</think>'

        self.text_answer_start = '<answer>'
        self.text_answer_end = '</answer>'

        self.text_system = '<system>'

        self.text_image = '<image>'

        self.end = self.tokenizer.convert_tokens_to_ids(self.text_end)

        self.pad = self.tokenizer.convert_tokens_to_ids(self.text_pad)
        self.unk = self.tokenizer.convert_tokens_to_ids(self.text_unk)

        self.user = self.tokenizer.convert_tokens_to_ids(self.text_user)
        self.assistant = self.tokenizer.convert_tokens_to_ids(self.text_assistant)

        self.think_start = self.tokenizer.convert_tokens_to_ids(self.text_think_start)
        self.think_end = self.tokenizer.convert_tokens_to_ids(self.text_think_end)

        self.answer_start = self.tokenizer.convert_tokens_to_ids(self.text_answer_start)
        self.answer_end = self.tokenizer.convert_tokens_to_ids(self.text_answer_end)

        self.system = self.tokenizer.convert_tokens_to_ids(self.text_system)
        self.image = self.tokenizer.convert_tokens_to_ids(self.text_image)

        self.vocab_size = len(self.tokenizer)

    def encode(
            self,
            text: str,
            unsqueeze: bool = False,
            covert_tensor: bool = False
    ) -> Union[torch.Tensor, List[int]]:
        # [x,x,x]
        encoded = self.tokenizer.encode(text, add_special_tokens=False)

        if unsqueeze:
            # tensor: [[x,x,x]]
            return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        else:
            # tensor: # [x,x,x]
            if covert_tensor:
                return torch.tensor(encoded, dtype=torch.long)

            return encoded

    def batch_encode(
            self,
            text: List[str],
            padding = False,
            truncation = False,
            covert_tensor: bool = False,
            return_attention_mask: bool = False
    ) -> Union[torch.Tensor, List[List[int]]]:
        encoded = self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            return_attention_mask=return_attention_mask
        )['input_ids']

        if covert_tensor:
            encoded = torch.tensor(encoded, dtype=torch.long)

        return encoded

    def decode(
            self,
            token: Union[torch.Tensor, List[int]],
            skip_special_tokens: bool = False
    ) -> str:
        return self.tokenizer.decode(token, skip_special_tokens=skip_special_tokens)

    def batch_decode(
            self,
            tokens: Union[torch.Tensor, List[int], List[List[int]]],
            skip_special_tokens: bool = False
    ) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=skip_special_tokens)

    def encode_to_token(self, text: str, unsqueeze=True, covert_tensor=True):
        warnings.warn('encode_to_token is deprecated. Please use `encode` instead.')
        return self.encode(text, unsqueeze, covert_tensor)

    def decode_to_text(self, token: torch.Tensor, skip_special_tokens: bool = False) -> str:
        warnings.warn('decode_to_text is deprecated. Please use `decode` instead.')
        return self.decode(token.squeeze(0), skip_special_tokens)

    def apply_chat_template(
            self,
            conversations: List[Dict[str, str]],
            tokenizer: bool = True,
            add_answer_tag_for_assistant: bool = True,
            unsqueeze=False,
            covert_tensor=False
    ):
        """
            [
                {"role":"system", "content":"system prompt"},
                {"role":"user", "content":"hello?"},
                {"role":"assistant", "content":"hello"},
                {"role":"user", "content":"hello hello?"},
                {"role":"assistant", "think":"thinking", "content":"hello hello"},
            ]
            <system>{system_prompt}</s><user>hello?</s><assistant>hello</s><user>hello hello?</s><assistant><think>thinking</think><answer>hello hello</answer></s>
        """

        chat_template = ''
        support_roles = {'system': self.text_system, 'user': self.text_user, 'assistant': self.text_assistant}
        for conversation in conversations:
            role = conversation['role']
            if role in support_roles:
                content = conversation['content']
                if add_answer_tag_for_assistant and role == 'assistant':
                    content = f"{self.text_answer_start}{content}{self.text_answer_end}"

                if 'think' in conversation:
                    content = f"{self.text_think_start}{conversation['think']}{self.text_think_end}{content}"

                chat_template = f"{chat_template}{support_roles[role]}{content}{self.text_end}"

        if tokenizer:
            return self.encode(chat_template, unsqueeze, covert_tensor)

        return chat_template

    def get_special_tokens_dict(self):
        return {
            self.text_end: self.end,
            self.text_pad: self.pad,
            self.text_unk: self.unk,
            self.text_user: self.user,
            self.text_assistant: self.assistant,
            self.text_think_start: self.think_start,
            self.text_think_end: self.think_end,
            self.text_answer_start: self.answer_start,
            self.text_answer_end: self.answer_end,
            self.text_system: self.system,
            self.text_image: self.image,
        }

