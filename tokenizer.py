import os
import torch
from typing import List, Dict, Union
from nano_tokenizer import NanoTokenizer

class Tokenizer:
    def __init__(self):
        self.tokenizer = NanoTokenizer(os.environ['TOKEN_DIR'])

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

        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        # [x,x,x]
        return self.tokenizer.encode(text)

    def batch_encode(self, text: List[str], padding = False) -> List[List[int]]:
        pad_id = self.pad if self.pad is not None else 0
        return self.tokenizer.batch_encode(
            text, padding, False, pad_id, self.text_pad
        )

    def decode(self, token: Union[torch.Tensor, List[int]]) -> str:
        if isinstance(token, torch.Tensor):
            token = token.view(-1).cpu().tolist()

        return self.tokenizer.decode(token)

    def batch_decode(self, tokens: Union[torch.Tensor, List[int], List[List[int]], List[torch.Tensor]]) -> List[str]:
        if isinstance(tokens, torch.Tensor):
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            tokens = tokens.cpu().tolist()
        elif isinstance(tokens, list) and len(tokens) > 0:
            if isinstance(tokens[0], torch.Tensor):
                tokens = [t.view(-1).cpu().tolist() for t in tokens]
            elif not isinstance(tokens[0], list):
                tokens = [tokens]

        return self.tokenizer.batch_decode(tokens)

    def apply_chat_template(
            self,
            conversations: List[Dict[str, str]],
            tokenizer: bool = True,
            add_answer_tag_for_assistant: bool = True
    ) -> Union[str, List[int]]:
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
            return self.encode(chat_template)

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

