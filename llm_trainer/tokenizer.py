from typing import List, Dict, Union
import os

import torch
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

        self.text_tool_call_start = '<tool_call>'
        self.text_tool_call_end = '</tool_call>'

        self.text_tool_response_start = '<tool_response>'
        self.text_tool_response_end = '</tool_response>'

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

        self.tool_call_start = self.tokenizer.convert_tokens_to_ids(self.text_tool_call_start)
        self.tool_call_end = self.tokenizer.convert_tokens_to_ids(self.text_tool_call_end)

        self.tool_response_start = self.tokenizer.convert_tokens_to_ids(self.text_tool_response_start)
        self.tool_response_end = self.tokenizer.convert_tokens_to_ids(self.text_tool_response_end)

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
                {"role":"assistant", "tool_call":'{"name": "calc", "arguments": {"expr": "1+1"}}'},
                {"role":"tool", "content":'{"result": 2}'},
                {"role":"assistant", "think":"thinking", "content":"hello hello"},
            ]
            <system>{system_prompt}</s><user>hello?</s><assistant><tool_call>...</tool_call></s><user><tool_response>...</tool_response></s><assistant><think>thinking</think><answer>hello hello</answer></s>
        """

        chat_template = ''
        support_roles = {
            'system': self.text_system,
            'user': self.text_user,
            'assistant': self.text_assistant,
            'tool': self.text_user,
            'tool_response': self.text_user
        }
        for conversation in conversations:
            role = conversation['role']
            if role in support_roles:
                content = conversation.get('content', '')
                
                # 处理 tool / tool_response 角色 (放入 <user><tool_response>...</tool_response></s>)
                if role in ('tool', 'tool_response'):
                    if not content.startswith(self.text_tool_response_start):
                        content = f"{self.text_tool_response_start}\n{content}\n{self.text_tool_response_end}"
                    chat_template = f"{chat_template}{support_roles[role]}{content}{self.text_end}"
                    continue

                # 处理 assistant 角色
                if role == 'assistant':
                    # 1. 优先处理工具调用 tool_call
                    if 'tool_call' in conversation:
                        tc_content = conversation['tool_call']
                        if not tc_content.startswith(self.text_tool_call_start):
                            tc_content = f"{self.text_tool_call_start}\n{tc_content}\n{self.text_tool_call_end}"
                        body = f"{content}\n{tc_content}".strip() if content else tc_content
                    else:
                        # 2. 普通内容生成，判断是否需要加 <answer>
                        if (
                            add_answer_tag_for_assistant
                            and content
                            and not content.startswith(self.text_tool_call_start)
                            and not content.startswith(self.text_answer_start)
                        ):
                            body = f"{self.text_answer_start}{content}{self.text_answer_end}"
                        else:
                            body = content

                    if 'think' in conversation and conversation['think']:
                        body = f"{self.text_think_start}{conversation['think']}{self.text_think_end}{body}"

                    chat_template = f"{chat_template}{support_roles[role]}{body}{self.text_end}"
                else:
                    # system / user 角色
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
            self.text_tool_call_start: self.tool_call_start,
            self.text_tool_call_end: self.tool_call_end,
            self.text_tool_response_start: self.tool_response_start,
            self.text_tool_response_end: self.tool_response_end,
        }

