**The dataset supports npy, pkl, and jsonl formats, with npy being recommended.**

**For dataset formats regarding different training stages, please refer to the code in the mock_xxx functions below.**

**WARN: If the jsonl format is used during pre-training, the entire file will be loaded into memory during dataset initialization.**


``` python
import random
import string

from utils import *
init_env()

from llm_trainer.dataset import *

def rs():
    chars = string.ascii_letters
    length = random.randint(1, 5)
    return ''.join(random.choices(chars, k=length))


def mock_pretrain_data():
    data = [f'hello{i}' for i in range(20)]

    raw_data = [f"{json.dumps({'text': i})}\n" for i in data]
    with open('./data/pretrain.jsonl', 'w') as f:
        f.writelines(raw_data)

    tokens = []
    for item in data:
        tokens.extend(TrainerTools().tokenizer.encode(item['text']))

    with open("./data/pretrain.pkl", 'wb') as f:
        pickle.dump(tokens, f)

    token_array = np.array(tokens, dtype=np.uint32)
    np.save('./data/pretrain.npy', token_array)


def mock_sft_data():
    data = []
    raw_data = []
    for i in range(20):
        t = [
            {'role': 'system', 'content': f'{rs()}{i}'},
            {'role': 'user', 'content': f'{rs()}{i}'},
            {'role': 'assistant', 'think': f'{rs()}{i}', 'content': f'{rs()}{i}'},
        ]
        raw_data.append(f'{json.dumps(t)}\n')
        data.append(TrainerTools().tokenizer.apply_chat_template(t))

    with open('./data/sft.jsonl', 'w') as f:
        f.writelines(raw_data)

    with open(f"./data/sft.pkl", 'wb') as f:
        pickle.dump(data, f)

    token_array = np.array(data, dtype=object)
    np.save('./data/sft.npy', token_array)


def mock_dpo_data():
    data = []
    raw_data = []
    for i in range(20):
        chosen = [
            {'role': 'system', 'content': f'{rs()}{i}'},
            {'role': 'user', 'content': f'{rs()}{i}'},
            {'role': 'assistant', 'think': f'{rs()}{i}', 'content': f'{rs()}{i}'},
        ]

        reject = [
            {'role': 'system', 'content': f'{rs()}{i}'},
            {'role': 'user', 'content': f'{rs()}{i}'},
            {'role': 'assistant', 'think': f'{rs()}{i}', 'content': f'{rs()}{i}'},
        ]

        chosen_t = TrainerTools().tokenizer.apply_chat_template(chosen)
        reject_t = TrainerTools().tokenizer.apply_chat_template(reject)

        item = {'chosen': chosen, 'rejected': reject}
        raw_data.append(f'{json.dumps(item)}\n')
        data.append({'chosen': chosen_t, 'rejected': reject_t})

    with open('./data/dpo.jsonl', 'w') as f:
        f.writelines(raw_data)

    with open(f"./data/dpo.pkl", 'wb') as f:
        pickle.dump(data, f)

    token_array = np.array(data, dtype=object)
    np.save('./data/dpo.npy', token_array)


def mock_rl_data(need_answer):
    data = []
    raw_data = []
    for i in range(20):
        prompt = [
            {'role': 'system', 'content': f'{rs()}{i}'},
            {'role': 'user', 'content': f'{rs()}{i}'},
            {'role': 'assistant', 'think': f'{rs()}{i}', 'content': f'{rs()}{i}'},
        ]

        if need_answer:
            answer = f'{i}'
        else:
            answer = None

        raw_item = {'prompt': prompt}
        if answer:
            raw_item['answer'] = answer

        item = {'prompt': TrainerTools().tokenizer.apply_chat_template(prompt)}
        if answer:
            item['answer'] = TrainerTools().tokenizer.encode(answer)

        raw_data.append(f'{json.dumps(raw_item)}\n')
        data.append(item)

    with open(f'./data/rl_answer_{need_answer}.jsonl', 'w') as f:
        f.writelines(raw_data)

    with open(f"./data/rl_answer_{need_answer}.pkl", 'wb') as f:
        pickle.dump(data, f)

    token_array = np.array(data, dtype=object)
    np.save(f'./data/rl_answer_{need_answer}.npy', token_array)


def test_pretrain_data():
    dataset = PretrainDataset('./data/pretrain.jsonl', 2, 2)
    for item in dataset:
        print(TrainerTools().tokenizer.decode(item), end='')

    print("\n============")

    dataset = PretrainDataset('./data/pretrain.pkl', 2, 2)
    for item in dataset:
        print(TrainerTools().tokenizer.decode(item), end='')

    print("\n============")

    dataset = PretrainDataset('./data/pretrain.npy', 2, 2)
    for item in dataset:
        print(TrainerTools().tokenizer.decode(item), end='')



def test_sft_data():
    dataset = SFTDataset('./data/sft.jsonl', 512)
    for item in dataset:
        print(TrainerTools().tokenizer.decode(item['inputs']))

    print("\n============")

    dataset = SFTDataset('./data/sft.pkl', 512)
    for item in dataset:
        print(TrainerTools().tokenizer.decode(item['inputs']))

    print("\n============")

    dataset = SFTDataset('./data/sft.npy', 512)
    for item in dataset:
        print(TrainerTools().tokenizer.decode(item['inputs']))


def test_dpo_data():
    dataset = DPODataset('./data/dpo.jsonl', 512)
    for item in dataset:
        print(TrainerTools().tokenizer.decode(item['chosen']))
        print(TrainerTools().tokenizer.decode(item['rejected']))

    print("\n============")

    dataset = DPODataset('./data/dpo.pkl', 512)
    for item in dataset:
        print(TrainerTools().tokenizer.decode(item['chosen']))
        print(TrainerTools().tokenizer.decode(item['rejected']))

    print("\n============")

    dataset = DPODataset('./data/dpo.npy', 512)
    for item in dataset:
        print(TrainerTools().tokenizer.decode(item['chosen']))
        print(TrainerTools().tokenizer.decode(item['rejected']))


def test_grpo_data():
    dataset = RLDataset('./data/rl_answer_True.jsonl')
    for item in dataset:
        print(TrainerTools().tokenizer.decode(item['prompt']))
        print(TrainerTools().tokenizer.decode(item['answer']))

    print("\n============")

    dataset = RLDataset('./data/rl_answer_True.pkl')
    for item in dataset:
        print(TrainerTools().tokenizer.decode(item['prompt']))
        print(TrainerTools().tokenizer.decode(item['answer']))

    print("\n============")

    dataset = RLDataset('./data/rl_answer_True.npy')
    for item in dataset:
        print(TrainerTools().tokenizer.decode(item['prompt']))


def test_ppo_data():
    dataset = RLDataset('./data/rl_answer_False.jsonl')
    for item in dataset:
        print(TrainerTools().tokenizer.decode(item['prompt']))

    print("\n============")

    dataset = RLDataset('./data/rl_answer_False.pkl')
    for item in dataset:
        print(TrainerTools().tokenizer.decode(item['prompt']))

    print("\n============")

    dataset = RLDataset('./data/rl_answer_False.npy')
    for item in dataset:
        print(TrainerTools().tokenizer.decode(item['prompt']))


mock_pretrain_data()
mock_sft_data()
mock_dpo_data()
mock_rl_data(True)
mock_rl_data(False)

test_pretrain_data()
test_sft_data()
test_dpo_data()
test_grpo_data()
test_ppo_data()
```
