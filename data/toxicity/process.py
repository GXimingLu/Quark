import random
import json
random.seed(42)


data = open('raw/prompts.jsonl', 'r').readlines()
print(f'total data: {len(data)}\n')

test = open('raw/nontoxic_prompts-10k.jsonl', 'r').readlines()
print(f'total test: {len(test)}')

train_val = [x for x in data if x not in test]
val_num = 5000
random.shuffle(train_val)
val = train_val[:val_num]
train = train_val[val_num:]
print(f'total val: {len(val)}')
print(f'total train: {len(train)}')

for split, name in zip([train, val, test], ['train', 'val', 'test']):
    with open(f'{name}.jsonl', 'w') as f:
        for s in split:
            f.write(s)