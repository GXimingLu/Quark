from typing import List
from copy import deepcopy


class DataPool:
    def __init__(self, tree_tokens, n_extra_tokens):
        self.tree_tokens = tree_tokens
        self.n_extra_tokens = n_extra_tokens

        self.cat_tokens = None
        self.prompt_pool, self.response_pool, self.score_pool = [], [], []

    def add(self, prompts: List[str], responses: List[str], scores: List[float]):
        self.prompt_pool.extend(prompts)
        self.response_pool.extend(responses)
        self.score_pool.extend(scores)

        data = zip(self.prompt_pool, self.response_pool, self.score_pool)
        data = [x for x in data if x[-1] is not None]
        sorted_data = sorted(data, key=lambda x: x[-1], reverse=True)
        self.prompt_pool, self.response_pool, self.score_pool = [list(x) for x in list(zip(*sorted_data))]

        cat_pos = [[i] * (len(sorted_data) // self.n_extra_tokens) for i in range(self.n_extra_tokens)]
        cat_pos = [y for x in cat_pos for y in x]
        cat_pos = cat_pos + [self.n_extra_tokens - 1] * (len(sorted_data) - len(cat_pos))
        self.cat_tokens = [self.tree_tokens[i] for i in cat_pos]

    def get_data(self):
        return deepcopy(self.prompt_pool), deepcopy(self.response_pool), deepcopy(self.cat_tokens)

