import json
from pathlib import Path
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from typing import Optional, List, Iterable, Dict, Any

from policy import Policy
from utils.utils import batchify, load_jsonl
from utils.perspective_api import PerspectiveWorker, make_generations_col

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


class Reward:
    def __init__(self, save_path: str, rate_limit: int, batch_size: int):
        self.path = save_path
        self.rate_limit = rate_limit
        self.batch_size = batch_size

    def get_reward(self, prompts: List[str], responses: List[str], epoch: str) -> List[float]:
        perspective_file = Path(self.path) / f'perspective_{epoch}.json'
        perspective = PerspectiveWorker(
            out_file=perspective_file,
            total=len(prompts),
            rate_limit=self.rate_limit
        )
        assert len(prompts) == len(responses), f'prompts({len(prompts)}) and responses({len(responses)}) mismatch'
        for i, r in enumerate(responses):
            perspective(f'generation-{i}', r)

        perspective.stop()
        assert os.path.exists(perspective_file), 'missing perspective file'
        data = pd.DataFrame.from_dict({'prompt': prompts})
        results = collate(data, responses, load_jsonl(perspective_file), os.path.join(self.path, f'reward_{epoch}.json'))
        rewards = [toxicity_to_reward(y['toxicity']) for x in results for y in x]

        return rewards


class DummyReward(Reward):
    def get_reward(self, prompts: List[str], responses: List[str], epoch: str) -> List[float]:
        return [np.random.normal() for x in prompts]


def toxicity_to_reward(score):
    if score is None:
        return None
    return 1 - score


def reward_to_toxicity(score):
    return 1 - score


def collate(dataset: Optional[pd.DataFrame],
            generations: List[str],
            responses: Iterable[Dict[str, Any]],
            output_file: str = ''):
    generations_col_iter = make_generations_col(generations, responses)
    if dataset is None:
        generations_col = list(tqdm(generations_col_iter, total=len(generations), desc='Collating files'))
        dataset = pd.DataFrame(generations_col)
    else:
        assert len(generations) % len(dataset) == 0
        n = len(generations) // len(dataset)
        generations_col = list(tqdm(batchify(generations_col_iter, n), total=len(dataset), desc='Collating files'))
        dataset['generations'] = generations_col

    if output_file:
        dataset.to_json(output_file, orient='records', lines=True)
    return generations_col
