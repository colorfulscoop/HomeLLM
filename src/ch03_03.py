# === 書籍共通のモジュールインポート === #
from typing import List
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import EarlyStoppingCallback
from transformers import StoppingCriteriaList, StoppingCriteria
import datasets
import peft
import torch
import jinja2
# === 書籍共通のモジュールインポートおわり === #

from ch03_01 import Prompt


def build_dataset(
    dataset_path: str,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    prompt: Prompt,
    shuffle: bool=False,
):
    dataset = datasets.load_dataset(dataset_path, split=split)  # ❶
    
    dataset_input = (
        dataset
        .map(lambda example: {"text": prompt.format(example=example)})  # ❷
        .map(lambda example: tokenizer(
            example["text"],
            max_length=1024,
            truncation=True)
        )  # ❸
        .remove_columns(["turns", "text"])  # ❹
    )
    if shuffle:
        dataset_input = dataset_input.shuffle()  # ❺
    
    return dataset_input

def test_build_dataset():
    pretrained_model = "cyberagent/open-calm-3b"
    tokenizer=AutoTokenizer.from_pretrained(pretrained_model)
    for split in ["train", "validation"]:
        print(f"Split: {split}")
        ds = build_dataset(
            dataset_path="data-shovel",
            split=split,
            tokenizer=AutoTokenizer.from_pretrained(pretrained_model),
            prompt=Prompt(),
        )
        print(ds)

if __name__ == "__main__":
    test_build_dataset()
