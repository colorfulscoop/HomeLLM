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


def test_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-3b")

    tokenizer_out = tokenizer("言語モデルについて説明します。")  # ❶
    print("Tokenizer output:", tokenizer_out)

    input_ids = tokenizer_out["input_ids"]

    decode_out = tokenizer.decode(input_ids)  # ❷
    print("Decode output:", decode_out)

    decode_out_each_id = [tokenizer.decode([i]) for i in input_ids]  # ❸
    print("Decode output for each id:", decode_out_each_id)

test_tokenizer()
