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


def print_next_tokens(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    top_n: int=5,
):
    model_input = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        model_output = model(**model_input)

    target_token_idx = -1
    dist_next_token = torch.softmax(
        model_output.logits[0, target_token_idx],
        dim=-1
    )
    
    input_ids = model_input["input_ids"][0]
    input_tokens = [tokenizer.decode([i]) for i in input_ids]
    context = ", ".join(input_tokens)

    sorted_token_ids = torch.argsort(dist_next_token, descending=True)
    for next_token_id in sorted_token_ids[:top_n]:
        next_token = tokenizer.decode([next_token_id])
        next_prob = dist_next_token[next_token_id]
        print(f"- P({repr(next_token)}| {context}) = {next_prob:.4f}")

def test_print_next_tokens():
    pretrained_model = "cyberagent/open-calm-3b"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model)

    print_next_tokens("言語モデルについて", model, tokenizer, top_n=5)

test_print_next_tokens()
