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


def print_sent_prob(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    add_bos_token: bool=True
):
    if add_bos_token:
        text = tokenizer.bos_token + text  # ❶
    
    model_input = tokenizer(text, return_tensors="pt")  # ❷
    print("Model input:", model_input)

    with torch.no_grad():  # ❸
        model_output = model(**model_input)
    print("Output shape:", model_output.logits.shape)

    input_ids = model_input["input_ids"][0]
    input_tokens = [tokenizer.decode([i]) for i in input_ids]
    for idx, token_id in enumerate(input_ids[:-1]):
        dist_next_token = torch.softmax(
            model_output.logits[0, idx],
            dim=-1
        )  # ❹
        
        next_token_id = input_ids[idx+1]
        next_prob = dist_next_token[next_token_id]
        next_token = input_tokens[idx+1]
        context = ", ".join(input_tokens[:idx+1])    
        print(f"- P({next_token} | {context}) = {next_prob:.8f}")

def test_print_sent_prob():
    pretrained_model = "cyberagent/open-calm-3b"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model)

    print_sent_prob(
        text="言語モデルについて説明します。",
        model=model,
        tokenizer=tokenizer,
        add_bos_token=True,
    )

test_print_sent_prob()
