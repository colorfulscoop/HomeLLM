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


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    temperature: float=1.0,
    top_p=1.0,
    seed=0
) -> str:
    torch.manual_seed(seed)  # ❶

    model_input = tokenizer(text, return_tensors="pt")
    output = model.generate(  # ❷
        **model_input,
        do_sample=True, 
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=50,
    )    
    output_str = tokenizer.decode(output[0])
    
    return output_str

def test_generate():
    pretrained_model = "cyberagent/open-calm-3b"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model)
    
    text = "技術書の執筆には、"
    
    print("=== Top-p ===")
    for top_p in [0.1, 0.9, 1.0]:
        gen_str = generate(model, tokenizer, text, top_p=top_p)
        print(f"[Top-p={top_p}]", gen_str)
    print()    

    print("=== 温度 ===")
    for temperature in [0.3, 0.7, 1.3]:
        gen_str = generate(model, tokenizer, text, temperature=temperature)
        print(f"[Temperature={temperature}]", gen_str)

test_generate()
