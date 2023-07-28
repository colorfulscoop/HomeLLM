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
from ch03_02 import Generator


def test_generate_chat_lora():
    model_path = "trained_model-shovel/model-best"

    config = peft.PeftConfig.from_pretrained(model_path)  # ❶
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path
    )  # ❷
    model = peft.PeftModel.from_pretrained(model, model_path)  # ❸
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    generator = Generator(tokenizer=tokenizer, model=model)
    
    prompt = Prompt()
    example = {
        "turns": [
            {"user": "ユーザ", "text": "こんにちは！最近暑いよね"},
            {"user": "アシスタント",
             "text": "最近はとっても暑いよね。キミはどんな夏を過ごしてるのかなぁ♪"},
            {"user": "ユーザ", "text": "今年の夏は、海に行ってみようかな！"},
            {"user": "アシスタント", "text": ""},
        ]
    }
    input_text = prompt.format(example=example, end="")
    gen_str = generator.generate(input_text, temperature=0.8)

    print(gen_str)

if __name__ == "__main__":
    test_generate_chat_lora()
