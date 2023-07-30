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


class WordStoppingCriteria(StoppingCriteria):  # ❶
    def __init__(self, ids: List[int]):
        self._ids = ids

    def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor,
            **kwargs
        ) -> bool:
        return all(
            input_ids[0][-1-i] == self._ids[-1-i]
            for i in range(len(self._ids))
        )  # ❷

@dataclass
class Generator:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    def generate(
        self,
        text: str,
        temperature: float=1.0,
        top_p: float=1.0,
        seed: int=0,
        max_new_tokens: int=30,
    ) -> str:
        torch.manual_seed(seed)
        
        model_input = self.tokenizer(text, return_tensors="pt")

        stopping_criteria_list = StoppingCriteriaList(
            [WordStoppingCriteria(self.tokenizer.encode("\n"))]
        )  # ❸

        output = self.model.generate(
            **model_input,
            do_sample=True, 
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria_list,
        )
        output_str = self.tokenizer.decode(output[0])

        return output_str

def test_generate_chat():
    pretrained_model = "cyberagent/open-calm-3b"
    generator = Generator(
        tokenizer=AutoTokenizer.from_pretrained(pretrained_model),
        model=AutoModelForCausalLM.from_pretrained(pretrained_model),
    )
    prompt = Prompt()
    example = {
        "turns": [
            {"user": "ユーザ", "text": "こんにちは！最近暑いよね"},
            {"user": "アシスタント", "text": ""},
        ]
    }
    input_text = prompt.format(example=example, end="")
    gen_str = generator.generate(input_text, temperature=0.8)
    print(gen_str)

if __name__ == "__main__":
    test_generate_chat()
