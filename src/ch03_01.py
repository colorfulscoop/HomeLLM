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


# ❶
TEMPLATE = """会話
{%- for turn in example["turns"] %}
- {{ turn.user }}: {{ turn.text }}
{%- endfor %}
"""

# ❷
@dataclass
class Prompt:
    template: str = TEMPLATE

    def format(self, end="\n", *args, **argv):
        tpl = jinja2.Template(self.template)
        output = tpl.render(*args, **argv)
        output += end
        return output

def test_prompt():
    prompt = Prompt()

    # ❸
    example = {
        "turns": [
            {"user": "ユーザ", "text": "こんにちは! 最近暑いよね"},
            {"user": "アシスタント", "text": ""},
        ]
    }
    prompt_out = prompt.format(
        example=example,
        end=""  # ❹
    )
    print(prompt_out)

if __name__ == "__main__":
    test_prompt()
