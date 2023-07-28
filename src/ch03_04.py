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
from ch03_03 import build_dataset


def train(
    prompt: Prompt,
    pretrained_model_path: str,
    dataset_path: str,
    train_args: TrainingArguments,
    lora_config: peft.LoraConfig,
):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_path)

    # ❶
    model = peft.get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # データセットをロード
    train_dataset = build_dataset(
        dataset_path=dataset_path,
        split="train",
        tokenizer=tokenizer,
        prompt=prompt,
        shuffle=True,
    )
    valid_dataset = build_dataset(
        dataset_path=dataset_path,
        split="validation",
        tokenizer=tokenizer,
        prompt=prompt,
        shuffle=False,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )  # ❷
    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        eval_dataset=valid_dataset,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2)  # ❸
        ]
    )
    trainer.train()

    # モデルを保存
    best_model_path = f"{train_args.output_dir}/model-best"
    trainer.save_model(best_model_path)
    

def main_train():
    pretrained_model = "cyberagent/open-calm-3b"
    dataset_path = "data-shovel"
    train_args = TrainingArguments(
        output_dir="trained_model-shovel",
        logging_strategy="epoch",
        # ❹
        num_train_epochs=20,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        gradient_accumulation_steps=1,
        # ❺
        load_best_model_at_end=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        # ❻
        fp16=True,
        fp16_full_eval=True,
    )
    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )  # ❼
    prompt = Prompt()
    train(
        prompt=prompt,
        pretrained_model_path=pretrained_model,
        dataset_path=dataset_path,
        train_args=train_args,
        lora_config=lora_config,
    )

if __name__ == "__main__":
    main_train()
