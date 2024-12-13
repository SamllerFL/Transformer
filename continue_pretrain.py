import os, sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(f"{ROOT_PATH}/train")
sys.path.append(f"{ROOT_PATH}/train/preprocess/dataset")

from typing import List, Optional, Union, Any, Dict, Mapping
from transformers import (
    set_seed,
    GPT2Config,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    AutoTokenizer,
    GPT2LMHeadModel,
)

import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from functools import partial
import argparse
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils import PreTrainedTokenizerBase
from dataclasses import dataclass


def build_train_valid_test_dataset():
    pass


class CustomTrainer(Trainer):
    """
    用于覆盖默认的方法，自定义自己的训练方法
    """

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=args.num_train_epochs, eta_min=1e-5
        )
        self._created_lr_scheduler = True


@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        内部训练数据输入的转换，按照transformer的规则转换数据
        """
        # Handle dict or lists with proper padding and conversion to tensor.
        convert_examples = [{"input_ids": e["text"]} for e in examples]
        if isinstance(convert_examples[0], Mapping):
            batch = self.tokenizer.pad(
                convert_examples,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch


def load_tokenizer() -> PreTrainedTokenizer:
    """
    根据的词表路径，导入tokenizer，并进行一些初始化动作
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.vocab_dir, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        # tokenizer.pad_token = "<pad>"
        # tokenizer.pad_token_id = len(tokenizer)
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        args.vocab_size += 1
    return tokenizer


def init_trainer():
    """
    初始化Trainer
    """
    # 1. 导入tokenizer
    tokenizer = load_tokenizer()

    # 2. 初始化模型
    model = GPT2LMHeadModel(GPT2Config(**args))
    model.resize_token_embeddings(len(tokenizer))

    # 3. 创建dataset
    train_dataset, valid_dataset, _ = build_train_valid_test_dataset(**args)
    return CustomTrainer(
        model=model,
        args=TrainingArguments(**args),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )


def do_train():
    """
    执行训练
    """
    # 设置seed保证可重新复现
    set_seed(0)

    # 训练
    trainer = init_trainer()
    print("*** Train ***")
    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    # 训练指标输出
    metrics = train_result.metrics
    max_train_samples = (
        args.max_train_samples
        if args.max_train_samples is not None
        else len(trainer.train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(trainer.train_dataset))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.compute_loss

    # 保存模型
    print("*** Save model ***")
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")


def parse_args():
    """
    解析传入的参数
    """
    parser = argparse.ArgumentParser(description="LLM trainer")
    parser.add_argument(
        "--data_config",
        "-d",
        type=str,
        required=False,
        default=f"{ROOT_PATH}/train/training/config/pretrain_args/test_data_config.yaml",
        help="LLM data config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    do_train()
