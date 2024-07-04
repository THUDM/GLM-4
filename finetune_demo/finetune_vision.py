# -*- coding: utf-8 -*-
import os
import jieba
import dataclasses as dc
import functools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Union
import numpy as np
import ruamel.yaml as yaml
import torch
import typer
from datasets import Dataset, Split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import PeftConfig, get_peft_config, get_peft_model
from rouge_chinese import Rouge
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    GenerationConfig,
    PreTrainedTokenizer,
    Seq2SeqTrainingArguments,
)
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer
from datasets import load_dataset, DatasetDict, NamedSplit
from typing import Optional
from PIL import Image

app = typer.Typer(pretty_exceptions_show_locals=False)


class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        output_ids = ([feature['output_ids'] for feature in features] if 'output_ids' in features[0].keys() else None)
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                        (
                                max_output_length + self.pad_to_multiple_of - 1) //
                        self.pad_to_multiple_of * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                        max_output_length - len(feature['output_ids'])
                )
                if isinstance(feature['output_ids'], list):
                    feature['output_ids'] = feature['output_ids'] + remainder
                else:
                    feature['output_ids'] = np.concatenate(
                        [feature['output_ids'], remainder]
                    ).astype(np.int64)
        return super().__call__(features, return_tensors)


class Seq2SeqTrainer(_Seq2SeqTrainer):
    # Not Support for apex
    def training_step(self, model: nn.Module, inputs: dict[str, Any]) -> torch.Tensor:

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)
        detached_loss = loss.detach() / self.args.gradient_accumulation_steps
        del inputs
        torch.cuda.empty_cache()
        return detached_loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict,
            prediction_loss_only: bool,
            ignore_keys=None,
            **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        with torch.no_grad():
            if self.args.predict_with_generate:
                output_ids = inputs.pop('output_ids', None)
            loss, generated_tokens, labels = super().prediction_step(
                model=model,
                inputs=inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                **gen_kwargs
            )

            if generated_tokens is not None:
                generated_tokens = generated_tokens[:, inputs["input_ids"].size()[1]:]

            if self.args.predict_with_generate:
                labels = output_ids

            del inputs, output_ids
            torch.cuda.empty_cache()

        return loss, generated_tokens, labels


@dc.dataclass
class DataConfig(object):
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    num_proc: Optional[int] = None

    @property
    def data_format(self) -> str:
        return Path(self.train_file).suffix

    @property
    def data_files(self) -> dict[NamedSplit, str]:
        return {
            split: data_file
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],
                [self.train_file, self.val_file, self.test_file],
            )
            if data_file is not None
        }


@dc.dataclass
class FinetuningConfig(object):
    data_config: DataConfig

    max_input_length: int
    max_output_length: int

    training_args: Seq2SeqTrainingArguments = dc.field(
        default_factory=lambda: Seq2SeqTrainingArguments(output_dir='./output')
    )
    peft_config: Optional[PeftConfig] = None

    def __post_init__(self):
        if not self.training_args.do_eval or self.data_config.val_file is None:
            self.training_args.do_eval = False
            self.training_args.evaluation_strategy = 'no'
            self.data_config.val_file = None
        else:
            self.training_args.per_device_eval_batch_size = (
                    self.training_args.per_device_eval_batch_size
                    or self.training_args.per_device_train_batch_size
            )

    @classmethod
    def from_dict(cls, **kwargs) -> 'FinetuningConfig':
        training_args = kwargs.get('training_args', None)
        if training_args is not None and not isinstance(
                training_args, Seq2SeqTrainingArguments
        ):
            gen_config = training_args.get('generation_config')
            if not isinstance(gen_config, GenerationConfig):
                training_args['generation_config'] = GenerationConfig(
                    **gen_config
                )
            kwargs['training_args'] = Seq2SeqTrainingArguments(**training_args)

        data_config = kwargs.get('data_config')
        if not isinstance(data_config, DataConfig):
            kwargs['data_config'] = DataConfig(**data_config)

        peft_config = kwargs.get('peft_config', None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs['peft_config'] = get_peft_config(config_dict=peft_config)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'FinetuningConfig':
        path = Path(path)
        parser = yaml.YAML(typ='safe', pure=True)
        parser.indent(mapping=2, offset=2, sequence=4)
        parser.default_flow_style = False
        kwargs = parser.load(path)
        return cls.from_dict(**kwargs)


def _load_datasets(
        data_dir: str,
        data_format: str,
        data_files: dict[NamedSplit, str],
        num_proc: Optional[int],
) -> DatasetDict:
    if data_format == '.jsonl':
        dataset_dct = load_dataset(
            data_dir,
            data_files=data_files,
            split=None,
            num_proc=num_proc,
        )
    else:
        raise NotImplementedError(f"Cannot load dataset in the '{data_format}' format.")
    return dataset_dct


class DataManager(object):
    def __init__(self, data_dir: str, data_config: DataConfig):
        self._num_proc = data_config.num_proc

        self._dataset_dct = _load_datasets(
            data_dir,
            data_config.data_format,
            data_config.data_files,
            self._num_proc,
        )

    def _get_dataset(self, split: NamedSplit) -> Optional[Dataset]:
        return self._dataset_dct.get(split, None)

    def get_dataset(
            self,
            split: NamedSplit,
            process_fn: Callable[[dict[str, Any]], dict[str, Any]],
            batched: bool = True,
            remove_orig_columns: bool = True,
    ) -> Optional[Dataset]:
        orig_dataset = self._get_dataset(split)
        if orig_dataset is None:
            return
        if remove_orig_columns:
            remove_columns = orig_dataset.column_names
        else:
            remove_columns = None
        return orig_dataset.map(
            process_fn,
            batched=batched,
            remove_columns=remove_columns,
            num_proc=self._num_proc,
            # This is default params of  orig_dataset.map, and you can change it smaller
            # https://github.com/THUDM/GLM-4/issues/277
            writer_batch_size=1000,
            batch_size=1000,
        )


def process_batch(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    batched_conv = batch['messages']
    batched_input_ids = []
    batched_attention_mask = []
    batched_position_ids = []
    batched_labels = []
    batched_images = []

    max_length = max_input_length + max_output_length

    for conv in batched_conv:
        input_ids = [151331, 151333]
        attention_mask = [1, 1]
        position_ids = list(range(len(input_ids)))
        loss_masks = [False, False]
        images = []

        for message in conv:
            if message.get('image'):
                image = Image.open(message['image']).convert('RGB')
                message['image'] = image

            loss_mask_val = False if message['role'] in ('system', 'user', 'observation') else True
            new_input_ids_all = tokenizer.apply_chat_template(
                [message],
                tokenize=True,
                return_dict=True,
                padding=True
            )
            new_input_ids = new_input_ids_all['input_ids'][0][2:]
            new_attention_mask = new_input_ids_all['attention_mask'][0][2:]
            new_position_ids = list(range(position_ids[-1] + 1, position_ids[-1] + 1 + len(new_input_ids)))
            if message.get('image'):  # Only One Image
                images.append(new_input_ids_all['images'])

            new_loss_masks = [loss_mask_val] * len(new_input_ids)
            input_ids += new_input_ids
            attention_mask += new_attention_mask
            position_ids += new_position_ids
            loss_masks += new_loss_masks

        input_ids.append(151336)  # EOS
        attention_mask.append(1)
        position_ids.append(len(position_ids))
        loss_masks.append(False)

        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)

        batched_input_ids.append(input_ids[:max_length])
        batched_attention_mask.append(attention_mask[:max_length])
        batched_position_ids.append(position_ids[:max_length])
        batched_labels.append(labels[:max_length])
        if images is not None:
            batched_images.append(images[0][0])

    del batched_conv, conv, input_ids, attention_mask, position_ids, loss_masks, message, new_input_ids, new_loss_masks, labels, input_id, mask
    torch.cuda.empty_cache()

    return {
        'input_ids': batched_input_ids,
        'attention_mask': batched_attention_mask,
        'position_ids': batched_position_ids,
        'labels': batched_labels,
        'images': batched_images
    }


def process_batch_eval(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    batched_conv = batch['messages']
    batched_input_ids = []
    batched_attention_mask = []
    batched_position_ids = []
    batched_output_ids = []
    batched_images = []

    for conv in batched_conv:

        if conv[0].get('image'):
            image = Image.open(conv[0]['image']).convert('RGB')
            conv[0]['image'] = image

        new_input_ids_all = tokenizer.apply_chat_template(
            conv,
            tokenize=True,
            return_dict=True,
            padding=True
        )

        input_ids = new_input_ids_all['input_ids'][0]
        attention_mask = new_input_ids_all['attention_mask'][0]
        position_ids = list(range(len(input_ids)))

        dialogue_parts = [0]
        for idx, token_id in enumerate(input_ids):
            if token_id == 151337:
                dialogue_parts.append(idx + 1)

        if not dialogue_parts or dialogue_parts[-1] != len(input_ids):
            dialogue_parts.append(len(input_ids))

            # Split the conversation into multiple dialogue segments
        for end_idx in range(1, len(dialogue_parts)):
            input_segment = input_ids[:dialogue_parts[end_idx]]
            attention_segment = attention_mask[:dialogue_parts[end_idx]]
            position_segment = position_ids[:dialogue_parts[end_idx]]
            output_segment = input_ids[dialogue_parts[end_idx - 1]:dialogue_parts[end_idx]]
            output_segment.append(151336)  # Add EOS token

            batched_input_ids.append(input_segment[:max_input_length])
            batched_attention_mask.append(attention_segment[:max_input_length])
            batched_position_ids.append(position_segment[:max_input_length])
            batched_output_ids.append(output_segment[:max_output_length])
            batched_images.append(new_input_ids_all['images'][0])

    del batched_conv, input_ids, attention_mask, position_ids, new_input_ids_all, output_segment
    torch.cuda.empty_cache()

    return {
        'input_ids': batched_input_ids,
        'attention_mask': batched_attention_mask,
        'position_ids': batched_position_ids,
        'output_ids': batched_output_ids,
        'images': batched_images
    }


def load_tokenizer_and_model(
        model_dir: str,
        peft_config: Optional[PeftConfig] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if peft_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            empty_init=False,
            use_cache=False,
            torch_dtype=torch.bfloat16  # Must use BFloat 16
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            empty_init=False,
            use_cache=False,
            torch_dtype=torch.bfloat16
        )
    return tokenizer, model


def compute_metrics(eval_preds: EvalPrediction, tokenizer):
    batched_pred_ids, batched_label_ids = eval_preds
    metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        pred_txt = tokenizer.decode(pred_ids).strip()
        label_txt = tokenizer.decode(label_ids).strip()
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
        for k, v in scores[0].items():
            metrics_dct[k].append(round(v['f'] * 100, 4))
        metrics_dct['bleu-4'].append(
            sentence_bleu([label_tokens], pred_tokens, smoothing_function=SmoothingFunction().method3))
    return {k: np.mean(v) for k, v in metrics_dct.items()}


@app.command()
def main(
        data_dir: Annotated[str, typer.Argument(help='')],
        model_dir: Annotated[
            str,
            typer.Argument(
                help='A string that specifies the model id of a pretrained model configuration hosted on huggingface.co, or a path to a directory containing a model configuration file.'
            ),
        ],
        config_file: Annotated[str, typer.Argument(help='')],
        auto_resume_from_checkpoint: str = typer.Argument(
            default='',
            help='If entered as yes, automatically use the latest save checkpoint. If it is a numerical example 12 15, use the corresponding save checkpoint. If the input is no, restart training'
        ),
):
    ft_config = FinetuningConfig.from_file(config_file)
    tokenizer, model = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config)
    data_manager = DataManager(data_dir, ft_config.data_config)

    train_dataset = data_manager.get_dataset(
        Split.TRAIN,
        functools.partial(
            process_batch,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    print('train_dataset:', train_dataset)

    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )

    if val_dataset is not None:
        print('val_dataset:', val_dataset)
    test_dataset = data_manager.get_dataset(
        Split.TEST,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if test_dataset is not None:
        print('test_dataset:', test_dataset)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    trainer = Seq2SeqTrainer(
        model=model,
        args=ft_config.training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding='longest',
            return_tensors='pt',
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
    )

    if auto_resume_from_checkpoint.upper() == "" or auto_resume_from_checkpoint is None:
        trainer.train()
    else:
        output_dir = ft_config.training_args.output_dir
        dirlist = os.listdir(output_dir)
        checkpoint_sn = 0
        for checkpoint_str in dirlist:
            if checkpoint_str.find("eckpoint") > 0 and checkpoint_str.find("tmp") == -1:
                checkpoint = int(checkpoint_str.replace("checkpoint-", ""))
                if checkpoint > checkpoint_sn:
                    checkpoint_sn = checkpoint
        if auto_resume_from_checkpoint.upper() == "YES":
            if checkpoint_sn > 0:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
                checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))
                print("resume checkpoint from checkpoint-" + str(checkpoint_sn))
                trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                trainer.train()
        else:
            if auto_resume_from_checkpoint.isdigit():
                if int(auto_resume_from_checkpoint) > 0:
                    checkpoint_sn = int(auto_resume_from_checkpoint)
                    model.gradient_checkpointing_enable()
                    model.enable_input_require_grads()
                    checkpoint_directory = os.path.join(output_dir, "checkpoint-" + str(checkpoint_sn))
                    print("resume checkpoint from checkpoint-" + str(checkpoint_sn))
                    trainer.train(resume_from_checkpoint=checkpoint_directory)
            else:
                print(auto_resume_from_checkpoint,
                      "The specified checkpoint sn(" + auto_resume_from_checkpoint + ") has not been saved. Please search for the correct checkpoint in the model output directory")

    if test_dataset is not None:
        trainer.predict(test_dataset)


if __name__ == '__main__':
    app()
