import argparse
import json
import logging
import math
import os
import random

import datasets
import nltk
import numpy as np
import torch
from datasets import load_metric, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from filelock import FileLock

from paraphraser_utils.scoring import StyleScorer
from paraphraser_utils.modeling_bart import BartStyleParaphraser
from paraphraser_utils.modeling_pegasus import PegasusStyleParaphraser
from transformers import (
    AdamW,
    Adafactor,
    BartConfig,
    BartTokenizer,
    PegasusConfig,
    PegasusTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='./style_dataset.py',
        help="The name of the dataset to use."
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
        choices=['neutralizer', 'emo_stylizer', 'phrase_stylizer',
                 'anger_stylizer', 'happiness_stylizer', 'sadness_stylizer']
    )
    parser.add_argument(
        "--silver_per",
        type=float,
        default=0.0,
        help="The percentage of silver data in training set.",
        choices=[0.0, 0.5, 1.0, 2.0]
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        default=True,
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--path_to_classifier_dir",
        type=str,
        default=None,
        help="Path to the trained style (style) classifier."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the testing dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--max_no_improvements", type=int, default=3,
                        help="Max number of training epochs to do early stopping.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        # type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--optimizer",
        # type=SchedulerType,
        default="adamw",
        help="The scheduler type to use.",
        choices=["adamw", "adafactor"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_bertscore",
        action="store_true",
        help="If passed, will use a bert score as an additional metrics for evaluation.",
    )
    parser.add_argument(
        "--use_style_loss",
        action="store_true",
        help="If passed, will use a bert score as an additional metrics for evaluation.",
    )
    parser.add_argument("--output_path", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Where to store the caches.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--num_beams",
        type=int,
        default=8,  # 20
        help="Number of beams to use for evaluation. This argument will be "
             "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    return parser.parse_args()


def evaluate(model, eval_dataloader, config, tokenizer, bertscore=None, return_preds=False):
    completed_steps = 0
    progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

    model.eval()
    gen_kwargs = {
        "max_length": args.max_length if args is not None else config.max_length,
        "num_beams": args.num_beams if args.num_beams > 8 else 8
    }

    pred_outs = {"predictions": [],
                 "predicted_emos": []}
    all_pred_emos, all_gold_emos = np.array([]), np.array([])
    for step, batch in enumerate(eval_dataloader):  # to set batch size = 1
        with torch.no_grad():
            model_kwargs = {
                "inputs": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
            # if args.use_style_loss:
            #    model_kwargs["emo_labels"] = batch["emo_labels"]

            generated_tokens = accelerator.unwrap_model(model).generate(
                **model_kwargs,
                **gen_kwargs,
            )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]
            if not args.pad_to_max_length:
                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            pred_emos = style_scorer.score(decoded_preds)
            pred_emos = pred_emos.detach().cpu().numpy()
            pred_emos = np.argmax(pred_emos, axis=1)

            if args.use_bertscore:
                bertscore.add_batch(predictions=decoded_preds, references=decoded_labels)

            all_pred_emos = np.concatenate([all_pred_emos, pred_emos], axis=0)
            all_gold_emos = np.concatenate([all_gold_emos, batch["emo_labels"].detach().cpu().numpy()], axis=0)

            if return_preds:
                pred_outs["predictions"].append(str(decoded_preds[0]))
                pred_outs["predicted_emos"].append(int(list(pred_emos)[0]))

            if step % args.gradient_accumulation_steps == 0 or step == len(eval_dataloader) - 1:
                completed_steps += 1

            if completed_steps % 100 == 0:
                progress_bar.update(100)
                logger.info(f'Decoded candidates: {decoded_preds[0]}')

            # if completed_steps >= args.max_train_steps:
            #    break

    style_result = {"accuracy": (all_pred_emos == all_gold_emos).astype(np.float32).mean().item()}
    style_result = {metric: round(score, 4) for metric, score in style_result.items()}
    total_score = style_result["accuracy"]

    if args.use_bertscore:
        bertscore_result = bertscore.compute(lang='en')
        bertscore_result = {metric: round(sum(score_list) / len(score_list), 4)
                            for metric, score_list in bertscore_result.items() if isinstance(score_list, list)}
        # total_score += bertscore_result["precision"]
        logger.info(bertscore_result)

    logger.info(style_result)
    if return_preds:
        return total_score, pred_outs
    return total_score


def load_model_tokenizer():
    # model, tokenizer, config = None, None, None
    # if 'bart' in args.model_name_or_path:
    if 'pegasus' in args.model_name_or_path:
        config = PegasusConfig.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir
        )
        tokenizer = PegasusTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir
        )
        # PegasusForConditionalGeneration
        model = PegasusStyleParaphraser.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir
        )
    else:
        config = BartConfig.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir
        )
        tokenizer = BartTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
            cache_dir=args.cache_dir
        )
        model = BartStyleParaphraser.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir
        )
    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    return model, tokenizer, config


def main():
    if args.silver_per != 0.0:
        args.dataset_config_name = f"{args.dataset_config_name}_{args.silver_per}"
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    model, tokenizer, config = load_model_tokenizer()

    # Preprocessing the datasets.
    # First we tokenize all the texts.

    # Temporarily set max_target_length for training.
    max_length = args.max_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples['input']
        targets = examples['target']

        if 'constraint' in examples:
            constraints = examples['constraint']
            model_inputs = tokenizer(inputs, constraints, max_length=max_length, padding=padding, truncation=True,
                                     return_token_type_ids=False)
        else:
            model_inputs = tokenizer(inputs, max_length=max_length, padding=padding, truncation=True,
                                     return_token_type_ids=False)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            target_labels = tokenizer(targets, max_length=max_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            target_labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in target_labels["input_ids"]
            ]

        model_inputs["labels"] = target_labels["input_ids"]
        model_inputs["emo_labels"] = [label2id[emo] for emo in examples["target_emo"]]
        return model_inputs

    column_names = raw_datasets["train"].column_names
    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache
        )
        eval_dataset = raw_datasets["validation"].map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache
        )
        test_dataset = raw_datasets["test"].map(
            preprocess_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache
        )

    # Log a few random samples from the training set:
    # logger.info(f"Len of train dataset: {len(train_dataset)}.")
    for index in random.sample(range(len(train_dataset)), 1):
        # logger.info(f"Index of sample: {index}.")
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_test_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = None
    if args.optimizer == 'adafactor':
        optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, relative_step=False)
    elif args.optimizer == 'adamw':
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    bertscore = None
    if args.use_bertscore:
        bertscore = load_metric("bertscore")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    best_total_score = None
    max_no_improvements = args.max_no_improvements
    no_improvements = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            model_kwargs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"]
            }
            if args.use_style_loss:
                model_kwargs["emo_labels"] = batch["emo_labels"]
                model_kwargs["style_classifier"] = style_scorer.classifier

            outputs = model(**model_kwargs)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

            if completed_steps % 100 == 0:
                progress_bar.update(100)

            if completed_steps >= args.max_train_steps:
                break

        # evaluate
        logger.info(f"Results on validation set from epoch {epoch}:")
        if args.use_bertscore:
            total_score = evaluate(model, eval_dataloader, config, tokenizer, bertscore)
        else:
            total_score = evaluate(model, eval_dataloader, config, tokenizer)

        if not best_total_score or total_score > best_total_score:
            best_total_score = total_score
            no_improvements = 0

            logger.info(f"Results on test set from epoch {epoch}:")
            logger.info(f"Save model after epoch {epoch}:")
            if args.output_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
        else:
            no_improvements += 1

        if no_improvements >= max_no_improvements:  # do early stopping
            break

    # do it on test
    logger.info("Results on test set after training:")
    logger.info("Make predictions on test set")
    _, pred_outs = evaluate(model, test_dataloader, config, tokenizer, bertscore, return_preds=True)

    save_preds_to_file(raw_datasets["test"], pred_outs)


def save_preds_to_file(raw_test_set, pred_outs):
    outs = {}
    for idx in range(len(raw_test_set)):
        outs[raw_test_set['id'][idx]] = {'input': raw_test_set['input'][idx],
                                         'target': raw_test_set['target'][idx],
                                         'target_emo': raw_test_set['target_emo'][idx],
                                         'pred': str(pred_outs['predictions'][idx]),
                                         'pred_emo': id2label[pred_outs['predicted_emos'][idx]],
                                         'context': raw_test_set['context'][idx]}
    with open(os.path.join(args.output_dir, f"test_preds.json"), 'w', encoding='utf-8') as jfile:
        json.dump(outs, jfile, indent=4)


if __name__ == "__main__":
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # set proper output dir
    dir_name = args.dataset_config_name if args.silver_per == 0.0 \
        else f'{args.dataset_config_name}{args.silver_per}'
    args.output_dir = os.path.join(os.path.join(args.output_path, args.output_dir), dir_name)
    args.path_to_classifier_dir = os.path.join(args.output_path, args.path_to_classifier_dir)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # load pre-trained style classifier
    style_scorer = StyleScorer(args.path_to_classifier_dir, args.cache_dir)
    label2id = style_scorer.classifier.config.label2id
    id2label = style_scorer.classifier.config.id2label

    logger.info(args)
    main()
