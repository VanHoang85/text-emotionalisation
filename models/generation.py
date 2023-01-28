import argparse
import json

import torch
import os
import re
import numpy as np
from tqdm import tqdm

from collections import OrderedDict
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    BartConfig,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    PegasusConfig)
from baseline import NaiveStyleTransfer
from paraphraser_utils.scoring import StyleScorer, PerplexityScorer, SimilarityScorer, GrammarScorer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def filter_indexes(emo_scores, sim_scores, grm_scores):
    # get indexes of pred with emo and sim scores higher than threshold
    emo_idxes = np.where(emo_scores > args.min_emo_score)[0]
    grm_idxes = np.where(grm_scores > args.min_grm_score)[0]
    sim_idxes = np.where(sim_scores > args.min_sim_score)[0]  # filter with min sim score
    filtered_idxes = np.sort(np.intersect1d(np.intersect1d(emo_idxes, sim_idxes), grm_idxes))

    # if no idx meet min reqs, back off
    if args.use_backoff:
        if len(filtered_idxes) == 0:
            filtered_idxes = np.intersect1d(emo_idxes, sim_idxes)

        if len(filtered_idxes) == 0:
            filtered_idxes = np.intersect1d(emo_idxes, grm_idxes)

        if len(filtered_idxes) == 0:
            filtered_idxes = emo_idxes

        if len(filtered_idxes) == 0:  # no filtering
            filtered_idxes = np.arange(len(emo_scores))
    return filtered_idxes


def filter_preds(preds: list, input_text: str) -> list:
    """ Filter out pred that shorter than input text, and preds with same emotion phrases """
    filtered_preds = []
    all_pred_tokens = []

    input_tokens = re.sub(r'[^\w\s]', '', input_text).strip().split()
    for pred in preds:
        pred_tokens = re.sub(r'[^\w\s]', '', pred).strip().split()

        if len(pred_tokens) >= len(input_tokens) and set(pred_tokens) not in all_pred_tokens:
            filtered_preds.append(pred)
            all_pred_tokens.append(set(pred_tokens))
    return filtered_preds


def rank_preds(preds: list, input_text: str, target_emo: int) -> list:
    """Rank predictions according to style, similarity score, and fluency"""
    preds = filter_preds(preds, input_text)
    emo_scores, sim_scores, ppl_scores, grm_scores = get_scores(preds, input_text, target_emo)
    ppl_normed = ppl_scorer.normalize_scores(ppl_scores)
    filtered_idxes = filter_indexes(emo_scores, sim_scores, grm_scores)

    pred_with_scores = []
    for index in list(filtered_idxes):
        pred_with_scores.append({"prediction": preds[index],
                                 "emo_score": round(float(emo_scores[index]), 3),
                                 "sim_score": round(float(sim_scores[index]), 3),
                                 "grm_score": round(float(grm_scores[index]), 3),
                                 "ppl_normd": round(float(ppl_normed[index]), 3),
                                 "ppl_score": round(float(ppl_scores[index]), 3),
                                 "total": round(float(emo_scores[index] + grm_scores[index]), 3)})  # sim_scores[index]

    # sort based on total score
    pred_with_scores = sorted(pred_with_scores, key=lambda item: item['total'], reverse=True)
    return pred_with_scores[:args.max_pred_return]


def get_emo_scores(preds: list, target_emo: int):
    emo_scores = np.array([])
    for pred in preds:
        emo_score = style_scorer.score(pred)
        emo_score = emo_score[:, target_emo]  # get target emo score for each pred
        emo_score = emo_score.detach().cpu().numpy()  # turn to numpy
        emo_scores = np.concatenate([emo_scores, emo_score], axis=0)
    return emo_scores


def get_grammar_scores(preds: list):
    grm_scores = np.array([])
    for pred in preds:
        grm_score = grm_scorer.score(pred)
        grm_scores = np.append(grm_scores, grm_score)
    return grm_scores


def get_scores(preds: list, input_text: str, target_emo: int):
    emo_scores = get_emo_scores(preds, target_emo)  # Score style / emo
    sim_scores = sim_scorer.score(preds, input_text) if args.score_sim else np.ones(len(emo_scores))  # Score similarity
    ppl_scores = ppl_scorer.score(preds, do_normalize=False) if args.score_ppl else np.ones(len(emo_scores))  # Score perplexity
    grm_scores = get_grammar_scores(preds) if args.score_grm else np.ones(len(emo_scores))  # Score grammar acceptability

    # make sure they are of the same len
    assert len(emo_scores) == len(sim_scores) == len(ppl_scores) == len(grm_scores)
    return emo_scores, sim_scores, ppl_scores, grm_scores


def get_responses(model, data, tokenizer) -> list:
    completed_steps = 0
    progress_bar = tqdm(range(len(data)), disable=not accelerator.is_local_main_process)

    model.eval()

    max_length = args.max_seq_length
    padding = "max_length"

    pred_outs = []
    for example in data:
        with torch.no_grad():
            out = {}
            if args.gen_all_emos and "emo_stylizer" in args.dataset_config_name:
                target_emos = ['anger', 'happiness', 'sadness']
            elif "neutralizer" in args.dataset_config_name:
                target_emos = ['neutral']
            elif 'constraint' in example:
                target_emos = [example['constraint']]
            else:
                raise ValueError("No target emotion detected!!!!")

            for emo in target_emos:
                if args.use_constraint and "emo_stylizer" in args.dataset_config_name:
                    model_inputs = tokenizer(example['input'], emo,
                                             max_length=max_length,
                                             padding=padding, truncation=True,
                                             return_tensors="pt").to(device)
                else:
                    model_inputs = tokenizer(example['input'],
                                             max_length=max_length,
                                             padding=padding, truncation=True,
                                             return_tensors="pt").to(device)

                generated_tokens = accelerator.unwrap_model(model).generate(
                    model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    **gen_kwargs,
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                out[emo] = decoded_preds

            pred_outs.append(out)
            completed_steps += 1
            if completed_steps % 100 == 0:
                progress_bar.update(100)
                for emo, ls in out.items():
                    print(f'Decoded candidates of {emo}: {ls[0]}')
    return pred_outs


def get_naive_responses(model: NaiveStyleTransfer, data) -> list:
    pred_outs = []
    for example in data:
        out = {}
        if args.gen_all_emos and "emo_stylizer" in args.dataset_config_name:
            target_emos = ['anger', 'happiness', 'sadness']
        elif "neutralizer" in args.dataset_config_name:
            target_emos = ['neutral']
        elif 'constraint' in example:
            target_emos = [example['constraint']]
        else:
            raise ValueError("No target emotion detected!!!!")

        for emo in target_emos:
            out[emo] = model.generate_batch(example['input'], emo, num=gen_kwargs['num_return_sequences'])
        pred_outs.append(out)
    return pred_outs


def get_target_responses(data):
    pred_outs = []
    for example in data:
        out = {}
        if args.gen_all_emos and "emo_stylizer" in args.dataset_config_name:
            target_emos = ['anger', 'happiness', 'sadness']
        elif "neutralizer" in args.dataset_config_name:
            target_emos = ['neutral']
        elif 'constraint' in example:
            target_emos = [example['constraint']]
        else:
            raise ValueError("No target emotion detected!!!!")

        for emo in target_emos:
            out[emo] = [example['target']]
        pred_outs.append(out)
    return pred_outs


def get_input_responses(data):
    pred_outs = []
    for example in data:
        out = {}
        if args.gen_all_emos and "emo_stylizer" in args.dataset_config_name:
            target_emos = ['anger', 'happiness', 'sadness']
        elif "neutralizer" in args.dataset_config_name:
            target_emos = ['neutral']
        elif 'constraint' in example:
            target_emos = [example['constraint']]
        else:
            raise ValueError("No target emotion detected!!!!")

        for emo in target_emos:
            out[emo] = [example['input']]
        pred_outs.append(out)
    return pred_outs


def load_model_tokenizer():
    if 'pegasus' in args.model_name_or_path:
        config = PegasusConfig.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir
        )
        tokenizer = PegasusTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=True,
            cache_dir=args.cache_dir
        )
        model = PegasusForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir
        )
    elif 'bart' in args.model_name_or_path:
        config = BartConfig.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir
        )
        tokenizer = BartTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=True,
            cache_dir=args.cache_dir
        )
        model = BartForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir
        )
    elif 'naive' in args.model_name_or_path:
        model = NaiveStyleTransfer()
        tokenizer, config = None, None
    else:
        model, tokenizer, config = None, None, None
    return model, tokenizer, config


def load_data_from_file():
    print(f'Loading data from {args.input_file}...')
    raw_data = {"id": [],
                "input": [],
                "target_emo": [],
                "context": [],
                "context_emo": []
                }
    if "emo_stylizer" in args.dataset_config_name:
        raw_data["constraint"] = []

    with open(args.input_file, 'r', encoding='utf-8') as jfile:
        data = json.load(jfile)

    for id_, utt_info in data.items():
        input_text = utt_info["emotion_sent"] if args.dataset_config_name == "neutralizer" \
          else utt_info["neutral_sent"]
        # input_text = utt_info["emotion_sent"]
        target_emo = "neutral" if args.dataset_config_name == "neutralizer" else utt_info["emo"]

        raw_data["id"].append(id_)
        raw_data["input"].append(input_text)
        raw_data["target_emo"].append(target_emo)

        if "context" in utt_info:
            raw_data["context"].append(utt_info["context"])
        if "context_emo" in utt_info:
            raw_data["context_emo"].append(utt_info["context_emo"])

        if "emo_stylizer" in args.dataset_config_name:
            raw_data["constraint"].append(target_emo)
    return Dataset.from_dict(raw_data)


def main():
    # to process input file
    if not args.input_file:
        raw_test_dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir)['test']
    else:
        raw_test_dataset = load_data_from_file()

    model, tokenizer, config = load_model_tokenizer()

    if 'pegasus' in args.model_name_or_path or 'bart' in args.model_name_or_path:
        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
        model = accelerator.prepare(model)
        all_predictions = get_responses(model, raw_test_dataset, tokenizer)
    elif args.model_name_or_path == 'naive':
        all_predictions = get_naive_responses(model, raw_test_dataset)
    elif args.model_name_or_path == 'input':
        all_predictions = get_input_responses(raw_test_dataset)
    else:
        all_predictions = get_target_responses(raw_test_dataset)

    completed = 0
    progress_bar = tqdm(range(len(all_predictions)))

    all_predictions_filtered = OrderedDict()
    for example, predictions in zip(raw_test_dataset, all_predictions):
        utt_info = {"input": example['input']}

        if 'context' in raw_test_dataset.column_names:
            utt_info["context"] = example['context']
        if 'context_emo' in raw_test_dataset.column_names:
            utt_info['context_emo'] = example['context_emo']
        if 'target' in raw_test_dataset.column_names:
            utt_info["target_sent"] = example['target']

        for emo, emo_preds in predictions.items():
            filtered_preds = rank_preds(preds=emo_preds,
                                        input_text=example['input'],
                                        target_emo=label2id[emo])
            utt_info[f"preds_{emo}"] = filtered_preds
        all_predictions_filtered.update({example['id']: utt_info})

        completed += 1
        if completed % 100 == 0:
            progress_bar.update(100)

    # save to file
    with open(os.path.join(args.output_dir, args.output_file), 'w', encoding='utf-8') as jfile:
        json.dump(all_predictions_filtered, jfile, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for generation")
    parser.add_argument("--dataset_name", type=str, default='./style_dataset.py',
                        help="The name of the dataset to use.")
    parser.add_argument("--dataset_config_name", type=str, default='emo_stylizer',
                        help="The configuration name of the dataset to use (via the datasets library).",
                        choices=['neutralizer', 'emo_stylizer', 'phrase_stylizer'])
    parser.add_argument("--model_name_or_path", type=str,
                        default=None,
                        required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--path_to_classifier_dir", type=str,
                        help="Path to the trained style (emotion) classifier.")
    parser.add_argument("--path_to_cola_classifier", type=str,
                        help="Path to the trained cola (grammar acceptability) classifier.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=60)
    parser.add_argument("--min_seq_length", type=int, default=5)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--num_return_sequences", type=int, default=20)
    parser.add_argument("--do_sample", action='store_true')
    parser.add_argument("--early_stopping", action='store_true')
    parser.add_argument("--temperature", type=float, default=1.0,
                        help='The value used to module the next token probabilities.')
    parser.add_argument("--top_k", type=int, default=50,  # 120
                        help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
    parser.add_argument("--top_p", type=float, default=1.0,  # 0.95
                        help='If set to float < 1, only the most probable tokens with probabilities that add up to '
                             'top_p or higher are kept for generation.')
    parser.add_argument("--num_beam_groups", type=int, default=1,  # 3
                        help='Number of groups to divide num_beams into in order to ensure diversity '
                             'among different groups of beams. ')
    parser.add_argument("--diversity_penalty", type=float, default=0.0,  # 0.2 to 0.8)
                        help='Value to control diversity for group beam search. that will be used by default in the '
                             '`generate` method of the model. 0 means no diversity penalty. '
                             'The higher the penalty, the more diverse are the outputs.')
    parser.add_argument("--use_constraint", action='store_true')
    parser.add_argument("--gen_all_emos", action='store_true')
    parser.add_argument("--score_ppl", action='store_true')
    parser.add_argument("--score_sim", action='store_true')
    parser.add_argument("--score_grm", action='store_true')
    parser.add_argument("--use_backoff", action='store_true',
                        help='If no predictions after filtering steps, backoff.')
    parser.add_argument("--min_sim_score", type=float, default=0.4)
    parser.add_argument("--min_emo_score", type=float, default=0.5)
    parser.add_argument("--min_ppl_score", type=float, default=0.5)
    parser.add_argument("--min_grm_score", type=float, default=0.5)
    parser.add_argument("--max_pred_return", type=int, default=5)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--output_dir", type=str, default='../outputs')
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    args = parser.parse_args()

    print(args)

    # make sure returned seqs smaller than num beams:
    if args.num_return_sequences > args.num_beams:
        args.num_return_sequences = args.num_beams

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not args.output_file and not args.input_file:
        args.output_file = f'test_{args.dataset_config_name}_all_preds.json'
    elif not args.output_file and args.input_file:
        args.output_file = f'{args.input_file}_all_preds.json'

    gen_kwargs = {
        "max_length": args.max_seq_length,
        "min_length": args.min_seq_length,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "do_sample": args.do_sample,
        "early_stopping": args.early_stopping,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "num_beam_groups": args.num_beam_groups,
        "diversity_penalty": args.diversity_penalty
    }

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    accelerator.wait_for_everyone()

    style_scorer = StyleScorer(args.path_to_classifier_dir, args.cache_dir)
    sim_scorer = SimilarityScorer() if args.score_sim else None
    ppl_scorer = PerplexityScorer() if args.score_ppl else None
    grm_scorer = GrammarScorer(args.path_to_cola_classifier) if args.score_grm else None

    label2id = style_scorer.classifier.config.label2id

    main()
