import os
import re
import math
import json
import spacy
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_metric
from collections import Counter
import matplotlib.pyplot as plt
from paraphraser_utils.scoring import DiversityScorer
from models.data_utils.get_edit_operations import get_all


def micro_evaluation():
    num_no_pred = 0
    sent_scores = []  # sentence-level evaluation
    bertscore_result = {'precision': [], 'recall': [], 'f1': []}
    grammar_scores = []

    for utt_info in tqdm(all_predictions_filtered.values()):
        for key in utt_info.keys():
            if 'preds' in key:
                score = []

                if not utt_info[key]:
                    num_no_pred += 1

                # get style transfer accuracy
                emo = label2id[key.split('_')[1]]
                emo_score = utt_info[key][0]['emo_score'] if utt_info[key] else 0
                pred_emo = emo if (emo_score > args.min_emo_score) else -1

                all_gold_emos.append(emo)
                all_pred_emos.append(pred_emo)

                # bertscore if there is target text
                # content preservation score
                if 'target_sent' in utt_info:
                    pred = utt_info[key][0]['prediction'] if utt_info[key] else ""
                    if pred:
                        results = bertscore.compute(predictions=[pred], references=[utt_info['target_sent']], lang="en")
                    else:
                        results = {'precision': [0.0], 'recall': [0.0], 'f1': [0.0]}
                    score.append(round(float(results['precision'][0]), 3))
                    bertscore_result['precision'].append(round(float(results['precision'][0]), 3))
                    bertscore_result['recall'].append(round(float(results['recall'][0]), 3))
                    bertscore_result['f1'].append(round(float(results['f1'][0]), 3))

                # fluency score
                grm_score = utt_info[key][0]['grm_score'] if utt_info[key] else 0
                judgement = 1 if (grm_score > args.min_grm_score) else 0
                grammar_juds.append(judgement)
                grammar_scores.append(grm_score)

                if args.micro_type == 'sum':
                    score.append(emo_score)
                    score.append(grm_score)
                    sent_scores.append(sum(score) / len(score))

                elif args.micro_type == 'binary':
                    score.append(1 if emo == pred_emo else 0)
                    score.append(judgement)
                    sent_scores.append(math.prod(score))

    assert len(all_pred_emos) == len(all_gold_emos) == len(grammar_juds)

    eval_results['num_no_pred'] = num_no_pred

    # grammar scores
    eval_results['grammar_scores'] = {'mean': round(float(np.mean(np.array(grammar_scores))), 3),
                                      'median': round(float(np.median(np.array(grammar_scores))), 3)}

    micro_score = sum(sent_scores) / len(sent_scores)
    eval_results['micro_score'] = round(micro_score, 3)
    return bertscore_result


def macro_evaluation(bertscore_result):
    """ Additionally, do also corpus-level evaluation """
    style_macro = {"accuracy": (np.array(all_pred_emos) == np.array(all_gold_emos)).astype(np.float32).mean().item()}
    style_macro = {metric: round(score, 4) for metric, score in style_macro.items()}
    eval_results['style_score'] = style_macro

    gold_juds = [1] * len(grammar_juds)
    grammar_macro = {"accuracy": (np.array(grammar_juds) == np.array(gold_juds)).astype(np.float32).mean().item()}
    grammar_macro = {metric: round(score, 4) for metric, score in grammar_macro.items()}
    eval_results['grammar_score'] = grammar_macro

    if bertscore_result['precision']:
        bertscore_macro = {metric: round(sum(score_list) / len(score_list), 4)
                           for metric, score_list in bertscore_result.items() if isinstance(score_list, list)}
        eval_results['bert_score'] = bertscore_macro
    else:
        bertscore_macro = {'precision': 1.0}

    macro_score = sum([style_macro['accuracy'] + grammar_macro['accuracy'] + bertscore_macro['precision']]) / 3
    eval_results['macro_score'] = round(macro_score, 3)


def score_diversity():
    diversity = {}
    type_counter = Counter()
    all_predictions = {'anger': [],
                       'happiness': [],
                       'sadness': [],
                       'all': []}
    diverse_1 = DiversityScorer(n=1)
    diverse_2 = DiversityScorer(n=2)
    nlp = spacy.load("en_core_web_sm")

    for utt_info in all_predictions_filtered.values():
        for key in utt_info.keys():
            input_sent = re.sub(r'[^\w\s]', '', utt_info['input']).strip()
            if 'preds' in key and utt_info[key]:
                emotion = key.split('_')[1]
                for pred in utt_info[key][:args.max_num_pred]:
                    pred_sent = re.sub(r'[^\w\s]', '', pred['prediction']).strip()
                    _, edit_type, e_phrases = get_all(pred_sent, input_sent, nlp)
                    type_counter.update([edit_type])
                    all_predictions[emotion].extend(e_phrases)
                    all_predictions['all'].extend(e_phrases)

    for emotion, all_phrases in all_predictions.items():
        score_uni = diverse_1.score(predictions=all_phrases)
        score_bi = diverse_2.score(predictions=all_phrases)
        diversity[emotion] = {'diversity-1': round(score_uni, 3),
                              'diversity-2': round(score_bi, 3)}

    eval_results['edit_types'] = type_counter
    eval_results['diversity'] = diversity
    eval_results['emo_phrases'] = all_predictions


def score_perplexity():
    all_ppl_scores = []
    eval_ppl = {}

    for utt_info in all_predictions_filtered.values():
        for key in utt_info.keys():
            if 'preds' in key and utt_info[key]:
                for pred in utt_info[key][:args.max_num_pred]:
                    all_ppl_scores.append(pred['ppl_score'])

    eval_ppl['mean'] = round(float(np.mean(all_ppl_scores)), 3)  # average ppl scores
    eval_ppl['median'] = round(float(np.median(all_ppl_scores)), 3)
    eval_results['perplexity'] = eval_ppl
    # plot_perplexity(all_ppl_scores)


def plot_perplexity(all_ppl_scores):
    all_ppl_scores = np.array(all_ppl_scores)
    # Plot the distribution of numpy data
    plt.hist(all_ppl_scores, bins=np.arange(min(all_ppl_scores), max(all_ppl_scores) + 0.75, 0.75), align='left')

    # Add axis labels
    plt.xlabel("Perplexity Score")
    plt.ylabel("Freq")
    plt.title(f"{args.prediction_file.split('_')[0].title()} Perplexity Distribution")
    # plt.legend()
    # plt.show()
    plt.savefig(os.path.join(args.output_dir, f'{args.prediction_file[:-5]}_ppl_plot.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for evaluation")
    parser.add_argument("--output_dir", type=str, default='../outputs')
    parser.add_argument("--prediction_file", type=str, default=None)
    parser.add_argument("--micro_type", type=str, default='binary', choices=['binary', 'sum'])
    parser.add_argument("--min_grm_score", type=float, default=0.5)
    parser.add_argument("--min_emo_score", type=float, default=0.5)
    parser.add_argument("--max_num_pred", type=int, default=5)
    args = parser.parse_args()

    label2id = {'neutral': 0, 'anger': 1, 'happiness': 2, 'sadness': 3}
    eval_results = {}
    all_pred_emos, all_gold_emos = [], []
    grammar_juds = []

    # load prediction file
    with open(os.path.join(args.output_dir, args.prediction_file), 'r', encoding='utf-8') as jfile:
        all_predictions_filtered = json.load(jfile)

    bertscore = load_metric("bertscore")
    bert_result = micro_evaluation()
    macro_evaluation(bert_result)
    score_perplexity()
    score_diversity()

    # write to file
    output_file = f'{args.prediction_file[:-5]}_eval_{args.micro_type}.json'
    with open(os.path.join(args.output_dir, output_file), 'w', encoding='utf-8') as jfile:
        json.dump(eval_results, jfile, indent=4)
