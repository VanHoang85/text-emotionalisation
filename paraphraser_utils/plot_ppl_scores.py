import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from paraphraser_utils.scoring import PerplexityScorer


def get_num_bins(all_model_scores):
    all_scores = []
    for scores in all_model_scores.values():
        all_scores.extend(scores)
    all_scores = np.array(all_scores)
    return int(((max_ppl_score - min(all_scores)) // 20) + 1)


def get_target_sent_scores(target_sents):
    scorer = PerplexityScorer()
    scores = scorer.score(target_sents, do_normalize=False)
    print(f'Target mean perplexity score {round(float(np.mean(scores)), 3)}')
    print(f'Target median perplexity score {round(float(np.median(scores)), 3)}')
    return scores


def main():
    all_model_scores = {}
    # target_sents = []
    for file in files:
        if not os.path.exists(os.path.join(dir_path, file)):
            print('File not exist:', file)
            continue

        with open(os.path.join(dir_path, file), 'r', encoding='utf-8') as jfile:
            data = json.load(jfile)

        model = file.split('_')[0]
        scores = []

        print(f'Processing {file}....')
        for utt_info in tqdm(data.values()):
            for key in utt_info.keys():
                if 'preds' in key and utt_info[key]:
                    for pred in utt_info[key][:5]:
                        scores.append(float(pred['ppl_score']))

            # if 'pegasus' in file:
            #     target_sents.append(utt_info['target_sent'])
        all_model_scores[model] = np.array(scores)
    # all_model_scores['target'] = get_target_sent_scores(target_sents)

    for model, scores_ls in all_model_scores.items():
        count = np.sum(np.where(np.array(scores_ls) >= max_ppl_score, 1, 0))
        print(f'{model} has {count} sentences with perplexity > {max_ppl_score}')

    num_bins = get_num_bins(all_model_scores)
    print(num_bins)
    kwargs = dict(alpha=0.5, bins=num_bins)

    plt.hist(all_model_scores['naive'][np.where(all_model_scores['naive'] < max_ppl_score)], **kwargs, color='g', label='naive')
    plt.hist(all_model_scores['bart'][np.where(all_model_scores['bart'] < max_ppl_score)], **kwargs, color='b', label='bart')
    plt.hist(all_model_scores['pegasus'][np.where(all_model_scores['pegasus'] < max_ppl_score)], **kwargs, color='r', label='pegasus')
    # plt.hist(all_model_scores['target'][np.where(all_model_scores['target'] < max_ppl_score)], **kwargs, color='y', label='target')

    plt.gca().set(title='Perplexity Distribution', ylabel='Frequency', xlabel='Perplexity Score')
    # plt.xlim(50, 75)
    plt.legend()
    plt.savefig(os.path.join(dir_path, f'ppl_plot_{max_ppl_score}_wbf.png'))


if __name__ == "__main__":
    dir_path = '../outputs'
    files = ['naive_phrases_test_emo_wbf.json',
             'bart_with_style_loss_test_emo_wbf.json',
             'pegasus_with_style_loss_test_emo_wbf.json']
    max_ppl_score = 1000

    main()
