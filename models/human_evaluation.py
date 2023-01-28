import os
import json
import numpy as np
import statistics
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from datasets import load_metric
from statsmodels.stats import inter_rater


def read_file(path_to_file):
    with open(path_to_file, 'r', encoding='utf-8') as jfile:
        data = json.load(jfile)
    return data


def get_scores(results: dict):
    scores_by_models = {}
    scores_by_emotions = {}

    for utt_id, utt_results in results.items():
        for aspect in ["emotion", "similarity", "fluency", "context"]:
            if aspect not in scores_by_models:
                scores_by_models[aspect] = {}
            if aspect not in scores_by_emotions:
                scores_by_emotions[aspect] = {}

            for worker, eval_result in utt_results[aspect].items():
                for key, val in eval_result.items():
                    if aspect in key:
                        model = key.split('_')[-1]

                        if model not in scores_by_models[aspect]:
                            scores_by_models[aspect][model] = []
                        scores_by_models[aspect][model].append(int(val))

                        if utt_results["target_emo"] not in scores_by_emotions[aspect]:
                            scores_by_emotions[aspect][utt_results["target_emo"]] = []
                        scores_by_emotions[aspect][utt_results["target_emo"]].append(int(val))
    return scores_by_models, scores_by_emotions


def plot_stacked_bar_chart(scores_by_models):
    for aspect, results in scores_by_models.items():
        models = ["ref", "std", "nel", "nps", "naive"]
        ones = np.array([results["target"].count(1), results["wst"].count(1), results["nst"].count(1), results["nps"].count(1), results["naive"].count(1)])
        twos = np.array([results["target"].count(2), results["wst"].count(2), results["nst"].count(2), results["nps"].count(2), results["naive"].count(2)])
        threes = np.array([results["target"].count(3), results["wst"].count(3), results["nst"].count(3), results["nps"].count(3), results["naive"].count(3)])
        fours = np.array([results["target"].count(4), results["wst"].count(4), results["nst"].count(4), results["nps"].count(4), results["naive"].count(4)])
        fives = np.array([results["target"].count(5), results["wst"].count(5), results["nst"].count(5), results["nps"].count(5), results["naive"].count(5)])

        # print(aspect)
        # print(ones, twos, threes, fours, fives)
        w = 0.6  # Define width of stacked chart

        # Plot stacked bar chart
        plt.bar(models, ones, w, color='red', label='1')
        plt.bar(models, twos, w, color='orange', label='2', bottom=ones)
        plt.bar(models, threes, w, color='yellow', label='3', bottom=ones + twos)
        plt.bar(models, fours, w, color='blue', label='4', bottom=ones + twos + threes)
        plt.bar(models, fives, w, color='green', label='5', bottom=ones + twos + threes + fours)

        plt.title(f"Score Distribution By {aspect.title()}")
        # plt.legend()
        # plt.show()
        plt.savefig(os.path.join(dir_path, f'score_dis_{aspect}.png'))


def calculate_spearman(human_results: dict, path_to_automatic_scores: str):
    bertscore = load_metric("bertscore")
    automatic_results = read_file(path_to_automatic_scores)
    outs = {}

    for idx, aspect in enumerate(['emotion', 'similarity', 'fluency']):
        human_scores, ranking_scores, automatic_scores = [], [], []

        for id_, utt_results in human_results.items():
            aspect_scores = utt_results[aspect]

            for score in aspect_scores.values():
                human_scores.append(float(score[f'{aspect}_target']))
                ranking_scores.append(float(automatic_results[id_]["reference"].split(':')[-1].split(',')[idx]))

                human_scores.append(float(score[f'{aspect}_wst']))
                ranking_scores.append(float(automatic_results[id_]["pred_pegasus_wst"].split(':')[-1].split(',')[idx]))

                human_scores.append(float(score[f'{aspect}_nst']))
                ranking_scores.append(float(automatic_results[id_]["pred_pegasus_nst"].split(':')[-1].split(',')[idx]))

                human_scores.append(float(score[f'{aspect}_nps']))
                ranking_scores.append(float(automatic_results[id_]["pred_pegasus_np"].split(':')[-1].split(',')[idx]))

                human_scores.append(float(score[f'{aspect}_naive']))
                ranking_scores.append(float(automatic_results[id_]["pred_naive"].split(':')[-1].split(',')[idx]))

                if aspect == 'similarity':
                    automatic_scores.append(bertscore.compute(
                        predictions=[automatic_results[id_]["reference"].split(':')[0]],
                        references=[automatic_results[id_]["reference"].split(':')[0]],
                        lang="en")['precision'])
                    automatic_scores.append(bertscore.compute(
                        predictions=[automatic_results[id_]["pred_pegasus_wst"].split(':')[0]],
                        references=[automatic_results[id_]["reference"].split(':')[0]],
                        lang="en")['precision'])
                    automatic_scores.append(bertscore.compute(
                        predictions=[automatic_results[id_]["pred_pegasus_nst"].split(':')[0]],
                        references=[automatic_results[id_]["reference"].split(':')[0]],
                        lang="en")['precision'])
                    automatic_scores.append(bertscore.compute(
                        predictions=[automatic_results[id_]["pred_pegasus_np"].split(':')[0]],
                        references=[automatic_results[id_]["reference"].split(':')[0]],
                        lang="en")['precision'])
                    automatic_scores.append(bertscore.compute(
                        predictions=[automatic_results[id_]["pred_naive"].split(':')[0]],
                        references=[automatic_results[id_]["reference"].split(':')[0]],
                        lang="en")['precision'])
                else:
                    automatic_scores.append(float(automatic_results[id_]["reference"].split(':')[-1].split(',')[idx]))
                    automatic_scores.append(float(automatic_results[id_]["pred_pegasus_wst"].split(':')[-1].split(',')[idx]))
                    automatic_scores.append(float(automatic_results[id_]["pred_pegasus_nst"].split(':')[-1].split(',')[idx]))
                    automatic_scores.append(float(automatic_results[id_]["pred_pegasus_np"].split(':')[-1].split(',')[idx]))
                    automatic_scores.append(float(automatic_results[id_]["pred_naive"].split(':')[-1].split(',')[idx]))

        assert len(human_scores) == len(ranking_scores) == len(automatic_scores)
        print(f'Number of scores: {len(human_scores)}')

        human_ranking_corr = spearmanr(human_scores, ranking_scores)
        human_automatic_corr = spearmanr(human_scores, automatic_scores)
        outs[aspect] = {"Human-Ranking Corr": human_ranking_corr,
                        "Human-Automatic Corr": human_automatic_corr}
        with open(os.path.join(dir_path, f"{aspect}_corr_scores.txt"), "w", encoding="utf-8") as file:
            file.write('\t'.join([str(s) for s in human_scores]))
            file.write('\n')
            file.write('\t'.join([str(s) for s in ranking_scores]))
            file.write('\n')
            file.write('\t'.join([str(s) for s in automatic_scores]))
    return outs


def calculate_fleiss_kappa(human_results: dict):
    outs = {}
    subj_rater_matrix_all = []
    score_mapping = {"1": "bad", "2": "bad", "3": "bad", "4": "good", "5": "good"}

    for aspect in ["emotion", "similarity", "fluency", "context"]:
        subj_rater_matrix_by_aspect = []

        for utt_results in human_results.values():
            aspect_results = utt_results[aspect]
            for model in ['target', 'wst', 'nst', 'nps', 'naive']:
                model_scores = []
                for rater_results in aspect_results.values():
                    model_scores.append(score_mapping[rater_results[f'{aspect}_{model}']])
                subj_rater_matrix_by_aspect.append(model_scores)
        subj_rater_matrix_all.extend(subj_rater_matrix_by_aspect)

        data, cats = inter_rater.aggregate_raters(np.array(subj_rater_matrix_by_aspect))
        fleiss_irr = inter_rater.fleiss_kappa(data, method='fleiss')
        outs[aspect] = fleiss_irr

    data, cats = inter_rater.aggregate_raters(np.array(subj_rater_matrix_all))
    fleiss_irr = inter_rater.fleiss_kappa(data, method='fleiss')
    outs["all_aspects"] = fleiss_irr

    with open(os.path.join(dir_path, "rater_scores.txt"), "w", encoding="utf-8") as file:
        for model_scores in subj_rater_matrix_all:
            for score in model_scores:
                file.write(score)
                file.write('\t')
            file.write('\n')
    return outs


def get_stats(scores_by_config: dict):
    outs = {}
    for aspect, results in scores_by_config.items():
        if aspect not in outs:
            outs[aspect] = {}

        for config, scores in results.items():
            outs[aspect][config] = {
                "mean": round(statistics.mean(scores), 3),
                "median": statistics.median(scores),
                "mode": statistics.mode(scores),
                "stdev": round(statistics.stdev(scores), 3),
                "variance": round(statistics.variance(scores), 3)
            }
    return outs


def main():
    results = read_file(os.path.join(dir_path, "amt_results.json"))
    scores_by_models, scores_by_emotions = get_scores(results)

    # stats calculation
    print("Calculate stats...")
    outs = {
        "by_model": get_stats(scores_by_models),
        "by_emotion": get_stats(scores_by_emotions)
    }

    # plot
    print("Plotting...")
    plot_stacked_bar_chart(scores_by_models)

    # correlation
    print("Calculating Spearman correlations...")
    outs["correlations"] = calculate_spearman(
        human_results=results,
        path_to_automatic_scores=os.path.join(dir_path, "human_eval_candidates.json"))

    # inter-annotator agreement
    print("Calculating Fleiss agreement...")
    outs["fleiss"] = calculate_fleiss_kappa(human_results=results)

    with open(os.path.join(dir_path, "human_eval_results.json"), "w", encoding="utf-8") as jfile:
        json.dump(outs, jfile, indent=4)


if __name__ == '__main__':
    dir_path = '../data/human_eval_files'
    main()
