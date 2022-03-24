import os
import re
import csv
import json
import random
import itertools as it
from collections import Counter
from human_eval_files.mapping import hit2utt, utt2emo
from human_eval_files.templates import templates


def read_file(path_to_file):
    with open(path_to_file, 'r', encoding='utf-8') as jfile:
        data = json.load(jfile)
    return data


def compare_models():
    naive_data = read_file(os.path.join(dir_path, 'pegasus_no_para_test_emo_wbf.json'))
    bart_file = read_file(os.path.join(dir_path, 'pegasus_with_style_loss_test_emo_wbf.json'))
    pegasus_file = read_file(os.path.join(dir_path, 'pegasus_no_style_loss_test_emo_wbf.json'))
    # pegasus_file = read_file(os.path.join(dir_path, 'pegasus_with_style_loss_test_emo_wbf.json'))
    outs = {}

    all_ids = list(pegasus_file.keys())
    random.shuffle(all_ids)
    max_num_pred = 10

    for id_ in all_ids:
        # print(pegasus_file[id_])
        input_text = pegasus_file[id_]["input"]
        if len(input_text.strip().split()) < 5 or len(input_text.strip().split()) > 15:
            continue

        for key in pegasus_file[id_].keys():
            if 'preds' in key:
                emo = key.split('_')[1]
                preds_pegasus = pegasus_file[id_][key][:max_num_pred]
                preds_bart = bart_file[id_][key][:max_num_pred]
                preds_naive = naive_data[id_][key][:max_num_pred]

                outs[id_] = {"input": input_text,
                             "context": pegasus_file[id_]["context"],
                             "context_emo": pegasus_file[id_]["context_emo"],
                             "target_sent": pegasus_file[id_]["target_sent"],
                             "target_emo": emo,
                             "pred_pegasus_wst": [f'{pred["prediction"]}: {pred["emo_score"]}, {pred["sim_score"]}, {pred["grm_score"]}' for pred in preds_bart],
                             "pred_pegasus_nst": [f'{pred["prediction"]}: {pred["emo_score"]}, {pred["sim_score"]}, {pred["grm_score"]}' for pred in preds_pegasus],
                             "pred_pegasus_np": [f'{pred["prediction"]}: {pred["emo_score"]}, {pred["sim_score"]}, {pred["grm_score"]}' for pred in preds_naive]}

    # save to file
    with open(os.path.join(dir_path, "compare_pegasus_wbf.json"), 'w', encoding='utf-8') as jfile:
        json.dump(outs, jfile, indent=4)


def check_len_condition(utterance: str):
    tokens = utterance.strip().split()
    return True if 5 <= len(tokens) <= 20 else False


def check_pred_condition(naive_preds: list, pegasus_preds: dict, reference: str):
    preds = set()
    preds.add(re.sub(r'[^\w\s]', '', reference).strip())
    preds.add(re.sub(r'[^\w\s]', '', naive_preds[0].strip().split(":")[0]).strip())
    for model_type in ["pred_pegasus_wst", "pred_pegasus_nst", "pred_pegasus_np"]:
        try:
            preds.add(re.sub(r'[^\w\s]', '', pegasus_preds[model_type][0].strip().split(":")[0]).strip())
        except IndexError:
            return False
    return True if len(preds) == 5 else False


def human_eval_file():
    naive_data = read_file(os.path.join(dir_path, 'compare_with_style_loss_wbf.json'))
    pegasus_data = read_file(os.path.join(dir_path, 'compare_pegasus_wbf.json'))
    outs = {}

    for id_, utt_info in pegasus_data.items():
        if not utt_info["context"]:
            continue

        if check_len_condition(utt_info["context"]) and check_len_condition(utt_info["input"]) \
                and check_pred_condition(naive_data[id_]["pred_naive"], utt_info, utt_info["target_sent"]):
            outs[id_] = {"input": utt_info["input"],
                         "context": utt_info["context"],
                         "context_emo": utt_info["context_emo"],
                         "target_emo": utt_info["target_emo"],
                         "reference": utt_info["target_sent"],
                         "pred_naive": naive_data[id_]["pred_naive"][0],
                         "pred_pegasus_wst": utt_info["pred_pegasus_wst"][0],
                         "pred_pegasus_nst": utt_info["pred_pegasus_nst"][0],
                         "pred_pegasus_np": utt_info["pred_pegasus_np"][0]}

    with open(os.path.join(dir_path, "human_eval_candidates.json"), 'w', encoding='utf-8') as jfile:
        json.dump(outs, jfile, indent=4)


def get_overlap(path_to_file):
    raw_data = read_file(path_to_file)
    data = {}

    for utt_info in raw_data.values():
        for key in utt_info.keys():
            if 'pred' in key:
                if key not in data:
                    data[key] = []
                data[key].extend([pred.split(':')[0] for pred in utt_info[key]])
    combinations = it.combinations(list(data.keys()), 2)
    for combo in combinations:
        ls1 = set(data[combo[0]])
        ls2 = set(data[combo[1]])
        overlap = ls1.intersection(ls2)
        print(f'Overlap between {combo[0]} (len {len(ls1)}) and {combo[1]} (len {len(ls2)}) is {len(overlap)}')

    ids = list(data.keys())
    all_overlap = set(data[ids[0]]).intersection(set(data[ids[1]])).intersection(set(data[ids[2]]))
    print(f'Overlap all: {len(all_overlap)}')


def amt_templates():
    path_to_file = "human_eval_files/human_eval_candidates.json"
    counter = Counter()
    used_ids = []
    mapping = {"anger": "angry", "happiness": "happy", "sadness": "sad"}
    data = read_file(path_to_file)
    all_ids = list(data.keys())
    random.shuffle(all_ids)

    for id_ in all_ids:
        emotion = mapping[data[id_]["target_emo"]]
        if emotion in counter and counter[emotion] >= 7:
            continue

        temps = templates(emotion, data[id_])
        counter.update([emotion])
        used_ids.append(id_)

        for aspect, temp in temps.items():
            dir_out = os.path.join("human_eval_files/human_eval_templates", emotion)
            if not os.path.exists(dir_out):
                os.mkdir(dir_out)

            with open(os.path.join(dir_out, f"{id_.replace(':', '_')}_{aspect}"), "w", encoding="utf-8") as file:
                file.write(temp)

        if sum(counter.values()) >= 21:
            break
    print(used_ids)


def read_amt_results():
    path_to_dir = "human_eval_files/amt_files"
    outs = {}

    for amt_file in os.listdir(path_to_dir):
        if not amt_file.endswith("csv"):
            continue

        with open(os.path.join(path_to_dir, amt_file), 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if not row["HITId"]:
                    continue

                utt_id, aspect = hit2utt[row["HITId"]].rsplit('_', maxsplit=1)
                worker_id = row["WorkerId"]

                if utt_id not in outs:
                    outs[utt_id] = {}

                outs[utt_id]["target_emo"] = utt2emo[utt_id]
                if aspect not in outs[utt_id]:
                    outs[utt_id][aspect] = {}

                result = {
                    "LifetimeApprovalRate": row["LifetimeApprovalRate"],
                    "WorkTimeInSeconds": row["WorkTimeInSeconds"],
                    f"{aspect}_target": row[f"Answer.{aspect}-target"],
                    f"{aspect}_wst": row[f"Answer.{aspect}-wst"],
                    f"{aspect}_nst": row[f"Answer.{aspect}-nst"],
                    f"{aspect}_nps": row[f"Answer.{aspect}-nps"],
                    f"{aspect}_naive": row[f"Answer.{aspect}-naive"]
                }
                outs[utt_id][aspect][worker_id] = result
    with open("human_eval_files/amt_results.json", "w", encoding="utf-8") as jfile:
        json.dump(outs, jfile, indent=4)


def get_automatic_scores(output_with_scores: str):
    output, scores = output_with_scores.strip().split(':')
    scores = scores.split(',')
    return output, scores


def get_human_scores(all_scores: dict, system: str, aspect: str):
    scores = []
    if system == "reference":
        system = "target"
    elif system == "np":
        system = "nps"

    for worker_eval in all_scores.values():
        scores.append(worker_eval[f'{aspect}_{system}'])
    assert len(scores) == 3
    return ', '.join(scores)


def merge_auto_human_results():
    automatic_results = read_file("human_eval_files/human_eval_candidates.json")
    human_results = read_file("human_eval_files/amt_results.json")
    outs = {}

    for id_ in human_results.keys():
        utt_info = automatic_results[id_]
        out = {
            "input": utt_info["input"],
            "context": utt_info["context"],
            "context_emo": utt_info["context_emo"],
            "target_emo": utt_info["target_emo"]
        }
        for system in ["reference", "wst", "nst", "np", "naive"]:
            if system == "reference":
                key = "reference"
            elif system == "naive":
                key = "pred_naive"
            else:
                key = f'pred_pegasus_{system}'
            candidate, automatic_scores = get_automatic_scores(utt_info[key])
            out[key] = {
                "candidate": candidate,
                "emotion": {
                    "automatic": automatic_scores[0].strip(),
                    "human": get_human_scores(human_results[id_]["emotion"], system, "emotion")
                },
                "similarity": {
                    "automatic": automatic_scores[1].strip(),
                    "human": get_human_scores(human_results[id_]["similarity"], system, "similarity")
                },
                "fluency": {
                    "automatic": automatic_scores[2].strip(),
                    "human": get_human_scores(human_results[id_]["fluency"], system, "fluency")
                },
                "context": {
                    "automatic": "Na",
                    "human": get_human_scores(human_results[id_]["context"], system, "context")
                },
            }
        outs.update({id_: out})

    with open("human_eval_files/merge_automatic_human_scores.json", "w", encoding="utf-8") as jfile:
        json.dump(outs, jfile, indent=4)


if __name__ == '__main__':
    dir_path = './outputs'
    # files = ['naive_phrases_test_emo_wbf.json',
    #          'bart_with_style_loss_test_emo_wbf.json',
    #          'pegasus_with_style_loss_test_emo_wbf.json']

    # compare_models()
    # get_overlap(os.path.join(dir_path, "compare_pegasus_wbf.json"))
    # human_eval_file()
    # amt_templates()
    # read_amt_results()
    merge_auto_human_results()
