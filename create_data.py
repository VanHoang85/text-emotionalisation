import os
import re
import json
import random
import pickle
import argparse
from tqdm import tqdm
from collections import Counter
from data_utils.pattern_matcher import PatternMatcher
from sklearn.model_selection import train_test_split


def write_to_json_file(filename, data, filepath):
    with open(os.path.join(filepath, filename), 'w', encoding='utf-8') as jfile:
        json.dump(data, jfile, indent=4)


def remove_unicode(sentence: str) -> str:
    return sentence.strip().lower().encode("ascii", "ignore").decode('utf-8')


def process_file(path_to_parallel_dir, file, emo_data, to_do_data):
    with open(os.path.join(path_to_parallel_dir, file), 'r', encoding='utf-8') as jfile:
        data = json.load(jfile)

    dataset = file[:-5].split('_')[0]
    emotion = file[:-5].split('_')[1]

    for id_, utt_info in data.items():
        emo_sent = remove_unicode(utt_info["emotion_sent"])
        if emo_sent not in all_utts:
            key = f"{dataset}_{id_}"
            utt = {"neutral_sent": utt_info["neutral_sent"] if "to_do" not in file else "",
                   "emotion_sent": emo_sent,
                   "emo": emotion,
                   "context": utt_info["context"] if "context" in utt_info else "",
                   "context_emo": utt_info["context_emo"] if "context_emo" in utt_info else "",
                   "phrases": utt_info["phrases"] if "phrases" in utt_info else ""}

            all_utts.append(emo_sent)
            if "to_do" not in file:
                emo_data.update({key: utt})
            elif len(utt['emotion_sent'].split(' ')) > args.min_len:
                to_do_data.update({key: utt})


def create_save_dataset(id_list: list, split: str, filepath: str, data):
    dataset = {}
    random.shuffle(id_list)
    for id_ in id_list:
        if id_ in data:
            if "classifier" in filepath:
                utt = {"utterance": data[id_]["emotion_sent"],
                       "emo": data[id_]["emo"]}
            else:
                utt = data[id_]
            dataset.update({id_: utt})
    write_to_json_file(f"{split}.json", dataset, filepath)


def split_train_val_test(data, test_size):
    id_list = list(data.keys())
    emo_list = [data[id_]["emo"] for id_ in id_list]
    train_id, test_id = train_test_split(id_list, test_size=test_size, shuffle=True, stratify=emo_list)

    emo_list = [data[id_]["emo"] for id_ in train_id]
    train_id, val_id = train_test_split(train_id, test_size=test_size, shuffle=True, stratify=emo_list)
    return train_id, val_id, test_id


def create_stylized_data(train_id, val_id, test_id, emo_data, to_do_data):
    print("Create stylized data")
    filepath = os.path.join(os.path.join(args.output_dir, "style"))

    create_save_dataset(train_id, split='train', filepath=filepath, data=emo_data)
    create_save_dataset(val_id, split='validation', filepath=filepath, data=emo_data)
    create_save_dataset(test_id, split='test', filepath=filepath, data=emo_data)

    write_to_json_file("to_do.json", to_do_data, filepath=filepath)


def check_emo_condition(id_, emo_data, utt, counter) -> bool:
    return True if (id_ not in emo_data and
                   len(utt['emotion_sent'].split(' ')) >= args.min_len and
                   counter[utt['emo']] < args.max_samples) \
        else False


def create_classifier_data(train_id, val_id, test_id, neutral_data: dict, emo_data: dict, to_do_data: dict):
    print("Create classifier data")
    counter = Counter()
    filepath = os.path.join(os.path.join(args.output_dir, "classifier"))

    # get ids for neutral utts
    neutral_train_id, neutral_val_id, neutral_test_id = split_train_val_test(neutral_data, args.classifier_test_size)
    todo_train_id, todo_val_id, toto_test_id = split_train_val_test(to_do_data, args.classifier_test_size)

    train_id = list(set(train_id + neutral_train_id + todo_train_id))
    val_id = list(set(val_id + neutral_val_id + todo_val_id))
    test_id = list(set(test_id + neutral_test_id + toto_test_id))

    # add neutral utts to data
    for id_, utt in neutral_data.items():
        if check_emo_condition(id_, emo_data, utt, counter):
            emo_data.update({id_: utt})
            counter.update([utt['emo']])

    # add more emo utts to data
    for id_, utt in to_do_data.items():
        if check_emo_condition(id_, emo_data, utt, counter):
            emo_data.update({id_: utt})
            counter.update([utt['emo']])

    create_save_dataset(train_id, split='train', filepath=filepath, data=emo_data)
    create_save_dataset(val_id, split='validation', filepath=filepath, data=emo_data)
    create_save_dataset(test_id, split='test', filepath=filepath, data=emo_data)


def read_emotion_dataset():
    emo_data, neutral_data, to_do_data = {}, {}, {}
    for folder in os.listdir(args.path_to_emotion_data_dir):
        path_to_folder = os.path.join(args.path_to_emotion_data_dir, folder)
        if not os.path.isdir(path_to_folder):
            continue

        print('Processing', folder)
        path_to_parallel_dir = os.path.join(path_to_folder, "parallel_corpus")
        for file in os.listdir(path_to_parallel_dir):
            if not file.endswith('json') or 'wo_ep' in file:
                continue
            process_file(path_to_parallel_dir, file, emo_data, to_do_data)

        # create classifier data, aka add neutral utterance for classifier data type
        path_to_parallel_dir = os.path.join(path_to_folder, "extra_lexicons")
        for file in os.listdir(path_to_parallel_dir):
            if "neutral_wo_ep" not in file:
                continue
            process_file(path_to_parallel_dir, file, neutral_data, to_do_data)

    # split into 3 sets
    train_id, val_id, test_id = split_train_val_test(emo_data, args.style_test_size)

    create_stylized_data(train_id, val_id, test_id, emo_data, to_do_data)  # create stylized emo data
    create_classifier_data(train_id, val_id, test_id, neutral_data, emo_data, to_do_data)  # create classifier data


def check_sent_condition(tokens: list) -> bool:
    return False if (len(tokens) < 7 or len(tokens) > 25) else True


def check_overlap_condition(overlap: set, tokens: list) -> bool:
    return True if len(overlap) / len(set(tokens)) <= 0.5 else False


def check_token_overlap(sent1_tokens: list, sent2_tokens: list) -> bool:
    overlap = set(sent1_tokens).intersection(set(sent2_tokens))
    return True if (check_overlap_condition(overlap, sent1_tokens)
                    and check_overlap_condition(overlap, sent2_tokens)) \
        else False


def check_len_diff(sent1_tokens: list, sent2_tokens: list) -> bool:
    return True if abs(len(sent1_tokens) - len(sent2_tokens)) < 5 else False


def check_para_condition(sentence1: str, sentence2: str) -> bool:
    sent1_tokens = re.sub(r'[^\w\s]', '', sentence1).strip().split()
    sent2_tokens = re.sub(r'[^\w\s]', '', sentence2).strip().split()
    if not check_sent_condition(sent1_tokens) and not check_sent_condition(sent2_tokens):
        return False
    if not check_len_diff(sent1_tokens, sent2_tokens):
        return False
    if not check_token_overlap(sent1_tokens, sent2_tokens):
        return False
    return True


def read_generic_datasets():
    references, paraphrases = [], []
    for folder in os.listdir(args.path_to_generic_data_dir):
        path_to_folder = os.path.join(args.path_to_generic_data_dir, folder)
        if not os.path.isdir(path_to_folder):
            continue

        print('Processing', folder)
        if 'nmt' in path_to_folder:
            for split in ['train', 'dev']:
                with open(os.path.join(path_to_folder, f'{split}.pickle'), 'rb') as file:
                    data = pickle.load(file)

                for data_point in data:
                    references.append(data_point[3].strip())
                    paraphrases.append(data_point[4].strip())

        elif 'parabank' in path_to_folder:
            with open(os.path.join(path_to_folder, 'parabank2.tsv'), 'r', encoding='utf-8') as file:
                for line in tqdm(file):
                    line = line.strip().split('\t')
                    if float(line[0]) > 0.039 or float(line[0]) < 0.031 or len(line) < 3:
                        continue

                    if check_para_condition(line[1], line[2]):
                        references.append(line[1].strip())
                        paraphrases.append(line[2].strip())

    assert len(references) == len(paraphrases)
    print(f"len para: {len(paraphrases)}")

    # split into sets
    idx_ls = [i for i in range(len(references))]
    train_idx, test_idx = train_test_split(idx_ls, test_size=args.generic_test_size, shuffle=True)
    train_idx, val_idx = train_test_split(train_idx, test_size=args.generic_test_size, shuffle=True)

    print(len(train_idx), len(val_idx), len(test_idx))

    filepath_out = os.path.join(os.path.join(args.output_dir, "generic"))
    if not os.path.exists(filepath_out):
        os.makedirs(filepath_out)

    splits = ['train', 'validation', 'test']
    idx_lists = [train_idx, val_idx, test_idx]

    for split, split_idx in zip(splits, idx_lists):
        with open(os.path.join(filepath_out, f'{split}.txt'), 'w', encoding='utf-8') as file:
            for idx in split_idx:
                file.write(references[idx])
                file.write('\t')
                file.write(paraphrases[idx])
                file.write('\n')


def read_generic_datasets_for_classifier():
    paraphrases = {}

    for folder in os.listdir(args.path_to_generic_data_dir):
        path_to_folder = os.path.join(args.path_to_generic_data_dir, folder)
        if not os.path.isdir(path_to_folder):
            continue

        print('Processing', folder)
        if 'nmt' in path_to_folder:
            c = 0
            for split in ['train', 'dev']:
                with open(os.path.join(path_to_folder, f'{split}.pickle'), 'rb') as file:
                    data = pickle.load(file)

                for data_point in data:
                    c += 1
                    paraphrases.update({f'nmt_{c}': {'sent1': data_point[3].strip(),
                                                     'sent2': data_point[4].strip()}})

        elif 'parabank' in path_to_folder:
            c = 0
            with open(os.path.join(path_to_folder, 'parabank2.tsv'), 'r', encoding='utf-8') as file:
                for line in tqdm(file):
                    line = line.strip().split('\t')
                    if float(line[0]) > 0.75 or len(line) < 3:
                        continue

                    if check_para_condition(line[1], line[2]):
                        c += 1
                        paraphrases.update({f'pb_{c}': {'sent1': line[1].strip(),
                                                        'sent2': line[2].strip()}})

    print(f"len para: {len(paraphrases)}")
    filepath_out = os.path.join(os.path.join(args.output_dir, "generic"))
    with open(os.path.join(filepath_out, 'classifier.json'), 'w', encoding='utf-8') as jfile:
        json.dump(paraphrases, jfile, indent=4)


def split_file(data, out_dir):
    c = 1
    part = {}
    size = len(data) // 99
    for id_, utt in data.items():
        if len(part) == size:
            with open(os.path.join(out_dir, f'generic_same_emos_p{c}.json'), 'w', encoding='utf-8') as jfile:
                json.dump(part, jfile, indent=4)
            c += 1
            part = {}
        part.update({id_: utt})
    if len(part) != 0:
        with open(os.path.join(out_dir, f'generic_same_emos_p{c}.json'), 'w', encoding='utf-8') as jfile:
            json.dump(part, jfile, indent=4)


def read_generic_for_silver_data():
    path_to_gold_file = './data/style/train.json'
    path_to_gold_neu_para_file = 'outputs/pegasus_para_train_neutralizer.json'
    path_to_gold_emo_para_file = 'outputs/pegasus_para_train_emo_stylizer.json'
    path_to_raw_silver_file = './data/style/generic_diff_emos_wn_final_filtered.json'

    pattern_matcher = PatternMatcher('./lexicons')

    count_gold, count_silver = Counter(), Counter()
    silver_data, gold_para_data = {}, {}

    with open(path_to_gold_file, 'r', encoding='utf-8') as jfile:
        gold_data = json.load(jfile)
    with open(path_to_gold_neu_para_file, 'r', encoding='utf-8') as jfile:
        raw_gold_para_neu_data = json.load(jfile)
    with open(path_to_gold_emo_para_file, 'r', encoding='utf-8') as jfile:
        raw_gold_para_emo_data = json.load(jfile)
    with open(path_to_raw_silver_file, 'r', encoding='utf-8') as jfile:
        raw_silver_data = json.load(jfile)

    for utt in gold_data.values():
        count_gold.update([utt["emo"]])

    # create gold data from paraphrased neutral sentences
    num_change_neu, num_change_emo = 0, 0
    for id_, utt_info in gold_data.items():
        if not raw_gold_para_neu_data[id_]["preds_neutral"]:  # in case no paraphrased
            gold_para_data.update({id_: utt_info})
            continue

        reference_neu = raw_gold_para_neu_data[id_]["input"]
        candidate_neu = raw_gold_para_neu_data[id_]["preds_neutral"][0]["prediction"]

        if abs(len(reference_neu.strip().split()) - len(candidate_neu.strip().split())) <= 3:
            utt_info["neutral_sent"] = candidate_neu
            gold_para_data.update({id_: utt_info})
            num_change_neu += 1
        else:
            # check if we can get from emo sent
            emo_key = f"preds_{utt_info['emo']}"
            if not raw_gold_para_emo_data[id_][emo_key]:
                continue

            reference_emo = raw_gold_para_emo_data[id_]["input"]
            candidate_emo = raw_gold_para_emo_data[id_][emo_key][0]["prediction"]

            if abs(len(reference_emo.strip().split()) - len(candidate_emo.strip().split())) <= 3:
                utt_info["emotion_sent"] = candidate_emo
                gold_para_data.update({id_: utt_info})
                num_change_emo += 1

        # gold_para_data.update({id_: utt_info})

    for id_, utt in tqdm(raw_silver_data.items()):
        emotion = utt["emo_sent1"] if utt["emo_sent1"] != "neutral" else utt["emo_sent2"]
        emotion_sent = utt["sent1"] if utt["emo_sent1"] != "neutral" else utt["sent2"]
        neutral_sent = utt["sent1"] if utt["emo_sent1"] == "neutral" else utt["sent2"]

        if emotion == "happiness":
            e_phrases = pattern_matcher.find_patterns(emotion_sent)
            if "happiness" not in e_phrases:
                continue
            cluster_ids = set(e_phrases.keys())
            bad_clusters = ["words", "cluster_73", "cluster_61", "cluster_168"]
            if len(cluster_ids.intersection(set(bad_clusters))) == len(cluster_ids):
                continue

        count_silver.update([emotion])
        new_utt = {"neutral_sent": neutral_sent.strip(),
                   "emotion_sent": emotion_sent.strip(),
                   "emo": emotion.strip(),
                   "context": "",
                   "context_emo": "",
                   "phrases": []}
        silver_data.update({id_: new_utt})

    print(f"Number of neutral candidates: {num_change_neu} out of {len(gold_para_data)}")
    print(f"Number of emotion candidates: {num_change_emo} out of {len(gold_para_data)}")
    print(len(silver_data))

    # how much silver data to get based on gold data
    percentages = [2.0]  # 0.5, 1.0,
    for percentage in percentages:
        path_to_silver_out_file = f'./data/style/train_{percentage}.json'
        all_silver_ids = list(silver_data.keys())
        random.shuffle(all_silver_ids)

        silver_outs = {}
        count_outs = Counter()

        for id_ in all_silver_ids:
            utt = silver_data.get(id_)
            emotion = utt['emo']

            if emotion in count_outs and count_outs.get(emotion) > count_gold.get(emotion) * percentage:
                continue

            count_outs.update([emotion])
            silver_outs.update({id_: utt})

            # check if reach max to break loop soon
            if len(count_outs.keys()) == 3:
                if count_outs.get("anger") < count_gold.get("anger") * percentage:
                    continue
                if count_outs.get("happiness") < count_gold.get("happiness") * percentage:
                    continue
                if count_outs.get("sadness") < count_gold.get("sadness") * percentage:
                    continue
                break

        silver_outs.update(gold_para_data)
        print(f'Data size outs: {count_outs}')
        print(len(silver_outs))
        with open(path_to_silver_out_file, 'w', encoding='utf-8') as jfile:
            json.dump(silver_outs, jfile, indent=4)


def create_new_testset():
    path_to_neutral_file = 'datasets/annotated/RECCON/extra/dailydialog_neutral.json'
    path_to_out_file = './data/style/new_test.json'
    count = Counter()
    testset = {}

    with open(path_to_neutral_file, 'r', encoding='utf-8') as jfile:
        data = json.load(jfile)

    all_ids = list(data.keys())
    random.shuffle(all_ids)
    for id_ in tqdm(all_ids):
        utt_info = data[id_]
        emotion = utt_info["context_emo"]
        neu_sent = utt_info["emotion_sent"]
        context = utt_info["context"]

        if len(neu_sent.strip().split()) > 25 or len(neu_sent.strip().split()) < 5 or \
                len(context.strip().split()) > 25 or len(context.strip().split()) < 5:
            continue
        # if emotion in count and count.get(emotion) >= 15:
        #    continue

        count.update([emotion])
        testset.update({id_: {"neutral_sent": neu_sent,
                              "emotion_sent": "",
                              "emo": "neutral",
                              "context": context,
                              "context_emo": emotion,
                              "phrases": []}})

        # if sum(count.values()) >= len(count.keys()) * 15:
        #    break

    print(len(testset))
    print(count)
    with open(path_to_out_file, 'w', encoding='utf-8') as jfile:
        json.dump(testset, jfile, indent=4)


def create_human_test():
    path_to_pred_file = './outputs/pegasus_new_test_all_emos.json'
    path_to_out_file = 'human_eval_files/human_evaluation.json'
    counter = Counter()
    outs = {}

    with open(path_to_pred_file, 'r', encoding='utf-8') as jfile:
        data = json.load(jfile)

    all_ids = list(data.keys())
    all_ids.remove('style_acc')
    random.shuffle(all_ids)

    for id_ in tqdm(all_ids):
        utt_info = data[id_]
        if utt_info['context_emo'] not in ['neutral', 'anger', 'happiness', 'sadness']:
            continue

        pred_emo = ''
        while f'preds_{pred_emo}' not in utt_info or len(utt_info[f'preds_{pred_emo}']) == 0:
            pred_emo = random.choice(['anger', 'happiness', 'sadness'])

        emo_com = utt_info['context_emo'] + '_' + pred_emo
        if emo_com in counter and counter.get(emo_com) >= 10:
            continue

        counter.update([emo_com])
        pred = random.choice(utt_info[f'preds_{pred_emo}'])
        outs.update({id_: {"input": utt_info["input"],
                           "context": utt_info["context"],
                           "context_emo": utt_info["context_emo"],
                           "pred": pred["prediction"],
                           "pred_emo": pred_emo}})

        if sum(counter.values()) >= 120:
            break

    print(len(outs))
    print(counter)
    with open(path_to_out_file, 'w', encoding='utf-8') as jfile:
        json.dump(outs, jfile, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-ie", "--path_to_emotion_data_dir", type=str,
                        default="/Users/vanhoang/PycharmProjects/affectiveNLG/datasets/annotated")
    parser.add_argument("-ig", "--path_to_generic_data_dir", type=str,
                        default="/Users/vanhoang/PycharmProjects/affectiveNLG/datasets/generic")
    parser.add_argument("-o", "--output_dir", type=str, default="./data")
    parser.add_argument("--min_len", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=15000)
    parser.add_argument("--style_test_size", type=float, default=0.15)
    parser.add_argument("--classifier_test_size", type=float, default=0.2)
    parser.add_argument("--generic_test_size", type=float, default=10000)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    all_utts = []
    # read_emotion_dataset()
    read_generic_datasets()
    # read_generic_datasets_for_classifier()
    # read_generic_for_silver_data()
    # create_new_testset()
    # create_human_test()
