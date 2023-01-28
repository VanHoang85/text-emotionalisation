import os
import sys
import json
import csv


def save_to_json(path_to_file, data):
    with open(path_to_file, 'w', encoding='utf-8') as jfile:
        json.dump(data, jfile, indent=4)


def meld_converter():
    in_path_to_meld = sys.argv[2]
    out_path_to_meld = sys.argv[3]

    splits = ['train', 'dev', 'test']
    corpus = {}

    for split in splits:
        with open(os.path.join(in_path_to_meld, f'{split}_sent_emo.csv'), encoding='utf-8') as csv_file:
            meld = csv.DictReader(csv_file)

            data_split = []
            prev_utt, prev_emo = '', ''
            for row in meld:
                if int(row["Utterance_ID"]) == 0:
                    prev_utt, prev_emo = '', ''  # assign empty context when moving to next dialog

                curr_utt, curr_emo = row['Utterance'], row['Emotion']
                utterance = {'id': f'dialog{row["Dialogue_ID"]}_{row["Utterance_ID"]}',
                             'speaker': row['Speaker'],
                             'utterance': curr_utt,
                             'context': prev_utt,
                             'context_emo': prev_emo,
                             'style': curr_emo}
                prev_utt, prev_emo = row['Utterance'], row['Utterance']
                data_split.append(utterance)
            corpus[f'{split}_friends'] = data_split
    save_to_json(os.path.join(out_path_to_meld, f'friends.json'), corpus)


def reccon_converter():
    in_path_to_reccon = sys.argv[2]
    out_path_to_reccon = sys.argv[3]

    corpus = {}
    for filename in os.listdir(in_path_to_reccon):
        if not filename.endswith('json'):
            continue

        with open(os.path.join(in_path_to_reccon, filename), 'r', encoding='utf-8') as jfile:
            reccon = json.load(jfile)

        data_split = []
        for idx, dialog in reccon.items():
            for turn in dialog[0]:
                prev_utt = dialog[0][turn['turn'] - 2]['utterance'] if turn['turn'] != 1 else ''
                prev_emo = dialog[0][turn['turn'] - 2]['style'] if turn['turn'] != 1 else ''

                utterance = {'id': f'{idx}_{turn["turn"]}',
                             'speaker': turn['speaker'],
                             'utterance': turn['utterance'],
                             'context': prev_utt,
                             'context_emo': prev_emo,
                             'style': turn['style']
                             }

                data_split.append(utterance)
        corpus[f'{filename[:-5]}'] = data_split
    save_to_json(os.path.join(out_path_to_reccon, f'dailydialog.json'), corpus)


def goemo_converter():
    in_path_to_goemo = sys.argv[2]
    out_path_to_goemo = sys.argv[3]

    if not os.path.exists(out_path_to_goemo):
        os.mkdir(out_path_to_goemo)

    with open(os.path.join(in_path_to_goemo, 'ekman_mapping.json'), encoding='utf-8') as jfile:
        ekman_mapping = json.load(jfile)

    with open(os.path.join(in_path_to_goemo, 'emotions.txt'), 'r', encoding='utf-8') as file:
        emotions_mapping = file.read()
    emotions_mapping = emotions_mapping.strip().split('\n')

    splits = ['train', 'dev', 'test']
    corpus = {}

    for split in splits:
        with open(os.path.join(in_path_to_goemo, f'{split}.tsv'), encoding='utf-8') as csv_file:
            goemo = csv.reader(csv_file, delimiter='\t')

            idx = 1
            data_split = []
            for row in goemo:
                emotions = row[1].strip().split(',')
                for emotion in emotions:
                    utterance = {'id': f'{split}_{idx}',
                                 'speaker': row[2],
                                 'utterance': row[0],
                                 'style': to_ekman(ekman_mapping, emotions_mapping[int(emotion)])}
                    data_split.append(utterance)
                    idx += 1
            corpus[f'{split}_goemo'] = data_split
    save_to_json(os.path.join(out_path_to_goemo, f'goemotions.json'), corpus)


def to_ekman(ekman_mapping: dict, emotion) -> str:
    for ekman_emo, emo_list in ekman_mapping.items():
        if emotion in emo_list:
            return ekman_emo
    return 'neutral'


def carer_converter():
    in_path_to_carer = sys.argv[2]
    out_path_to_carer = sys.argv[3]

    if not os.path.exists(out_path_to_carer):
        os.mkdir(out_path_to_carer)

    splits = ['train', 'val', 'test']
    corpus = {}

    for split in splits:
        with open(os.path.join(in_path_to_carer, f'{split}.txt'), encoding='utf-8') as file:
            idx = 1
            data_split = []
            for line in file:
                line = line.strip().split(';')
                utterance = {'id': f'{split}_{idx}',
                             'utterance': line[0],
                             'style': line[1]}
                data_split.append(utterance)
                idx += 1
        corpus[f'{split}_carer'] = data_split
    save_to_json(os.path.join(out_path_to_carer, f'carer.json'), corpus)


def empa_converter():
    in_path_to_empa = sys.argv[2]
    out_path_to_empa = sys.argv[3]

    if not os.path.exists(out_path_to_empa):
        os.mkdir(out_path_to_empa)

    with open(os.path.join(in_path_to_empa, 'ekman_mapping.json'), encoding='utf-8') as jfile:
        ekman_mapping = json.load(jfile)

    splits = ['train', 'valid', 'test']
    corpus = {}

    for split in splits:
        with open(os.path.join(in_path_to_empa, f'{split}.csv'), encoding='utf-8') as file:
            header = file.readline().strip().split(',')

            data_split = []
            prev_utt, prev_emo = '', ''
            for line in file:
                line = line.strip().split(',')
                if int(line[header.index("utterance_idx")]) == 1:
                    prev_utt, prev_emo = '', ''

                curr_utt = line[header.index('utterance')].replace('_comma_', ', ')
                curr_emo = to_ekman(ekman_mapping, line[header.index('context')])
                utterance = {'id': f'{line[header.index("conv_id")]}_{line[header.index("utterance_idx")]}',
                             'speaker': line[header.index('speaker_idx')],
                             'utterance': curr_utt,
                             'context': prev_utt,
                             'context_emo': prev_emo,
                             'style': curr_emo}
                prev_utt, prev_emo = curr_utt, curr_emo
                data_split.append(utterance)

            corpus[f'{split}_empadialog'] = data_split
    save_to_json(os.path.join(out_path_to_empa, f'empadialog.json'), corpus)


if __name__ == '__main__':
    corpus_to_convert = sys.argv[1]

    if corpus_to_convert == 'meld':
        meld_converter()
    elif corpus_to_convert == 'reccon':
        reccon_converter()
    elif corpus_to_convert == 'goemo':
        goemo_converter()
    elif corpus_to_convert == 'carer':
        carer_converter()
    elif corpus_to_convert == 'empadialog':
        empa_converter()
    else:
        raise sys.exit("Corpus name not accepted.")
