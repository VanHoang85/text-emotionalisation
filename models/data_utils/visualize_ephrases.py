import os
import json
import spacy
import numpy as np
from tqdm import tqdm
from collections import Counter
from get_edit_operations import get_all
from pattern_matcher import PatternMatcher
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../..')
from models.paraphraser_utils.scoring import DiversityScorer


def filter_e_phrases(phrases: list) -> list:
    return [phrase for phrase in phrases if pattern_matcher.find_patterns(phrase)]


def get_phrases_from_pred():
    nlp = spacy.load("en_core_web_sm")
    files_to_process = ['bart_finetune_test_emo_stylizer.json',
                        'bart_noloss_finetune_test_emo_stylizer.json',
                        'pegasus_finetune_test_emo_stylizer.json',
                        'pegasus_finetune_test_emo_stylizer0.5.json',
                        'pegasus_finetune_test_emo_stylizer1.0.json',
                        'pegasus_finetune_test_emo_stylizer2.0.json',
                        'pegasus_finetune_test_emo_stylizer2.0_mixed.json']

    for filename in files_to_process:
        print(f'Processing {filename} ...')
        with open(os.path.join(path_to_pred_data, filename), 'r', encoding='utf-8') as jfile:
            data = json.load(jfile)

        type_counter = Counter()
        phrases = {'anger': [], 'happiness': [], 'sadness': []}
        for utt_info in tqdm(data.values()):
            if 'input' in utt_info:
                neutral_sent = utt_info['input']
                keys = list(utt_info.keys())
                for key in keys:
                    if 'preds' not in key:
                        continue

                    preds = utt_info[key]
                    if '_' in key:
                        emotion = key.split('_')[1]
                    else:
                        emotion = utt_info['target_emo']

                    if emotion == 'neutral':
                        continue
                    for pred in preds:
                        emotion_sent = pred['prediction']
                        _, edit_type, e_phrases = get_all(emotion_sent, neutral_sent, nlp)
                        phrases[emotion].extend(e_phrases)
                        type_counter.update([edit_type])

        for emo, phrase_ls in phrases.items():
            all_phrases[emo][filename[:-5]] = phrase_ls  # filter_e_phrases(phrase_ls)
            word_counter = Counter()
            for phrase in phrase_ls:
                word_counter.update([word for word in phrase.split()])

            with open(f'../outputs/word_counter/count_{filename}.txt', 'a', encoding='utf-8') as file:
                file.write(f'{emo.upper()}\n')
                for word, count in word_counter.most_common():
                    file.write(f'{word}: {count}\n')

        # write type edit to file
        with open('../../outputs/count_edit_type.txt', 'a', encoding='utf-8') as file:
            file.write(filename)
            file.write('\n')
            for edit_type, count in type_counter.items():
                file.write(f'{edit_type}: {count}\n')
            file.write('=' * 20)
            file.write('\n')


def get_phrases_from_gold():
    phrases = {'anger': [], 'happiness': [], 'sadness': []}
    word_counter = Counter()

    files_to_process = ['train.json', 'validation.json', 'test.json']
    for filename in files_to_process:
        print(f'Processing {filename} ...')
        with open(os.path.join(path_to_gold_data, filename), 'r', encoding='utf-8') as jfile:
            data = json.load(jfile)

        for utt_info in tqdm(data.values()):
            if utt_info['emo'] != 'neutral':
                phrases[utt_info['emo']].extend(utt_info['phrases'])

    for emo, phrase_ls in phrases.items():
        all_phrases[emo]['gold'] = phrase_ls  # filter_e_phrases(phrase_ls)
        for phrase in phrase_ls:
            word_counter.update([word for word in phrase.split()])

    with open(f'../outputs/word_counter/count_gold.txt', 'w', encoding='utf-8') as file:
        for word, count in word_counter.most_common():
            file.write(f'{word}: {count}\n')


def clustering():
    clustered_phrases = {}
    for emotion in all_phrases.keys():
        for filename, phrases in all_phrases[emotion].items():
            embeddings = embedder.encode(phrases, convert_to_numpy=True)

            # Normalize the embeddings to unit length
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            # Perform kmean clustering
            # , affinity='cosine', linkage='average', distance_threshold=0.4)
            clustering_model = AgglomerativeClustering(n_clusters=None,
                                                       distance_threshold=1.3)
            clustering_model.fit(embeddings)
            cluster_assignment = clustering_model.labels_

            clusters = {}
            for sentence_id, cluster_id in enumerate(cluster_assignment):
                cluster_id = 'cluster_' + str(cluster_id + 1)
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(phrases[sentence_id])

            if emotion not in clustered_phrases:
                clustered_phrases[emotion] = {}
            clustered_phrases[emotion][filename] = clusters
            print(f'{len(clusters)} clusters for {emotion} in {filename}')

    with open(os.path.join(path_to_pred_data, 'all_phrases_clustered.json'), 'w', encoding='utf-8') as jfile:
        json.dump(clustered_phrases, jfile, indent=4, sort_keys=True)


def visualize():
    # now visualize for each emotion
    print('Visualize all phrases ...')

    pca = PCA(n_components=50, random_state=7)
    tsne = TSNE(n_components=2, perplexity=10, random_state=6, learning_rate='auto', n_iter=1500, init='pca')

    for emotion in all_phrases.keys():
        file_ids, phrases = [], []

        for filename, phrase_ls in all_phrases[emotion].items():
            if 'noloss' in filename:
                continue
            phrases.extend(phrase_ls)
            file_ids.extend([filename.split('_')[0]] * len(phrase_ls))

        embeddings = embedder.encode(phrases, convert_to_numpy=True)
        pca_results = pca.fit_transform(embeddings)
        tsne_results = tsne.fit_transform(pca_results)

        data = {'tsne-2d-one': tsne_results[:, 0], 'tsne-2d-two': tsne_results[:, 1], 'file_id': file_ids}
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="file_id",
            style="file_id",
            # palette=sns.color_palette("hls", 10),
            palette="deep",
            data=data,
            legend="full",
            # alpha=0.3
        )
        plt.title(f"{emotion.title()}")
        plt.savefig(os.path.join(path_to_pred_data, f'{emotion}.png'))


def score_diversity():
    for emotion in all_phrases.keys():
        for filename, phrases in all_phrases.get(emotion).items():
            score_uni = diverse_1.score(predictions=phrases)
            score_bi = diverse_2.score(predictions=phrases)

            with open(f'../outputs/word_counter/count_diversity.txt', 'a', encoding='utf-8') as file:
                file.write('=' * 50)
                file.write(f'\nFor {emotion}:\n')
                file.write('-' * 10)
                file.write(f'\nDiversity unigram of {filename}: {round(score_uni, 3)}\n')
                file.write(f'\nDiversity bigram of {filename}: {round(score_bi, 3)}\n')
                file.write('-' * 10)
                file.write('\n\n\n')


if __name__ == '__main__':
    path_to_gold_data = '../../data/style'
    path_to_pred_data = '../../outputs'
    path_to_lexicon_dir = '../lexicons'
    all_phrases_filename = 'all_ephrases.json'

    pattern_matcher = PatternMatcher(path_to_lexicon_dir)
    # embedder = SentenceTransformer('all-MiniLM-L6-v2')
    diverse_1 = DiversityScorer(n=1)
    diverse_2 = DiversityScorer(n=2)

    # first, get or load list of all ephrases from gold data and predicted outputs
    print('Loading all phrases ...')
    if not os.path.exists(os.path.join(path_to_pred_data, all_phrases_filename)):
        all_phrases = {'anger': {}, 'happiness': {}, 'sadness': {}}
        get_phrases_from_gold()
        get_phrases_from_pred()

        with open(os.path.join(path_to_pred_data, all_phrases_filename), 'w', encoding='utf-8') as jfile:
            json.dump(all_phrases, jfile, indent=4)
    else:
        with open(os.path.join(path_to_pred_data, all_phrases_filename), 'r', encoding='utf-8') as jfile:
            all_phrases = json.load(jfile)

    score_diversity()
    # clustering()
    # visualize()
