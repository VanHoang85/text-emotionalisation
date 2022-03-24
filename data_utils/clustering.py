# https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/clustering/agglomerative.py
import os
import sys
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


def load_corpus(path_to_file: str) -> list:
    corpus = []
    with open(path_to_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().split('\t')
            for phrase in line:
                corpus.append(phrase.strip())
    return corpus


def main():
    path_to_indir = sys.argv[1]
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    emotions = ['anger', 'happiness', 'sadness']

    for emotion in tqdm(emotions):
        filename = f'dailydialog_{emotion}.txt'
        corpus = load_corpus(os.path.join(path_to_indir, filename))
        corpus_embeddings = embedder.encode(corpus)

        # Normalize the embeddings to unit length
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

        # Perform kmean clustering
        # , affinity='cosine', linkage='average', distance_threshold=0.4)
        clustering_model = AgglomerativeClustering(n_clusters=None,
                                                   distance_threshold=1.3)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            cluster_id = 'cluster_' + str(cluster_id + 1)
            if cluster_id not in clustered_sentences:
                clustered_sentences[cluster_id] = []
            clustered_sentences[cluster_id].append(corpus[sentence_id])

        with open(os.path.join(path_to_indir, f'{filename[:-4]}_clustered.json'), 'w', encoding='utf-8') as jfile:
            json.dump(clustered_sentences, jfile, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
