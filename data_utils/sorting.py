import os
import json
import re
import spacy
from lexicons import patterns
from pattern_matcher import PatternMatcher


def normalize_sent(sentence: str) -> str:
    sentence_normed = re.sub('\\s*\'\\s*', '\'', sentence.lower())
    sentence_normed = sentence_normed.replace(' !', '!').replace(' .', '.').replace(' ,', ',')
    return sentence_normed


def sort_sentence(sentence: str, pattern_dict: dict):
    doc = nlp(sentence)
    pos_ls = []

    for token in doc:
        pos_ls.append(token.pos_)
    pattern = ', '.join(pos_ls)

    if pattern not in pattern_dict:
        pattern_dict[pattern] = []

    if sentence not in pattern_dict[pattern]:
        pattern_dict[pattern].append(sentence)


def main():
    pattern_matcher = PatternMatcher("..\\lexicons")
    emotions = ['anger', 'sadness', 'happiness']
    for emo in emotions:
        filename_out = f'{emo}_clustered_sorted.json'
        data = patterns.load_patterns(emo)

        pattern_dict = {}
        for cluster_id in data.keys():
            phrases_list = data[cluster_id]['phrases']
            for phrase in phrases_list:
                emo_patterns = pattern_matcher.find_patterns(phrase)
                if not emo_patterns and emo not in emo_patterns.keys():
                    sort_sentence(normalize_sent(phrase), pattern_dict)

        i = 1
        out_dict = {}
        for pattern, phrases_list in pattern_dict.items():
            out_dict['cluster_' + str(i)] = {'pattern': pattern,
                                             'phrases': phrases_list}
            i += 1

        with open(os.path.join(path_to_out_dir, filename_out), 'w', encoding='utf-8') as jfile:
            json.dump(out_dict, jfile, indent=4)


if __name__ == '__main__':
    path_to_out_dir = "..\\lexicons\\patterns_clusters"
    nlp = spacy.load("en_core_web_sm")

    if not os.path.exists(path_to_out_dir):
        os.mkdir(path_to_out_dir)
    main()
