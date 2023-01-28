import os
import json
from tqdm import tqdm
from pattern_matcher import PatternMatcher

import sys
sys.path.append('../..')
from models.lexicons import patterns
from models.paraphraser_utils.scoring import StyleScorer


class EmoWordMatcher:
    def __init__(self):
        self.emotion_words = self.get_emotion_words()

    def get_emotion_words(self):
        return {'anger': set(patterns.anger_adjs + patterns.anger_nouns + patterns.anger_advs +
                             patterns.anger_exclamations + patterns.anger_extra),
                'happiness': set(patterns.happiness_adjs + patterns.happiness_advs + patterns.happiness_verbs +
                                 patterns.happiness_names + patterns.happiness_exclamations + patterns.happiness_extra),
                'sadness': set(patterns.sadness_adjs + patterns.sadness_advs + patterns.sadness_exclamations +
                               patterns.sadness_nouns + patterns.sadness_verbs)}

    def matcher(self, sentence: str, emotion: str) -> bool:
        sentence = set(sentence.strip().split(' '))
        return True if len(sentence.intersection(self.emotion_words.get(emotion))) > 0 else False


def filter_same_emos(num):
    path_to_original_file = f'../data/style/generic_original/generic_same_emos_p{num}.json'  # file to filter
    path_to_neutralized_file = f'../data/style/generic_neutralized/pegasus_neutralizer_generic_p{num}.json'
    path_to_out_file = os.path.join('../data/style/generic_neutralized', f'pegasus_neutralizer_generic_p{num}_filtered.json')

    filtered_data = {}

    with open(path_to_original_file, 'r', encoding='utf-8') as jfile:
        original_data = json.load(jfile)
    with open(path_to_neutralized_file, 'r', encoding='utf-8') as jfile:
        neutralized_data = json.load(jfile)

    for id_, info in tqdm(original_data.items()):
        emotion = info['emo']
        sentence1 = info['sent1']
        sentence2 = info['sent2']

        # check if it's possible to get neutralized sentences
        if not neutralized_data[id_]['preds_neutral_sent1'] and not not neutralized_data[id_]['preds_neutral_sent2']:
            continue

        # check with emotion words
        if not emo_word_matcher.matcher(sentence1, emotion) and not emo_word_matcher.matcher(sentence2, emotion):
            continue

        # check with emotion patterns
        e_phrases1 = pattern_matcher.find_patterns(sentence1)
        e_phrases2 = pattern_matcher.find_patterns(sentence2)
        if (not e_phrases1 or emotion not in e_phrases1) and (not e_phrases2 or emotion not in e_phrases2):
            continue

        # check emotion score for each sentence
        emo_score1 = style_scorer.score(sentence1)
        emo_score1 = emo_score1[:, label2id[emotion]]
        emo_score1 = float(emo_score1.detach().cpu().numpy())

        emo_score2 = style_scorer.score(sentence2)
        emo_score2 = emo_score2[:, label2id[emotion]]
        emo_score2 = float(emo_score2.detach().cpu().numpy())
        if emo_score1 < min_emo_score and emo_score2 < min_emo_score:
            continue

        sentence1_neu = neutralized_data[id_]['preds_neutral_sent1'][0]["prediction"] \
            if neutralized_data[id_]['preds_neutral_sent1'] else ""
        sentence2_neu = neutralized_data[id_]['preds_neutral_sent2'][0]["prediction"] \
            if neutralized_data[id_]['preds_neutral_sent2'] else ""

        if sentence2_neu:
            emo_score_neu2 = style_scorer.score(sentence2_neu)
            emo_score_neu2 = emo_score_neu2[:, label2id["neutral"]]
            emo_score_neu2 = float(emo_score_neu2.detach().cpu().numpy())
        else:
            emo_score_neu2 = 0.0
        if emo_score1 < min_emo_score and emo_score_neu2 < min_emo_score:
            continue

        if sentence1_neu:
            emo_score_neu1 = style_scorer.score(sentence1_neu)
            emo_score_neu1 = emo_score_neu1[:, label2id["neutral"]]
            emo_score_neu1 = float(emo_score_neu1.detach().cpu().numpy())
        else:
            emo_score_neu1 = 0.0
        if emo_score2 < min_emo_score and emo_score_neu1 < min_emo_score:
            continue

        if emo_score_neu1 < min_emo_score and emo_score_neu2 < min_emo_score:
            continue

        # if all passed, add to dict
        updated_info = {"sent1": sentence1 if emo_score1 >= min_emo_score else "",
                        "sent2": sentence2 if emo_score2 >= min_emo_score else "",
                        "emo": emotion,
                        "sent1_neu": sentence1_neu if emo_score_neu1 >= min_emo_score else "",
                        "sent2_neu": sentence2_neu if emo_score_neu2 >= min_emo_score else ""}
        if (updated_info["sent1"] and updated_info["sent2_neu"]) or (updated_info["sent2"] and updated_info["sent1_neu"]):
            filtered_data.update({id_: updated_info})

    with open(path_to_out_file, 'w', encoding='utf-8') as jfile:
        json.dump(filtered_data, jfile, indent=4)


def filter_diff_emos():
    path_to_in_file = '../data/style/generic_diff_emos_wn_filtered.json'  # file to filter
    path_to_out_file = os.path.join('../../outputs', 'generic_diff_emos_wn_final_filtered.json')

    filtered_data = {}

    with open(path_to_in_file, 'r', encoding='utf-8') as jfile:
        data = json.load(jfile)

    for id_, info in data.items():
        emotion = info['emo_sent1'] if info['emo_sent1'] != 'neutral' else info['emo_sent2']
        sentence = info['sent1'] if info['emo_sent1'] != 'neutral' else info['sent2']
        sentence_neu = info['sent1'] if info['emo_sent1'] == 'neutral' else info['sent2']

        # check with emotion words
        if not emo_word_matcher.matcher(sentence, emotion):
            continue

        # check with emotion patterns
        e_phrases = pattern_matcher.find_patterns(sentence)
        if not e_phrases or emotion not in e_phrases:
            continue

        # check emotion score
        emo_score = style_scorer.score(sentence)
        emo_score = emo_score[:, label2id[emotion]]
        emo_score = float(emo_score.detach().cpu().numpy())
        if emo_score < min_emo_score:
            continue

        # check emotion score
        emo_score_neu = style_scorer.score(sentence_neu)
        emo_score_neu = emo_score_neu[:, label2id["neutral"]]
        emo_score_neu = float(emo_score_neu.detach().cpu().numpy())
        if emo_score_neu < min_emo_score and emotion != "sadness":
            continue

        filtered_data.update({id_: info})

    with open(path_to_out_file, 'w', encoding='utf-8') as jfile:
        json.dump(filtered_data, jfile, indent=4)


if __name__ == '__main__':
    path_to_lexicon_dir = '../lexicons'
    path_to_classifier_dir = "/speech/dbwork/mul/spielwiese3/dehoang/outputs/classifier"
    cache_dir = "/speech/dbwork/mul/spielwiese3/dehoang/caches"
    min_emo_score = 0.9

    emo_word_matcher = EmoWordMatcher()
    pattern_matcher = PatternMatcher(path_to_lexicon_dir)
    style_scorer = StyleScorer(path_to_classifier_dir, cache_dir)
    label2id = style_scorer.classifier.config.label2id

    filter_diff_emos()
    # for num in range(1, 11):
    #    filter_same_emos(num)
