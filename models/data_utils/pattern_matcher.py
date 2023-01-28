# DESCRIPTION: To convert the original datasets to new format

import argparse
import json
import os
import re
import spacy
from tqdm import tqdm
from spacy.matcher import Matcher

import sys
sys.path.append('../..')

from models.lexicons import patterns
from models.lexicons.nrc import NRC
from models.lexicons.wna import WordNetAffect
# python -m spacy download en_core_web_sm


class PatternMatcher:
    def __init__(self, path_to_lexicon_dir: str):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        self.single_word_patterns, self.multi_word_patterns = {}, {}
        self.load_patterns(path_to_lexicon_dir)

    def load_patterns(self, path_to_lexicon_dir: str):
        if os.path.exists(os.path.join(path_to_lexicon_dir, 'single_word_patterns.json')) and \
                os.path.exists(os.path.join(path_to_lexicon_dir, 'multi_word_patterns.json')):
            with open(os.path.join(path_to_lexicon_dir, 'single_word_patterns.json'), 'r', encoding='utf--8') as jfile:
                self.single_word_patterns = json.load(jfile)
            with open(os.path.join(path_to_lexicon_dir, 'multi_word_patterns.json'), 'r', encoding='utf--8') as jfile:
                self.multi_word_patterns = json.load(jfile)
        else:
            self.create_patterns(path_to_lexicon_dir)

        # add patterns to matcher
        for emotion in self.single_word_patterns.keys():
            self.matcher.add(emotion + "_words", self.single_word_patterns[emotion])
            self.matcher.add(emotion + "_phrases", self.multi_word_patterns[emotion])

            phrase_patterns = patterns.load_patterns(emotion)
            for cid in phrase_patterns.keys():
                if phrase_patterns[cid]['pattern']:
                    self.matcher.add(emotion + "_" + cid, [phrase_patterns[cid]['pattern']], greedy='LONGEST')

    def find_patterns(self, text: str) -> dict:
        p_patterns = {}
        doc = self.nlp(text)
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            emotion, cid = self.nlp.vocab.strings[match_id].split("_", maxsplit=1)  # Get string representation
            span = doc[start:end]  # The matched span

            if emotion not in p_patterns:
                p_patterns[emotion] = []
            if cid + ": " + span.text not in p_patterns[emotion]:
                p_patterns[emotion].append(cid + ": " + span.text)
        return p_patterns

    def create_patterns(self, path_to_lexicon_dir: str):
        single_words_exps, multi_words_exps = {}, {}
        for emotion in ['anger', 'happiness', 'sadness', 'disgust', 'fear', 'surprise']:
            single_words_exps[emotion] = []
            multi_words_exps[emotion] = []
            self.single_word_patterns[emotion] = []
            self.multi_word_patterns[emotion] = []

        self.get_from_wna(path_to_lexicon_dir, single_words_exps, multi_words_exps)
        self.get_from_nrc(path_to_lexicon_dir, single_words_exps, multi_words_exps)
        self.get_extra(path_to_lexicon_dir, single_words_exps)

        with open(os.path.join(path_to_lexicon_dir, 'single_word_patterns.json'), 'w', encoding='utf--8') as jfile:
            json.dump(self.single_word_patterns, jfile, indent=4)

        with open(os.path.join(path_to_lexicon_dir, 'multi_word_patterns.json'), 'w', encoding='utf--8') as jfile:
            json.dump(self.multi_word_patterns, jfile, indent=4)

    def get_from_wna(self, path_to_lexicon_dir: str, single_words_exps: dict, multi_words_exps: dict):
        wna = WordNetAffect(os.path.join(path_to_lexicon_dir, 'WordNetAffectEmotionLists'))
        for emotion in wna.emo2word2synsets.keys():
            emo_list = wna.get_all_words_from_emotion(emotion)
            for word in emo_list:
                pos, phrase = word.strip().split('#')
                if len(phrase.strip().split('_')) == 1:
                    if phrase.lower() not in single_words_exps[emotion]:
                        single_words_exps[emotion].append(phrase)
                        pattern = [{"LOWER": phrase, "POS": self.convert_pos(pos)}]
                        self.single_word_patterns[emotion].append(pattern)
                else:
                    phrase = phrase.replace('_', ' ')
                    if phrase.lower() not in multi_words_exps[emotion]:
                        multi_words_exps[emotion].append(phrase)
                        pattern = [{"LOWER": token} for token in phrase.lower().split(' ')]
                        self.multi_word_patterns[emotion].append(pattern)

    def get_from_nrc(self, path_to_lexicon_dir: str, single_words_exps: dict, multi_words_exps: dict):
        nrc_threshold = 0.6
        nrc = NRC(os.path.join(path_to_lexicon_dir, 'NRC-Emotion-Intensity-Lexicon-v1.txt'))
        for phrase in nrc.nrc_dict.keys():
            for emotion, score in nrc.get_associations(phrase).items():
                if float(score) >= nrc_threshold and emotion not in ['anticipation', 'trust']:
                    if len(phrase.strip().split(' ')) == 1:
                        if phrase.lower() not in single_words_exps[emotion]:
                            single_words_exps[emotion].append(phrase)
                            pattern = [{"LEMMA": phrase}]
                            self.single_word_patterns[emotion].append(pattern)
                    else:
                        if phrase.lower() not in multi_words_exps[emotion]:
                            multi_words_exps[emotion].append(phrase)
                            pattern = [{"LOWER": token} for token in phrase.lower().split(' ')]
                            self.multi_word_patterns[emotion].append(pattern)

    def get_extra(self, path_to_lexicon_dir: str, single_words_exps: dict):
        with open(os.path.join(path_to_lexicon_dir, 'emo_words.json'), 'r', encoding='utf-8') as jfile:
            eword_dict = json.load(jfile)
        for emotion, emo_list in eword_dict.items():
            for token in emo_list:
                if token not in single_words_exps[emotion]:
                    single_words_exps[emotion].append(token)
                    pattern = [{"LOWER": token}]
                    self.single_word_patterns[emotion].append(pattern)

    def convert_pos(self, pos: str) -> str:
        if pos == 'a':
            return 'ADJ'
        elif pos == 'n':
            return 'NOUN'
        elif pos == 'v':
            return 'VERB'
        elif pos == 'r':
            return 'ADV'


def normalize_sent(sentence: str) -> str:
    sentence_normed = re.sub('\\s*\'\\s*', '\'', sentence)
    sentence_normed = re.sub('\\s{2,}', ' ', sentence_normed)
    sentence_normed = sentence_normed.replace(' !', '!').replace(' .', '.').replace(' ,', ',')
    return sentence_normed


def get_neutral_sent(emo_sent: str, phrases: list) -> str:
    neutral_sent = emo_sent
    for phrase in phrases:
        phrase = phrase.split(": ", maxsplit=1)[1].strip()
        neutral_sent = neutral_sent.replace(phrase, '<del>')
    return neutral_sent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--path_to_lexicon_dir', type=str, default='../lexicons')
    parser.add_argument('-i', '--path_to_input_dir', type=str, help='Input dir',
                        default='../datasets/annotated/GoEmotions/extra')
    args = parser.parse_args()

    path_to_output_dir = args.path_to_input_dir + '_lexicons'
    if not os.path.exists(path_to_output_dir):
        os.mkdir(path_to_output_dir)

    pattern_matcher = PatternMatcher(args.path_to_lexicon_dir)

    for filename in os.listdir(args.path_to_input_dir):
        if filename.endswith('.json'):
            print('Processing', filename)
            suffix = filename.strip().split('_')[1][:-5]
            with open(os.path.join(args.path_to_input_dir, filename), 'r', encoding='utf-8') as jfile:
                data = json.load(jfile)

            data_w_ep, data_wo_ep = {}, {}
            for _id, utt in tqdm(data.items()):
                emo_sent = normalize_sent(utt['emotion_sent'])
                emo_patterns = pattern_matcher.find_patterns(emo_sent)
                if emo_patterns:
                    if suffix in emo_patterns.keys() or suffix == 'neutral':
                        utt['neutral_sent'] = get_neutral_sent(emo_sent, emo_patterns[suffix]) \
                            if suffix != 'neutral' else ""
                        utt['emotion_sent'] = emo_sent
                        utt.update({'phrases': emo_patterns[suffix] if suffix != 'neutral' else emo_patterns})
                        data_w_ep.update({_id: utt})
                    else:
                        data_wo_ep.update({_id: utt})
                else:
                    data_wo_ep.update({_id: utt})

            # save to files:
            with open(os.path.join(path_to_output_dir, f'{filename[:-5]}_w_ep.json'), 'w', encoding='utf-8') as jfile:
                json.dump(data_w_ep, jfile, indent=4)
            with open(os.path.join(path_to_output_dir, f'{filename[:-5]}_wo_ep.json'), 'w', encoding='utf-8') as jfile:
                json.dump(data_wo_ep, jfile, indent=4)


if __name__ == '__main__':
    main()
