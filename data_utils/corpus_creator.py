import os
import re
import sys
import json
from typing import List


def get_emo_words(utterance: str) -> List[str]:
    e_phrases = []
    positions = [match.start() for match in re.finditer(u'#', utterance)]
    for num in range(len(positions)//2):
        e_phrases.append(utterance[positions[num*2]: positions[num*2 + 1] + 1].strip())
    return e_phrases


def get_neutral_sent(utterance: str, e_words: List[str]) -> str:
    neutral = utterance
    for phrase in e_words:
        neutral = neutral.replace(phrase, '')
    return neutral.strip()


def save_phrases(emo_phrases: dict, path_to_out_dir, dataset: str):
    if not os.path.exists(path_to_out_dir):
        os.mkdir(path_to_out_dir)

    for emotion, phrases in emo_phrases.items():
        phrases = list(set(phrases))
        with open(os.path.join(path_to_out_dir, f'{dataset}_{emotion}.txt'), 'w', encoding='utf-8') as file:
            file.write('\n'.join(phrases).replace('#', ''))


def save_corpus(corpus: dict, path_to_out_dir, dataset: str):
    if not os.path.exists(path_to_out_dir):
        os.mkdir(path_to_out_dir)

    for emotion, data in corpus.items():
        with open(os.path.join(path_to_out_dir, f'{dataset}_{emotion}.json'), 'w', encoding='utf-8') as jfile:
            json.dump(data, jfile, indent=4)


def remove_unicodes(utterance: str) -> str:
    return utterance.strip().replace('\u2014', '- ').replace('\u201c', "'").replace('\u201d', "'")\
        .replace('\u2018', "'").replace('\u2019', "'").replace('\u2026', '... ')


def normalize_sent(sentence: str) -> str:
    sentence_normed = re.sub('\\s*\'\\s*', '\'', sentence)
    sentence_normed = sentence_normed.replace(' !', '!').replace(' .', '.').replace(' ,', ',')
    return sentence_normed


def emotion_mapping(emotion):
    if emotion == 'joy':
        return 'happiness'
    elif emotion == 'happines':
        return 'happiness'
    elif emotion == 'excited':
        return 'happiness'
    elif emotion == 'happy':
        return 'happiness'
    elif emotion == 'angry':
        return 'anger'
    elif emotion == 'frustrated':
        return 'anger'
    elif emotion == 'sad':
        return 'sadness'
    elif emotion == 'surprised':
        return 'surprise'
    return emotion


def create_corpus():

    emo_phrases, corpus, extra = {}, {}, {}
    with open(path_to_jfile, 'r', encoding='utf-8') as jfile:
        dataset = json.load(jfile)

    for name, data in dataset.items():
        for utt in data:
            emotion = emotion_mapping(utt['style'])
            utterance = normalize_sent(remove_unicodes(utt['utterance'].replace('#', '@')))
            context = normalize_sent(remove_unicodes(utt['context'].replace('#', '@')))
            context_emo = emotion_mapping(utt['context_emo'])

            if emotion == 'neutral' or '@' not in utterance:
                if emotion not in extra:
                    extra[emotion] = {}
                extra[emotion].update({utt['id']: {'neutral_sent': '',
                                                   'emotion_sent': utterance,
                                                   'context': context,
                                                   'context_emo': context_emo,
                                                   'type': ''}})
            else:
                e_words = get_emo_words(utterance)
                neutral_sent = get_neutral_sent(utterance, e_words)

                # add to the emo_phrases dict
                if emotion not in emo_phrases:
                    emo_phrases[emotion] = [phrase.replace('#', '').strip() for phrase in e_words]
                else:
                    emo_phrases[emotion] += [phrase.replace('#', '').strip() for phrase in e_words]

                # add to corpus
                if emotion not in corpus:
                    corpus[emotion] = {}  # strict or paraphrasing
                corpus[emotion].update({utt['id']: {'neutral_sent': neutral_sent,
                                                    'emotion_sent': utterance.replace('#', ''),
                                                    'type': ''}})

    save_phrases(emo_phrases, path_to_out_dir=os.path.join(path_to_annotated_dir, 'e-phrases'),
                 dataset=os.path.basename(path_to_jfile).split('.')[0])
    save_corpus(corpus=corpus, path_to_out_dir=os.path.join(path_to_annotated_dir, 'parallel_corpus'),
                dataset=os.path.basename(path_to_jfile).split('.')[0])
    save_corpus(corpus=extra, path_to_out_dir=os.path.join(path_to_annotated_dir, 'extra'),
                dataset=os.path.basename(path_to_jfile).split('.')[0])


if __name__ == '__main__':
    path_to_jfile = sys.argv[1]  # path to the json file to create corpus
    # path_to_jfile = 'C:\\Users\\7000026141\PycharmProjects\\affectiveNLG\datasets\\annotated\dailydialog.json'
    path_to_annotated_dir = os.path.dirname(path_to_jfile)

    create_corpus()
