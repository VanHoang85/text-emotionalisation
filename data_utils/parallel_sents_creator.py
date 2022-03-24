import sys
import os
import json
import spacy
from tqdm import tqdm
from get_edit_operations import get_all
from pattern_matcher import PatternMatcher


def load_emo_corpus(emotion):
    path_to_extra = os.path.join(path_to_corpus, 'extra')
    for filename in os.listdir(path_to_extra):
        if emotion in filename:
            with open(os.path.join(path_to_extra, filename), 'r', encoding='utf-8') as jfile:
                return json.load(jfile)


def filter_e_phrases(phrases: list, pattern_matcher: PatternMatcher) -> list:
    return [phrase for phrase in phrases if pattern_matcher.find_patterns(phrase)]


def remove_unicode(sentence: str) -> str:
    return sentence.strip().lower().encode("ascii", "ignore").decode('utf-8')


def get_utt_info(_id: str, utt: dict, emotion_corpus: dict, nlp, pattern_matcher):
    emotion_sent = remove_unicode(utt["emotion_sent"])
    neutral_sent = ''
    context, context_emo = '', ''

    if utt["type"]:
        neutral_sent = remove_unicode(utt["suggested"] if "suggested" in utt else utt["neutral_sent"])

    if _id in emotion_corpus and "context" in emotion_corpus[_id]:
        context = remove_unicode(emotion_corpus[_id]["context"])
        context_emo = emotion_corpus[_id]["context_emo"]

    if neutral_sent:
        operations, edit_type, e_phrases = get_all(emotion_sent, neutral_sent, nlp)
        return {"neutral_sent": neutral_sent,
                "emotion_sent": emotion_sent,
                "operations": ', '.join(operations),
                "type": edit_type,
                "context": context,
                "context_emo": context_emo,
                "phrases": filter_e_phrases(e_phrases, pattern_matcher)}
    else:
        return {"neutral_sent": neutral_sent,
                "emotion_sent": emotion_sent,
                "context": context,
                "context_emo": context_emo,
                "phrases": pattern_matcher.find_patterns(emotion_sent)}


def main():
    path_to_dir_in = os.path.join(path_to_corpus, dir_in)
    path_to_out_dir = os.path.join(path_to_corpus, dir_out)

    data_parallel, data_to_do, wo_phrases = {}, {}, {}
    all_utts = []

    dataset = None
    emotions = ['anger', 'happiness', 'sadness']

    nlp = spacy.load("en_core_web_sm")
    pattern_matcher = PatternMatcher(path_to_lexicon_dir)

    for filename in os.listdir(path_to_dir_in):
        if filename.endswith('.json'):
            print("Processing", filename, "....")
            dataset = filename[:-5].split('_')[0]
            emotion = filename[:-5].split('_')[1]

            if emotion not in emotions:
                continue

            emotion_corpus = load_emo_corpus(emotion)
            if emotion not in data_parallel:
                data_parallel[emotion] = {}
                data_to_do[emotion] = {}
                wo_phrases[emotion] = {}

            with open(os.path.join(path_to_dir_in, filename), 'r', encoding='utf-8') as jfile:
                data_file = json.load(jfile)

            for _id, utt in tqdm(data_file.items()):
                emotion_sent = remove_unicode(utt["emotion_sent"])
                if emotion_sent in all_utts:  # check if duplicate
                    continue

                e_phrases = pattern_matcher.find_patterns(emotion_sent)
                all_utts.append(emotion_sent)

                if e_phrases:
                    utterance = get_utt_info(_id, utt, emotion_corpus, nlp, pattern_matcher)

                    if utt["type"]:
                        data_parallel[emotion].update({_id: utterance})
                    else:
                        data_to_do[emotion].update({_id: utterance})
                else:
                    wo_phrases[emotion].update({_id: {"neutral_sent": "",
                                                      "emotion_sent": emotion_sent}})

    for emotion in data_parallel.keys():
        with open(os.path.join(path_to_out_dir, f'{dataset}_{emotion}.json'), 'w', encoding='utf-8') as jfile:
            json.dump(data_parallel[emotion], jfile, indent=4)

        with open(os.path.join(path_to_out_dir, f'{dataset}_{emotion}_to_do.json'), 'w', encoding='utf-8') as jfile:
            json.dump(data_to_do[emotion], jfile, indent=4)

        with open(os.path.join(path_to_out_dir, f'{dataset}_{emotion}_wo_ep.json'), 'w', encoding='utf-8') as jfile:
            json.dump(wo_phrases[emotion], jfile, indent=4)


if __name__ == '__main__':
    path_to_corpus = sys.argv[1]  # path to corpus dir
    path_to_lexicon_dir = '../lexicons'
    # C:\Users\7000026141\PycharmProjects\affectiveNLG\datasets\annotated\RECCON
    # /Users/vanhoang/PycharmProjects/affectiveNLG/datasets/annotated/RECCON
    dir_in = 'parallel_corpus_auto'
    dir_out = 'parallel_corpus'

    if not os.path.exists(os.path.join(path_to_corpus, dir_out)):
        os.mkdir(os.path.join(path_to_corpus, dir_out))
    main()
