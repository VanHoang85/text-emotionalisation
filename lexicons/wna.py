import os
from typing import Dict, List


class WordNetAffect:
    def __init__(self, path_to_wna_dir):
        self.path_to_wna_dir = path_to_wna_dir
        self.emo2word2synsets: Dict[str: Dict[str: List]] = {}
        self.synset2words: Dict[str: List] = {}
        self.load_wna()

    def load_wna(self):
        if not os.path.isdir(self.path_to_wna_dir):
            raise SystemExit("Expect a dir of lists of wordnet affect")

        for filename in os.listdir(self.path_to_wna_dir):
            if filename.endswith('.txt'):
                emotion = filename[:-4]
                self.emo2word2synsets[emotion] = {}

                with open(os.path.join(self.path_to_wna_dir, filename), 'r', encoding='utf-8') as file:
                    for line in file:
                        line = line.strip().split()  # whitespace
                        pos, synset = line[0].split('#')

                        if synset not in self.synset2words:
                            self.synset2words[synset] = []

                        for token in line[1:]:
                            word = pos + '#' + token
                            if word not in self.emo2word2synsets[emotion]:
                                self.emo2word2synsets[emotion][word] = []
                            if synset not in self.emo2word2synsets[emotion][word]:
                                self.emo2word2synsets[emotion][word].append(synset)
                            if word not in self.synset2words[synset]:
                                self.synset2words[synset].append(word)

    def get_all_words_from_synset(self, synset: str, with_pos=True) -> List[str]:
        if with_pos:
            return self.synset2words[synset]
        else:
            return [word.strip().split('#')[1] for word in self.synset2words[synset]]

    def get_all_synsets_from_word(self, emotion: str, word: str) -> List[str]:
        if '#' not in word:
            raise Exception("Need to include pos tag in the word.")
        return self.emo2word2synsets[emotion][word]

    def get_all_words_from_emotion(self, emotion: str) -> List[str]:
        return list(set(self.emo2word2synsets[emotion].keys()))

    def get_all_emotions_from_word(self, token: str, pos: str) -> List[str]:
        return [emotion for emotion, word2synsets in self.emo2word2synsets.items() if pos + '#' + token in word2synsets]

    def get_all_alternative_words(self, emotion: str, word: str) -> List[str]:
        alternative_words = []
        synsets = self.get_all_synsets_from_word(emotion, word)
        for synset in synsets:
            alternative_words.extend(self.get_all_words_from_synset(synset, with_pos=False))
        return list(set(alternative_words))
