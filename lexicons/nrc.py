import json


class NRC:
    def __init__(self, path_to_nrc_lex):
        self.path_to_nrc_lex = path_to_nrc_lex
        self.nrc_dict = self.create_nrc_json() if self.path_to_nrc_lex.endswith('.txt') else self.load_nrc()

    def create_nrc_json(self):
        nrc_dic = {}
        with open(self.path_to_nrc_lex, 'r', encoding='utf-8') as file:
            _ = file.readline()  # get rid of first line
            for line in file:
                word, emo, score = line.strip().split('\t')
                if emo == 'joy':
                    emo = 'happiness'  # re-assign to match file name

                if word not in nrc_dic:
                    nrc_dic[word] = {}
                nrc_dic[word].update({emo: score})
        return nrc_dic

    def load_nrc(self) -> dict:
        with open(self.path_to_nrc_lex, 'r', encoding='utf-8') as jfile:
            nrc = json.load(jfile)
        return nrc

    def check_exist(self, word: str) -> bool:
        return True if word in self.nrc_dict else False

    def get_associations(self, word: str) -> dict:
        return self.nrc_dict.get(word) if self.check_exist(word) else {}
