# coding=utf-8
# DISCLAIMER:
# https://github.com/machelreid/lewis/blob/master/roberta-tagger/preprocess-roberta-tagger.py

import os
import sys
import re
import json
import spacy
import Levenshtein


def normalize(sentence: str) -> str:
    return re.sub('\\s*\'\\s*', '\'', sentence.strip().lower())


def read_file(filename):
    with open(os.path.join(path_to_parallel_dir, filename), 'r', encoding='utf-8') as jfile:
        parallel_data = json.load(jfile)
    return parallel_data


def write_to_file(filename, data):
    with open(os.path.join(path_to_parallel_dir, filename), 'w', encoding='utf-8') as jfile:
        json.dump(data, jfile, indent=4)


def get_e_phrases(tokens: list, operations: list, pos_ls: list) -> list:
    phrases, phrase_indexes = [], []
    idx = 0
    start, end = None, None
    while idx < len(tokens):
        if operations[idx] == 'INSERT' or operations[idx] == "REPLACE":
            if start is None:
                start = idx
            end = idx
        else:
            if start is not None:
                phrase_indexes.append((start, end))
                start, end = None, None
        idx += 1

    if start is not None:  # add the last one
        phrase_indexes.append((start, end))

    for start, end in phrase_indexes:
        if pos_ls[end] == 'PRON':
            start = start - 1
            end = end - 1
        elif pos_ls[start] == 'PUNCT':
            start = start + 1
            end = end + 1

        phrases.append(' '.join(tokens[start:end + 1]))
    return list(filter(None, phrases))


def add_edit_ops(emotion_tokens, neutral_tokens) -> list:
    # do tokenization
    common_tokens = list(set(emotion_tokens + neutral_tokens))
    # if len(emotion_tokens) < len(neutral_tokens):
    #     print(emotion_tokens)
    #     print(neutral_tokens)
    #     raise ValueError("Len style sent can't shorter than neutral sent.")

    # map each token to a unique char
    characters = list(
        set(
            "1234567890-qwertyuiopasdfghjklzxcvbQWERTYUIOOASDFGHJKLZXCVBNMnm,./;[]-+_=|}{><?!!@#$%^&*()iœ∑´®†\¨ˆøπ¬˚∆˙©ƒ∂ßåååååΩ≈ç√∫˜µ≤øˆ˙©ƒ¨∂®ß´∑ß´\†∂®¨あグノシーは、ネット上に存在する様々な情報を独自のアルゴリズムで収集し、評価付けを行い、ユーザーに届ける情報キュレーションサービスである。CEOの福島は、「ユーザーが面白いと思うコンテンツを、ニュースに限らず配信していった結果、自然とエンターテインメント性の強いメディアになった」と述べている†ƒ√˚©電気通信事業者（でんきつうしんじぎょうしゃ）とは、一般に固定電話や携帯電話等の電気通信サービスを提供する会社の総称。「音声やデータを運ぶ」というところから通信キャリア（または単にキャリア）や通信回線事業者（または単に回線事業者）と呼ばれることもある。携帯電話専業の会社については携帯会社と呼ぶことが多いがDocomoなどの携帯電話回線会社とAppleなどの携帯電話製造会社との混同されるため呼ばれ方が変わりつつある。回線事業者または回線会社として扱われる。∆˙˚∫∆…√©∆ç®™£∞¢§¶•ªªªª¸˛Ç◊ı˜Â¯Â¯˘¿ÆÚÒÔÓ˝ÏÎÍÅÅÅÅŒ„´‰ˇÁ¨ˆØ∏∏∏∏””’»±—‚·°‡ﬂﬁ›‹⁄`乙 了 又 与 及 丈 刃 凡 勺 互 弔 井 升 丹 乏 匁 屯 介 冗 凶 刈 匹 厄 双 孔 幻 斗 斤 且 丙 甲 凸 丘 斥 仙 凹 召 巨 占 囚 奴 尼 巧 払 汁 玄 甘 矛 込 弐 朱 吏 劣 充 妄 企 仰 伐 伏 刑 旬 旨 匠 叫 吐 吉 如 妃 尽 帆 忙 扱 朽 朴 汚 汗 江 壮 缶 肌 舟 芋 芝 巡 迅 亜 更 寿 励 含 佐 伺 伸 但 伯 伴 呉 克 却 吟 吹 呈 壱 坑 坊 妊 妨 妙 肖 尿 尾 岐 攻 忌 床 廷 忍 戒 戻 抗 抄 択 把 抜 扶 抑 杉 沖 沢 沈 没 妥 狂 秀 肝 即 芳 辛 迎 邦 岳 奉 享 盲 依 佳 侍 侮 併 免 刺 劾 卓 叔 坪 奇 奔 姓 宜 尚 屈 岬 弦 征 彼 怪 怖 肩 房 押 拐 拒 拠 拘 拙 拓 抽 抵 拍 披 抱 抹 昆 昇 枢 析 杯 枠 欧 肯 殴 況 沼 泥 泊 泌 沸 泡 炎 炊 炉 邪 祈 祉 突 肢 肪 到 茎 苗 茂 迭 迫 邸 阻 附 斉 甚 帥 衷 幽 為 盾 卑 哀 亭 帝 侯 俊 侵 促 俗 盆 冠 削 勅 貞 卸 厘 怠 叙 咲 垣 契 姻 孤 封 峡 峠 弧 悔 恒 恨 怒 威 括 挟 拷 挑 施 是 冒 架 枯 柄 柳 皆 洪 浄 津 洞 牲 狭 狩 珍 某 疫 柔 砕 窃 糾 耐 胎 胆 胞 臭 荒 荘 虐 訂 赴 軌 逃 郊 郎 香 剛 衰 畝 恋 倹 倒 倣 俸 倫 翁 兼 准 凍 剣 剖 脅 匿 栽 索 桑 唆 哲 埋 娯 娠 姫 娘 宴 宰 宵 峰 貢 唐 徐 悦 恐 恭 恵 悟 悩 扇 振 捜 挿 捕 敏 核 桟 栓 桃 殊 殉 浦 浸 泰 浜 浮 涙 浪 烈 畜 珠 畔 疾 症 疲 眠 砲 祥 称 租 秩 粋 紛 紡 紋 耗 恥 脂 朕"
        )
    )
    emotion_mapping = ''.join([characters[common_tokens.index(token)] for token in emotion_tokens])
    neutral_mapping = ''.join([characters[common_tokens.index(token)] for token in neutral_tokens])
    full_operations = ['KEEP'] * len(emotion_tokens) if len(emotion_tokens) > len(neutral_tokens) \
        else ['KEEP'] * len(neutral_tokens)

    # edit_operations = Levenshtein.editops(source_string=neutral_mapping, destination_string=emotion_mapping)
    edit_operations = Levenshtein.editops(neutral_mapping, emotion_mapping)
    for op in edit_operations:
        try:
            if op[2] < len(full_operations):
                full_operations[op[2]] = op[0].upper()
        except IndexError:
            print(emotion_tokens)
            print(neutral_tokens)
            print(edit_operations)
            sys.exit()
    return full_operations


def add_editing_type(operations: list) -> str:
    insert_num = operations[:-1].count('INSERT')
    replace_num = operations[:-1].count('REPLACE')
    delete_num = operations[:-1].count('DELETE')

    if insert_num > 0 and replace_num == 0:
        if operations[0] == 'INSERT':
            return 'insertion-s'
        elif operations[-1] == 'INSERT' or operations[-2] == 'INSERT':
            return 'insertion-e'
        else:
            return 'insertion-m'

    elif insert_num == 0 and replace_num > 0:
        return 'replacement'
    elif insert_num > 0 and replace_num > 0:
        return 'paraphrase'
    elif insert_num == 0 and replace_num == 0 and delete_num > 0:
        return 'deletion'
    elif insert_num == 0 and replace_num == 0 and delete_num == 0:
        return 'no-change'
    else:
        return 'NULL'


def get_all(emotion_sent: str, neutral_sent: str, nlp):
    emotion_pos_ls = [token.pos_ for token in nlp(normalize(emotion_sent))]
    emotion_tokens = [token.text for token in nlp(normalize(emotion_sent))]
    neutral_tokens = [token.text for token in nlp(normalize(neutral_sent))]

    operations = add_edit_ops(emotion_tokens, neutral_tokens)
    edit_type = add_editing_type(operations)
    e_phrases = get_e_phrases(emotion_tokens, operations, emotion_pos_ls)
    return operations, edit_type, e_phrases


def main():
    nlp = spacy.load("en_core_web_sm")
    for filename in os.listdir(path_to_parallel_dir):
        if not filename.endswith('json') or filename.count('_') > 1:
            continue

        parallel_data_with_ops = {}
        parallel_data = read_file(filename)
        for _id, utt in parallel_data.items():
            operations, edit_type, e_phrases = get_all(utt['emotion_sent'], utt['neutral_sent'], nlp)
            parallel_data_with_ops[_id] = {"neutral_sent": utt['neutral_sent'],
                                           "emotion_sent": utt['emotion_sent'],
                                           "operations": ', '.join(operations),
                                           "type": edit_type,
                                           "context": utt["context"] if "context" in utt else "",
                                           "context_emo": utt["context_emo"] if "context_emo" in utt else "",
                                           "phrases": e_phrases}
        write_to_file(f'{filename[:-5]}_with_ops.json', parallel_data_with_ops)


if __name__ == '__main__':
    # "/Users/vanhoang/PycharmProjects/affectiveNLG/datasets/annotated/RECCON/parallel_corpus_gold"
    path_to_parallel_dir = sys.argv[1]
    main()
