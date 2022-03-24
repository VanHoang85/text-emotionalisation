import os
import sys
import re
import json
from tqdm import tqdm
import language_tool_python


def normalize_sent(sentence: str) -> str:
    sent_normed = re.sub('\\s{2,}', ' ', sentence.replace('<del>', ''))
    return re.sub('^\\W+', '', sent_normed).strip().capitalize()


def main():
    path_to_dataset_dir = sys.argv[1]
    path_to_input_dir = os.path.join(path_to_dataset_dir, 'extra_lexicons')
    path_to_output_dir = os.path.join(path_to_dataset_dir, 'parallel_corpus_auto')

    if not os.path.exists(path_to_output_dir):
        os.mkdir(path_to_output_dir)

    checker = language_tool_python.LanguageTool('en-US')
    suffixes = ['anger_w_ep.json', 'happiness_w_ep.json', 'sadness_w_ep.json']
    for filename in os.listdir(path_to_input_dir):
        if filename.split('_', maxsplit=1)[1] in suffixes:
            print('Processing', filename)
            with open(os.path.join(path_to_input_dir, filename), 'r', encoding='utf-8') as jfile:
                data = json.load(jfile)

            data_all_good, data_to_check = {}, {}
            for _id, utterance in tqdm(data.items()):
                sent_to_check = normalize_sent(utterance['neutral_sent'])

                if not sent_to_check:  # if empty string
                    utterance['neutral_sent'] = ""
                    data_to_check.update({_id: utterance})
                else:
                    checks = checker.check(sent_to_check)
                    if checks:
                        utterance['rules'] = []
                        for rule in checks:
                            utterance['rules'].append(rule.ruleId + ': ' + rule.message)
                        utterance['suggested'] = checker.correct(sent_to_check)
                        data_to_check.update({_id: utterance})
                    else:
                        utterance['neutral_sent'] = sent_to_check
                        data_all_good.update({_id: utterance})

            # save to files:
            data_name = '_'.join(filename.split('_', 2)[:2])
            with open(os.path.join(path_to_output_dir, f'{data_name}.json'), 'w', encoding='utf-8') as jfile:
                json.dump(data_all_good, jfile, indent=4)
            with open(os.path.join(path_to_output_dir, f'{data_name}_to_check.json'), 'w', encoding='utf-8') as jfile:
                json.dump(data_to_check, jfile, indent=4)


if __name__ == '__main__':
    main()
