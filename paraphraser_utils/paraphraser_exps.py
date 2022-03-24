import os
import json
import torch
import argparse
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_response(input_text, model, tokenizer):
    batch = tokenizer([input_text], truncation=True, padding='longest', return_tensors="pt").to(device)
    translated = model.generate(**batch,
                                **gen_kwargs)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text


def update_outs(model_name, in_text, emo, out_texts):
    if in_text not in outs:
        outs.update({in_text: {'emo': emo}})

    if model_name not in outs[in_text]:
        outs[in_text].update({model_name: {'texts': [],
                                           'nums': []}})

    for num, text in enumerate(out_texts):
        # and text.lower().strip() != in_text.lower().strip()
        if text not in outs[in_text][model_name]['texts']:
            outs[in_text][model_name]['texts'].append(text)
            outs[in_text][model_name]['nums'].append(num + 1)


def pegasus():
    # https://huggingface.co/tuner007/pegasus_paraphrase
    print('Run Pegasus...')
    model_name = 'tuner007/pegasus_paraphrase'
    model = PegasusForConditionalGeneration.from_pretrained(model_name, cache_dir=args.cache_dir).to(device)
    tokenizer = PegasusTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)

    for line in data:
        emo, utterance = line.strip().split('\t')
        out_texts = get_response(utterance, model, tokenizer)
        update_outs('pegasus', utterance, emo, out_texts)


def bart():
    # https://huggingface.co/eugenesiow/bart-paraphrase
    print('Run Bart...')
    model_name = 'eugenesiow/bart-paraphrase'
    model = BartForConditionalGeneration.from_pretrained(model_name, cache_dir=args.cache_dir).to(device)
    tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)

    for line in data:
        emo, utterance = line.strip().split('\t')
        out_texts = get_response(utterance, model, tokenizer)
        update_outs('bart', utterance, emo, out_texts)


def t5():
    print('Run T5...')
    # https://huggingface.co/Vamsi/T5_Paraphrase_Paws
    model_name = "Vamsi/T5_Paraphrase_Paws"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=args.cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)

    for line in data:
        emo, utterance = line.strip().split('\t')
        out_texts = get_response("paraphrase: " + utterance + " </s>", model, tokenizer)
        update_outs('t5', utterance, emo, out_texts)


def write_outs():
    print('Done! Write outputs...')
    args.output_dir = os.path.join(args.output_dir, f'all_texts_compared_{args.output_file}')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for input_txt, txt_infos in outs.items():
        with open(os.path.join(args.output_dir, f'{input_txt}.txt'), 'w', encoding='utf-8') as file:
            emotion = txt_infos['emo']
            file.write(f'Input Text: {input_txt}\n')
            file.write(f'Emotion: {emotion}\n')
            file.write('='*50)
            file.write('\n\n')

            for key, out_txts in txt_infos.items():
                if key != 'emo':
                    file.write(f'{key.upper()}:\n')
                    for txt, num in zip(out_txts['texts'], out_txts['nums']):
                        file.write(f'{num}. {txt}\n')
                    file.write('-' * 50)
                    file.write('\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for generation")
    parser.add_argument("--max_seq_length", type=int, default=60)
    parser.add_argument("--min_seq_length", type=int, default=5)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--num_return_sequences", type=int, default=20)
    parser.add_argument("--do_sample", default=False, action='store_true')
    parser.add_argument("--early_stopping", default=False, action='store_true')
    parser.add_argument("--temperature", type=float, default=1.0,
                        help='The value used to module the next token probabilities.')
    parser.add_argument("--top_k", type=int, default=50,  # 120
                        help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
    parser.add_argument("--top_p", type=float, default=1.0,  # 0.95
                        help='If set to float < 1, only the most probable tokens with probabilities that add up to '
                             'top_p or higher are kept for generation.')
    parser.add_argument("--num_beam_groups", type=int, default=1,  # 3
                        help='Number of groups to divide num_beams into in order to ensure diversity '
                             'among different groups of beams. ')
    parser.add_argument("--diversity_penalty", type=float, default=0.0,  # 0.2 to 0.8)
                        help='Value to control diversity for group beam search. that will be used by default in the '
                             '`generate` method of the model. 0 means no diversity penalty. '
                             'The higher the penalty, the more diverse are the outputs.')
    parser.add_argument("--cache_dir", type=str, default='/speech/dbwork/mul/spielwiese3/dehoang/caches')
    parser.add_argument("--input_dir", type=str, default='../data')
    parser.add_argument("--input_file", type=str, default='para_exps.txt')
    parser.add_argument("--output_dir", type=str,
                        default='/speech/dbwork/mul/spielwiese3/dehoang/outputs/para_exps')
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    # make sure returned seqs smaller than num beams:
    if args.num_return_sequences > args.num_beams:
        args.num_beams = args.num_return_sequences + 10

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not args.output_file:
        args.output_file = f'para_exps_{args.num_return_sequences}.json'

    gen_kwargs = {
        "max_length": args.max_seq_length,
        "min_length": args.min_seq_length,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "do_sample": args.do_sample,
        "early_stopping": args.early_stopping,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "num_beam_groups": args.num_beam_groups,
        "diversity_penalty": args.diversity_penalty
    }

    if not os.path.exists(os.path.join(args.output_dir, f'para_exps_{args.output_file}.json')):
        outs = {}
        with open(os.path.join(args.input_dir, args.input_file), 'r', encoding='utf-8') as f:
            data = f.read().strip().split('\n')

        # run all models
        pegasus()
        bart()
        # t5()
        with open(os.path.join(args.output_dir, f'para_exps_{args.output_file}.json'), 'w', encoding='utf-8') as jfile:
            json.dump(outs, jfile, indent=4)
    else:
        with open(os.path.join(args.output_dir, f'para_exps_{args.output_file}.json'), 'r', encoding='utf-8') as jfile:
            outs = json.load(jfile)

    write_outs()
