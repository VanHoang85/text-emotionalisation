import os
import json
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def score(sent1, sent2) -> float:
    sent1_emb = model.encode(sent1)
    sent2_emb = model.encode(sent2)
    return util.pytorch_cos_sim(sent1_emb, sent2_emb).detach().cpu().numpy()


def plot():
    # Plot the distribution of numpy data
    plt.hist(all_scores, bins=np.arange(min(all_scores), max(all_scores) + 0.75, 0.75), align='left')

    # Add axis labels
    plt.xlabel("Sim Score")
    plt.ylabel("Freq")
    plt.title("Similarity Distribution")
    plt.legend()
    # plt.show()
    plt.savefig('sim_plot.png')


if __name__ == "__main__":
    in_dir = os.path.join('../data', 'style')
    splits = ['train.json', 'validation.json', 'test.json']
    model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

    all_scores = []
    all_samples = {}
    for split in splits:
        print('Processing', split, '...')
        with open(os.path.join(in_dir, split), 'r', encoding='utf-8') as jfile:
            data = json.load(jfile)

        for utt in tqdm(data.values()):
            cos = float(score(sent1=utt['neutral_sent'], sent2=utt['emotion_sent']))
            all_scores.append(round(cos, 3))

            if round(cos, 3) not in all_samples.keys():
                all_samples[round(cos, 3)] = []
            all_samples[round(cos, 3)].append({"neutral_sent": utt["neutral_sent"],
                                               "emotion_sent": utt["emotion_sent"]})

    print('Do plotting....')
    all_scores = np.array(all_scores)
    plot()  # now plot the similarity scores

    all_samples = {k: v for k, v in sorted(all_samples.items(), key=lambda key: key)}
    with open('sent_pair_with_scores.json', 'w', encoding='utf-8') as jfile:
        json.dump(all_samples, jfile, indent=4)
