import os.path
import re
import torch
from torch.nn.functional import softmax
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    RobertaConfig
)
from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PerplexityScorer:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval().to(DEVICE)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def score(self, sentences, do_normalize=True):
        pp_scores = []
        for sentence in sentences:
            tokenized_input = self.tokenizer.encode(sentence)
            tensor_input = torch.tensor([tokenized_input]).to(DEVICE)
            loss = self.model(tensor_input, labels=tensor_input)[0]
            pp_scores.append(np.exp(loss.detach().cpu().numpy()))
        return self.normalize_scores(np.array(pp_scores)) if do_normalize else np.array(pp_scores)

    def normalize_scores(self, scores: np.array):
        """Normalize scores in range [0, 1]. Note: smaller perplexity is better"""
        try:
            return (scores - np.max(scores)) / (np.min(scores) - np.max(scores))
        except ValueError:
            print(scores)
            return np.array([0.0] * len(scores))


class SimilarityScorer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def score(self, sentences: list, reference: str) -> np.array:
        sim_scores = []
        ref_emb = self.model.encode(reference)
        sent_embs = self.model.encode(sentences)

        for idx in range(len(sent_embs)):
            cos_score = util.pytorch_cos_sim(ref_emb, sent_embs[idx]).detach().cpu().numpy()
            sim_scores.append(cos_score)
        return np.array(sim_scores)


class StyleScorer:
    def __init__(self, path_to_classifier_dir, cache_dir):
        self.config = RobertaConfig.from_pretrained(path_to_classifier_dir, cache_dir=cache_dir)
        self.tokenizer = RobertaTokenizer.from_pretrained(path_to_classifier_dir, cache_dir=cache_dir)
        self.classifier = RobertaForSequenceClassification.from_pretrained(path_to_classifier_dir,
                                                                           cache_dir=cache_dir,
                                                                           config=self.config)
        self.classifier.eval().to(DEVICE)

    def score(self, sentence, target=None):
        """By default, return softmax"""
        tokenized_inputs = self.tokenizer(sentence, padding='max_length',
                                          truncation=True, return_tensors="pt").to(DEVICE)
        outputs = self.classifier(**tokenized_inputs)
        preds = outputs.logits

        # return accuracy if given targets
        if target:
            preds = preds.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
            return (preds == np.array(target)).astype(np.float32).mean().item()  # accuracy
        return softmax(preds, dim=1)  # else softmax score


class DiversityScorer:
    def __init__(self, n):
        self.ngram = n

    def preds2ngrams(self, predictions: list) -> list:
        ngrams = []
        for pred in predictions:
            tokens = re.sub(r'[^\w\s]', '', pred).strip().split()
            ngrams.extend([tuple(tokens[i:i + self.ngram]) for i in range(len(tokens) - self.ngram + 1)])
        return ngrams

    def score(self, predictions: list) -> float:
        ngrams = self.preds2ngrams(predictions)
        return len(set(ngrams)) / len(ngrams) if len(ngrams) > 0 else 0.0


# https://github.com/martiansideofthemoon/style-transfer-paraphrase/blob/master/style_paraphrase/evaluation/scripts/acceptability.py
class GrammarScorer:
    def __init__(self, path_to_cola_classifier: str):
        self.roberta = RobertaModel.from_pretrained(
            path_to_cola_classifier,
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path=os.path.join(path_to_cola_classifier, 'cola-bin')
        )
        # self.unk_bpe = self.roberta.bpe.encode(" <unk>").strip()

    def score(self, sentence: str):
        """ Always return acceptable score"""
        tokenized_inputs = self.roberta.bpe.encode(self.detokenize(sentence))
        batch = collate_tokens(
            [self.roberta.task.source_dictionary.encode_line("<s> " + tokenized_inputs + " </s>", append_eos=False)],
            pad_idx=1
        )
        batch = batch[:, :512]

        with torch.no_grad():
            prediction = self.roberta.predict('sentence_classification_head', batch.long())[0]

        pred_prob = torch.exp(prediction).max(axis=0)[0].item()
        pred_label = self.label_fn(prediction.argmax(axis=0).item())

        return pred_prob if pred_label == 'acceptable' else 1.0 - pred_prob

    def detokenize(self, x):
        return x.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" )", ")").replace(
            "( ", "(")

    def label_fn(self, label):
        return self.roberta.task.label_dictionary.string(
            [label + self.roberta.task.target_dictionary.nspecial]
        )
