# -*- coding: utf-8 -*-
import pickle as pkl
from importlib import import_module
from utils_fasttext import build_iterator

import torch
from sklearn.metrics import f1_score


def load_model(dataset="./"):
    model_name = "FastText"
    embedding = 'random'
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    tokenizer = lambda x: [y for y in x]
    vocab = pkl.load(open(config.vocab_path, 'rb'))
    config.n_vocab = len(vocab)
    config.batch_size = 1
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))

    return config, tokenizer, model, vocab


def load_dataset(sens, pad_size=32):
    contents = []

    UNK, PAD = '<UNK>', '<PAD>'

    def biGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        return (t1 * 14918087) % buckets

    def triGramHash(sequence, t, buckets):
        t1 = sequence[t - 1] if t - 1 >= 0 else 0
        t2 = sequence[t - 2] if t - 2 >= 0 else 0
        return (t2 * 14918087 * 18408749 + t1 * 14918087) % buckets

    for content in sens:
        label = 0
        words_line = []
        token = tokenizer(content)
        seq_len = len(token)
        if pad_size:
            if len(token) < pad_size:
                token.extend([PAD] * (pad_size - len(token)))
            else:
                token = token[:pad_size]
                seq_len = pad_size
        # word to id
        for word in token:
            words_line.append(vocab.get(word, vocab.get(UNK)))

        # fasttext ngram
        buckets = config.n_gram_vocab
        bigram = []
        trigram = []
        # ------ngram------
        for i in range(pad_size):
            bigram.append(biGramHash(words_line, i, buckets))
            trigram.append(triGramHash(words_line, i, buckets))
        # -----------------
        contents.append((words_line, int(label), seq_len, bigram, trigram))

    return contents


def eval(sens):
    predicts = []
    model.eval()
    test_data = load_dataset(sens, config.pad_size)
    test_iter = build_iterator(test_data, config)

    with torch.no_grad():
        for sen, (texts, _) in zip(sens, test_iter):
            outputs = model(texts)
            pred = torch.argmax(outputs.data, 1).cpu().numpy()
            # print(f"{sen} 是 {id2target[pred[0]]} .")
            predicts.append(pred[0])

    return predicts


if __name__ == "__main__":
    id2target = {0: "陈述句", 1: "疑问句"}
    config, tokenizer, model, vocab = load_model("./")

    sens, labels = [], []
    with open("data/test.txt", "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.rstrip()
            sen, label = line.split("\t")
            sens.append(sen)
            labels.append(int(label))

    preds = eval(sens)
    assert len(preds) == len(labels)
    print('FastText : %.4f' % f1_score(preds, labels, average="macro"))  # 0.9624
