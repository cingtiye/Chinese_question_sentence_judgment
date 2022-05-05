# -*- coding: utf-8 -*-
import os
import sys
import pickle

import jieba

id2tgt = {0: "非疑问句", 1: "疑问句"}
vectorizer = pickle.load(open("model/vectorizer.pickle", "rb"))
clf = pickle.load(open("model/model.pickle", "rb"))

x = [
    "这个仓库可以起到作用吗",
    "明天能不能早起",
    "今天还过去吗",
    "你喜欢看《中国》吗",
    "明天早起不",
    "明天早起吗",
    "这部电影好看不？",
    "这部电影真的很好看？"
]
sen = list(map(jieba.lcut, x))
sen = list(map(lambda x: " ".join(x), sen))
tfidf = vectorizer.transform(sen)
preds = clf.predict(tfidf)
for s, pred in zip(x, preds):
    print(f"{s} 是 {id2tgt[pred]}")
