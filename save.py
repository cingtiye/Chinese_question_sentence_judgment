# -*- coding: utf-8 -*-
import os
import sys
import pickle
import random

import jieba

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score

from run import txt2data, delete_end_symbol


def get_features(max_features):
    """获取 TfidfVector特征
    Args:
        max_features (int): 最大特征数，超参数
    Return:
        x_train, x_val, y_train, y_val
    """
    clf = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        max_features=max_features,
        # ngram_range=(1, 3),
    )

    clf.fit(document)
    pickle.dump(clf, open("model/vectorizer.pickle", "wb"))

    tfidf = clf.transform(document)
    x_train, x_val, y_train, y_val = train_test_split(tfidf,
                                                      label,
                                                      test_size=0.3,
                                                      shuffle=True,
                                                      random_state=2022)

    return x_train, x_val, y_train, y_val


def save_model():
    clf1 = LinearSVC()
    clf2 = DecisionTreeClassifier()
    clf3 = AdaBoostClassifier(random_state=2022)

    clf5 = VotingClassifier(estimators=[('svc', clf1), ('tree', clf2), ('ada', clf3)])
    clf5.fit(x_train, y_train)
    pickle.dump(clf5, open("model/model.pickle", "wb"))

    val_pre = clf5.predict(x_val)
    f1 = f1_score(y_val, val_pre, average='macro')
    print('TF-IDF + VotingHard : %.4f' % f1)

    return


if __name__ == "__main__":
    document, label = txt2data("data/all_data.txt", participle=True)
    document = delete_end_symbol(document, p=0.5)
    x_train, x_val, y_train, y_val = get_features(max_features=500)
    save_model()
