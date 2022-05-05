# -*- coding: utf-8 -*-
import os
import sys
import random

import jieba

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score


def txt2data(file_name, participle=True):
    """从 all_data.txt 读取数据
    Args:
        file_name (str): all_data.txt 文件地址,标签为0表示非疑问句，标签为1表示问句
        participle (bool): 是否分词
    Return:
        X, y
    """
    X, y = [], []
    with open(file_name, "r", encoding="utf-8") as fp:
        for line in fp:
            sample = line.rstrip().split("\t")
            if len(sample) != 2:
                continue

            sen = sample[0]
            if participle:
                sen = jieba.lcut(sen)

            sen = " ".join(sen)
            X.append(sen)
            y.append(int(sample[1]))

        fp.close()

    return X, y


def get_features(max_features):
    """获取 TfidfVector特征
    Args:
        max_features (int): 最大特征数，超参数
    Return:
        x_train, x_val, y_train, y_val
    """
    tfidf = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        max_features=max_features,
        # ngram_range=(1, 3),
    ).fit_transform(document)

    x_train, x_val, y_train, y_val = train_test_split(tfidf,
                                                      label,
                                                      test_size=0.3,
                                                      shuffle=True,
                                                      random_state=2022)

    return x_train, x_val, y_train, y_val


def run_model(x_train, x_val, y_train, y_val):
    # clf1 = LinearSVC()
    # clf1.fit(x_train, y_train)
    # val_pre = clf1.predict(x_val)
    # f1 = f1_score(y_val, val_pre, average='macro')
    # print('TF-IDF + LinearSVC : %.4f' % f1)
    #
    # clf2 = DecisionTreeClassifier()
    # clf2.fit(x_train, y_train)
    # val_pre = clf2.predict(x_val)
    # f1 = f1_score(y_val, val_pre, average='macro')
    # print('TF-IDF + DecisionTree : %.4f' % f1)
    #
    # clf3 = AdaBoostClassifier(random_state=2022)
    # clf3.fit(x_train, y_train)
    # val_pre = clf3.predict(x_val)
    # f1 = f1_score(y_val, val_pre, average='macro')
    # print('TF-IDF + AdaBoost : %.4f' % f1)

    # clf4 = VotingClassifier(estimators=[('svc', clf1), ('tree', clf2), ('ada', clf3)],
    #                         voting='soft', weights=[1, 1, 1])
    # clf4.fit(x_train, y_train)
    # val_pre = clf4.predict(x_val)
    # f1 = f1_score(y_val, val_pre, average='macro')
    # print('TF-IDF + VotingSoft : %.4f' % f1)

    clf1 = LinearSVC()
    clf2 = DecisionTreeClassifier()
    clf3 = AdaBoostClassifier(random_state=2022)
    clf5 = VotingClassifier(estimators=[('svc', clf1), ('tree', clf2), ('ada', clf3)])
    clf5.fit(x_train, y_train)
    val_pre = clf5.predict(x_val)
    f1 = f1_score(y_val, val_pre, average='macro')
    print('TF-IDF + VotingHard : %.4f' % f1)

    return


def delete_end_symbol(sens, p):
    """以概率p删除句子末尾符号"""
    data = []
    for sen in sens:
        if random.random() <= p:
            sen = sen[:-1]
        data.append(sen)

    return data


if __name__ == "__main__":
    document, label = txt2data("data/all_data.txt", participle=True)
    for p in [0.0, 0.3, 0.5, 1.0]:
        for max_fea in [10, 100, 500, 1000, None]:
            print(f"================ p={p}, max_fea={max_fea} ========================")
            document = delete_end_symbol(document, p=p)
            x_train, x_val, y_train, y_val = get_features(max_fea)
            run_model(x_train, x_val, y_train, y_val)
            print(f"============================ END =================================")
