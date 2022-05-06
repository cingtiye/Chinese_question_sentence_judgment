# -*- coding: utf-8 -*-
import os
import sys
import random

import jieba
from sklearn.model_selection import train_test_split


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


def delete_end_symbol(sens, p):
    """以概率p删除句子末尾符号"""
    data = []
    for sen in sens:
        sen = sen.strip()
        if len(sen) <= 1:
            continue
        if random.random() <= p:
            sen = sen[:-1]
        data.append(sen)

    return data


def data2txt(output_file, X, Y, mode="w"):
    with open(output_file, mode, encoding="utf-8") as fo:
        for x, y in zip(X, Y):
            x = x.replace(" ", "").strip()
            if len(x) <= 1:
                continue
            fo.write(x + "\t" + str(y) + "\n")
        fo.close()

    return


if __name__ == "__main__":
    root_dir = "E:/work/Chinese_question_sentence_judgment"
    document, label = txt2data(os.path.join(root_dir, "data/all_data.txt"), participle=False)
    document = delete_end_symbol(document, p=0.5)

    x_train, x_val, y_train, y_val = train_test_split(document,
                                                      label,
                                                      test_size=0.3,
                                                      shuffle=True,
                                                      random_state=2022)
    data2txt(
        os.path.join(root_dir, "FastText/data/train.txt"),
        x_train,
        y_train,
        mode="w",
    )
    data2txt(
        os.path.join(root_dir, "FastText/data/dev.txt"),
        x_val,
        y_val,
        mode="w",
    )
    data2txt(
        os.path.join(root_dir, "FastText/data/test.txt"),
        x_val,
        y_val,
        mode="w",
    )
