# -*- coding: utf-8 -*-
import os
import sys
import re
import json
from collections import Counter


def get_data_from_c3_public(file_dir, delete_end_punctuation=False):
    """Get data from c3_public"""
    data, label = [], []

    file_type = "json"
    for file in os.listdir(file_dir):
        if file.split(".")[-1] == file_type:
            with open(os.path.join(file_dir, file), "r", encoding="utf-8") as fp:
                samples = json.load(fp)
                for sample in samples:
                    for sen in sample[0]:
                        sens = split_sen(sen.split("：")[-1])
                        # print(sens)
                        for _sen in sens:
                            if len(_sen) <= 1:
                                continue
                            if _sen[-1] in ("?", "？"):
                                label.append(tag2id["interrogative_sentence"])
                            else:
                                label.append(tag2id["declarative_sentence"])
                            if delete_end_punctuation:
                                data.append(_sen[:-1])
                            else:
                                data.append(_sen)

                    _sen = sample[1][0]["question"][-1]
                    if _sen[-1] in ("?", "？"):
                        label.append(tag2id["interrogative_sentence"])
                        if delete_end_punctuation:
                            data.append(_sen[:-1])
                        else:
                            data.append(_sen)

            fp.close()

    assert len(data) == len(label)
    print(dict(Counter(label)))
    return data, label


def get_data_from_cmrc2018_public(file_dir, delete_end_punctuation=False):
    """Get data from cmrc2018_public"""
    data, label = [], []

    file_type = "json"
    for file in os.listdir(file_dir):
        if file.split(".")[-1] == file_type:
            with open(os.path.join(file_dir, file), "r", encoding="utf-8") as fp:
                samples = json.load(fp)
                samples = samples["data"]
                for sample in samples:
                    for para in sample["paragraphs"]:
                        for ques in para["qas"]:
                            sen = ques["question"]
                            if len(sen) <= 1:
                                continue
                            if sen[-1] in ("?", "？"):
                                label.append(tag2id["interrogative_sentence"])
                                if delete_end_punctuation:
                                    data.append(sen[:-1])
                                else:
                                    data.append(sen)
                            else:
                                label.append(tag2id["interrogative_sentence"])
                                if delete_end_punctuation:
                                    data.append(sen)
                                else:
                                    data.append(sen + "？")

            fp.close()

    assert len(data) == len(label)
    print(dict(Counter(label)))

    return data, label


def split_sen(sen):
    sens = re.split(pat, sen)
    len_sen = 0
    for i in range(len(sens)):
        len_sen += len(sens[i])
        if len_sen < len(sen) and sen[len_sen] in set('。？！?.!'):
            sens[i] += sen[len_sen]
            len_sen += 1

    return sens


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
    pat = r'？|。|！|\?|\.|!'
    tag2id = {"interrogative_sentence": 1, "declarative_sentence": 0}
    X1, y1 = get_data_from_c3_public("data/c3_public")
    X2, y2 = get_data_from_cmrc2018_public("data/cmrc2018_public")

    data2txt("data/all_data.txt", X1, y1)
    data2txt("data/all_data.txt", X2, y2, mode="a")
