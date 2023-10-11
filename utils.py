import pandas as pd
import jieba
from tqdm import tqdm
import numpy as np
import re

def load_stop_words(stop_words_name):
    stop_words = open(stop_words_name).readlines()
    stop_words = [line.strip() for line in stop_words]
    s = [" ", "#", "##", "nbsp", "\u3000", "{"]
    for _ in s:
        stop_words.append(_)
    return stop_words


def update_jieba_words(dict_file):
    for i in range(len(dict_file)):
        words = open(dict_file[i]).readlines()
        for w in words:
            jieba.add_word(w.strip())


def choose_update_jieba_words(useTrainWords, useTestWords):
    words_file = []
    if useTrainWords is True:
        words_file = ['./data/add_train_words.txt']
    if useTestWords is True:
        words_file.append('./data/add_test_words.txt')
    update_jieba_words(words_file)


def split_count(datas, stop_words):
    # 分词并统计词频
    dicts = {}
    data = []
    for text in tqdm(datas):
        seg_res = []
        seg_list = list(jieba.cut(text, cut_all=False))
        for seg_item in seg_list:
            if seg_item in stop_words:
                continue
            seg_res.append(seg_item)
            if seg_item in dicts.keys():
                dicts[seg_item] = dicts[seg_item] + 1
            else:
                dicts[seg_item] = 1
        data.append(seg_res)
    return dicts, data


def load_HXbank_dict(voc_dict_path):
    voc_dict = {}
    dict_list = open(voc_dict_path).readlines()
    for item in dict_list:
        item = item.split(",")
        voc_dict[item[0]] = item[1].strip()
    return voc_dict


def read_csv_column_pandas(file_path, column_name):
    df = pd.read_csv(file_path)  # 使用pandas读取CSV文件
    if column_name in df.columns:
        data_list = df[column_name].tolist()  # 提取"Name"列的数据并转换为列表
        return data_list
    else:
        print("index error !!!")
        return None


def load_HXbank_data():
    file_path = "./data/train.csv"
    texts = read_csv_column_pandas(file_path, 'text')
    labels = read_csv_column_pandas(file_path, 'negative')

    zipped_data = list(zip(texts, labels))
    np.random.shuffle(zipped_data)
    texts, labels = zip(*zipped_data)
    return [texts, labels]


def load_HXbank_ner_data():
    file_path = "./data/train.csv"
    text = read_csv_column_pandas(file_path, 'text')
    label = read_csv_column_pandas(file_path, 'key_entity')

    zipped_data = list(zip(text, label))
    np.random.shuffle(zipped_data)
    text, label = zip(*zipped_data)

    texts = []
    labels = []
    for i in range(len(label)):
        if str(label[i]) != 'nan':
            labels.append(label[i])
            texts.append(text[i])
    return [texts, labels]

def remove_punctuation(input_string):
    # 使用正则表达式匹配中文标点符号和英文标点符号，并替换为空字符串
    clean_string = re.sub(r'[^\w\s]', '', input_string).replace(" ", "")
    return clean_string


def get_ner_texts_labels(texts, entities, max_l):
    all_text = []
    labels = []
    for i, item in enumerate(entities):
        label = np.array([0] * max_l)
        s = remove_punctuation(texts[i])
        text = s[:max_l-2]
        all_text.append(text)
        words = str(item).split(";")
        for w in words:
            w = remove_punctuation(w)
            j = text.find(w)
            if j >= 0:
                label[j+1:j+len(w)+1] = 1
        labels.append(label)
    return all_text, labels


def load_weibo_data():
    file_path = "./data/weibo_senti_100k.csv"
    texts = read_csv_column_pandas(file_path, 'text')
    labels = read_csv_column_pandas(file_path, 'negative')

    data_path2 = "./data/online_shopping.csv"
    data_texts2 = read_csv_column_pandas(data_path2, "text")
    data_labels2 = read_csv_column_pandas(data_path2, "negative")

    texts = texts + data_texts2
    labels = labels + data_labels2

    zipped_data = list(zip(texts, labels))
    np.random.shuffle(zipped_data)
    texts, labels = zip(*zipped_data)
    return [texts, labels]


def get_k_fold_idx(lens, fold_i, K_FOLD):
    idx = [id for id in range(lens)]

    fold_size = lens // K_FOLD
    val_start = fold_i * fold_size
    val_end = (fold_i + 1) * fold_size

    train_idx = idx[:val_start] + idx[val_end:]
    test_idx = idx[val_start:val_end]

    return train_idx, test_idx


def use_tsc(input_text):
    texts = [input_text]
    labels = [1]
    zipped_data = list(zip(texts, labels))
    np.random.shuffle(zipped_data)
    texts, labels = zip(*zipped_data)
    return [texts, labels]


def use_ner(input_text):
    texts = [input_text]
    labels = [input_text[0]]
    text = remove_punctuation(input_text)
    return [texts, labels], text
