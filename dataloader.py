import torch
from torch.utils.data import Dataset
from utils import load_stop_words, choose_update_jieba_words, split_count
import numpy as np
from utils import get_ner_texts_labels

def get_jieba_seg_res(data_texts, useTrainWords, useTestWords):
    # 加载stop word
    stop_words_name = "./data/hit_stopwords.txt"
    stop_words = load_stop_words(stop_words_name)

    # 更新jieba字典
    choose_update_jieba_words(useTrainWords, useTestWords)

    # 划分
    _, seg_res = split_count(data_texts, stop_words)
    return seg_res


class HXbank_dataset(Dataset):
    def __init__(self, voc_dict, datas, args):
        self.voc_dict = voc_dict
        self.texts = datas[0]
        self.seg_texts = get_jieba_seg_res(self.texts, args.useTrainWords, args.useTestWords)
        self.labels = datas[1]
        self.max_words = args.max_words

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        seg_text = self.seg_texts[item]
        print(">>> 划分结果：" + str(seg_text))
        label = self.labels[item]
        words_idx = []
        for word in seg_text:
            if word in self.voc_dict.keys():
                words_idx.append(int(self.voc_dict[word]))
            else:
                words_idx.append(int(self.voc_dict["<UNK>"]))
        return words_idx[:self.max_words], label, int(self.voc_dict["<PAD>"])


def to_float_tensor(data):
    return torch.FloatTensor(data)


def to_long_tensor(data):
    return torch.LongTensor(data)


def collate_fn(batch):
    max_len = max([len(_[0]) for _ in batch])
    batch_words_mask = np.zeros((len(batch), max_len))

    batch_words_idx = []
    batch_labels = []
    for item in batch:
        words_idx = item[0]
        for i in range(max_len - len(item[0])):
            words_idx.append(item[2])
        batch_words_idx.append(words_idx)
        batch_words_mask[:len(item[0])] = 1
        batch_labels.append(item[1])

    batch_words_idx = to_long_tensor(batch_words_idx)
    batch_words_mask = to_long_tensor(batch_words_mask)
    batch_labels = to_float_tensor(batch_labels)
    return batch_words_idx, batch_words_mask, batch_labels


class HXbank_ner_dataset(Dataset):
    def __init__(self, datas, args, bert_tokenizer):
        self.texts, self.labels = get_ner_texts_labels(datas[0], datas[1], args.max_l)
        self.words = bert_tokenizer.batch_encode_plus(self.texts, max_length=args.max_l, padding='max_length',
                                                      truncation=True)
        self.words_ids = self.words.input_ids
        self.words_attention_mask = self.words.attention_mask

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        word = self.words_ids[item]
        words_mask = self.words_attention_mask[item]
        label = list(self.labels[item])
        return word, words_mask, label


def collate_fn_ner(batch):
    batch_words_idx = []
    batch_words_mask = []
    batch_labels = []
    for item in batch:
        batch_words_idx.append(item[0])
        batch_words_mask.append(item[1])
        batch_labels.append(item[2])
        # if 1 in item[2]:
        #     print(len(item[2]) - item[2][::-1].index(1) - 1)
        # else:
        #     print("xx")

    batch_words_idx = to_long_tensor(batch_words_idx)
    batch_words_mask = to_long_tensor(batch_words_mask)
    batch_labels = to_long_tensor(batch_labels)
    batch_labels = batch_labels.view(-1)
    return batch_words_idx, batch_words_mask, batch_labels
