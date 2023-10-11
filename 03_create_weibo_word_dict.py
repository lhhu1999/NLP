import os

from utils import load_stop_words, choose_update_jieba_words, split_count, read_csv_column_pandas

UNK = "<UNK>"
PAD = "<PAD>"


def add_unk_pad(dict_data):
    for i in range(len(dict_data)):
        dict_data[i].update({UNK: len(dict_data[i]), PAD: len(dict_data[i]) + 1})


def write_dict(dict_filename, dict_data):
    for i in range(len(dict_filename)):
        f1 = open(dict_filename[i], "w")
        for item in dict_data[i].keys():
            f1.writelines("{},{}\n".format(item, dict_data[i][item]))
        f1.close()


if __name__ == '__main__':
    data_path = "./data/weibo_senti_100k.csv"
    data_texts = read_csv_column_pandas(data_path, "text")
    data_labels = read_csv_column_pandas(data_path, "negative")

    data_path2 = "./data/online_shopping.csv"
    data_texts2 = read_csv_column_pandas(data_path2, "text")
    data_labels2 = read_csv_column_pandas(data_path2, "negative")

    data_texts = data_texts + data_texts2
    data_labels = data_labels + data_labels2

    # 加载stop word
    stop_words_name = "./data/hit_stopwords.txt"
    stop_words = load_stop_words(stop_words_name)

    # 更新jieba字典
    useTrainWords = True
    useTestWords = False
    choose_update_jieba_words(useTrainWords, useTestWords)

    # 初步划分和统计词频
    voc_dict, _ = split_count(data_texts, stop_words)

    # 根据词频排序并根据阈值截取字典
    voc_list_10 = sorted([_ for _ in voc_dict.items() if _[1] > 59], key=lambda x: x[1], reverse=True)
    voc_list_20 = [_ for _ in voc_list_10 if _[1] > 69]
    voc_list_30 = [_ for _ in voc_list_20 if _[1] > 79]
    voc_list_50 = [_ for _ in voc_list_30 if _[1] > 99]

    ####################################################
    hx_data_path = "./data/train.csv"
    hx_data_texts = read_csv_column_pandas(hx_data_path, "text")
    hx_voc_dict, _ = split_count(hx_data_texts, stop_words)
    hx_voc_list_5 = sorted([_ for _ in hx_voc_dict.items() if _[1] > 19], key=lambda x: x[1], reverse=True)

    found_s10 = [item[0] for item in voc_list_10]
    found_s20 = [item[0] for item in voc_list_20]
    found_s30 = [item[0] for item in voc_list_30]
    found_s50 = [item[0] for item in voc_list_50]
    for _ in hx_voc_list_5:
        if _[0] not in found_s10:
            voc_list_10.append(_)
        if _[0] not in found_s20:
            voc_list_20.append(_)
        if _[0] not in found_s30:
            voc_list_30.append(_)
        if _[0] not in found_s50:
            voc_list_50.append(_)
    ####################################################3

    # 生成新的字典
    dict_data = []
    for item in [voc_list_10, voc_list_20, voc_list_30, voc_list_50]:
        dict_data.append({word_count[0]: idx for idx, word_count in enumerate(item)})

    # 添加UNK和 PAD，并保存字典文件
    os.makedirs("./data/weibo_dict/", exist_ok=True)
    dict_filename = ["./data/weibo_dict/weibo_words_voc_dict_60", "./data/weibo_dict/weibo_words_voc_dict_70",
                     "./data/weibo_dict/weibo_words_voc_dict_80", "./data/weibo_dict/weibo_words_voc_dict_100"]

    add_unk_pad(dict_data)
    write_dict(dict_filename[1:2], dict_data[1:2])

    print("success")
