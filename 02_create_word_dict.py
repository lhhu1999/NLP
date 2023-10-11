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
    data_path = "./data/train.csv"
    data_ids = read_csv_column_pandas(data_path, "id")
    data_texts = read_csv_column_pandas(data_path, "text")
    data_labels = read_csv_column_pandas(data_path, "negative")

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
    voc_list_2 = sorted([_ for _ in voc_dict.items() if _[1] > 9], key=lambda x:x[1], reverse=True)
    voc_list_3 = [_ for _ in voc_list_2 if _[1] > 19]
    voc_list_4 = [_ for _ in voc_list_3 if _[1] > 29]
    voc_list_5 = [_ for _ in voc_list_3 if _[1] > 39]

    # 生成新的字典
    dict_data = []
    for item in [voc_list_2, voc_list_3, voc_list_4, voc_list_5]:
        dict_data.append({word_count[0]: idx for idx, word_count in enumerate(item)})

    # 添加UNK和 PAD，并保存字典文件
    os.makedirs("./data/dict/", exist_ok=True)
    if useTrainWords == True and useTestWords == True:
        dict_filename = ["./data/dict/add_train_test_words_voc_dict_2", "./data/dict/add_train_test_words_voc_dict_3",
                     "./data/dict/add_train_test_words_voc_dict_4", "./data/dict/add_train_test_words_voc_dict_5"]
    elif useTrainWords == True and useTestWords == False:
        dict_filename = ["./data/dict/add_train_words_voc_dict_10", "./data/dict/add_train_words_voc_dict_20",
                         "./data/dict/add_train_words_voc_dict_30", "./data/dict/add_train_words_voc_dict_40"]
    else:
        dict_filename = ["./data/dict/words_voc_dict_2", "./data/dict/words_voc_dict_3",
                         "./data/dict/words_voc_dict_4", "./data/dict/words_voc_dict_5"]
    add_unk_pad(dict_data)
    write_dict(dict_filename, dict_data)

    print("success")
