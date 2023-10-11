import pandas as pd
import re

def read_csv_column_pandas(file_path, column_name):
    df = pd.read_csv(file_path)
    if column_name in df.columns:
        data_list = df[column_name].tolist()
        return data_list
    else:
        print("index error !!!")
        return None

def split_word(data_list):
    all_words = []
    for item in data_list:
        words = str(item).split(";")
        for word in words:
            word = re.sub(r'\?+', '', word)  # 去掉？？？
            if word != "" and word != "nan" and len(word)<4:
                all_words.append(word)
    all_words = list(set(all_words))
    return all_words


if __name__ == "__main__":
    train_file_path = "./data/train.csv"
    test_file_path = "./data/Test_Data.csv"
    train_entity = read_csv_column_pandas(train_file_path, "entity")
    test_entity = read_csv_column_pandas(test_file_path, "entity")
    train_add_words = split_word(train_entity)
    test_add_words = split_word(test_entity)

    with open("./data/add_train_words.txt", 'w') as f:
        for item in train_add_words:
            f.write(item + '\n')
        f.close()

    with open("./data/add_test_words.txt", 'w') as f:
        for item in test_add_words:
            f.write(item + '\n')
        f.close()

    print("success")
