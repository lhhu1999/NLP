import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
from dataloader import HXbank_ner_dataset, collate_fn_ner
from torch.utils.data import DataLoader
from model_ner import NER
from utils import use_ner
from transformers import BertTokenizer, BertModel

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0', help='Disables CUDA training.')
parser.add_argument('--batch-size', type=int, default=32, help='Number of batch_size')
parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--lr-decay', type=float, default=0.5, help='Learning rate decay.')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--max-l', type=int, default=64, help='The maximum length of the words.')
parser.add_argument('--K-FOLD', type=int, default=5, help='The 5 cross validation.')
parser.add_argument('--epochs', type=int, default=120, help='Number of epochs to train.')
parser.add_argument('--lr-step-size', type=int, default=30, help='Period of learning rate decay.')
parser.add_argument('--lr-gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay.')
parser.add_argument('--count-decline', default=30, help='Early stopping.')

args = parser.parse_args()

if args.device == 'cuda:0':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print("cuda is not available!!!")
        device = torch.device('cpu')
else:
    device = torch.device('cpu')
print('>>>>>> 实体抽取模型运行在：', device)


def test(model, data_loader):
    model.eval()
    pred_labels = []
    with torch.no_grad():
        for batch_words_idx, batch_words_mask, batch_labels in tqdm(data_loader, colour='green', desc='Test'):
            batch_words_idx = batch_words_idx.to(args.device)
            batch_words_mask = batch_words_mask.to(args.device)

            out = model(batch_words_idx, batch_words_mask)

            ys = out.to('cpu').data.numpy()
            pred_labels += list(map(lambda x: np.argmax(x), ys))  # (32)预测结果
            res = pred_labels
    return res


def run(model, test_loader, output_text):
    model_path = './save_model/ner_best_checkpoint_4.pth'
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print(">>> Model loaded successfully ...")
    else:
        print("Model loading failed !!!")
        return 0

    res = test(model, test_loader)
    print(">>> 处理后文本：" + output_text)
    print(">>> 预测标签：" + str(res[1:len(output_text)+1]))
    s = []
    ss = ""
    flag = 0
    for i,p in enumerate(res[1:len(output_text)+1]):
        if p == 1:
            ss = ss + output_text[i]
            flag = 1
        else:
            if flag == 1:
                s.append(ss)
                ss = ""
                flag = 0
    for ss in s:
        print(">>> 预测实体：" + ss)


def run_ner(sss):
    bert_tokenizer = BertTokenizer.from_pretrained('bert_tokenizer')
    bert_model = BertModel.from_pretrained('bert_model')

    datas, output_text = use_ner(sss)
    data_iter = HXbank_ner_dataset(datas, args, bert_tokenizer)

    data_loader = DataLoader(data_iter, batch_size=args.batch_size, collate_fn=collate_fn_ner, drop_last=False)
    NER_model = NER(bert_model).to(args.device)
    run(NER_model, data_loader, output_text)


if __name__ == '__main__':
    bert_tokenizer = BertTokenizer.from_pretrained('bert_tokenizer')
    bert_model = BertModel.from_pretrained('bert_model')

    while True:
        print(">>>>>> 请输入内容：", end='')
        sss = input()
        datas, output_text = use_ner(sss)
        data_iter = HXbank_ner_dataset(datas, args, bert_tokenizer)

        data_loader = DataLoader(data_iter, batch_size=args.batch_size, collate_fn=collate_fn_ner, drop_last=False)
        NER_model = NER(bert_model).to(args.device)
        run(NER_model, data_loader, output_text)

        print("是否继续使用(y/n)：", end='')
        next = input()
        if next == "n" or next == "N":
            break

    print("已结束。")
