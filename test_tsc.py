import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
from dataloader import HXbank_dataset, collate_fn
from torch.utils.data import DataLoader
from model_tsc import TSC
from utils import load_HXbank_dict, use_tsc

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0', help='Disables CUDA training.')
parser.add_argument('--voc-dict-path', default='./data/weibo_dict/5_dict/weibo_words_voc_dict_70', help='Select the dictionary')
parser.add_argument('--useTrainWords', default=True, help='Choose train set words')
parser.add_argument('--useTestWords', default=False, help='Choose test set words')
parser.add_argument('--batch-size', type=int, default=32, help='Number of batch_size')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--lr-decay', type=float, default=0.5, help='Learning rate decay.')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--max-words', type=int, default=100, help='The maximum length of the words.')
parser.add_argument('--word-dim', type=int, default=512, help='Encoding dimension of word')
parser.add_argument('--K-FOLD', type=int, default=5, help='The 5 cross validation.')
parser.add_argument('--epochs', type=int, default=70, help='Number of epochs to train.')
parser.add_argument('--lr-step-size', type=int, default=10, help='Period of learning rate decay.')
parser.add_argument('--lr-gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay.')
parser.add_argument('--count-decline', default=10, help='Early stopping.')

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
print('>>>>>> 情感分析模型运行在：', device)


def test(model, data_loader):
    model.eval()
    res = []
    with torch.no_grad():
        for batch_words_idx, batch_words_mask, batch_labels in tqdm(data_loader, colour='green', desc='Test'):
            batch_words_idx = batch_words_idx.to(args.device)
            batch_words_mask = batch_words_mask.to(args.device)

            out = model(batch_words_idx, batch_words_mask)
            ys = out.to('cpu').data.numpy()
            res = ys[0]
    return res


def run(model, test_loader):
    model_path = './save_model/5_dict/hx_best_model_70.pth'
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print(">>> Model loaded successfully ...")
    else:
        print("Model loading failed !!!")
        return 0

    res = test(model, test_loader)
    print(">>> 积极概率：%.4f   消极概率：%.4f"%(res[0],res[1]))


def run_tsc(sss):
    voc_dict = load_HXbank_dict(args.voc_dict_path)

    datas = use_tsc(sss)
    data_iter = HXbank_dataset(voc_dict, datas, args)
    data_loader = DataLoader(data_iter, batch_size=args.batch_size, collate_fn=collate_fn)
    TSC_model = TSC(args, len(voc_dict)).to(args.device)
    run(TSC_model, data_loader)


if __name__ == '__main__':
    ###############################################
    ##         lr:1e-4, lr_step_size:10          ##
    ###############################################
    voc_dict = load_HXbank_dict(args.voc_dict_path)

    while True:
        print(">>>>>> 请输入内容：", end='')
        sss = input()

        datas = use_tsc(sss)
        data_iter = HXbank_dataset(voc_dict, datas, args)
        data_loader = DataLoader(data_iter, batch_size=args.batch_size, collate_fn=collate_fn)
        TSC_model = TSC(args, len(voc_dict)).to(args.device)
        run(TSC_model, data_loader)

        print("是否继续使用(y/n)：", end='')
        next = input()
        if next == "n" or next == "N":
            break

    print("已结束。")
