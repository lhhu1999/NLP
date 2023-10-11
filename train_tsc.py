import torch
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from dataloader import HXbank_dataset, collate_fn
from evaluate import evaluate, save_result
from torch.utils.data import Subset, DataLoader
from model_tsc import TSC
from utils import load_HXbank_dict, load_HXbank_data, get_k_fold_idx

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda:0', help='Disables CUDA training.')
parser.add_argument('--voc-dict-path', default='./data/weibo_dict/weibo_words_voc_dict_80', help='Select the dictionary')
parser.add_argument('--batch-size', type=int, default=32, help='Number of batch_size')
parser.add_argument('--useTrainWords', default=True, help='Choose train set words')
parser.add_argument('--useTestWords', default=False, help='Choose test set words')
parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--lr-decay', type=float, default=0.5, help='Learning rate decay.')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--max-words', type=int, default=100, help='The maximum length of the words.')
parser.add_argument('--word-dim', type=int, default=512, help='Encoding dimension of word')
parser.add_argument('--K-FOLD', type=int, default=5, help='The 5 cross validation.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
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
print('>>>>>> The code run on the ', device)
criterion = nn.CrossEntropyLoss()


def train(model, optimizer, data_loader):
    model.train()
    pred_labels = []
    predictions = []
    all_labels = []
    total_loss = []
    for batch_words_idx, batch_words_mask, batch_labels in tqdm(data_loader, desc='Train'):
        batch_words_idx = batch_words_idx.to(args.device)
        batch_words_mask = batch_words_mask.to(args.device)
        batch_labels = batch_labels.to(args.device)

        optimizer.zero_grad()
        out = model(batch_words_idx, batch_words_mask)

        loss = criterion(out.float(), batch_labels.reshape(batch_labels.shape[0]).long())
        total_loss.append(loss.cpu().detach())
        ys = out.to('cpu').data.numpy()
        pred_labels += list(map(lambda x: np.argmax(x), ys))  # (32)预测结果
        predictions += list(map(lambda x: x[1], ys))  # (32)标签为正的值
        all_labels += batch_labels.cpu().numpy().reshape(-1).tolist()

        loss.backward()
        optimizer.step()
    res = evaluate(total_loss, pred_labels, predictions, all_labels)
    return res


def test(model, data_loader):
    model.eval()
    pred_labels = []
    predictions = []
    all_labels = []
    total_loss = []
    with torch.no_grad():
        for batch_words_idx, batch_words_mask, batch_labels in tqdm(data_loader, colour='green', desc='Test'):
            batch_words_idx = batch_words_idx.to(args.device)
            batch_words_mask = batch_words_mask.to(args.device)
            batch_labels = batch_labels.to(args.device)

            out = model(batch_words_idx, batch_words_mask)

            loss = criterion(out.float(), batch_labels.reshape(batch_labels.shape[0]).long())
            total_loss.append(loss.cpu().detach())
            ys = out.to('cpu').data.numpy()
            pred_labels += list(map(lambda x: np.argmax(x), ys))  # (32)预测结果
            predictions += list(map(lambda x: x[1], ys))  # (32)标签为正的值
            all_labels += batch_labels.cpu().numpy().reshape(-1).tolist()

    res = evaluate(total_loss, pred_labels, predictions, all_labels)
    return res


def run(model, optimizer, train_loader, test_loader, fold_i):
    max_AUC = 0.5
    decline = 0
    for epoch in range(1, args.epochs + 1):
        print('****** epoch:{} ******'.format(epoch))
        res_train = train(model, optimizer, train_loader)
        save_result("Train", res_train, epoch)

        res_test = test(model, test_loader)
        save_result("Test", res_test, epoch)

        if res_test[3] > max_AUC:
            max_AUC = res_test[3]
            decline = 0
            save_path = './save_model/hx_best_model_80_{}.pth'.format(fold_i)
            torch.save(model.state_dict(), save_path)
        else:
            decline = decline + 1
            if decline >= args.count_decline:
                print("EarlyStopping !!!")
                break

        if epoch % args.lr_step_size == 0:       # 每隔n次学习率衰减一次
            optimizer.param_groups[0]['lr'] *= args.lr_gamma


if __name__ == '__main__':
    print("##############################################################################################")
    print("useTrainWords: " + str(args.useTrainWords) + "; useTestWords: " +str(args.useTestWords))
    print("voc_dict: " + args.voc_dict_path)
    print("##############################################################################################")

    voc_dict = load_HXbank_dict(args.voc_dict_path)
    datas = load_HXbank_data()
    data_iter = HXbank_dataset(voc_dict, datas, args)

    for fold_i in range(1,args.K_FOLD):
        train_idx, test_idx = get_k_fold_idx(len(data_iter), fold_i, args.K_FOLD)
        train_iter = Subset(data_iter, train_idx)
        test_iter = Subset(data_iter, test_idx)

        train_loader = DataLoader(train_iter, batch_size=args.batch_size, collate_fn=collate_fn, drop_last=False)
        test_loader = DataLoader(test_iter, batch_size=args.batch_size, collate_fn=collate_fn, drop_last=False)

        TSC_model = TSC(args, len(voc_dict)).to(args.device)
        TSC_optimizer = optim.AdamW(TSC_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        run(TSC_model, TSC_optimizer, train_loader, test_loader, fold_i)
