from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import argparse
import torch.nn as nn
import numpy as np
import os
from utils import load_HXbank_ner_data, get_k_fold_idx
from dataloader import HXbank_ner_dataset, collate_fn_ner
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
from model_ner import NER
from evaluate import evaluate, save_result

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
print('>>>>>> The code run on the ', device)
class_weights = torch.tensor([1.0, 9.0]).to(args.device)
criterion = nn.CrossEntropyLoss(weight=class_weights)


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
    os.makedirs('./save_model/', exist_ok=True)
    model_path = './save_model/ner_weibo_best_model_XX.pth'
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print(">>> Continue training ...")

    max_AUPR = 0.5
    decline = 0
    for epoch in range(1, args.epochs + 1):
        print('****** epoch:{} ******'.format(epoch))
        res_train = train(model, optimizer, train_loader)
        save_result("Train", res_train, epoch)
        res_test = test(model, test_loader)
        save_result("Test", res_test, epoch)

        if res_test[4] > max_AUPR:
            max_AUPR = res_test[4]
            decline = 0
            save_path = './save_model/ner_best_checkpoint_{}.pth'.format(fold_i)
            torch.save(model.state_dict(), save_path)
        else:
            decline = decline + 1
            if decline >= args.count_decline:
                print("EarlyStopping !!!")
                break

        if epoch % args.lr_step_size == 0:       # 每隔n次学习率衰减一次
            optimizer.param_groups[0]['lr'] *= args.lr_gamma


if __name__ == '__main__':
    bert_tokenizer = BertTokenizer.from_pretrained('bert_tokenizer')
    bert_model = BertModel.from_pretrained('bert_model')

    datas = load_HXbank_ner_data()
    data_iter = HXbank_ner_dataset(datas, args, bert_tokenizer)

    for fold_i in range(args.K_FOLD):
        train_idx, test_idx = get_k_fold_idx(len(data_iter), fold_i, args.K_FOLD)
        train_iter = Subset(data_iter, train_idx)
        test_iter = Subset(data_iter, test_idx)

        train_loader = DataLoader(train_iter, batch_size=args.batch_size, collate_fn=collate_fn_ner, drop_last=False)
        test_loader = DataLoader(test_iter, batch_size=args.batch_size, collate_fn=collate_fn_ner, drop_last=False)

        NER_model = NER(bert_model).to(args.device)
        NER_optimizer = optim.AdamW(NER_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        run(NER_model, NER_optimizer, train_loader, test_loader, fold_i)
