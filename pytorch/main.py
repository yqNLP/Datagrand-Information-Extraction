import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import argparse
from dataset import DGDataset, DGTestDataset
from model import NER
from dataset import tag2id
from utils import compute_accuracy
import numpy as np
from utils import write_to_file
from utils import valid

# hyperparameter
parser = argparse.ArgumentParser(description='BiLSTM-CRF')
parser.add_argument('--train_data', type=str, default='./data/new_train.txt', help='train data file')
parser.add_argument('--valid_data', type=str, default='./data/new_valid.txt', help='valid data file')
parser.add_argument('--test_data', type=str, default='./data/test.txt', help='test data file')
parser.add_argument('--batch_size', type=int, default=256, help='#sample of minibatch')
parser.add_argument('--epoch', type=int, default=60, help='#epoch of training')
parser.add_argument('--mode', type=str, default='train', help='train/test')
parser.add_argument('--isValid', type=bool, default=True, help='whether to valid')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--save_path', type=str, default="./model_save/Bilstm_2_layers.pt")
parser.add_argument('--result_file', type=str, default="./result/result.txt")

# model
parser.add_argument('--vocab_size', type=int, default=21226, help='#vocab size')
parser.add_argument('--pretrain_embedding', type=bool, default=False, help='whether use pretrain word embedding')
parser.add_argument('--embedding_dim', type=int, default=300, help='embedding dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--hidden_dim', type=int, default=200, help='BiLSTM hidden dim')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--max_len', type=int, default=200, help='maximum length')
parser.add_argument('--num_of_layer', type=int, default=2, help='num of lstm layer')

args = parser.parse_args()

if args.gpu_id>=0 and torch.cuda.is_available():
    args.use_gpu = True
    device = torch.device("cuda:{}".format(args.gpu_id))
    print("using GPU...")
else:
    args.use_gpu = False
    print("using CPU...")

# Init model
ner = NER(vocab_size=args.vocab_size,
          embedding_dim=200,
          hidden_dim=200,
          tag2id=tag2id,
          embedding=None,
          dropout=0.2,
          num_layers=args.num_of_layer,
          max_len=args.max_len)
if args.use_gpu:
    ner.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(ner.parameters(), lr=args.lr)

def main():
    print(args)
    if args.mode=='train':
        train()
    if args.mode=='test':
        test()

def train():
    # data
    train_data = DGDataset(file_path=args.train_data, tag2id=tag2id, maxlen=args.max_len)
    train_dl = DataLoader(train_data, batch_size=args.batch_size, num_workers=2)

    valid_data = DGDataset(file_path=args.valid_data, tag2id=tag2id, maxlen=args.max_len)
    valid_dl = DataLoader(valid_data, batch_size=args.batch_size, num_workers=2)

    num_of_batch = len(train_data)//args.batch_size + 1
    print("start training...")
    for epoch in range(args.epoch):
        if epoch==args.epoch-1:
            torch.save(ner.state_dict(), args.save_path)
        running_loss = 0.0
        for i, data in enumerate(train_dl):
            sent2id, label, length = data
            if args.use_gpu:
                sent2id = sent2id.to(device)
                label = label.to(device)

            optimizer.zero_grad()

            # pred = ner(sent2id)

            # pred:[batch, seq_len, len(tag2id)], label:[batch, seq_len]
            # print(length)

            # loss = criterion(pred.view(-1, len(tag2id)), label.view(-1))
            loss = ner.neg_log_likelihood(sent2id, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i%10 == 0 and i!=0:
                print('epoch {}/{}, batch {}/{}, loss:{:.3f}'.format(epoch+1, args.epoch, i+1, num_of_batch,running_loss/2))
                running_loss = 0.0

        # valid
        if args.isValid:
            print("===========epoch{},valid===========".format(epoch+1))
            with torch.no_grad():
                preds = []
                lengths = []
                sent2ids = []
                for sent2id, label, length in valid_dl:
                    sent2ids.append(sent2id)
                    if args.use_gpu:
                        sent2id = sent2id.to(device)

                    # shape:[batch, seqlen, len(tag2id)]
                    # pred = ner(sent2id)
                    # shape:[batch, seqlen]
                    # pred = torch.argmax(pred, dim=-1)
                    pred = ner(sent2id)

                    # preds.append(pred.cpu().numpy())
                    preds.append(pred)
                    lengths.append(length.numpy())
                # acc = compute_accuracy(preds, labels, lengths)
                # print("epoch{}: ACC{:.3f}".format(epoch+1, acc))
            preds = np.concatenate(preds, axis=0)
            lengths = np.concatenate(lengths, axis=0)
            sent2ids = np.concatenate(sent2ids, axis=0)
            write_to_file(preds, lengths, sent2ids, "./result/valid_result.txt", tag2id)
            valid(result_file="./result/valid_result.txt", label_file=args.valid_data)

def test():
    print("testing...")
    if not args.use_gpu:
        state_dict = torch.load(args.save_path, map_location='cpu')
    else:
        state_dict = torch.load(args.save_path)
    ner.load_state_dict(state_dict)
    ner.eval()
    test_data = DGTestDataset(file_path=args.test_data, maxlen=args.max_len)
    test_dl = DataLoader(test_data, batch_size=args.batch_size, num_workers=2)

    preds = []
    lengths = []
    sent2ids = []
    print("===========testing===========")
    with torch.no_grad():
        for sent2id, length in test_dl:
            sent2ids.append(sent2id)
            if args.use_gpu:
                sent2id = sent2id.to(device)
            # shape:[batch, seqlen, len(tag2id)]
            # pred = ner(sent2id)
            # shape:[batch, seqlen]
            # pred = torch.argmax(pred, dim=-1)
            pred = ner(sent2id)

            # preds.append(pred.cpu().numpy())
            preds.append(pred)
            lengths.append(length.numpy())

    preds = np.concatenate(preds, axis=0)
    lengths = np.concatenate(lengths, axis=0)
    sent2ids = np.concatenate(sent2ids, axis=0)

    write_to_file(preds, lengths, sent2ids, args.result_file, tag2id)
if __name__ == '__main__':
    main()