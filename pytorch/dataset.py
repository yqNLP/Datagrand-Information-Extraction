import torch
from torch.utils.data import Dataset

tag2id = {'/o': 0,
          '/a': 1,
          '/b': 2,
          '/c': 3}


class DGDataset(Dataset):
    def __init__(self, file_path, tag2id, maxlen):
        super(DGDataset, self).__init__()
        self.sents = []
        self.labels = []
        self.length = []

        # f = open("./train_length.txt", 'w+')
        print("loading data: {}......".format(file_path))
        with open(file_path, mode='r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                sent = []
                label = []
                line = line.strip()
                line = line.split(' ')
                for piece in line:
                    # 过滤空的:''
                    if len(piece)==0:
                        continue
                    assert len(piece) > 2
                    str2int = [int(str) for str in piece[:-2].split('_')]
                    sent.extend(str2int)
                    try:
                        tag = tag2id[piece[-2:]]
                    except:
                        print("tag not found in tag2id.",line)
                        break
                    label.extend([tag]*len(str2int))

                # print(str(len(label)), file=f)
                self.length.append( min(len(label), maxlen) )
                # padding and cut
                sent = self._pad_and_cut(sent, pad_token=21225, seq_len=maxlen)
                label = self._pad_and_cut(label, pad_token=-1, seq_len=maxlen)

                # to tensor
                sent = torch.Tensor(sent).long()
                label = torch.Tensor(label).long()

                self.sents.append(sent)
                self.labels.append(label)

        self.sents = torch.stack(self.sents, dim=0)
        self.labels = torch.stack(self.labels, dim=0)


    def _pad_and_cut(self, seq, seq_len, pad_token):
        if len(seq)>seq_len:
            ret = seq[:seq_len]
        else:
            ret = seq + [pad_token]*(seq_len-len(seq))
        return ret

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sents[index], self.labels[index], self.length[index]

class DGTestDataset(Dataset):
    def __init__(self, file_path, maxlen):
        super(DGTestDataset, self).__init__()
        self.sents = []
        self.length = []

        # f = open("./test_length.txt", 'w+')
        print("loading data: {}......".format(file_path))
        with open(file_path, mode='r', encoding='utf-8') as fr:
            lines = fr.readlines()

            for line in lines:
                # line:str
                line = line.strip()
                # sent:list
                sent = line.split('_')
                sent2id = [int(w) for w in sent]

                # print(str(len(sent2id)), file=f)
                self.length.append( min(len(sent2id), maxlen) )
                # padding and cut
                sent2id = self._pad_and_cut(sent2id, pad_token=21225, seq_len=maxlen)

                # to tensor
                sent2id = torch.Tensor(sent2id).long()

                self.sents.append(sent2id)

        self.sents = torch.stack(self.sents, dim=0)

    def _pad_and_cut(self, seq, seq_len, pad_token):
        # bug
        if len(seq)>seq_len:
            ret = seq[:seq_len]
        else:
            ret = seq + [pad_token]*(seq_len-len(seq))
        return ret

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, index):
        return self.sents[index], self.length[index]


if __name__ == '__main__':
    # unit test

    train_data = DGDataset('./data/train.txt', tag2id, maxlen=200)
    # print(train_data)

    test_data = DGTestDataset('./data/test.txt', maxlen=200)
    # print(test_data)




