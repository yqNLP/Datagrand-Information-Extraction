import sys, pickle, os, random
import numpy as np


tag2label = {'/o': 0,
             '/a': 1,
             '/b': 2,
             '/c': 3}

tag2label_BIO = {"O": 0,
                 "B-A": 1, "I-A": 2,
                 "B-B": 3, "I-B": 4,
                 "B-C": 5, "I-C": 6
                 }


def read_train_corpus(file_path, maxlen):
        train_data = []

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
                        tag = tag2label[piece[-2:]]
                    except:
                        print("tag not found in tag2id.",line)
                        break
                    if tag==0:
                        label.extend([tag]*len(str2int))
                    else:
                        label.append(tag * 2 - 1)
                        if len(str2int) > 1:
                            label.extend([tag*2]*(len(str2int)-1))

                length = min(len(label), maxlen)
                # padding and cut
                sent = _pad_and_cut(sent, pad_token=21225, seq_len=maxlen)
                label = _pad_and_cut(label, pad_token=0, seq_len=maxlen)
                train_data.append((sent, label, length))
        return train_data


def data_augmentation(train_data, maxlen):
    def check_seqLen(row):
        if row[2]<maxlen/2:
            return True
        else:
            return False

    short_data = list(filter(check_seqLen, train_data))
    filter_len = len(short_data)
    print("seqs that less than maxlen/2:{}/{}".format(filter_len, len(train_data)))

    data_augment = []
    for j in range(2):
        for i in range(filter_len):
            rand_index = np.random.randint(low=0, high=filter_len)
            row0, row1 = [], []
            len1 = short_data[i][2]
            len2 = short_data[rand_index][2]

            row0.extend(short_data[i][0][:len1])
            row0.extend(short_data[rand_index][0][:len2])
            row0 = _pad_and_cut(row0, seq_len=maxlen, pad_token=21225)

            row1.extend(short_data[i][1][:len1])
            row1.extend(short_data[rand_index][1][:len2])
            row1 = _pad_and_cut(row1, seq_len=maxlen, pad_token=0)

            row = (row0, row1, len1+len2)
            data_augment.append(row)
    ret_data = []
    ret_data.extend(train_data)
    ret_data.extend(data_augment)
    return ret_data

# def read_train_corpus(file_path, maxlen):
#         train_data = []
#
#         print("loading data: {}......".format(file_path))
#         with open(file_path, mode='r', encoding='utf-8') as fr:
#             lines = fr.readlines()
#             for line in lines:
#                 sent = []
#                 label = []
#                 line = line.strip()
#                 line = line.split(' ')
#                 for piece in line:
#                     # 过滤空的:''
#                     if len(piece)==0:
#                         continue
#                     assert len(piece) > 2
#                     str2int = [int(str) for str in piece[:-2].split('_')]
#                     sent.extend(str2int)
#                     try:
#                         tag = tag2label[piece[-2:]]
#                     except:
#                         print("tag not found in tag2id.",line)
#                         break
#                     label.extend([tag]*len(str2int))
#
#                 length = min(len(label), maxlen)
#                 # padding and cut
#                 sent = _pad_and_cut(sent, pad_token=21225, seq_len=maxlen)
#                 label = _pad_and_cut(label, pad_token=0, seq_len=maxlen)
#                 train_data.append((sent, label, length))
#         return train_data

def read_test_corpus(file_path, maxlen):
    test_data = []

    print("loading data: {}......".format(file_path))
    with open(file_path, mode='r', encoding='utf-8') as fr:
        lines = fr.readlines()

        for line in lines:
            # line:str
            line = line.strip()
            # sent:list
            sent = line.split('_')
            sent2id = [int(w) for w in sent]

            length = min(len(sent2id), maxlen)
            # padding and cut
            sent2id = _pad_and_cut(sent2id, pad_token=21225, seq_len=maxlen)

            # 0代表label
            test_data.append((sent2id, 0, length))

    return test_data

def _pad_and_cut(seq, seq_len, pad_token):
    if len(seq)>seq_len:
        ret = seq[:seq_len]
    else:
        ret = seq + [pad_token]*(seq_len-len(seq))
    return ret


def random_embedding(vocab_size, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat



def batch_yield(data, batch_size, shuffle=False):
    """
    :param data:
    :param batch_size:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)

    seqs, labels, lengths = [], [], []
    for (sent, label, length) in data:
        if len(seqs) == batch_size:
            yield seqs, labels, lengths
            seqs, labels, lengths = [], [], []

        seqs.append(sent)
        labels.append(label)
        lengths.append(length)

    if len(seqs) != 0:
        yield seqs, labels, lengths


if __name__=="__main__":
    # train_data = read_train_corpus(file_path='./data/train.txt', maxlen=566)

    test_data = read_test_corpus(file_path='./data/test.txt', maxlen=566)

    num = 0
    for i in range(len(test_data)):
        if(test_data[i][2]<=256):
           num += 1
        else:
            print(test_data[i][2])
    print("seqs that shorter than 256: {}/{}".format(num, len(test_data)))

    # train
    # seqs that shorter than 200: 16941 / 17000
    # seqs that shorter than 150: 16787 / 17000
    # seqs that shorter than 128: 16595 / 17000

    # test
    # seqs that shorter than 200: 2989 / 3000
    # seqs that shorter than 150: 2963 / 3000
    # seqs that shorter than 128: 2921 / 3000
    # pass