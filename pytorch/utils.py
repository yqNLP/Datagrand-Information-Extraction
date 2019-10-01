from random import shuffle
import numpy as np


def micro_f1(sub_lines, ans_lines, split = ' '):
    correct = []
    total_sub = 0
    total_ans = 0
    for sub_line, ans_line in zip(sub_lines, ans_lines):
        sub_line = set(str(sub_line).split(split))
        ans_line = set(str(ans_line).split(split))
        c = sum(1 for i in sub_line if i in ans_line) if sub_line != {''} else 0
        total_sub += len(sub_line) if sub_line != {''} else 0
        total_ans += len(ans_line) if ans_line != {''} else 0
        correct.append(c)
    p = np.sum(correct) / total_sub if total_sub != 0 else 0
    r = np.sum(correct) / total_ans if total_ans != 0 else 0
    f1 = 2*p*r / (p + r) if (p + r) != 0 else 0
    print('total sub:', total_sub)
    print('total ans:', total_ans)
    # print('correct: ', np.sum(correct), correct)
    print('precision: ', p)
    print('recall: ',r)
    return 'f1',f1


def findLargesetIndex(filePath):
    # find the largest index of the corpus
    with open(filePath, mode='r', encoding='utf-8') as fr:
        lines = fr.readlines()
        largestIndex = 0
        for i, line in enumerate(lines):
            # list
            if i%10000==0:
                print("processing line {}".format(i))
            line = line.strip().split('_')
            line = filter(lambda x: x!='', line)
            line2id = [int(ele) for ele in line]
            for num in line2id:
                largestIndex = num if largestIndex<num else largestIndex
    return largestIndex


def partition(train_file='./data/train.txt', new_train_file='./data/new_train.txt', new_valid_file='./data/new_valid.txt'):
    # divide the original train_file into train and valid, train:valid = 9:1

    with open(train_file, mode='r', encoding='utf-8') as fr:
        lines_read = fr.readlines()
        lines = [line for line in lines_read]

        # important
        lines[-1] += '\n'

        # shuffle
        shuffle(lines)

        num_len = len(lines)
        train_lines = lines[:int(0.9*num_len)]
        valid_lines = lines[int(0.9*num_len):]
        with open(new_train_file, 'w', encoding='utf-8') as fw:
            for l in train_lines:
                fw.write(l)
        with open(new_valid_file, 'w', encoding='utf-8') as fw:
            for l in valid_lines:
                fw.write(l)

def compute_accuracy(preds, labels, lengths):
    """
    :param preds: [pred], pred=[batch, seqlen]
    :param labels: [label], pred=[batch, seqlen]
    :param lengths: [length], length=[batch]
    :return:acc
    """
    assert len(preds)==len(labels)
    assert  len(preds)==len(lengths)

    # shape:[N, seqlen]
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    # shape:[N]
    lengths = np.concatenate(lengths, axis=0)

    num_sent = preds.shape[0]
    num_dom = 0
    num_right = 0
    for i in range(num_sent):
        length = lengths[i]
        pred = preds[i][:length]
        label = labels[i][:length]
        num_right += np.sum(pred==label)
        num_dom += length

    return num_right/num_dom


def write_to_file(preds, lengths, sent2ids, file, tag2id):

    id2tag = {}
    for (tag, id) in tag2id.items():
        id2tag[id] = tag
    assert len(preds)==len(lengths)
    assert len(sent2ids)==len(lengths)
    with open(file, mode="w", encoding='utf-8') as fw:
        for i in range(len(preds)):
            pred = preds[i]
            pred = [id2tag[ele] for ele in pred]
            sent2id = sent2ids[i]
            sent2id = [str(ele) for ele in sent2id]
            length = lengths[i]
            piece = []
            if len(pred)<length:
                print(len(pred), length)
            # assert len(pred)==length
            for j in range(length):
                if j==length-1:
                    piece.append(sent2id[j])
                    piece_to_write = "".join(piece)
                    fw.write(piece_to_write+pred[j]+"\n")
                    piece = []
                else:
                    if pred[j]!=pred[j+1]:
                        piece.append(sent2id[j])
                        piece_to_write = "".join(piece)
                        fw.write(piece_to_write + pred[j] + "  ")
                        piece = []
                    else:
                        piece.append(sent2id[j]+"_")

def valid(result_file, valid_file):
    results = []
    labels = []
    with open(result_file, mode="r", encoding='utf-8') as fr:
        result_lines = fr.readlines()
        for line in result_lines:
            line = line.strip()
            results.append(line)
    with open(valid_file, mode="r", encoding='utf-8') as fr:
        valid_lines = fr.readlines()
        for line in valid_lines:
            line = line.strip()
            labels.append(line)
    micro_f1(results, labels, split=' ')

if __name__=='__main__':
    # filePath = './data/corpus.txt'
    # id = findLargesetIndex(filePath)
    # print("largestIndex: {}".format(id))
    # largest: 21224
    # partition()

    # preds = [np.array([[1,2,3,9], [1,2,3,10]]), np.array([[1,2,3,9], [1,2,3,10]])]
    # labels = [np.array([[1,2,3,7], [1,2,4,7]]), np.array([[1,2,3,9], [1,2,3,10]])]
    # lengths = [np.array([3, 3]), np.array([3, 3])]
    # acc = compute_accuracy(preds, labels, lengths)
    # print(acc)
    micro_f1(['11_13/a 12_15/b', '13/a 14/a 15/b'], ['11_13/a 12_15/b', '13/a 14/b'], split=' ')

