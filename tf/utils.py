import numpy as np
import argparse
from data import tag2label_BIO
from data import tag2label

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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

def write_to_file(preds, lengths, sent2ids, file):
    def is_valid(pred1, pred2):
        if (pred1, pred2) in [(1,2), (3,4), (5,6)]:
            return False
        else:
            return True
    id2tag = {}
    for (tag, id) in tag2label.items():
        id2tag[id] = tag
    assert len(preds)==len(lengths)
    assert len(sent2ids)==len(lengths)
    with open(file, mode="w", encoding='utf-8') as fw:
        for i in range(len(preds)):
            pred = preds[i]
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
                    fw.write(piece_to_write+id2tag[(pred[j]+1)//2]+"\n")
                    piece = []
                else:
                    if (pred[j]!=pred[j+1] and is_valid(pred[j], pred[j+1])) or (pred[j]==pred[j+1] and pred[j]%2==1):
                        piece.append(sent2id[j])
                        piece_to_write = "".join(piece)
                        fw.write(piece_to_write + id2tag[(pred[j]+1)//2] + "  ")
                        piece = []
                    else:
                        piece.append(sent2id[j]+"_")



def valid(result_file, label_file):
    def filter_other(chunk):
        if chunk[-2:]=='/o':
            return False
        else:
            return True
    results = []
    labels = []
    with open(result_file, mode="r", encoding='utf-8') as fr:
        result_lines = fr.readlines()
        for line in result_lines:
            line = line.strip().split()
            line = list(filter(filter_other, line))
            line = "  ".join(line)
            results.append(line)
    with open(label_file, mode="r", encoding='utf-8') as fr:
        valid_lines = fr.readlines()
        for line in valid_lines:
            line = line.strip().split()
            line = list(filter(filter_other, line))
            line = "  ".join(line)
            labels.append(line)
    micro_f1(results, labels, split=' ')

if __name__ == "__main__":
    sent2ids = [[1,2,3,4,7,5,6], [1,2,3,4,7,5,6]]
    lengths = [6,5]
    preds = [[1,2,0,1,2,2,0], [1,0,0,5,6,0,0]]
    file = "./test.txt"
    write_to_file(preds, lengths, sent2ids, file)
    # valid(result_file='./test.txt', label_file='./result.txt')