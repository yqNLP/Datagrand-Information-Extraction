import collections
import numpy as np
import copy

tag2label = {'/o': 0,
             '/a': 1,
             '/b': 2,
             '/c': 3}

tag2label_BIO = {"O": 0,
                 "B-A": 1, "I-A": 2,
                 "B-B": 3, "I-B": 4,
                 "B-C": 5, "I-C": 6
                 }


def pad_and_cut(seq, seq_len, pad_token):
    if len(seq) > seq_len:
        ret = seq[:seq_len]
    else:
        ret = seq + [pad_token] * (seq_len - len(seq))
    return ret


def sent2id(sent, vocab):
    sent2id = []
    for w in sent:
        if w in vocab:
            sent2id.append(vocab[w])
        else:
            sent2id.append(1)

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

def write_to_file(preds, lengths, sent2ids, file, vocab):
    def is_valid(pred1, pred2):
        if (pred1, pred2) in [(1,2), (3,4), (5,6)]:
            return False
        else:
            return True
    id2word = {value:key for key,value in vocab.items()}
    id2tag = {}
    for (tag, id) in tag2label.items():
        id2tag[id] = tag
    assert len(preds)==len(lengths)
    assert len(sent2ids)==len(lengths)
    with open(file, mode="w", encoding='utf-8') as fw:
        for i in range(len(preds)):
            pred = preds[i]
            sent2id = sent2ids[i]
            # sent2id = [str(ele) for ele in sent2id]
            sent = []
            for ele in sent2id:
                if ele in sent2id:
                    sent.append(id2word[ele])
                else:
                    sent.append("[UNK]")

            length = lengths[i]
            piece = []
            if len(pred)<length:
                print(len(pred), length)
            # assert len(pred)==length
            for j in range(length):
                if j==length-1:
                    piece.append(sent[j])
                    piece_to_write = "".join(piece)
                    fw.write(piece_to_write+id2tag[(pred[j]+1)//2]+"\n")
                    piece = []
                else:
                    if (pred[j]!=pred[j+1] and is_valid(pred[j], pred[j+1])) or (pred[j]==pred[j+1] and pred[j]%2==1):
                        piece.append(sent[j])
                        piece_to_write = "".join(piece)
                        fw.write(piece_to_write + id2tag[(pred[j]+1)//2] + "  ")
                        piece = []
                    else:
                        piece.append(sent[j]+"_")



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


def get_sent_length_label(examples):
    sent2ids = []
    lengths = []
    labels = []
    for example in examples:
        sent2ids.append(example.text_id)
        lengths.append(example.length)
        labels.append(example.label_id)
    return sent2ids, lengths, labels

def read_vocab(vocab_file):
    vocab = collections.OrderedDict()
    index = 0

    with open(vocab_file, mode="r", encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip()
            vocab[line] = index
            index += 1
    return vocab

def data_augmentation(train_data, maxlen, dup):
    def check_seqLen(row):
        if row[2]<maxlen/2:
            return True
        else:
            return False

    short_data = list(filter(check_seqLen, train_data))
    filter_len = len(short_data)
    print("seqs that less than maxlen/2:{}/{}".format(filter_len, len(train_data)))

    data_augment = []
    for j in range(dup):
        for i in range(filter_len):
            rand_index = np.random.randint(low=0, high=filter_len)
            row0, row1 = [], []
            len1 = short_data[i][2]
            len2 = short_data[rand_index][2]

            row0.extend(short_data[i][0][:len1])
            row0.extend(short_data[rand_index][0][:len2])
            row0 = pad_and_cut(row0, seq_len=maxlen, pad_token=0)

            row1.extend(short_data[i][1][:len1])
            row1.extend(short_data[rand_index][1][:len2])
            row1 = pad_and_cut(row1, seq_len=maxlen, pad_token=0)

            row = (row0, row1, len1+len2)
            data_augment.append(row)
    ret_data = []
    ret_data.extend(train_data)
    ret_data.extend(data_augment)
    return ret_data

def data_augmentation_v3(train_data, maxlen, dup):
    print("+++++++++++++++++++++++using data augment v3+++++++++++++++++++++++++++")
    def check_seqLen(row):
        if row[2]<maxlen/2:
            return True
        else:
            return False

    short_data = list(filter(check_seqLen, train_data))
    filter_len = len(short_data)
    print("seqs that less than maxlen/2:{}/{}".format(filter_len, len(train_data)))

    data_augment = []
    for j in range(dup):
        for i in range(filter_len):
            rand_index = np.random.randint(low=0, high=filter_len)
            row0, row1 = [], []
            len1 = short_data[i][2]
            len2 = short_data[rand_index][2]

            row0.extend(short_data[i][0][:len1])
            row0.extend(short_data[rand_index][0][:len2])
            row0 = pad_and_cut(row0, seq_len=maxlen, pad_token=0)

            row1.extend(short_data[i][1][:len1])
            row1.extend(short_data[rand_index][1][:len2])
            row1 = pad_and_cut(row1, seq_len=maxlen, pad_token=0)

            row = (row0, row1, len1+len2)
            data_augment.append(row)
        if j%2==0:
            data_augment.extend(train_data)

    return data_augment

def split(fr, fw):
    line_list = []
    with open(fr, mode="r", encoding="utf-8") as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip().split()
            line_list.append(line)

    len1 = 0
    len_all = len(line_list)
    with open(fw, mode="w", encoding='utf-8') as fw:
        for line in line_list:
            if len(line)>=5:
                len1 += 1
                line1 = "  ".join(line[:len(line)//2])
                line2 = "  ".join(line[len(line)//2:])
                line3 = "  ".join(line)
                fw.write(line1+"\n")
                fw.write(line2+"\n")
                fw.write(line3+"\n")
            else:
                line = "  ".join(line)
                fw.write(line+"\n")
    print("num of line that split:{}/{}".format(len1, len_all))

def mask(train_data):
    """
    :param train_data:[row], row=(sent, label, length)
    :return:
    """

    train_data_original = copy.deepcopy(train_data)
    for row in train_data:
        sent2id = row[0]
        index = np.random.randint(low=0, high=row[2], size=row[2]//12+1)
        for ind in index:
            sent2id[ind] = 0

    train_data_original.extend(train_data)

    return train_data_original

# def mask(fr, fw):
#     line_list = []
#     with open(fr, mode="r", encoding="utf-8") as fr:
#         lines = fr.readlines()
#         for line in lines:
#             line = line.strip().split()
#             line_list.append(line)

def data_augmentation_v2(train_data, maxlen, dup):
    def check_seqLen(row):
        if row[2]<maxlen/2:
            return True
        else:
            return False

    short_data = list(filter(check_seqLen, train_data))
    filter_len = len(short_data)
    print("seqs that less than maxlen/2:{}/{}".format(filter_len, len(train_data)))

    data_augment = []
    for j in range(dup):
        for i in range(filter_len):
            rand_index = np.random.randint(low=0, high=filter_len)
            row0, row1 = [], []
            len1 = short_data[i][2]
            len2 = short_data[rand_index][2]

            row0.extend(short_data[i][0][:len1])
            row0.extend(short_data[rand_index][0][:len2])
            row0 = pad_and_cut(row0, seq_len=maxlen, pad_token=0)

            row1.extend(short_data[i][1][:len1])
            row1.extend(short_data[rand_index][1][:len2])
            row1 = pad_and_cut(row1, seq_len=maxlen, pad_token=0)

            row = (row0, row1, len1+len2)
            data_augment.append(row)
        data_augment.extend(train_data)


    return data_augment


def create_more_data():
    from BERT_NER import NerProcessor
    vocab = read_vocab("./data/dg_vocab.txt")
    processor = NerProcessor(max_len=200, vocab=vocab)

    train_data_aug = processor.get_train_data("./data/train.txt", True, 8)
    train_data = np.stack(train_data_aug)
    sent2ids = train_data[:,0]
    labels = train_data[:,1]
    lengths = train_data[:,2]

    write_to_file(file="./data/train_0829_aug.txt", lengths=lengths, preds=labels, sent2ids=sent2ids, vocab=vocab)

def reconstuct(fr, fw, dict_file):
    import json
    old_lines = []
    with open(file=fr, mode="r", encoding="utf-8") as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip()
            old_lines.append(line)
    with open(file=dict_file, mode="r", encoding="utf-8") as fr:
        index_map = json.load(fr)
    with open(file=fw, mode="w", encoding="utf-8") as fw:
        for i, line in enumerate(old_lines):
            fw.write(line)
            if i < len(old_lines)-1:
                if index_map[str(i)]!=index_map[str(i+1)]:
                    fw.write("\n")
                else:
                    fw.write("  ")
            else:
                fw.write("\n")

def spilt_and_write(maxlen):
    test_map_file = "./data/test_map_0829.json"
    result_file = "./result_25epoch.txt"
    file_split = "./data/test_0829.txt"

    from BERT_NER import NerProcessor
    import numpy as np
    vocab = read_vocab("./DG_BERT/dg_vocab.txt")
    processor = NerProcessor(max_len=566, vocab=vocab)
    train_data = processor._read_train_data(file_path=result_file, maxlen=566)

    train_data = np.stack(train_data)
    original_sent2ids = train_data[:,0]
    original_labels = train_data[:,1]
    original_lengths = train_data[:,2]

    now2orig = {}
    sent2ids, labels, lengths = [], [], []
    j = 0
    nums = 0
    cnt = 0
    for i in range(len(original_lengths)):
        sent2id = original_sent2ids[i]
        label = original_labels[i]
        length = original_lengths[i]
        if length<=maxlen:
            sent2ids.append(sent2id)
            labels.append(label)
            lengths.append(length)

            now2orig[j] = i
            j += 1
        else:
            nums += 1
            while length>0:

                if length>maxlen:
                    if label[maxlen-1] != 0:
                        cnt += 1

                if length>maxlen:
                    index = maxlen
                    while label[index-1] != 0:
                        index -= 1

                    sent2ids.append(sent2id[:index])
                    labels.append(label[:index])
                    lengths.append(len(label[:index]))

                    sent2id = sent2id[index:]
                    label = label[index:]
                    length -= index
                    if index != maxlen:
                        print("index:{}".format(index))

                else:
                    sent2ids.append(sent2id[:maxlen])
                    labels.append(label[:maxlen])
                    lengths.append(length)

                    sent2id = sent2id[maxlen:]
                    label = label[maxlen:]
                    length -= maxlen

                now2orig[j] = i
                j += 1

    print("num of lines that longer than {}: {}".format(maxlen, nums))
    print("num of times that fails: {}".format(cnt))
    import json

    with open(file=test_map_file, mode="w", encoding="utf-8") as fp:
        json.dump(now2orig, fp)

    id2word = {value:key for key,value in vocab.items()}
    with open(file=file_split, mode="w", encoding="utf-8") as fw:
        for j, sent2id in enumerate(sent2ids):
            for i in range(lengths[j]):
                w = sent2id[i]
                fw.write(id2word[w])
                if i<lengths[j]-1:
                    fw.write("_")
            fw.write("\n")

    # write_to_file(file=file_split, lengths=lengths, preds=labels, sent2ids=sent2ids, vocab=vocab)

def view_length():
    test_spilt_file = "./data/test_0829_split.txt"
    from BERT_NER import NerProcessor
    vocab = read_vocab("./data/dg_vocab.txt")
    processor = NerProcessor(max_len=566, vocab=vocab)

    test_data = processor.get_train_data("./result_25epoch.txt", False, 8)
    train_data = np.stack(test_data)
    sent2ids = train_data[:,0]
    labels = train_data[:,1]
    lengths = train_data[:,2]

    print(max(lengths))
    id2word = {value:key for key,value in vocab.items()}
    with open(file=test_spilt_file, mode="w", encoding="utf-8") as fw:
        for j, sent2id in enumerate(sent2ids):
            for i in range(lengths[j]):
                w = sent2id[i]
                fw.write(id2word[w])
                if i<lengths[j]-1:
                    fw.write("_")
            fw.write("\n")

    # write_to_file(file="./data/test_0829.txt", lengths=lengths, preds=labels, sent2ids=sent2ids, vocab=vocab)


if __name__ == "__main__":
    # vocab = read_vocab("./data/dg_vocab.txt")
    # sent = ["1", "2", "3", "[PAD]", "123456"]
    # sent2ids = sent2id(sent, vocab)
    # split(fr="./data/train.txt", fw="./data/train_split.txt")
    # create_more_data()
    # reconstuct(fr="./data/test_0829.txt", fw="./data/valid_rec.txt", dict_file="./data/valid_map.json")
    # view_length()
    # spilt_and_write(maxlen=200)
    # test_0829.txt

    test_map_file = "./data/test_map_0829.json"
    file_split = "./result/test_0829.txt"
    file_reconstrcut = "./result/test_0829_rec.txt"

    reconstuct(fr=file_split, fw=file_reconstrcut, dict_file=test_map_file)