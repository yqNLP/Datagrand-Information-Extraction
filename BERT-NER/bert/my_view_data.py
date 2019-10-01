import seaborn as sns
import matplotlib.pyplot as plt

def view_corpus(corpusPath):
    len_list = []
    len_max = 0
    with open(corpusPath, mode="r", encoding='utf-8') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines):
            if i%10000==0:
                print("processing line {}".format(i))
            line = line.strip().split('_')
            len_list.append(len(line))
            if(len(line)>len_max):
                len_max = len(line)
    sns.distplot(len_list)
    plt.show()
    print("max line length: {}".format(len_max))

def create_vocab(vocab_file):
    with open(vocab_file, mode="w", encoding="utf-8") as fw:
        other = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        for ele in other:
            fw.write(ele+"\n")
        for i in range(21226):
            fw.write(str(i)+"\n")

def preprocess_corpus(old_file, new_file, maxlen):
    write_list = []
    with open(old_file, mode="r", encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            line = line.strip().split("_")
            write_list.append(line[:maxlen])
            line = line[maxlen:]
            while len(line)>0:
                write_list.append(line[:maxlen])
                line = line[maxlen:]


    with open(new_file, mode="w", encoding="utf-8") as fw:
        for line in write_list:
            line = " ".join(line)
            fw.write(line+"\n")


if __name__=="__main__":
    # view_corpus(corpusPath="./data/corpus.txt")
    create_vocab("./data/dg_vocab.txt")
    # preprocess_corpus(old_file="./data/corpus.txt", new_file="./data/corpus_process.txt", maxlen=200)
