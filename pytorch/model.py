import torch
import torch.nn as nn
from torchcrf import CRF

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding, dropout=0.2):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding) if embedding is not None else nn.Embedding(vocab_size, embedding_dim, padding_idx=21225)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        """
        :param input:[batch, seq_len]
        :return: [batch, seq_len, embedding_dim]
        """
        return self.dropout(self.embedding(input))


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=1):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim//2,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)


    def forward(self, input):
        """
        :param input: [batch, seq_len, embedding_dim]
        :return: [batch, seq_len, num_layers]
        """
        output, _ = self.lstm(input)
        return output


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

    def forward(self, *input):
        pass


class NER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tag2id, max_len, embedding=None, dropout=0.2, num_layers=1):
        super(NER, self).__init__()
        self.max_len = max_len
        self.embedding = Embedding(vocab_size=vocab_size,
                                   embedding_dim=embedding_dim,
                                   embedding=embedding,
                                   dropout=dropout)
        self.BiLSTM = BiLSTM(embedding_dim=embedding_dim,
                             hidden_dim=hidden_dim,
                             num_layers=num_layers)
        self.crf = CRF(num_tags=4, batch_first=True)

        self.linear = nn.Linear(hidden_dim, len(tag2id))

    def _get_lstm_feature(self, sents):

        # shape: [batch, seq_len, embedding_dim]
        embeds = self.embedding(sents)

        lstm_out = self.BiLSTM(embeds)

        # shape:[batch, seq_len, len(tag2id)]
        emission = self.linear(lstm_out)

        return emission

    def neg_log_likelihood(self, sents, tags):
        """
        :param sents: [batch_size, seq_len]
        :param tags: [batch_size, seq_len]
        :return: scalar
        """
        mask = sents.lt(21225)

        emission = self._get_lstm_feature(sents)
        loss = self.crf(emission, tags, mask=mask, reduction='mean')
        return -loss


    def forward(self, sents):
        """
        :param input: [batch, seq_len]
        :return: [batch, seq_len, len(tag2id)]
        """

        mask = sents.lt(21225)

        emission = self._get_lstm_feature(sents)
        out = self.crf.decode(emission, mask)
        return out


if __name__=='__main__':
    # unit test
    # emb = Embedding(vocab_size=10, embedding_dim=5, dropout=0.2)
    # sent = torch.Tensor([[1,2,3], [1,2,3]]).long()
    # out = emb(sent)
    # print(out)
    # input = torch.Tensor([[[-0.0000, -1.4126, -2.0737, -0.2818, -0.0000],
    #                      [-0.0000, -1.8608, -1.3309,  1.5317,  3.0769],
    #                      [-2.0207,  0.0418,  1.7475, -0.0000, -1.0618]],
    #                     [[-1.2715, -1.4126, -2.0737, -0.2818, -2.0624],
    #                      [-1.5657, -1.8608, -1.3309,  0.0000,  3.0769],
    #                      [-2.0207,  0.0418,  0.0000, -2.2774, -1.0618]]]).long()
    # print(input.size())
    ner = NER(vocab_size=21226, embedding_dim=100, hidden_dim=200, tag2id = {'/o': 0, '/a': 1, '/b': 2, '/c': 3}, max_len=6)
    # sent = torch.randint(low=0, high=1000, size=(4, 30))
    # tags = torch.randint(low=0, high=3, size=(4,30))
    sent = torch.Tensor([[1,3,4,21225, 21225, 21225], [1,3,4,6, 21225, 21225]]).long()
    tag = torch.Tensor([[1,3,3,0, 0, 0], [1,3,2,1, 0, 0]]).long()

    out = ner.forward(sent)
    print(out)
    # loss = ner.neg_log_likelihood(sent, tag)
    # print(loss)
