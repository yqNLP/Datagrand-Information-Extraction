import tensorflow as tf
import numpy as np
import os, argparse, time
from model import BiLSTM_CRF
from data import read_train_corpus, read_test_corpus, random_embedding, data_augmentation
from utils import str2bool

## hyperparameters
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Datagrand NER task')
parser.add_argument('--train_data', type=str, default='new_train.txt', help='train data source')
parser.add_argument('--valid_data', type=str, default='new_valid.txt', help='valid data source')
parser.add_argument('--test_data', type=str, default='test.txt', help='test data source')

parser.add_argument('--batch_size', type=int, default=512, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=50, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='', help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='train', help='train/test')

parser.add_argument('--gpu_id', type=str, default='1', help='#gpu id')
parser.add_argument('--vocab_size', type=int, default=21226, help='#vocab size')
parser.add_argument('--max_len', type=int, default=200, help='#maxlen')
parser.add_argument('--model_path', type=str, default='./model_save', help='path to save model')
parser.add_argument('--restore_path', type=str, default='./model_save/model.ckpt-4655', help='path to restore model')
parser.add_argument('--data_augment', type=str2bool, default=False, help='whether to use data augment')

parser.add_argument('--result_path', type=str, default='result.txt', help='path to save result')
parser.add_argument('--valid_result', type=str, default='valid_result.txt', help='path to save valid result')
parser.add_argument('--num_layers', type=int, default=2, help='#num of lstm layers')

args = parser.parse_args()

## Session configuration
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

## get char embeddings
# word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(args.vocab_size, args.embedding_dim)
else:
    embedding_path = './data/word2vec.npy'
    print("loading pretrain vector...")
    embeddings = np.array(np.load(embedding_path), dtype='float32')

## training model
if args.mode == 'train':
    print("loading training data...")
    train_path = os.path.join('./data', args.train_data)
    train_data = read_train_corpus(file_path=train_path, maxlen=args.max_len)
    print(args.data_augment)
    if args.data_augment:
        train_data = data_augmentation(train_data, maxlen=args.max_len)
    print("loading valid data...")
    valid_path = file_path=os.path.join('./data', args.valid_data)
    valid_data = read_train_corpus(file_path=valid_path, maxlen=args.max_len)

    print("building model...")
    result_path = os.path.join('./result', args.result_path)
    valid_result_path = os.path.join('./result', args.valid_result)
    model_path = os.path.join(args.model_path, 'model.ckpt')

    model = BiLSTM_CRF(args,
                       embeddings,
                       model_path=model_path,
                       result_path=result_path,
                       valid_result=valid_result_path,
                       config=config)
    model.build_graph()

    ## train model on the whole training data
    print("train data: {}".format(len(train_data)))
    print("start trainging...")
    model.train(train=train_data, dev=valid_data)  # use test_data as the dev_data to see overfitting phenomena

## testing model
elif args.mode == 'test':
    print("loading testing data...")
    test_path = os.path.join('./data', args.test_data)
    test_data = read_test_corpus(file_path=test_path, maxlen=args.max_len)

    # ckpt_file = tf.train.latest_checkpoint(args.model_path)
    ckpt_file = args.restore_path
    print("loading file from {}...".format(ckpt_file))

    result_path = os.path.join('./result', args.result_path)
    valid_result_path = os.path.join('./result', args.valid_result)

    model = BiLSTM_CRF(args,
                       embeddings,
                       model_path=ckpt_file,
                       result_path=result_path,
                       valid_result=valid_result_path,
                       config=config)
    model.build_graph()
    print("test data: {}".format(len(test_data)))
    print("start testing...")
    model.test(test_data)

