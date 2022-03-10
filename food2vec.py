import gensim
from gensim.models import Word2Vec

from tqdm import tqdm
import json
import datetime
import random
import argparse
import os

import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_config():
    parser = argparse.ArgumentParser()

    """path, model option"""
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--json_path', type= str, default= './data/small_major_set_data.json')
    parser.add_argument('--model_path', type= str, default= './model/small_major_set_sh10_ep_10.model')
    parser.add_argument('--word', type=str, default='포테이토피자')


    """hyperparameter"""
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--shuffle', type=float, default= 10)
    parser.add_argument('--min_count', type=int, default= 1)
    parser.add_argument('--window', type=int, default= 100)
    parser.add_argument('--sg', type=int, default= 1)
    parser.add_argument('--workers', type=int, default= 4)

    
    args= parser.parse_args()

    return args

def train(args, data, model):

    for i in range(args.shuffle):
        for i in range(len(data)):
            random.shuffle(data[i])

        model.train(data, total_examples= model.corpus_count, epochs= args.epochs)
    # print(model.wv.vectors.shape)
    print(model.wv.most_similar(args.word))    

if __name__ == '__main__':

    args= get_config()

    with open(args.json_path, 'r') as f:
        data= json.load(f)
    
    model= Word2Vec(window= args.window, min_count= args.min_count, workers= args.workers, sg= args.sg)
    model.build_vocab(data)

    start= datetime.datetime.now()

    train(args, data, model)

    model.save(args.model_path)
    
