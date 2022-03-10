from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import random 
from tqdm import tqdm
import json

import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type= str, default= './model/small_major_set_sh10_ep_10.model')
    parser.add_argument('--save_model_path', type= str, default= './model/region_small_major_set_sh10_ep_10.model')

    parser.add_argument('--json_path', type= str, default= './data/region_major_set_data.json')

    parser.add_argument('--word', type=str, default='포테이토피자')
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--min_count', type=int, default= 1)
    parser.add_argument('--window', type=int, default= 1)
    parser.add_argument('--sg', type=int, default= 1)
    parser.add_argument('--workers', type=int, default= 4)

    args= parser.parse_args()

    return args

def retrain(args, data, model):
    model.build_vocab(data, update= True)
    model.train(data, total_examples= model.corpus_count, epochs= args.epochs)

if __name__ == '__main__':
    args= get_config()
    MODEL_PATH= args.model_path

    with open(args.json_path, 'r') as f:
        data= json.load(f)

    model= Word2Vec.load(args.model_path)
    # model= Word2Vec(window= args.window, min_count= args.min_count, workers= args.workers, sg= args.sg)

    retrain(args, data, model)

    print(model.wv.most_similar(args.word))
    model.save(args.save_model_path)

