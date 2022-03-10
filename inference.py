from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import random 
from tqdm import tqdm

import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type= str, default= './model/small_major_set_sh10_ep_10.model')
    parser.add_argument('--word', type=str, default='포테이토 피자')
    
    args= parser.parse_args()

    return args

if __name__ == '__main__':
    args= get_config()
    MODEL_PATH= args.model_path

    model= Word2Vec.load(args.model_path)

    print(model.wv.most_similar(args.word))

