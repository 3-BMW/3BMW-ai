import streamlit as st
import pandas as pd
from gensim.models import Word2Vec
import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type= str, default= './model/region_small_major_set_sh10_ep_10.model')
    # parser.add_argument('--word', type=str, default='포테이토피자')
    
    args= parser.parse_args()

    return args

if __name__ == '__main__':
    args= get_config()

    st.set_page_config(layout= 'wide')
    st.title('BMW TEST')

    if 'model' not in st.session_state:
        model= Word2Vec.load(args.model_path)

    upload_file= st.file_uploader('choose a file', type= ['png', 'jpeg', 'jpg'])

    keyword= st.text_input('단어를 입력해주세요.')
    cnt= st.text_input('추천받고 싶은 해시태크의 수를 입력해주세요. (최대 10개)')

    if st.button('생성'):
        with st.spinner('loading inference'):
            item= model.wv.most_similar(keyword)[:int(cnt)]
            for i in range(int(cnt)):
                st.text(item[i])
