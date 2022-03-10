import os
import gc
import pickle
import gzip
import math

from PIL import Image # Pillow, OpenCV, PIL 
from io import BytesIO
from socket import timeout
from urllib.request import urlopen
import urllib.parse
import urllib.request

import cx_Oracle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.applications.xception import preprocess_input, decode_predictions

class Embed_predictor():
    def __init__(self, cuda_divice_num = 1):
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_divice_num)
        self.vector_shape = (813, 640, 3)
        self.model = self.make_embedding_model(self.vector_shape)
        
    def make_embedding_model(self, vector_shape):
        base_model = Xception(weights = 'imagenet',    # 이미지넷으로 트레이닝 된 모델(Xception)을 불러옴
                              include_top = False,     # 상단의 Fully-connected layer를 포함할 것인지 아닌지 결정 
                              input_shape = vector_shape)

        base_model.trainable = False                   # Feature extraction 파라미터들은 이미지넷으로 학습된 값들을 
                                                       # 그대로 사용할 것이기 때문에 trainable의 속성을 False로 함

        # Add Layer Embedding
        xception_model = tf.keras.Sequential([
                                              base_model,
                                              GlobalMaxPooling2D()
                                             ])
        
        return xception_model
    
    
    def load_img_from_url(self, url):
        resp = urlopen(url)
        im_src = Image.open(BytesIO(resp.read()))
        mode = im_src.mode

        if mode != 'RGB':
            im = im_src.convert('RGB')
        else:
            im = im_src

        img_vector = np.array(im, dtype=np.uint8)
        return img_vector, mode
        
    def get_embedding(self, img_name):
        img_vector, mode = self.load_img_from_url(img_name)

        try:
            img_vector = np.expand_dims(img_vector, axis=0)
            preprocess_image = preprocess_input(img_vector) 
            return self.model.predict(preprocess_image).reshape(-1)

        except Exception as e:
            return np.zeros(2048)
            print("e: ", e)        

    def return_df(self, img_shape_df):
        emb_img_list = []
        for img_url in img_shape_df['IMAGE']:
            img_emb = self.get_embedding(img_url)
            emb_img_list.append(img_emb)
            
        lf_img_list = pd.DataFrame(emb_img_list)
        img_emb_df = pd.concat([img_shape_df[['PROD_CD', 'IMAGE']], lf_img_list], axis=1)
        img_emb_df = img_emb_df.sort_values('PROD_CD')
        img_emb_df = img_emb_df.reset_index(drop = True)                
            
        return img_emb_df