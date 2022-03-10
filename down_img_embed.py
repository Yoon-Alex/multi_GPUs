#-*- coding: utf-8 -*-
import os
import gc
import pickle
import gzip
import math
os.system("source switch-cuda.sh 11.0")
os.putenv('NLS_LANG', '.UTF8')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cx_Oracle
import pandas as pd
import numpy as np

import itertools
import datetime
from multiprocessing import Pool

from predictor import Embed_predictor

WORKING_DIR = "/home/ec2-user/venv/py3.7.10/img_embed/batch_new_img/"
NOT_FOUND_DIR = '/home/ec2-user/venv/py3.7.10/img_embed/not_embed/'
file_name = 'img_shape_df'
tdy = datetime.datetime.now().strftime("%Y%m%d")    
n_cores = round(os.cpu_count() * 0.5)

def export_raw_data():
    # db connection 
    connection = cx_Oracle.connect('EBIZPWUSER/lfpw99@10.49.8.97:1527/LFDW')

    raw_data = pd.read_sql("""                       
    WITH MART AS ( 
    SELECT  A.PROD_CD 
            , 'http://nimg.lfmall.co.kr'||B.IMG_PATH1 AS IMAGE
      FROM  ( 
            SELECT  DISTINCT PROD_CD 
              FROM  ( 
                    SELECT  PROD_CD 
                      FROM  CMS_ANLY.LF_ORIGIN_PROD_N_MATCH_INF

                    UNION ALL 

                    SELECT  PROD_CD        
                      FROM  WLGF_LST_PRODUCT_DETAIL A
                     WHERE  TO_CHAR(REG_DT , 'YYYYMMDD') >= TO_CHAR(SYSDATE-1, 'YYYYMMDD')
                            OR TO_CHAR(UP_DT , 'YYYYMMDD') >= TO_CHAR(SYSDATE-1, 'YYYYMMDD')                
                    ) A 
            ) A 
      JOIN  WLGF_LST_PRODUCT_DETAIL B 
        ON  A.PROD_CD = B.PROD_CD 
    )
    SELECT  A.PROD_CD
        ,   A.PROD_NM
        ,   B.IMAGE
      FROM  (
             SELECT /*+FULL(A) PARALLEL(A 4) USE_HASH(A B C D)*/
                    A.PROD_CD
                 ,  A.PROD_NM
                 ,  STANDARDCATEGORYID
                 ,  TRIM(BRAND_NM) BRAND_NM
                 ,  TRIM(BRAND_ENM) BRAND_ENM
                 ,  D.TBRAND_CD AS TBRAND_CD
                 ,  TRIM(TBRAND_HNM) TBRAND_HNM
                 ,  TRIM(TBRAND_ENM) TBRAND_ENM
                 ,  B.REG_DT
               FROM WLGF_STB_PRODUCT_D A
                 ,  WLGF_STB_ITEM B
                 ,  WLGF_STB_BRAND C
                 ,  WLGF_LST_DISPLAY_TBRAND D
              WHERE A.PROD_CD = B.ITEM_CD
                AND A.BRAND_CD = C.BRAND_CD
                AND C.TBRAND_CD = D.TBRAND_CD      
                AND PROD_STS_CD = '90'
                AND BUY_ABLEYN = 'Y'
                AND PROD_TYPE = '10'                                
                AND A.FORMAL_GB IS NULL -- 입점상품만              
            ) A 
      JOIN  MART B 
        ON  A.PROD_CD = B.PROD_CD       
    """
    , con = connection
    )
    print(len(raw_data))
    
    return raw_data

def load_img_from_url(url):
    resp = urlopen(url)
    im_src = Image.open(BytesIO(resp.read()))
    mode = im_src.mode
    
    if mode != 'RGB':
        im = im_src.convert('RGB')
    else:
        im = im_src
    
    img_vector = np.array(im, dtype=np.uint8)
    return img_vector, mode

def img_shape_process(data):   
    img_shape_lst = []
    for i in range(len(data)):
        try:
            img_vector, mode = load_img_from_url(data.iloc[i]['IMAGE'])
            img_shape_lst.append(img_vector.shape)            
        
        except Exception as f:
            img_shape_lst.append(404) # HTTPError와 동일하게 404로 뱉음
        
    return img_shape_lst

def paralle_process(df, func, n_cores):
    df_split = np.array_split(df, n_cores)    
    pool = Pool(n_cores)    
    lst = pool.map(func, df_split)
    first_list = list(itertools.chain(*lst)) # 22-02-21 변경
    pool.close()
    pool.join()
    return first_list

def has_comfile(file_name, img_shape_df):
    dirpath = os.getcwd()
    with gzip.open(WORKING_DIR + 'img_shape_df' + '_' + tdy + ".pickle", 'wb') as f:
        pickle.dump(img_shape_df, f)

def make_data_img():
    raw_data = export_raw_data()

    raw_data['IMG_SHAPE'] = \
    pd.Series(paralle_process(raw_data, img_shape_process, n_cores))

    # 데이터 찾을 수 없는 경우, not found 경우, 813 아닌 경우 제외 
    no_data_img = raw_data[raw_data.IMG_SHAPE != (813, 640, 3)]
    raw_data_img = raw_data[raw_data.IMG_SHAPE == (813, 640, 3)]

    # write data 
    if len(no_data_img) > 0 :
        no_data_img['PROD_CD'].reset_index(drop=True).to_csv(NOT_FOUND_DIR+'not_embed_'+tdy+'_bf.csv', encoding='utf-8-sig', index = False)
    
    has_comfile(file_name, raw_data_img)
        
def write_data(img_emb_df): 
    if os.path.isfile(WORKING_DIR + 'img_rslt' + '_' + tdy + ".pickle"):
        os.remove(WORKING_DIR + 'img_rslt' + '_' + tdy + ".pickle")
        with gzip.open(WORKING_DIR + 'img_rslt' + '_' + tdy + ".pickle", 'wb') as f:
            pickle.dump(img_emb_df, f)
            
    else:
        with gzip.open(WORKING_DIR + 'img_rslt' + '_' + tdy + ".pickle", 'wb') as f:
            pickle.dump(img_emb_df, f)        

def predict_GPUs(num, df):
    predictor = Embed_predictor(num)
    return predictor.return_df(df.reset_index())

def paralle_GPUs(df, func, n_cores = 4):
    df_split = np.array_split(df, n_cores)    
    a_args = [0,1,2,3] # GPU list 
    
    with Pool(4) as pool:
        lst = pool.starmap(func, zip(a_args, df_split))
    
    return lst   

def main(): 
    # 파일 
    if os.path.isfile(WORKING_DIR + 'img_shape_df' + '_' + tdy + ".pickle"):
        with gzip.open(WORKING_DIR + 'img_shape_df' + '_' + tdy + ".pickle", "rb") as f:
            raw_data_img = pickle.load(f)    

    else :     
        make_data_img()    
        with gzip.open(WORKING_DIR + 'img_shape_df' + '_' + tdy + ".pickle", "rb") as f:
            raw_data_img = pickle.load(f)    

    img_shape_df = raw_data_img.reset_index(drop=True)
    result = paralle_GPUs(img_shape_df, predict_GPUs, 4)
    img_emb_df = pd.concat(result).reset_index(drop=True)
    
    # NOT EMBED FILE
    if len(img_emb_df[img_emb_df.loc[:,0:].sum(axis = 1) == 0]) > 0 :
        img_emb_df[img_emb_df.loc[:,0:].sum(axis = 1) == 0].PROD_CD.to_csv(NOT_FOUND_DIR+'not_embed_'+tdy+'_af.csv', encoding = 'utf-8-sig', index = False)

    print("Embbeding ", img_emb_df.shape[0], ", Success")
    # WRITE EMBEDDED FILE 
    write_data(img_emb_df)
    print("----------------------------SUCCESS WORD EMBEDDING---------------------------------")

if __name__=='__main__':
    main()    
    