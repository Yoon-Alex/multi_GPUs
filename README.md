# multi_GPUs
각 인스턴스 당 GPU를 할당, 분산 처리(predict)을 할 수 있도록 구성한 작업. 

## predictor.py
선언 방법
from predictor import Embed_predictor

분산 처리 활용. 
```python
from multiprocessing import Pool

def predict_GPUs(num, df):
    predictor = Embed_predictor(num)
    return predictor.return_df(df.reset_index())

def paralle_GPUs(df, func, n_cores = 4):
    df_split = np.array_split(df, n_cores)    
    a_args = [0,1,2,3] # GPU list 
    
    with Pool(4) as pool:
        lst = pool.starmap(func, zip(a_args, df_split))
    
    return lst  
```

테이블 구조 예시 
|PROD_CD|IMAGE|PROD_NM|
|------|---|---|
|A상품코드|이미지url|상품명|
|B상품코드|이미지url|상품명|
|C상품코드|이미지url|상품명|

```python
data # dataframe 
result = paralle_GPUs(img_shape_df, predict_GPUs, 4)
```
