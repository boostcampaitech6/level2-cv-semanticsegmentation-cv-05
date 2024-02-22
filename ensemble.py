import os
import pandas as pd
import numpy as np

# 앙상블 할 폴더에서 파일 목록 가져오기
submission_files = [file for file in os.listdir('./ensemble_voting') if not file.startswith('.')]

# 데이터프레임 미리 읽기
dfs = [pd.read_csv(f'./ensemble_voting/{file}') for file in submission_files]

# 각 파일의 최소 행 수 계산
min_rows = min(df.shape[0] for df in dfs)

# 결과를 저장할 리스트 초기화
image_name = []
classes = []
rles = []
vote_threshold = len(submission_files) // 2

def decode_rle_to_mask(rle, shape=(2048, 2048)):
    s = list(map(int, rle.split()))
    starts, lengths = s[::2], s[1::2]
    starts = np.array(starts) - 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def encode_mask_to_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

for idx in range(min_rows):
    if idx % 50 == 0:
        print(f"Processing {idx + 1}/{min_rows}")
    sum_result = np.zeros((2048, 2048), dtype=np.uint8)
    
    for df in dfs:
        rle = df.loc[idx, 'rle']
        mask = decode_rle_to_mask(rle)
        sum_result += mask
    
    result_mask = np.where(sum_result > vote_threshold, 1, 0)
    rle = encode_mask_to_rle(result_mask)
    
    image_name.append(dfs[0].loc[idx, 'image_name'])
    classes.append(dfs[0].loc[idx, 'class'])
    rles.append(rle)

# 결과 DataFrame 생성 및 저장
result_df = pd.DataFrame({
    "image_name": image_name,
    "class": classes,
    "rle": rles,
})
result_df.to_csv("ensemble.csv", index=False)

print("Ensemble completed and saved to 'ensemble.csv'.")
