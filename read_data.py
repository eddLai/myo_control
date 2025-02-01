#%%
import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV 檔案
file_path = "data/grandma's_hand.csvemg.csv"
column_names = ["Timestamp", "Channel_A", "Channel_B", "Channel_C", 
                "Channel_D", "Channel_E", "Channel_F", "Channel_G", "Channel_H"]

# 讀取 CSV，並直接指定欄位名稱
data = pd.read_csv(file_path, sep=r"\s+", header=None, names=column_names)

# 確保時間戳記為 datetime 格式（將 UNIX 時間轉為可讀格式）
data["Timestamp"] = pd.to_datetime(data["Timestamp"], unit='s')
# %%
