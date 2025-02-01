#%%
import numpy as np
shared_memory_items = [
    ["emg", (1000, 8), np.double],       # EMG 數據的共享記憶體項
    ["emg_count", (1, 1), np.int32],     # 計數共享記憶體項
    ["imu", (250, 10), np.double],       # IMU 數據的共享記憶體項
    ["imu_count", (1, 1), np.int32]      # 計數共享記憶體項
]
#%%
import multiprocessing as mp
for item in shared_memory_items:
    item.append(mp.Lock())