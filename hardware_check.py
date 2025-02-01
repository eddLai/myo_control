from libemg._streamers._myo_streamer import Myo, emg_mode
from libemg.shared_memory_manager import SharedMemoryManager
import numpy as np
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------------------------------
# 1. 初始化 Myo 與共享記憶體變數
# -------------------------------
# 建立 Myo 物件（此處使用 FILTERED 模式，也可選 RAW）
myo = Myo(mode=emg_mode.FILTERED)
smm = SharedMemoryManager()

# 建立共享記憶體變數：
# "emg" 用來儲存一個 1000x8 的 NumPy 陣列，
# "emg_count" 用來計數（形狀為 (1, 1) 的整數）
lock = threading.Lock()  # 使用鎖以確保線程安全
smm.create_variable("emg", (1000, 8), np.double, lock)
smm.create_variable("emg_count", (1, 1), np.int32, lock)

# 連接到 Myo 裝置
myo.connect()
# 設定 serial timeout 為極短時間（非阻塞讀取）
myo.bt.ser.timeout = 0.001

# -------------------------------
# 2. 註冊 EMG 資料回呼
# -------------------------------
def write_emg(emg_data):
    emg_arr = np.array(emg_data)
    # 將新資料與現有資料垂直堆疊，並只保留最新的 1000 筆
    current = smm.variables["emg"]["data"]
    new_data = np.vstack((emg_arr, current))
    new_data = new_data[:current.shape[0], :]
    with smm.variables["emg"]["lock"]:
        smm.variables["emg"]["data"][:] = new_data
        smm.variables["emg_count"]["data"][:] = smm.variables["emg_count"]["data"] + emg_arr.shape[0]

# 當有 EMG 資料進來時會呼叫 write_emg
myo.add_emg_handler(write_emg)

# -------------------------------
# 3. 資料接收線程：持續呼叫 myo.run() 處理藍牙封包
# -------------------------------
def myo_data_thread():
    while True:
        myo.run()

data_thread = threading.Thread(target=myo_data_thread, daemon=True)
data_thread.start()

# -------------------------------
# 4. 圖形化顯示與鍵盤事件處理（使用 8 個垂直排列的子圖）
# -------------------------------
# 建立一個包含 8 個子圖的圖形（8 行 1 列）
fig, axs = plt.subplots(8, 1, figsize=(9, 6), sharex=True)
# 為 8 個通道分配 8 種不同的顏色
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff8800']
lines = []
x_data = np.arange(1000)

# 對每個子圖進行設定
for i, ax in enumerate(axs):
    # 每個子圖畫出一條線，預設資料皆為 0
    line, = ax.plot(x_data, np.zeros(1000), color=colors[i], lw=1.5)
    lines.append(line)
    ax.set_ylim(-200, 200)
    # 設定子圖標題或 Y 軸標籤（字型大小調整）
    ax.set_ylabel(f"Ch {i}", fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True)
    # 除最後一個子圖外，不顯示 X 軸標籤
    if i < len(axs) - 1:
        ax.set_xticklabels([])

# 為整張圖設置 X 軸標籤與標題
axs[-1].set_xlabel("Sample", fontsize=10)
fig.suptitle("All 8 EMG Channels", fontsize=12)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# 定義動畫更新函式，每 50 毫秒更新一次所有子圖資料
def update_plot(frame):
    with smm.variables["emg"]["lock"]:
        data = smm.variables["emg"]["data"].copy()  # data 的形狀為 (1000, 8)
    for i, line in enumerate(lines):
        line.set_ydata(data[:, i])
    return lines

# 定義振動命令函式
def vibrate_command():
    try:
        myo.vibrate(2)
    except AttributeError as e:
        if "'NoneType' object has no attribute 'typ'" in str(e):
            print("Vibration command sent (ignored None reply).")
        else:
            print("Vibration error:", e)
    except Exception as e:
        print("Vibration error:", e)

# 當按下 'v' 鍵時，啟動一個獨立線程執行振動命令
def on_key_press(event):
    if event.key == 'v':
        print("Vibration triggered!")
        threading.Thread(target=vibrate_command, daemon=True).start()

fig.canvas.mpl_connect('key_press_event', on_key_press)

# 傳入 cache_frame_data=False 以抑制 FuncAnimation 快取警告
ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=True, cache_frame_data=False)

plt.show()
