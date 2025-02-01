from libemg._streamers._myo_streamer import Myo, emg_mode
from libemg.shared_memory_manager import SharedMemoryManager
import numpy as np
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import time

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
# 3. 控制資料接收的 Event 與資料接收線程
# -------------------------------
# 建立一個全局事件，當 set 時允許資料接收，否則暫停
data_receiving_allowed = threading.Event()
data_receiving_allowed.set()  # 初始允許資料接收

def myo_data_thread():
    while True:
        # 等待直到允許接收資料
        data_receiving_allowed.wait()
        myo.run()

data_thread = threading.Thread(target=myo_data_thread, daemon=True)
data_thread.start()

# -------------------------------
# 4. GUI 顯示：使用 8 個垂直排列的子圖來分離呈現各通道資料
# -------------------------------
# 建立一個包含 8 個子圖的圖形（8 行 1 列），共用 X 軸
fig, axs = plt.subplots(8, 1, figsize=(9, 6), sharex=True)
# 為了讓下方有空間放置按鈕，調整底部邊界
fig.subplots_adjust(bottom=0.25)
# 為 8 個通道分配 8 種不同的顏色
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff8800']
lines = []
x_data = np.arange(1000)

for i, ax in enumerate(axs):
    line, = ax.plot(x_data, np.zeros(1000), color=colors[i], lw=1.5)
    lines.append(line)
    ax.set_ylim(-200, 200)
    ax.set_ylabel(f"Ch {i}", fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True)
    if i < len(axs) - 1:
        ax.set_xticklabels([])

axs[-1].set_xlabel("Sample", fontsize=10)
fig.suptitle("All 8 EMG Channels", fontsize=12)
fig.tight_layout(rect=[0, 0.07, 1, 0.95])

# -------------------------------
# 5. 定義動畫更新函式，每 50 毫秒更新一次所有子圖資料
# -------------------------------
def update_plot(frame):
    with smm.variables["emg"]["lock"]:
        data = smm.variables["emg"]["data"].copy()  # data shape: (1000, 8)
    for i, line in enumerate(lines):
        line.set_ydata(data[:, i])
    return lines

# -------------------------------
# 6. 定義振動命令函式（修改後版本：使用計數器處理等待中的請求）
# -------------------------------
vibrate_lock = threading.Lock()
vibrate_pending = False  # 用布林值記錄是否已有排隊請求

def vibrate_command():
    global vibrate_pending
    # 嘗試以非阻塞方式取得振動鎖
    if not vibrate_lock.acquire(blocking=False):
        if not vibrate_pending:
            vibrate_pending = True
            print("Vibration already in progress; queued extra request.")
        else:
            print("Vibration already in progress; extra request already queued.")
        return
    try:
        # 暫停資料接收，避免振動回應干擾資料解析
        data_receiving_allowed.clear()
        try:
            myo.vibrate(2)
            # 模擬振動持續一段時間（例如 0.3 秒）
            time.sleep(0.3)
        except AttributeError as e:
            if "'NoneType' object has no attribute 'typ'" in str(e):
                print("Vibration command sent (ignored None reply).")
            else:
                print("Vibration error:", e)
        except Exception as e:
            print("Vibration error:", e)
        finally:
            data_receiving_allowed.set()
    finally:
        vibrate_lock.release()
    # 振動命令結束後，檢查是否有排隊請求
    if vibrate_pending:
        vibrate_pending = False
        print("Executing queued vibration command.")
        threading.Thread(target=vibrate_command, daemon=True).start()

# -------------------------------
# 7. 在 GUI 中加入一個按鈕以觸發振動
# -------------------------------
button_ax = fig.add_axes([0.4, 0.01, 0.2, 0.06])
vibrate_button = Button(button_ax, 'Vibrate', color='lightgray', hovercolor='0.975')

def vibrate_callback(event):
    print("Vibrate button clicked!")
    threading.Thread(target=vibrate_command, daemon=True).start()

vibrate_button.on_clicked(vibrate_callback)

# -------------------------------
# 8. 啟動動畫，每 50 毫秒更新一次圖表
# -------------------------------
ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=True, cache_frame_data=False)

plt.show()
