import tkinter as tk
from tkinter import ttk
import threading
import time
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 假設使用 libemg._streamers._myo_streamer:
from libemg._streamers._myo_streamer import Myo, emg_mode
from libemg.shared_memory_manager import SharedMemoryManager


class MyoGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Myo GUI - 分成兩種 Plot 模式 (EMG / IMU)")
        self.geometry("1200x700")

        self.is_closing = False  # 在 quit 時標記避免更新

        # --- Myo & SharedMemory ---
        self.myo = Myo(mode=emg_mode.FILTERED)
        self.smm = SharedMemoryManager()
        self.lock = threading.Lock()

        self.smm.create_variable("emg", (1000, 8), np.double, self.lock)
        self.smm.create_variable("emg_count", (1, 1), np.int32, self.lock)

        self.data_receiving_allowed = threading.Event()
        self.data_receiving_allowed.set()
        self.vibrate_lock = threading.Lock()

        # 介面佈局：左側放按鈕 & 模式切換，右側一塊 frame 做為顯示區
        self.left_frame = ttk.Frame(self, width=300)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.right_frame = ttk.Frame(self)
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # 建立左側按鈕
        self.create_left_panel()

        # 建立兩種繪圖 (EMG / IMU)，初始只顯示 EMG
        self.create_plot_area_emg()  
        self.create_plot_area_imu()

        # 預設顯示 EMG
        self.display_mode = tk.StringVar(value="EMG")
        # 顯示 EMG Plot、隱藏 IMU Plot
        self.show_emg_plot()

        # 嘗試連線
        try:
            print("Connecting Myo...")
            self.myo.connect()
            self.myo.bt.ser.timeout = 0.001
            print("Myo connected.")
        except Exception as e:
            print("Connection Error:", e)

        # 如果要啟用 IMU
        try:
            self.myo.write_attr(0x1d, b'\x01\x00')  # 啟用 IMU
        except:
            pass

        # 註冊回呼 (用 lambda 包起來可避免 self 指向錯誤)
        self.myo.add_emg_handler(lambda d: self.on_emg(d))
        self.myo.add_imu_handler(lambda q,a,g: self.on_imu(q,a,g))

        # 資料接收線程
        self.data_thread = threading.Thread(target=self.myo_data_loop, daemon=True)
        self.data_thread.start()

    # ---------------------------
    # 左側按鈕與模式切換
    # ---------------------------
    def create_left_panel(self):
        # 模式切換 (EMG / IMU)
        mode_frame = ttk.Frame(self.left_frame)
        mode_frame.pack(pady=5, fill=tk.X)
        self.rb_mode = tk.StringVar(value="EMG")
        rb_emg = ttk.Radiobutton(mode_frame, text="EMG模式", variable=self.rb_mode, value="EMG", command=self.on_mode_changed)
        rb_imu = ttk.Radiobutton(mode_frame, text="IMU模式", variable=self.rb_mode, value="IMU", command=self.on_mode_changed)
        rb_emg.pack(side=tk.LEFT)
        rb_imu.pack(side=tk.LEFT)

        btn_raw = ttk.Button(self.left_frame, text="Start Raw", command=self.cmd_start_raw)
        btn_raw.pack(pady=5, fill=tk.X)
        btn_filtered = ttk.Button(self.left_frame, text="Start Filtered", command=self.cmd_start_filtered)
        btn_filtered.pack(pady=5, fill=tk.X)
        btn_raw_unf = ttk.Button(self.left_frame, text="Start Raw Unfiltered", command=self.cmd_start_raw_unfiltered)
        btn_raw_unf.pack(pady=5, fill=tk.X)

        btn_mc_start = ttk.Button(self.left_frame, text="MC Start Collection", command=self.cmd_mc_start)
        btn_mc_start.pack(pady=5, fill=tk.X)
        btn_mc_end = ttk.Button(self.left_frame, text="MC End Collection", command=self.cmd_mc_end)
        btn_mc_end.pack(pady=5, fill=tk.X)

        # 輸入振動等級
        vib_frame = ttk.Frame(self.left_frame)
        vib_frame.pack(pady=5, fill=tk.X)
        ttk.Label(vib_frame, text="Vibrate Lv:").pack(side=tk.LEFT)
        self.vibrate_level = tk.StringVar(value="2")
        self.level_spin = ttk.Spinbox(vib_frame, from_=1, to=3, textvariable=self.vibrate_level, width=5)
        self.level_spin.pack(side=tk.LEFT, padx=5)
        btn_vib = ttk.Button(self.left_frame, text="Vibrate", command=self.cmd_vibrate)
        btn_vib.pack(pady=5, fill=tk.X)

        btn_power = ttk.Button(self.left_frame, text="Power Off", command=self.cmd_power_off)
        btn_power.pack(pady=5, fill=tk.X)
        btn_disconn = ttk.Button(self.left_frame, text="Disconnect", command=self.cmd_disconnect)
        btn_disconn.pack(pady=5, fill=tk.X)
        btn_quit = ttk.Button(self.left_frame, text="Quit", command=self.quit_app)
        btn_quit.pack(pady=5, fill=tk.X)

    def on_mode_changed(self):
        mode = self.rb_mode.get()
        if mode == "EMG":
            self.show_emg_plot()
        else:
            self.show_imu_plot()

    # ---------------------------
    # 建立兩種 Plot Area
    # ---------------------------
    def create_plot_area_emg(self):
        # EMG Figure
        self.fig_emg = plt.Figure(figsize=(6,6))
        self.axs_emg = [self.fig_emg.add_subplot(8,1,i+1) for i in range(8)]
        self.lines_emg = []
        self.x_data_emg = np.arange(1000)
        colors = ['b','g','r','c','m','y','k','#ff8800']
        for i, ax in enumerate(self.axs_emg):
            line, = ax.plot(self.x_data_emg, np.zeros(1000), color=colors[i], lw=1.5)
            self.lines_emg.append(line)
            ax.set_ylim(-200, 200)
            ax.set_ylabel(f"Ch {i}", fontsize=9)
            ax.grid(True)
        self.axs_emg[-1].set_xlabel("Sample", fontsize=10)
        self.fig_emg.suptitle("EMG Mode")

        # 將圖嵌入 right_frame
        self.canvas_emg = FigureCanvasTkAgg(self.fig_emg, master=self.right_frame)
        self.canvas_emg_widget = self.canvas_emg.get_tk_widget()

        # 每 50ms 更新 EMG
        def update_emg(frame):
            if self.is_closing: return self.lines_emg
            with self.smm.variables["emg"]["lock"]:
                data = self.smm.variables["emg"]["data"].copy()
            for i, line in enumerate(self.lines_emg):
                line.set_ydata(data[:, i])
            return self.lines_emg

        self.anim_emg = animation.FuncAnimation(self.fig_emg, update_emg, interval=50, blit=False)

    def create_plot_area_imu(self):
        # IMU Figure
        self.fig_imu = plt.Figure(figsize=(6,6))
        self.ax_imu = self.fig_imu.add_subplot(1,1,1)
        self.ax_imu.set_ylim(-30000, 30000)  # IMU加速度根據實際數值
        self.fig_imu.suptitle("IMU Mode")
        self.canvas_imu = FigureCanvasTkAgg(self.fig_imu, master=self.right_frame)
        self.canvas_imu_widget = self.canvas_imu.get_tk_widget()

        # 我們這裡示範三條加速度
        self.x_data_imu = np.arange(1000)
        self.imu_data = np.zeros((1000,3))
        self.lines_imu = []
        colors = ['red','green','blue']
        labels = ['Ax','Ay','Az']
        for c in range(3):
            line, = self.ax_imu.plot(self.x_data_imu, np.zeros(1000), color=colors[c], lw=1, label=labels[c])
            self.lines_imu.append(line)
        self.ax_imu.legend()

        def update_imu(frame):
            if self.is_closing: return self.lines_imu
            # 當 self.imu_data 在 on_imu() 更新
            for c, line in enumerate(self.lines_imu):
                line.set_ydata(self.imu_data[:, c])
            return self.lines_imu

        self.anim_imu = animation.FuncAnimation(self.fig_imu, update_imu, interval=50, blit=False)

    # ---------------------------
    # 分別顯示/隱藏 EMG Plot 與 IMU Plot
    # ---------------------------
    def show_emg_plot(self):
        # 隱藏 IMU plot
        self.canvas_imu_widget.pack_forget()
        self.anim_imu.event_source.stop()  # 暫停 imu 動畫

        # 顯示 EMG plot
        self.canvas_emg_widget.pack(fill=tk.BOTH, expand=True)
        self.anim_emg.event_source.start() # 啟動 emg 動畫

    def show_imu_plot(self):
        # 隱藏 EMG plot
        self.canvas_emg_widget.pack_forget()
        self.anim_emg.event_source.stop()

        # 顯示 IMU plot
        self.canvas_imu_widget.pack(fill=tk.BOTH, expand=True)
        self.anim_imu.event_source.start()

    # ---------------------------
    # 線程
    # ---------------------------
    def myo_data_loop(self):
        while not self.is_closing:
            self.data_receiving_allowed.wait(timeout=0.05)
            if self.is_closing:
                break
            if self.data_receiving_allowed.is_set():
                try:
                    self.myo.run()
                except Exception as e:
                    print("myo.run error (ignored):", e)
        print("myo_data_loop ended.")

    # ---------------------------
    # 回呼
    # ---------------------------
    def on_emg(self, emg_data):
        if self.is_closing:
            return
        emg_arr = np.array(emg_data)
        current = self.smm.variables["emg"]["data"]
        new_data = np.vstack((emg_arr, current))[:current.shape[0], :]
        with self.smm.variables["emg"]["lock"]:
            self.smm.variables["emg"]["data"][:] = new_data
            self.smm.variables["emg_count"]["data"][:] = self.smm.variables["emg_count"]["data"] + emg_arr.shape[0]

    def on_imu(self, quat, acc, gyro):
        if self.is_closing:
            return
        # 更新 self.imu_data
        new_acc = np.array(acc)
        self.imu_data = np.vstack((new_acc, self.imu_data))[:1000,:]

    # ---------------------------
    # Myo 功能
    # ---------------------------
    def cmd_start_raw(self):
        def task():
            print("Start Raw mode...")
            self.execute_myo_command(self.myo.start_raw)
            print("Raw mode started.")
        threading.Thread(target=task, daemon=True).start()

    def cmd_start_filtered(self):
        def task():
            print("Start Filtered mode...")
            self.execute_myo_command(self.myo.start_filtered)
            print("Filtered mode started.")
        threading.Thread(target=task, daemon=True).start()

    def cmd_start_raw_unfiltered(self):
        def task():
            print("Start Raw Unfiltered mode...")
            self.execute_myo_command(self.myo.start_raw_unfiltered)
            print("Raw Unfiltered mode started.")
        threading.Thread(target=task, daemon=True).start()

    def cmd_mc_start(self):
        def task():
            print("MC Start Collection...")
            self.execute_myo_command(self.myo.mc_start_collection)
            print("MC Start Collection done.")
        threading.Thread(target=task, daemon=True).start()

    def cmd_mc_end(self):
        def task():
            print("MC End Collection...")
            self.execute_myo_command(self.myo.mc_end_collection)
            print("MC End Collection done.")
        threading.Thread(target=task, daemon=True).start()

    def cmd_vibrate(self):
        threading.Thread(target=self.vibrate_command, daemon=True).start()

    def vibrate_command(self):
        if not self.vibrate_lock.acquire(blocking=False):
            print("Vibration already in progress; ignoring request.")
            return
        try:
            self.data_receiving_allowed.clear()
            if self.is_closing:
                return
            try:
                level_str = self.vibrate_level.get().strip()
                level = int(level_str) if level_str.isdigit() else 2
                print(f"Sending vibrate({level})...")
                self.myo.vibrate(level)
                time.sleep(0.3)
            except Exception as e:
                print("Vibration error:", e)
            finally:
                print("Releasing vibrate lock and resuming data.")
                if not self.is_closing:
                    self.data_receiving_allowed.set()
        finally:
            self.vibrate_lock.release()

    def cmd_power_off(self):
        def task():
            print("Powering off Myo...")
            self.execute_myo_command(self.myo.power_off)
            print("Power off done.")
        threading.Thread(target=task, daemon=True).start()

    def cmd_disconnect(self):
        def task():
            print("Disconnecting Myo...")
            self.execute_myo_command(self.myo.disconnect)
            print("Disconnected.")
        threading.Thread(target=task, daemon=True).start()

    # ---------------------------
    # 統一的 Myo 命令
    # ---------------------------
    def execute_myo_command(self, cmd_fn, *args, **kwargs):
        self.data_receiving_allowed.clear()
        try:
            cmd_fn(*args, **kwargs)
        except Exception as e:
            print("Myo command error:", e)
        finally:
            if not self.is_closing:
                self.data_receiving_allowed.set()

    # ---------------------------
    # Quit
    # ---------------------------
    def quit_app(self):
        print("Quitting the application...")
        self.is_closing = True
        self.data_receiving_allowed.clear()

        # 停止動畫
        if self.anim_emg: 
            self.anim_emg.event_source.stop()
        if self.anim_imu:
            self.anim_imu.event_source.stop()

        try:
            print("Disconnecting Myo...")
            self.myo.disconnect()
            print("Myo disconnected.")
        except Exception as e:
            print("Error while disconnecting:", e)

        if self.data_thread.is_alive():
            self.data_thread.join(timeout=1.0)
            print("data_thread joined (or timed out).")

        self.destroy()
        print("Main window destroyed.")


if __name__ == "__main__":
    app = MyoGUI()
    app.mainloop()
