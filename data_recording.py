from libemg.streamers import myo_streamer
from libemg.data_handler import OnlineDataHandler

if __name__ == "__main__":
    streamer, sm = myo_streamer()
    # streamer.run()
    # streamer.m.vibrate(0)
    # streamer.clean_up()
    odh = OnlineDataHandler(sm)
    odh.visualize()
    # odh.analyze_hardware()
    # odh.log_to_file(block=True, file_path="data/grandma's_hand.csv")
    # odh.visualize_heatmap()
    # if keyboard.is_pressed("s"):
    #     odh.log_to_file()