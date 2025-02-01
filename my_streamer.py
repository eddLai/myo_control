import socket
import multiprocessing
import pickle
from pymyo import Myo, emg_mode
from libemg.data_handler import OnlineDataHandler

def streamer():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    m = Myo(mode=emg_mode.FILTERED)
    m.connect()

    # On every new sample it, simply pickle it and write it over UDP
    def write_to_socket(emg, movement):
        data_arr = pickle.dumps(list(emg))
        sock.sendto(data_arr, ('127.0.0.1' 12345))
    
    m.add_emg_handler(write_to_socket)

    while True:
        m.run()

if __name__ == "__main__" :
    # Create streamer in a seperate Proces so that the main thread is free
    p = multiprocessing.Process(target=streamer, daemon=True)
    p.start()

    # Code leveraging the data goes here:
    odh = OnlineDataHandler(emg_arr=True, port=12345, ip='127.0.0.1')
    odh.start_listening()

    # Do stuff with data...