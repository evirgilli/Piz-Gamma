import numpy as np
import time
import sys
import select
import tty
import termios
import shutil
import os
from datetime import datetime
from DPP_comm import ser, FLAGS, DPP_PreConfig, DPP_Send, DPP_GetBG, DPP_Unpack_Board, DPP_Stop

# --- Initialization and Setup ---
DPP_Stop()
time.sleep(0.5)

if not ser.is_open:
    ser.open()
ser.reset_input_buffer()
ser.reset_output_buffer()

DPP_GetBG()
filtered_region = DPP_PreConfig('pmt') # Set filtered region for PMT: alternatively, use 'sipm'


# Acquisition window (ms) - adjust as needed
max_duration = 2000

# Select data acquisition channel
channel = FLAGS["FLAG_START_RAWeX_CHA"] # Alternatives: [# _RAW2_ ; # _RAW_# ; _CHB ; # _All] - See in DPP_comm.py

keep_updating = True  # Controls the main loop

# Total microseconds in one day:
MICROS_PER_DAY = 24 * 60 * 60 * 1_000_000

# File Initialization
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
binary_filename = f"/home/paolo/Scrivania/Piz Gamma/Pizgamma_2025/{current_time}_Data.bin"
binary_file = open(binary_filename, 'wb')

# Function to check for ESC key without blocking
def check_esc():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        key = sys.stdin.read(1)
        if key == '\x1b':  # ESC key
            return True
    return False

# Save terminal settings for cleanup
original_settings = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin.fileno())  # Enable non-blocking input

# Start Acquisition
DPP_Send(channel, max_duration, False)
timeold = file_start_time = time.time()
analysis_start_time = 0
frame=0

# Assuming total_events is the number of rows in your events_tmp array
structured_dtype = np.dtype([('timestamp', np.uint64), ('height', np.uint16)])


try:
    while keep_updating:
        #frame+=1
        #print('\r',frames, end='', flush=True)

        if check_esc():
            print("\nESC pressed. Stopping acquisition...")
            keep_updating = False
            break

        elapsed_time = time.time() - file_start_time

        if elapsed_time >= 60:  # 30 seconds elapsed
            file_start_time = time.time()
            binary_file.close()
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            binary_filename = f"/home/paolo/Scrivania/Piz Gamma/Pizgamma_2025/{current_time}_Data.bin"
            binary_file = open(binary_filename, 'wb')
            print("Creating new file: ",binary_filename)

        # --- Buffer reading and data saving
        buf = ser.read(4096)

        # Parse incomming message
        board = DPP_Unpack_Board(buf[0:16])
        BASE_timestamp_ms = np.frombuffer(buf[0:4], np.uint32)[0]

        # when the first payload arrive, set analysis start time
        if (analysis_start_time == 0):
            analysis_start_time = BASE_timestamp_ms
            # Get full Unix timestamp in microseconds and remove the date portion:
            unix_start_time = int(time.time() * 1e6) % MICROS_PER_DAY

        # event array - 800events x 5 bytes
        total_events = buf[16] + (buf[17]<<8)
        events = np.frombuffer(buf[20:20+800*5], dtype = np.uint8).reshape( 800, 5)

        # for SPI - TS of the data ready event in ms
        DR_timestamp_ms = np.frombuffer(buf[4096-4:4096], np.uint32)[0]/1000

        events_tmp = np.empty(total_events, dtype=structured_dtype)
        for i in range(total_events):
            # timestamp 24bit, overflows at 16s
            bt02 = (events[i][0] << 16) + (events[i][1] << 8) + (events[i][2])
            bt34 = (events[i][3] <<  8) + (events[i][4] )

            ts_us =  (bt02)
            h1    =  (bt34) & 0b1111111111111

            events_tmp['timestamp'][i] = np.uint64(unix_start_time+ts_us+((BASE_timestamp_ms-analysis_start_time-DR_timestamp_ms)*1000))
            events_tmp['height'][i] = np.uint16(h1)

        binary_file.write(events_tmp.tobytes())

finally:
    # Cleanup terminal settings
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_settings)
    binary_file.close()
    DPP_Stop()
    print("\nDONE!")

