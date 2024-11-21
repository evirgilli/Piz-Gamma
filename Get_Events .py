import numpy as np
import matplotlib.pyplot as plt
import time
import keyboard
import sys
import select
import traceback


import DPP_comm as dpp #import DPP_GetRawEvents, Dfilter, DPP_Config, DPP_GetBG, DPP_GetTao, DPP_Set, DPP_GetRaw, DPP_GetEvent, DPP_Unpack_Board, FLAGS, ser
from DPP_comm import ser, FLAGS, DPP_PreConfig

dpp.DPP_Stop()  # Added this line to handle the fact that sometimes the board is not stopped properly and, in that case, doesn't initialize DPP_GetBG() correctly.
time.sleep(0.5)
if (ser.is_open == False):  ser.open()
ser.reset_input_buffer()
ser.reset_output_buffer()
time.sleep(0.5) #Give the board a moment to settle.
# dpp.DPP_GetTao() 
dpp.DPP_GetBG()

DPP_PreConfig('pmt')


channel = FLAGS["FLAG_START_RAW_CHA"] # _CHB # _All
max_duration = 1000


# start aquisition
if (ser.is_open == False):  ser.open()
ser.write(bytearray([channel, 0x00, (max_duration>>8)&0xFF, (max_duration)&0xFF]))
timeold = time.time()


# Helper function to check terminal input
def is_key_pressed():
    """Non-blocking check if a key is pressed."""
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    return bool(dr)

print("Press 'ESC' and Enter to stop acquisition...")

try:
    hist2ddata = []

        # untill ESC is pressed
    while True:
        # Check for ESC key press
        if is_key_pressed():
            key = sys.stdin.read(1)
            if key == '\x1b':  # ESC key ASCII code
                print("ESC pressed. Stopping acquisition.")
                break

        buf = ser.read(20 + 1024*4)
        timenew = time.time()
        dat16 = np.frombuffer(buf[0:20], np.uint16) #.byteswap()
        total_events = dat16[8]
        total_rejected = dat16[9]
        flux = total_events/(timenew-timeold)
        timeold = timenew

        events = np.frombuffer(buf[20:], np.uint32) #.byteswap()
        print(total_events)
        print(events)

        # events_parced = []
        test_event = [0,0,0,0]
        number_of_registered_events = 0
        for i in range(total_events):
            # timestamp 20bit, overflows at 1s
            ts = (events[i][0]<<12)+(events[i][1]<<4)+(events[i][2]>>4)
            ch = (events[i][2]>>3)&1
            h1 = ((events[i][2] & 0b111)<<8) + events[i][3]

            # events_parced.append([i,ts,ch,h1])
            hist2ddata.append(h1)
            if i == 2:
                test_event = [i,ts,ch,h1]


        # print(dat16)
        board = dpp.DPP_Unpack_Board(buf[0:16])
        print(board)
        b = "Tot: " + str(total_events) + " Filtered: " + str(total_rejected) + " Flux: " + str(round(flux,3)) + "\t"
        b+= "T " + str(board["BME_Temperature"]/100) + "C, P " + str(board["BME_Pressure"]) + "mB, RH " + str(board["BME_Humidity"])
        
        # print(b)
        b += "\t Event[2]>>TS: " + str(board["BASE_timestamp_ms"] + (c>>12)/1000)  + " Channel: " + str((c>>11)&1) + " Height: " +  str(c&0x7FF) 
        # c = events[int(total_events/2)+2]
        # b += "\t Event[-2]>>TS: " + str(board["BASE_timestamp_ms"] + (c>>12)/1000)  + " Channel: " + str((c>>11)&1) + " Height: " +  str(c&0x7FF) + "\n"

        # print("{:032b}".format(c),"{:020b}".format(c>>12),"{:01b}".format((c>>11)&1), "{:011b}".format(c&0x7FF)  )
        print(b)        

    dpp.DPP_Stop()


except Exception as e:
    print(f"Error during acquisition: {e}")
    print("Exception occurred at:")
    traceback.print_exc()

finally:
    # Cleanup
    ser.close()
    print("Stopped and cleaned up.")
