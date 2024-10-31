import numpy as np
import matplotlib.pyplot as plt
import time
import keyboard

import DPP_comm as dpp #import DPP_GetRawEvents, Dfilter, DPP_Config, DPP_GetBG, DPP_GetTao, DPP_Set, DPP_GetRaw, DPP_GetEvent, DPP_Unpack_Board, FLAGS, ser
from DPP_comm import ser, FLAGS

# dpp.DPP_GetTao()
dpp.DPP_GetBG()

channel = FLAGS["FLAG_START_RAW_CH1"] # _CH2 # _All
max_duration = 2000

_filterSiPM = dpp.Dfilter(peak_threshold=100, peak_rising_time_max=10, p2p_distance_min=15)
_filterPMT = dpp.Dfilter(peak_threshold=50, peak_rising_time_max=5, p2p_distance_min=7)

# SIPM main config:
# DPP_Config (duration=100,tao=8.5, filter = _filterSiPM, DAC_level=230, GND_offset=200, ramp_direction=1)

# PMT main config
dpp.DPP_Config (duration=100,tao=5.8, filter = _filterPMT, DAC_level=150, GND_offset=30, ramp_direction=0)

# dummy SDD config
# DPP_Config (FIR_BG_manual_A=170, FIR_BG_manual_B=170, use_manual_background=1, dynamic_background=0,duration=100, tao=11, peak_threshold=20, ramp_direction=0, use_dummy=1)


# start aquisition
if (ser.is_open == False):  ser.open()
ser.write(bytearray([channel, 0x00, (max_duration>>8)&0xFF, (max_duration)&0xFF]))
timeold = time.time()



# untill ESC is pressed
while not keyboard.is_pressed('esc'):

    buf = ser.read(20 + 1024*4)
    timenew = time.time()
    dat16 = np.frombuffer(buf[0:20], np.uint16) #.byteswap()
    total_events = dat16[8]
    total_rejected = dat16[9]
    flux = total_events/(timenew-timeold)
    timeold = timenew

    events = np.frombuffer(buf[20:], np.uint32) #.byteswap()
    
    # print(dat16)
    board = dpp.DPP_Unpack_Board(buf[0:16])
    print(board)
    b = "Tot: " + str(total_events) + " Load: " + str(total_rejected/5) + " Flux: " + str(round(flux,3)) + "\t"
    b+= "T " + str(board["BME_Temperature"]/100) + "C, P " + str(board["BME_Pressure"]) + "mB, RH " + str(board["BME_Humidity"])
    
    # print(b)
    c = events[2]
    b += "\t Event[2]>>TS: " + str(board["BASE_timestamp_ms"] + (c>>12)/1000)  + " Channel: " + str((c>>11)&1) + " Height: " +  str(c&0x7FF) 
    # c = events[int(total_events/2)+2]
    # b += "\t Event[-2]>>TS: " + str(board["BASE_timestamp_ms"] + (c>>12)/1000)  + " Channel: " + str((c>>11)&1) + " Height: " +  str(c&0x7FF) + "\n"

    # print("{:032b}".format(c),"{:020b}".format(c>>12),"{:01b}".format((c>>11)&1), "{:011b}".format(c&0x7FF)  )
    print(b)        

XRmeas = bytearray([FLAGS["FLAG_STOP"], 0x00, 0x01, 0x40])
ser.write(XRmeas)
ser.close





print("Done")
