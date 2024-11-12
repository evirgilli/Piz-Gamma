import numpy as np
import matplotlib.pyplot as plt
import time
import keyboard

import DPP_comm as dpp #import DPP_GetRawEvents, Dfilter, DPP_Config, DPP_GetBG, DPP_GetTao, DPP_Set, DPP_GetRaw, DPP_GetEvent, DPP_Unpack_Board, FLAGS, ser
from DPP_comm import ser, FLAGS, DPP_PreConfig

# dpp.DPP_GetTao() 
dpp.DPP_GetBG()

DPP_PreConfig('sipm')


channel = FLAGS["FLAG_START_RAW_CHA"] # _CHB # _All
max_duration = 1000


# start aquisition
if (ser.is_open == False):  ser.open()
ser.write(bytearray([channel, 0x00, (max_duration>>8)&0xFF, (max_duration)&0xFF]))
timeold = time.time()

hist2ddata = []

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






print("Done")
