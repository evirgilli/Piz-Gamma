import serial
import serial.tools.list_ports
import numpy as np
import keyboard
import time
Serial_ports = serial.tools.list_ports.comports()

# need to be synced to dpp.h
FLAGS = {
	"FLAG_SETTINGS"		 : 81,
    "FLAG_SAVE_SETTINGS" : 82,
	"FLAG_GET_TAO"		 : 90,
	"FLAG_GET_BG"		 : 91,

	"FLAG_START_SINGLE"  : 100,
	"FLAG_START_REPEATED": 101,

	"FLAG_START_RAW_ALL" : 110,
	"FLAG_START_RAW_CH1" : 111,
	"FLAG_START_RAW_CH2" : 112,
	
    "FLAG_START_DBG_ALL_GOOD" : 113,
	"FLAG_START_DBG_CH1_GOOD" : 114,
	"FLAG_START_DBG_CH2_GOOD" : 115,

    "FLAG_START_DBG_ALL" : 116,
	"FLAG_START_DBG_CH1" : 117,
	"FLAG_START_DBG_CH2" : 118,

	"FLAG_RAW"	 		 : 120,
	"FLAG_RAW_SYNC_CH1"  : 121,
	"FLAG_RAW_SYNC_CH2"  : 122,
	"FLAG_RAW_SYNC_BOTH"  : 123,

	"FLAG_STOP"	 		 : 127
}
PORTS = {
	"USB" : 0,	
	"SPI" : 1,	
	"UART": 2	    
}

#for cm in ['COM5','COM23','COM24','COM25','COM26','COM27','COM28','COM26']:
if not Serial_ports:
    print("No available ports found.")
    ser = None
else:
 for port in Serial_ports:
    try:
        ser = serial.Serial(port.device, 115200, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)
        print('Connected ', port)
        break
    except:
        # ports = serial.tools.list_ports.comports()
        # for port, desc, hwid in sorted(ports):
        #     print("{}: {} [{}]".format(port, desc, hwid))
        ser = None



def Dfilter(peak_threshold = 50, bg_duration_min = 25, scan_step = 4, peak_rising_time_max = 10, p2p_distance_min = 15,
               use_syncrounous_events = False, offset_correct_mode = 0, offset_min = 0, offset_max = 10000):
    filter = {
        "peak_threshold": peak_threshold,
        "bg_duration_min": bg_duration_min,
        "scan_step": scan_step,
        "peak_rising_time_max": peak_rising_time_max,
        "p2p_distance_min": p2p_distance_min,
        "use_syncrounous_events": ord("0") + use_syncrounous_events ,
        "offset_correct_mode": ord("0") + offset_correct_mode ,
        "offset_min": offset_min,
        "offset_max": offset_max
    }
    return filter
def Dfilter_Set(peak_threshold = 0, bg_duration_min = 0, scan_step = 0, peak_rising_time_max = 0, p2p_distance_min = 0,
               use_syncrounous_events = -ord("0"), offset_correct_mode = -ord("0"), offset_min = 0, offset_max = 0):
               
    filter = {
        "peak_threshold": peak_threshold,
        "bg_duration_min": bg_duration_min,
        "scan_step": scan_step,
        "peak_rising_time_max": peak_rising_time_max,
        "p2p_distance_min": p2p_distance_min,
        "use_syncrounous_events": ord("0") + use_syncrounous_events ,
        "offset_correct_mode": ord("0") + offset_correct_mode ,
        "offset_min": offset_min,
        "offset_max": offset_max
    }
    return filter


def DPP_Config( duration=100, run_type = 0, use_dummy = False, filter = Dfilter(peak_threshold = 40),GND_offset = 31,
                DAC_level = 127, ramp_direction = 0, generate_reset = False, use_manual_background = False,
                adjust_background = True, FIR_BG = [200,200], tao=9.0, output_port = PORTS["USB"]):
    XRsettings = bytearray(32)

    XRsettings[0] = FLAGS["FLAG_SETTINGS"]    # Settings marker
    XRsettings[1] = run_type
    XRsettings[2] = (duration>>8)&0xFF
    XRsettings[3] = (duration   )&0xFF
    XRsettings[4] = ord("0")  + int(use_dummy == True)

    XRsettings[5] = filter["peak_threshold"]
    XRsettings[6] = filter["bg_duration_min"]
    XRsettings[7] = filter["scan_step"]
    XRsettings[8] = filter["peak_rising_time_max"]
    XRsettings[9] = filter["p2p_distance_min"]
    XRsettings[10] = filter["use_syncrounous_events"]
    XRsettings[11] = filter["offset_correct_mode"]
    XRsettings[12] = (filter["offset_min"]>>8)&0xFF
    XRsettings[13] = (filter["offset_min"]   )&0xFF
    XRsettings[14] = (filter["offset_max"]>>8)&0xFF
    XRsettings[15] = (filter["offset_max"]   )&0xFF

    XRsettings[16] = GND_offset
    XRsettings[17] = DAC_level
    XRsettings[18] = ord("0")  + int(ramp_direction)
    XRsettings[19] = 0 #is_running
    XRsettings[20] = ord("0")  + int(generate_reset == True)
    XRsettings[21] = 0 #gain
    XRsettings[22] = ord("0")  + int(use_manual_background == True)
    XRsettings[23] = ord("0")  + int(adjust_background == True)

    XRsettings[24] = (FIR_BG[0]>>8)&0xFF
    XRsettings[25] = (FIR_BG[0]   )&0xFF
    
    XRsettings[26] = (FIR_BG[1]>>8)&0xFF
    XRsettings[27] = (FIR_BG[1]   )&0xFF
    XRsettings[28] = int(tao * 10)
    XRsettings[29] = ord("0")  + int(output_port)

    if (ser.is_open == False):  ser.open()

    ser.write(XRsettings)
    buf = ser.read(2)
    print(buf)
    ser.close()




def DPP_Set( duration=0, run_type = 0, use_dummy = -ord("0"), filter = Dfilter_Set(),GND_offset = 0,
                DAC_level = 0, ramp_direction = 0, generate_reset = -ord("0"), use_manual_background = -ord("0"),
                adjust_background = -ord("0"), FIR_BG = [0,0], tao=0, output_port = PORTS["USB"]):
    
    DPP_Config(duration=duration, run_type = run_type, use_dummy = use_dummy, filter = filter, GND_offset = GND_offset,
                DAC_level = DAC_level, ramp_direction = ramp_direction, generate_reset = generate_reset, use_manual_background =use_manual_background,
                adjust_background = adjust_background, FIR_BG =FIR_BG, tao=tao, output_port = output_port)
    



def DPP_GetTao( ):
    XRtao = bytearray([FLAGS["FLAG_GET_TAO"] , 0x00, 0x01, 0x40])

    if (ser.is_open == False):  ser.open()
    ser.write(XRtao)

    buf = ser.read(16)
    t1 = int(buf[4:9])/1000
    t2 = int(buf[10:15])/1000

    print("Tao =", t1,",", t2)    
    ser.close
    return t1, t2

def DPP_GetBG( ):
    XRbg = bytearray([FLAGS["FLAG_GET_BG"] , 0x00, 0x01, 0x40])

    if (ser.is_open == False):  ser.open()
    ser.write(XRbg)

    buf = ser.read(16)
    t1 = int(buf[4:9])
    t2 = int(buf[10:15])

    print("Backgrounds =", t1,",", t2)    
    ser.close
    return t1, t2

def DPP_GetRaw():
    XRraw = bytearray([FLAGS["FLAG_RAW"], 0x00, 0x01, 0x40])
    if (ser.is_open == False):  ser.open()

    ser.write(XRraw)
    buf = ser.read(8000*5)
    dat = np.frombuffer(buf, np.int16) #.byteswap()
    ser.close

    x = np.arange(1000,step=0.25)
    chA = dat[0000:4000] #.copy()
    chA_Filtered = dat[4000:8000] #.copy()
    chB = dat[8000:12000] #.copy()
    chB_Filtered = dat[12000:16000] #.copy()
    chC = dat[16000:20000] #.copy()

    return x, chA, chA_Filtered, chB, chB_Filtered , chC

def DPP_GetDebugEvents(channel=0, max_duration = 100, only_good=0):
    XRraw = bytearray([FLAGS["FLAG_START_DBG_CH1_GOOD"], 0x00, 0x01, 0x40])
    if (channel == 1): 
        if (only_good == 1): 
            XRraw[0] = FLAGS["FLAG_START_DBG_CH2_GOOD"]
        else:
            XRraw[0] = FLAGS["FLAG_START_DBG_CH2"]
    else:
        if (only_good == 1): 
            XRraw[0] = FLAGS["FLAG_START_DBG_CH1_GOOD"]
        else:
            XRraw[0] = FLAGS["FLAG_START_DBG_CH1"]

    XRraw[2] = (max_duration>>8)&0xFF
    XRraw[3] = (max_duration   )&0xFF

    if (ser.is_open == False):  ser.open()

    ser.write(XRraw)
    buf = ser.read(20+4000*4)
    ser.close
    dat16 = np.frombuffer(buf[20:], np.int16) #.byteswap()
    board = DPP_Unpack_Board(buf[0:16])

    # dat16_cpy = dat16.copy()
    events = []
    raw_dat = dat16.reshape((125,2,32))
    ch_a = raw_dat[:,0,:].flatten()
    ch_b = raw_dat[:,1,:].flatten()

    x = np.arange(125*32)
    for i in range (125):
        pos1 = raw_dat[i,0,0]
        h1 = raw_dat[i,0,1]
        e1 = 0
        ofst1 = raw_dat[i,0,2]
        bg1 = raw_dat[i,0,3]
        peak1 = raw_dat[i,0,4:32]
        if h1<10:
            e1 = h1
            h1 = 0


        pos2 = raw_dat[i,1,0]
        h2 = raw_dat[i,1,1]
        e2 = 0
        ofst2 = raw_dat[i,1,2]
        bg2 = raw_dat[i,1,3]
        peak2 = raw_dat[i,1,4:32]
        if h2<10:
            e2 = h2
            h2 = 0

        ch_a[32*i] = 0
        ch_a[32*i+1] = 0
        ch_a[32*i+2] = 0
        ch_a[32*i+3] = 0

        ch_b[32*i] = 0
        ch_b[32*i+1] = 0
        ch_b[32*i+2] = 0
        ch_b[32*i+3] = 0

        events.append ( ((pos1, h1, e1, ofst1, bg1, peak1), (pos2, h2, e2, ofst2, bg2, peak2)))

    return x,ch_a,ch_b,events, board


def DPP_GetEvent(channel=0, max_duration = 100):
    XRevent = bytearray([0x00, 0x00, 0x01, 0x40])
    if (channel == 0): 
        XRevent[0] = FLAGS["FLAG_RAW"]
    if (channel == 1): 
        XRevent[0] = FLAGS["FLAG_RAW_SYNC_CH1"]
    if (channel == 2): 
        XRevent[0] = FLAGS["FLAG_RAW_SYNC_CH2"]
    if (channel == 3): 
        XRevent[0] = FLAGS["FLAG_RAW_SYNC_BOTH"]

    XRevent[2] = (max_duration>>8)&0xFF
    XRevent[3] = (max_duration   )&0xFF

    if (ser.is_open == False):  ser.open()

    ser.write(XRevent)
    buf = ser.read(8000*5)
    dat = np.frombuffer(buf, np.int16) #.byteswap()
    ser.close

    x = np.arange(1000,step=0.25)
    chA = dat[0000:4000] #.copy()
    chA_Filtered = dat[4000:8000] #.copy()
    chB = dat[8000:12000] #.copy()
    chB_Filtered = dat[12000:16000] #.copy()
    chC = dat[16000:20000] #.copy()

    return x, chA, chA_Filtered, chB, chB_Filtered , chC

def DPP_GetHist(duration = 1000, sync_events = 0):
    XRmeas = bytearray([FLAGS["FLAG_START_SINGLE"], 0x00, 0x01, 0x40])
    XRmeas[1] = 0 #ord("0") + int(sync_events)
    XRmeas[2] = (duration>>8)&0xFF
    XRmeas[3] = (duration   )&0xFF

    if (ser.is_open == False):  ser.open()

    ser.write(XRmeas)
    buf = ser.read(4096*2)
    ser.close
    
    board, stat_1 = DPP_Unpack_Hist(buf[0:96])
    board, stat_2 = DPP_Unpack_Hist(buf[4096+0:4096+96])

    hist_1 = np.frombuffer(buf[96:4096], np.int16) #.byteswap()
    hist_2 = np.frombuffer(buf[4096+96:4096+4096], np.int16) #.byteswap()

    return hist_1, hist_2, stat_1, stat_2, board



def DPP_Unpack_Board(buf):

    dat16 = np.frombuffer(buf, np.int16) #.byteswap()
    dat32 = np.frombuffer(buf, np.uint32) #.byteswap()
    board_state = {
        "BASE_timestamp_ms" : dat32[0],
        "ADC_R_thermistor"  : dat16[2],
        "ADC_HV_value"      : dat16[3],
        "BME_Pressure"      : dat16[4],
        "BME_Temperature"   : dat16[5],
        "BME_Humidity"      : dat16[6],
        "dac_hv_setting"    : dat16[7]
    }
    return  board_state

def DPP_Unpack_Hist(buf):
    board = DPP_Unpack_Board(buf[0:16])
    dat32 = np.frombuffer(buf[16:96], np.int32) 
    stat = {
        "total_events"      : dat32[4],
        "total_resets"      : dat32[5],

        "actual_time_ms"    : dat32[6],
        "BG_Initial"        : dat32[7],
        "BG_Final"          : dat32[8],
        "tao_meas_x1000"    : dat32[9],

        "proc_time"         : dat32[10],
        "dead_time"         : dat32[11],
        "rejected"          : dat32[12:20]
    }
    return  board, stat





def DPP_GetRawEvents(duration = 1000, channel = 0):
    XRmeas = bytearray([FLAGS["FLAG_START_RAW_CH1"], 0x00, 0x01, 0x40])
    if (channel == 1):
        XRmeas[0] = FLAGS["FLAG_START_RAW_CH2"]
    if (channel == 2):
        XRmeas[0] = FLAGS["FLAG_START_RAW_ALL"]

    XRmeas[1] = 0 #ord("0") + int(sync_events)
    XRmeas[2] = (duration>>8)&0xFF
    XRmeas[3] = (duration   )&0xFF

    if (ser.is_open == False):  ser.open()
    ser.write(XRmeas)
    timeold = time.time()
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
        board = DPP_Unpack_Board(buf[0:16])
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

    
    return  board


def DPP_Start_Raw(duration = 1000, channel = 0):
    XRmeas = bytearray(32)
    if (channel == 0): XRmeas[0] = FLAGS["FLAG_START_RAW_CH1"]
    if (channel == 1): XRmeas[0] = FLAGS["FLAG_START_RAW_CH2"]
    if (channel == 2): XRmeas[0] = FLAGS["FLAG_START_RAW_ALL"]

    XRmeas[1] = 0 #ord("0") + int(sync_events)
    XRmeas[2] = (duration>>8)&0xFF
    XRmeas[3] = (duration   )&0xFF

    if (ser.is_open == False):  ser.open()
    ser.write(XRmeas)
    ser.close

def DPP_Stop():
    XRmeas = bytearray(32)
    XRmeas[0] = FLAGS["FLAG_STOP"]

    if (ser.is_open == False):  ser.open()
    ser.write(XRmeas)
    ser.close
    
def DPP_Save():
    XRmeas = bytearray(32)
    XRmeas[0] = FLAGS["FLAG_SAVE_SETTINGS"]

    if (ser.is_open == False):  ser.open()
    ser.write(XRmeas)
    ser.close


# DPP_Config(duration=50, filter= Dfilter(peak_threshold=44, offset_max=8000))

# DPP_Set(tao=7)

# DPP_GetTao()

# DPP_GetBG()
