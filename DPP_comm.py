import serial
import serial.tools.list_ports
import numpy as np
import keyboard
import time

# last update: 29/07/2024
Serial_ports = serial.tools.list_ports.comports()
VID = "0483"
PID = "5740"

def WR(arr):
    ser.write(arr)







# need to be synced to dpp.h
FLAGS = {
    "FLAG_SETTINGS"			    :  15,	#// 0000 1111
	"FLAG_SAVE_SETTINGS"		:   8,	#// 0000 1000
	"FLAG_GET_TAO"			    :  10,	#// 0000 1010
	"FLAG_GET_BG"				:  12,	#// 0000 1100

	"FLAG_START_WAVEFORM"		:  16,	#// 0001 0000
	"FLAG_START_WAVEFORM_CHA"   :  18,	#// 0001 0010
	"FLAG_START_WAVEFORM_CHB"   :  20,	#// 0001 0100
	"FLAG_START_WAVEFORM_ALL"   :  22,	#// 0001 0110

	"FLAG_START_SINGLE" 		:  32,	#// 0010 0000
	"FLAG_START_REPEATED"		:  33,	#// 0010 0001

	"FLAG_START_RAW_CHA" 		:  66,	#// 0100 0010
	"FLAG_START_RAW_CHB" 		:  68,	#// 0100 0100
	"FLAG_START_RAW_ALL" 		:  70,	#// 0100 0110

	"FLAG_START_RAWeX_CHA"    	:  67,	#// 0100 0011
	"FLAG_START_RAWeX_CHB" 	    :  69,	#// 0100 0101
	"FLAG_START_RAWeX_ALL" 	    :  71,	#// 0100 0111

	"FLAG_START_RAW2_CHA" 	    :  74,	#// 0100 1010
	"FLAG_START_RAW2_CHB" 	    :  76,	#// 0100 1100
	"FLAG_START_RAW2_ALL" 	    :  78,	#// 0100 1110

	"FLAG_START_DBG_CHA" 		: 130,	#// 1000 0010
	"FLAG_START_DBG_CHB" 		: 132,	#// 1000 0100
	# "FLAG_START_DBG_ALL" 		: 134,	#// 1000 0110

	"FLAG_START_DBG_CHA_GOOD"	: 138,	#// 1000 1010
	"FLAG_START_DBG_CHB_GOOD"	: 140,	#// 1000 1100
	# "FLAG_START_DBG_ALL_GOOD"	: 142,	#// 1000 1110

	"FLAG_STOP"	 			    : 7,	#// 0000 0111
}

PORTS = {
	"USB" : 0,	
	"SPI" : 1,	
	"UART": 2	    
}

# for cm in ['/dev/ttyS0','COM6','COM13','COM23','COM24','COM25','COM26','COM27','COM28','COM26']:
    # try:
        # ser = serial.Serial(cm, 115200, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)
        # print('Connected ', cm)
        # break
    # except:
        # # ports = serial.tools.list_ports.comports()
        # # for port, desc, hwid in sorted(ports):
        # #     print("{}: {} [{}]".format(port, desc, hwid))
        # # print('NOT CONNECTED ',cm)
        # # print(str(e))
        # ser = None
if not Serial_ports:
    print("No available ports found.")
    ser = None
else:
 for port in Serial_ports:
  if f"VID:PID={VID}:{PID}" in port.hwid:
    try:
        ser = serial.Serial(port.device, 115200, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)
        print('Connected ', port)
        break
    except:
        # ports = serial.tools.list_ports.comports()
        # for port, desc, hwid in sorted(ports):
        #     print("{}: {} [{}]".format(port, desc, hwid))
        ser = None
  else:
     print(f"Skipping port {port.device}: VID/PID not matching: {port.hwid} ")


def DPP_PreConfig(type = 'sipm'):
    filter = {}
    port = PORTS["USB"]

    filterSiPM = Dfilter(peak_threshold=40, peak_height_min=200, peak_rising_time_max=8, p2p_distance_min=15, use_syncrounous_events=False)
    filterPMT  = Dfilter(peak_threshold=100, peak_height_min=50, peak_rising_time_max=7, p2p_distance_min=15, use_syncrounous_events=False)
    filterSDD  = Dfilter(peak_threshold=25, peak_height_min=100, peak_rising_time_max=5, p2p_distance_min=7)

    filterGAGG = Dfilter(peak_threshold=25, peak_height_min=30, peak_rising_time_max=8, p2p_distance_min=15)

    # CsI scintillator or default for SiPM
    if (type.lower() == 'sipm_csi') or (type.lower() == 'sipm'):
        filter = filterSiPM
        DPP_Config (duration=100,tao=12, filter = filter, DAC_level=222, GND_offset=220, ramp_direction=1, output_port=port, peak2_full_evo=False) #, use_manual_background=False, FIR_BG=[-1485,-1500], adjust_background=True)
        
    
    # plastic scintillator 
    if (type.lower() == 'sipm_plastic'):
        filter = filterSiPM
        DPP_Config (duration=100,tao=8.4, filter = filterSiPM, DAC_level=230, GND_offset=220, ramp_direction=1, output_port=port)

    if (type.lower() == 'pmt'):
        filter = filterPMT
        DPP_Config (duration=100,tao=6.4  , filter = filterPMT, DAC_level=150, GND_offset=30, ramp_direction=0, output_port=port, adjust_background=False, use_manual_background=True, FIR_BG=[323,323])

    if (type.lower() == 'test'):
        filter = filterSiPM
        DPP_Config (duration=100,tao=11  , filter = filterSDD, DAC_level=150, GND_offset=30, ramp_direction=0, output_port=port)
        
    # CsI scintillator or default for SiPM
    if (type.lower() == 'sipm_single'):
        filterSiPM["peak_threshold"] = 100
        DPP_Config (duration=100,tao=11, filter = filterSiPM, DAC_level=240, GND_offset=220, ramp_direction=1, output_port=port)

    # GAGG scintillator 
    if (type.lower() == 'sipm_gagg'):
        filter = filterGAGG
        DPP_Config (duration=100, tao=8.5, filter = filterGAGG, DAC_level=218, GND_offset=220, ramp_direction=1, output_port=port) #, use_manual_background=True, FIR_BG=[-1480,-1480], adjust_background=False)


    return [filter["peak_threshold"],filter["peak_height_min"]]





def Dfilter(peak_threshold = 50, peak_height_min = 00, bg_duration_min = 25, scan_step = 4, peak_rising_time_max = 10, p2p_distance_min = 15,
               use_syncrounous_events = False, offset_correct_mode = 0, offset_min = 0, offset_max = 10000):
    filter = {
        "peak_threshold": peak_threshold,
        "peak_height_min": peak_height_min,
        "bg_duration_min": bg_duration_min,
        "scan_step": scan_step,
        "peak_rising_time_max": peak_rising_time_max,
        "p2p_distance_min": p2p_distance_min,
        "use_syncrounous_events": ord("0") + use_syncrounous_events ,
        "offset_correct_mode": ord("0") + offset_correct_mode ,
        "offset_min": offset_min,
        "offset_max": offset_max
    }
    if filter["peak_height_min"]<filter["peak_threshold"]: 
        filter["peak_height_min"]=filter["peak_threshold"]
    return filter

def Dfilter_Set(peak_threshold = 0, peak_height_min = 00,  bg_duration_min = 0, scan_step = 0, peak_rising_time_max = 0, p2p_distance_min = 0,
               use_syncrounous_events = -ord("0"), offset_correct_mode = -ord("0"), offset_min = 0, offset_max = 0):
               
    filter = {
        "peak_threshold": peak_threshold,
        "peak_height_min": peak_height_min,
        "bg_duration_min": bg_duration_min,
        "scan_step": scan_step,
        "peak_rising_time_max": peak_rising_time_max,
        "p2p_distance_min": p2p_distance_min,
        "use_syncrounous_events": ord("0") + use_syncrounous_events ,
        "offset_correct_mode": ord("0") + offset_correct_mode ,
        "offset_min": offset_min,
        "offset_max": offset_max
    }
    if filter["peak_height_min"]<filter["peak_threshold"]: 
        filter["peak_height_min"]=filter["peak_threshold"]

    return filter


def DPP_Config( duration=100, run_type = 0, use_dummy = False, filter = Dfilter(peak_threshold = 40),GND_offset = 31,
                DAC_level = 127, ramp_direction = 0, generate_reset = False, use_manual_background = False,
                adjust_background = True, FIR_BG = [200,200], tao=9.0, output_port = PORTS["USB"], peak2_full_evo = False):
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

    XRsettings[30] = filter["peak_height_min"]
    
    if (ser.is_open == False):  ser.open()

    ser.write(XRsettings)
    buf = ser.read(2)
    print(buf)
    ser.close()




def DPP_Set( duration=0, run_type = 0, use_dummy = -ord("0"), filter = Dfilter_Set(),GND_offset = 0,
                DAC_level = 0, ramp_direction = 0, generate_reset = -ord("0"), use_manual_background = -ord("0"),
                adjust_background = -ord("0"), FIR_BG = [0,0], tao=0, output_port = PORTS["USB"], peak2_full_evo = -ord("0")):
    
    DPP_Config(duration=duration, run_type = run_type, use_dummy = use_dummy, filter = filter, GND_offset = GND_offset,
                DAC_level = DAC_level, ramp_direction = ramp_direction, generate_reset = generate_reset, use_manual_background =use_manual_background,
                adjust_background = adjust_background, FIR_BG =FIR_BG, tao=tao, output_port = output_port, peak2_full_evo=peak2_full_evo)
    



def DPP_GetTao( ):
    XRtao = bytearray(32)
    XRtao[0] = FLAGS["FLAG_GET_TAO"]    # Settings marker
    XRtao[1] = 0
    XRtao[2] = 0x01
    XRtao[3] = 0x40

    if (ser.is_open == False):  ser.open()
    ser.write(XRtao)

    buf = ser.read(16)
    t1 = int(buf[4:9])/1000
    t2 = int(buf[10:15])/1000

    print("Tao =", t1,",", t2)    
    ser.close
    return t1, t2

def DPP_GetBG( ):
    XRbg = bytearray(32)
    XRbg[0] = FLAGS["FLAG_GET_BG"]    # Settings marker
    XRbg[1] = 0
    XRbg[2] = 0x01
    XRbg[3] = 0x40

    if (ser.is_open == False):  ser.open()
    ser.write(XRbg)

    buf = ser.read(16)
    t1 = int(buf[4:9])
    t2 = int(buf[10:15])

    print("Backgrounds =", t1,",", t2)    
    ser.close
    return t1, t2

def DPP_Send( cmd = FLAGS["FLAG_STOP"], duration = 1000, close_port = True):
    XRsend = bytearray(32)
    XRsend[0] = cmd #FLAGS["FLAG_START_DBG_CHA_GOOD"]    # Settings marker
    XRsend[1] = 0
    XRsend[2] = (duration>>8)&0xFF
    XRsend[3] = (duration   )&0xFF
    if (ser.is_open == False):  ser.open()
    ser.write(XRsend)

    if (close_port):
        ser.close




def DPP_GetRaw():
    XRraw = bytearray(32)
    XRraw[0] = FLAGS["FLAG_RAW"]    # Settings marker
    XRraw[1] = 0
    XRraw[2] = 0x01
    XRraw[3] = 0x40

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

def DPP_GetDebugEvents(channel=FLAGS["FLAG_START_DBG_CHA_GOOD"], max_duration = 100, only_good=0):
    XRraw = bytearray(32)
    XRraw[0] = channel #FLAGS["FLAG_START_DBG_CHA_GOOD"]    # Settings marker
    XRraw[1] = 0
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


def DPP_GetWaveform(channel=FLAGS["FLAG_START_WAVEFORM"], max_duration = 100):
    XRevent = bytearray(32)
    XRevent[0] = channel
    XRevent[1] = 0
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
    XRmeas = bytearray(32)
    XRmeas[0] = FLAGS["FLAG_START_SINGLE"]    # Settings marker
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

def DPP_StartRepeatedHist(duration = 1000, sync_events = 0):
    XRmeas = bytearray(32)
    XRmeas[0] = FLAGS["FLAG_START_REPEATED"]    # Settings marker
    XRmeas[1] = 0 #ord("0") + int(sync_events)
    XRmeas[2] = (duration>>8)&0xFF
    XRmeas[3] = (duration   )&0xFF
    if (ser.is_open == False):  ser.open()
    ser.write(XRmeas)

def DPP_GetRepeatedHist():
    buf = ser.read(4096*2)
    board, stat_1 = DPP_Unpack_Hist(buf[0:96])
    board, stat_2 = DPP_Unpack_Hist(buf[4096+0:4096+96])

    hist_1 = np.frombuffer(buf[96:4096], np.int16) #.byteswap()
    hist_2 = np.frombuffer(buf[4096+96:4096+4096], np.int16) #.byteswap()

    return hist_1, hist_2, stat_1, stat_2, board

def DPP_StopRepeatedHist():
    XRmeas = bytearray(32)
    XRmeas[0] = FLAGS["FLAG_STOP"]    # Settings marker
    XRmeas[1] = 0 #ord("0") + int(sync_events)
    XRmeas[2] = 0
    XRmeas[3] = 0
    if (ser.is_open == False):  ser.open()
    ser.write(XRmeas)
    ser.close


def DPP_Unpack_Board(buf):

    dat08 = np.frombuffer(buf, np.uint8) #.byteswap()
    dat16 = np.frombuffer(buf, np.int16) #.byteswap()
    dat32 = np.frombuffer(buf, np.uint32) #.byteswap()
    board_state = {
        "BASE_timestamp_ms" : dat32[0],
        "ADC_R_thermistor"  : dat16[2],
        "ADC_HV_value"      : dat16[3],
        "BME_Pressure"      : dat16[4],
        "BME_Temperature"   : dat16[5],
        "BME_Humidity"      : dat08[12],
        "dac_hv_setting"    : dat08[13],
        "cpu_load_ch1"      : dat08[14],
        "cpu_load_ch2"      : dat08[15]        
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





def DPP_GetRawEvents(duration = 1000, channel = FLAGS["FLAG_START_RAW_CHA"] ):
    XRmeas = bytearray(32)
    XRmeas[0] = channel

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
    if (channel == 0): XRmeas[0] = FLAGS["FLAG_START_RAW_CHA"]
    if (channel == 1): XRmeas[0] = FLAGS["FLAG_START_RAW_CHB"]
    if (channel == 2): XRmeas[0] = FLAGS["FLAG_START_RAW_ALL"]

    XRmeas[1] = 0 #ord("0") + int(sync_events)
    XRmeas[2] = (duration>>8)&0xFF
    XRmeas[3] = (duration   )&0xFF

    if (ser.is_open == False):  ser.open()
    ser.write(XRmeas)
    # ser.close

def DPP_Start_RawEx(duration = 1000, channel = 0):
    XRmeas = bytearray(32)
    if (channel == 0): XRmeas[0] = FLAGS["FLAG_START_RAWeX_CHA"]
    if (channel == 1): XRmeas[0] = FLAGS["FLAG_START_RAWeX_CHB"]
    if (channel == 2): XRmeas[0] = FLAGS["FLAG_START_RAWeX_ALL"]

    XRmeas[1] = 0 #ord("0") + int(sync_events)
    XRmeas[2] = (duration>>8)&0xFF
    XRmeas[3] = (duration   )&0xFF

    if (ser.is_open == False):  ser.open()
    ser.write(XRmeas)
    # ser.close

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
# DPP_Stop()

# DPP_PreConfig('sipm')

# DPP_Save()
