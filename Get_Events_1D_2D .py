import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors 
import time
import keyboard
from datetime import datetime
from scipy.optimize import curve_fit
import math

# import DPP_comm as dpp #import DPP_GetRawEvents, Dfilter, DPP_Config, DPP_GetBG, DPP_GetTao, DPP_Set, DPP_GetRaw, DPP_GetEvent, DPP_Unpack_Board, FLAGS, ser
from DPP_comm import ser, FLAGS, DPP_PreConfig, DPP_Send, DPP_GetBG, DPP_Unpack_Board, DPP_Stop

# dpp.DPP_GetTao() 
DPP_GetBG()

DPP_PreConfig('sipm')
# filtered_region = DPP_PreConfig('sipm')
# filtered_region = DPP_PreConfig('pmt')


def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))



channel = FLAGS["FLAG_START_RAW2_CHA"] # _RAW2_ # _RAW_# _CHB # _All
# channel = FLAGS["FLAG_START_RAW_CHA"] # _CHB # _All




acquisition_tot = 200*1000
max_duration = 5000
coincidence_window = 2  # max peaks offset, points.

if (channel == FLAGS["FLAG_START_RAW_CHA"]) or (channel == FLAGS["FLAG_START_RAW_CHB"]) or (channel == FLAGS["FLAG_START_RAW_ALL"]):
    use_both = 0
else:
    use_both = 1



# Get the current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Construct the filename with the current time and date
filenameEvents = f"{current_time}_Events.txt"
filenameHist = f"{current_time}_hist.txt"
filenameFULL = f"{current_time}_Full.txt"



# start aquisition
DPP_Send(channel, max_duration, False)

# if (ser.is_open == False):  ser.open()
# ser.write(bytearray([channel, 0x00, (max_duration>>8)&0xFF, (max_duration)&0xFF]))
timeold = time.time()

# test_event0 = [0,0,0,0,0,0]

hist2ddata_x = []
hist2ddata_y = []
events_parced_tot = []
tot_registered_events = 0

# untill ESC is pressed
while not keyboard.is_pressed('esc') and (tot_registered_events<acquisition_tot):

    buf = ser.read(20 + 1024*4)
    timenew = time.time()
    dat16 = np.frombuffer(buf[0:20], np.uint16) #.byteswap()
    total_events = dat16[8]
    total_rejected = dat16[9]
    flux = total_events/(timenew-timeold)
    timeold = timenew
    
    board = DPP_Unpack_Board(buf[0:16])

    if (use_both):
        events = np.frombuffer(buf[20:20+total_events*6], dtype = np.uint8).reshape( total_events, 6)
    else:
        events = np.frombuffer(buf[20:20+total_events*4], dtype = np.uint8).reshape( total_events, 4)

    test_event = [0,0,0,0,0,0,0]
    number_of_registered_events = 0
    of_offset = 0
    ts_old = 0

    for i in range(total_events):
        # timestamp 20bit, overflows at 1s
        bt03 = (events[i][0] << 24) + (events[i][1] << 16) + (events[i][2] << 8) + (events[i][3])
        if (use_both):
            bt45 = (events[i][4] <<  8) + (events[i][5]      )
        else:
            bt45 = 0

        ts_ms =  (bt03>>22) & 0b1111111111
        ts_us =  (bt03>>12) & 0b1111111111
        ch    =  (bt03>>11) & 0b1
        h1    =  (bt03    ) & 0b11111111111
        dp    =  (bt45>>11) & 0b11111 #(np.int8(events[i][4])>>3)
        h2    =  (bt45    ) & 0b11111111111
        
        # # Correct TS overflow
        if (ts_ms<ts_old):
            of_offset += 1000
        ts_old = ts_ms

        ts_ms += of_offset
        ts_ms += board["BASE_timestamp_ms"] 

        if i == 2:
            test_event = [i, ts_ms, ts_us,ch,h1,dp,h2]

        if (dp>=-coincidence_window) and (dp<=coincidence_window)  or (use_both == 0):# :and (abs(h2-h1)<200)
            events_parced_tot.append([tot_registered_events, ts_ms, ts_us, ch, h1, dp, h2])
            tot_registered_events +=1
            number_of_registered_events += 1
            hist2ddata_x.append(h1)
            hist2ddata_y.append(h2)
        
    

    
    b = "Tot: " + str(total_events) + " (" +str(total_events - number_of_registered_events) + ") Load: " + str(board["cpu_load_ch1"]) + '/'+ str(board["cpu_load_ch2"]) + " Flux: " + str(round(flux,3)) 
    b+= "\t T=" + str(board["BME_Temperature"]/100) + "C, P=" + str(board["BME_Pressure"]) + "mB, RH=" + str(board["BME_Humidity"]) + "% "
    
    if total_events>2:
        c = test_event
        b += "\t Event[2]>>TS:" + "%8d"%(c[1])  +"ms: " + "%5d"%(c[2])  + "us, Channel: " + str(c[3]) + " Heights: " +  "%4d"%(c[4])  + "/" + "%4d"%(c[6]) + " DIF:"  +  "%3d"%(c[5])
    else:
        b += "\t [NO EVENTS]"

    print('\t',len(hist2ddata_x),'\t', b, end='\r')        

DPP_Stop()


# merge events to 1d histogram
if (use_both):
    dat = np.multiply(np.add(hist2ddata_x, hist2ddata_y), 0.5).astype(int)
else:
    dat = np.copy(hist2ddata_x).astype(int)
frq, edges = np.histogram(dat, range(0, 2000, 1))

np.savetxt(filenameEvents, dat, delimiter=',', fmt='%d')   # save all the events
np.savetxt(filenameHist, frq, fmt='%d')   # x,y,z equal sized 1D arrays
np.savetxt(filenameFULL, events_parced_tot, fmt='%d')   # x,y,z equal sized 1D arrays

# Get a gaussian fit for the peak
try:
    peak_C = np.argmax(frq)
    peak_w = 200
    popt0, pcov0 = curve_fit(Gauss, edges[peak_C-peak_w:peak_C+peak_w], frq[peak_C-peak_w:peak_C+peak_w],  p0=[np.max(frq)/2., peak_C, 20.])
    HIST_FIT = Gauss(edges[:-1], popt0[0],popt0[1],popt0[2])
    print("\nFit:", popt0, round(popt0[2]/popt0[1]*100,2), peak_C)
except (RuntimeError, ValueError):
    HIST_FIT = [0]*len(edges[:-1])
    print('\nFit failed')

fig = plt.figure(figsize=(10, 5))
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1)
plot_max =  pow(10,math.floor(math.log10(max(frq))+1))

ax1.plot(edges[:-1], frq,ms=4)
ax1.plot(edges[:-1], HIST_FIT,'-')
ax1.set_yscale('log')
ax1.set_xlim(0, 2000)
ax1.set_ylim(1, plot_max )  # Set only the lower limit
ax1.get_xaxis().set_visible(False)
# ax1.set_xticks([])

ax2 = plt.subplot2grid((2, 2), (1, 0), sharex=ax1)
ax2.plot(edges[:-1], frq,ms=4)
ax2.plot(edges[:-1], HIST_FIT,'-')
ax2.set_ylim(bottom=0)  # Set only the lower limit
# ax2.set_xlabel('Height (val)')

ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
ax3.hist2d(hist2ddata_x, hist2ddata_y, bins = 100,norm = colors.LogNorm()) #,norm = colors.LogNorm(), cmap ="Greens") 
ax3.set_title('X-product')

# plt.subplots_adjust(hspace=0.0)

plt.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0.1)  # Specifically remove vertical spacing

plt.show()

print("\nDONE!")