import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors 
import matplotlib.dates as md
import matplotlib.gridspec as gridspec
import time
import datetime
import keyboard
from datetime import datetime
from scipy.optimize import curve_fit
import math
# import threading as thrd

from DPP_comm import ser, FLAGS, DPP_PreConfig, DPP_Send, DPP_GetBG, DPP_Unpack_Board, DPP_Stop

# --- Initialization and Setup ---
### Safely stop and reset the board in case of errors or improper state
DPP_Stop() 
time.sleep(0.5)

if (ser.is_open == False):  ser.open()
ser.reset_input_buffer()
ser.reset_output_buffer()
#time.sleep(0.5) #Give the board a moment to settle.
###

DPP_GetBG()

# filtered_region = DPP_PreConfig('sipm')
#filtered_region = DPP_PreConfig('sdd')
filtered_region = DPP_PreConfig('pmt')

max_duration = 2000
gain = 1460/600

channel = FLAGS["FLAG_START_RAWeX_CHA"] # _RAW2_ # _RAW_# _CHB # _All
#channel = FLAGS["FLAG_START_RAW_CHA"] # _CHB # _All


def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def on_close(event):
    global keep_updating_plot
    global pause_plot

    DPP_Stop()

    keep_updating_plot = 0
    pause_plot = 0

def press(event):
    global keep_updating_plot
    global pause_plot
    global log_plot
    global axs

    if event.key == 'escape':
        keep_updating_plot = 0
        pause_plot = 0

    if event.key == 'l':
        if (log_plot == 0):
            log_plot = 1
            axs[0].set_yscale('log')
            axs[1].set_yscale('linear')
        else:
            log_plot = 0
            axs[0].set_yscale('linear')
            axs[1].set_yscale('linear')




do_replot = 1
keep_updating_plot = 1
pause_plot = 0
log_plot = 0

plt.ion()  # turning interactive mode on

HIST_RESOLUTION=2048#4096

HISTX = np.arange(HIST_RESOLUTION)
HIST = [0]*HIST_RESOLUTION
FULX = [0]*10
FULX_H = [0]*10


fig = plt.figure()
fig.canvas.manager.set_window_title('DPP Spectra')

fig.canvas.mpl_connect('key_press_event', press)
fig.canvas.mpl_connect('close_event', on_close)

# Define a GridSpec with 2 rows and 2 columns
gs = gridspec.GridSpec( 2, 2, hspace=0, width_ratios = [1, 2] )
# Add subplots to the grid
axH1 = fig.add_subplot(gs[0, 0])  # First row, first column
axH2 = fig.add_subplot(gs[1, 0], sharex=axH1)  # Second row, first column
axF1 = fig.add_subplot(gs[0, 1])  # First row, second column
axF2 = fig.add_subplot(gs[1, 1], sharex=axF1)  # Second row, second column


graphH = axH1.plot(HISTX, HIST,ms=4,scalex=False)[0]
fr1a   = axH1.axvspan(0, filtered_region[0]-1, alpha=0.2, color='r')
fr1b   = axH1.axvspan(filtered_region[0], filtered_region[1], alpha=0.2, color='g')
txta   = axH1.text(HIST_RESOLUTION*0.75, 0, "a", ha='left')
axH1.set_xlabel("Spectrum 1min")

graphHF= axH2.plot(HISTX, HIST,ms=4,scalex=False)[0]
fr1a   = axH2.axvspan(0, filtered_region[0]-1, alpha=0.2, color='r')
fr1b   = axH2.axvspan(filtered_region[0], filtered_region[1], alpha=0.2, color='g')
txta   = axH2.text(HIST_RESOLUTION*0.75, 0, "a", ha='left')
axH2.set_xlabel("Spectrum 1h")



graphFT = axF1.plot(HIST,ms=4,scalex=False, label='Flux Total', marker='.',color = 'g')[0]
graphFH = axF1.plot(HIST,ms=4,scalex=False, label='Flux HE', marker='.',color = 'r')[0]
axF1.legend()
axF1.xaxis.set_major_formatter(md.DateFormatter('%M:%S'))

graphR  = axF2.plot(HIST,ms=4,scalex=False, label='Resistance', marker='.',color = 'b')[0]
graphHV = axF2.plot(HIST,ms=4,scalex=False, label='Bias V', marker='.',color = 'm')[0]
graphT  = axF2.plot(HIST,ms=4,scalex=False, label='Temp', marker='.',color = 'k')[0]
axF2.legend()
axF2.xaxis.set_major_formatter(md.DateFormatter('%M:%S'))



# txtb = plt.text(HIST_RESOLUTION*0.75, 0, "a", ha='left')
txtb = axF1.text(0, 0, "a", ha='left')
axF1.set_xlabel("Flux(1/s) vs Analysis time(s)")

plt.pause(0.25)

coincidence_window = 2  # max peaks offset, points.


# Get the current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Construct the filename with the current time and date
filenameEvents = f"{current_time}_Events.txt"
filenameHist = f"{current_time}_hist.txt"
filenameFULL = f"{current_time}_Full.txt"

# start aquisition
DPP_Send(channel, max_duration, False)

# ser.write(bytearray([channel, 0x00, (max_duration>>8)&0xFF, (max_duration)&0xFF]))
timeold = datetime.now()


Global_hist_array = np.zeros((60,HIST_RESOLUTION))     # set of histograms by minutes
Global_hist_index = 0                       # current index in the hist array
Global_flux_array = np.empty((0, 9))        # global collection of flux values and ambiental conditions
Global_event_array = np.empty((0,2))       # collection of events (optional)

first_event_index = 0
last_event_index = 0
last_flux_ts = 0
analysis_start_time = 0
# payload_index = 0
# untill ESC is pressed
# while not keyboard.is_pressed('esc') and (tot_registered_events<acquisition_tot):
while(keep_updating_plot>0):

    buf = ser.read(4096)
    timenew = datetime.now()
    
    # Parse incomming message
    board = DPP_Unpack_Board(buf[0:16])
    BASE_timestamp_ms = np.frombuffer(buf[0:4], np.uint32)[0]
    # when the first payload arrive, set analysis start time
    if (analysis_start_time == 0): 
        analysis_start_time = BASE_timestamp_ms
        last_flux_ts = BASE_timestamp_ms/1000

    total_events = buf[16] + (buf[17]<<8)
    total_rejected = buf[18] + (buf[19]<<8)

    # event array - 800events x 5 bytes
    events = np.frombuffer(buf[20:20+800*5], dtype = np.uint8).reshape( 800, 5)

    # fluxes arrays - 23 fluxes x 2byte + 23 fluxes x 1byte
    flux_x200ms_F= np.frombuffer(buf[4020:4020+23*2], np.uint16)
    flux_x200ms_H= np.frombuffer(buf[4020+23*2:4020+23*3], np.uint8) 

    # number of filled fluxes
    flux_total = buf[4096-7] 
    # TS of the end of the first flux window
    flux_timestamp_ms = BASE_timestamp_ms + np.frombuffer(buf[4096-6:4096-4], np.uint16)[0]
    flux_timestamp_s = flux_timestamp_ms/1000

    # for SPI - TS of the data ready event in ms
    DR_timestamp_ms = np.frombuffer(buf[4096-4:4096], np.uint32)[0]/1000
    ### parsing done

    # OPTIONAL 
    flux_avg = total_events/(timenew-timeold).total_seconds()
    flux_rej = total_rejected/(timenew-timeold).total_seconds()
    timeold = timenew
    # Printout some statistics, Optional
    b = "Tot: " + str(total_events) + "/" + str(flux_total) + " (" +str(total_rejected) + ") Load: " + str(board["cpu_load_ch1"]) + '/'+ str(board["cpu_load_ch2"]) + " Flux: " + str(round(flux_avg,3)) 
    b+= "\tT=" + str(board["BME_Temperature"]/100) + "C, P=" + str(board["BME_Pressure"]) + "mB, RH=" + str(board["BME_Humidity"]) + "% "
    b+= "\tST " + "%.3f"%(analysis_start_time/1000) + "s BTS:" +  "%.3f"%(BASE_timestamp_ms/1000-analysis_start_time/1000) + "s FTS " + "%d"%(flux_timestamp_ms-BASE_timestamp_ms) + "ms DRDY " + "%.3f"%(DR_timestamp_ms) +"ms "



    ### Start Fluxes part ###
    # Add new fluxes to the main array
    flux_tmp = np.empty((23, 9))
    for i in range(flux_total):

        _duration =  0.2 if i != 0 else flux_timestamp_s - last_flux_ts

        fl_l = flux_x200ms_F[i]/_duration if _duration != 0 else 0
        fl_h = flux_x200ms_H[i]/_duration if _duration != 0 else 0
        
        new_data = [(flux_timestamp_s + 0.2*i), fl_l, fl_h, board["ADC_R_thermistor"]/100, board["ADC_HV_value"]/1000, board["BME_Temperature"]/100, board["BME_Pressure"], board["BME_Humidity"], flux_rej]
        flux_tmp[i,:] = new_data

    last_flux_ts = flux_tmp[flux_total-1,0]

    # add new fluxes to the global array
    Global_flux_array = np.vstack((Global_flux_array, flux_tmp[:flux_total,:]))
    ### End Fluxes part ###


    ### Start Events part ###
    old_hist_index = Global_hist_index
    Global_hist_index = timenew.minute

    # when new timeslot started
    if (old_hist_index != Global_hist_index):
        next_index = Global_hist_index+1
        if next_index >= 60: next_index = 0
        Global_hist_array[next_index,:] = 0


    events_tmp = np.empty((total_events,2))
    for i in range(total_events):
        # timestamp 24bit, overflows at 16s
        bt02 = (events[i][0] << 16) + (events[i][1] << 8) + (events[i][2])
        bt34 = (events[i][3] <<  8) + (events[i][4] )

        ts_us =  (bt02)
        ts_ns =  (bt34>>14) & 0b11
        ch    =  (bt34>>13) & 0b1
        h1    =  (bt34) & 0b1111111111111
        if h1>HIST_RESOLUTION-1: h1 = HIST_RESOLUTION-1

        events_tmp[i,0] = BASE_timestamp_ms+ts_us/1000-analysis_start_time
        events_tmp[i,1] = h1

        # add new height into the global histogram array
        Global_hist_array[Global_hist_index,h1] +=1
    # Add all events with timestamps into global array
    Global_event_array = np.vstack((Global_event_array, events_tmp))



    # OPTIONAL PRINT
    if total_events>0:
        b += "\t Event[" + str(total_events)+"] @ " + "%5.3f"%(Global_event_array[-1,0]-BASE_timestamp_ms+analysis_start_time) + "ms, H= " +  "%4d"%(Global_event_array[-1,1]) + ";" 
    else:
        b += "\t [NO EVENTS]"
    print(" ",len(Global_event_array),'\t', b) #, end='\r')        
    # End of printing



    # Update the plot 

    graphH.set_data(HISTX[:-1], Global_hist_array[Global_hist_index,:-1])
    axH1.relim()  
    axH1.autoscale_view()

    total_hist = np.sum(Global_hist_array, axis=0)
    graphHF.set_data(HISTX[:-1], total_hist[:-1])
    axH2.relim()  
    axH2.autoscale_view()

    time_axis = Global_flux_array[:,0] - analysis_start_time/1000.
    graphFT.set_data(time_axis, Global_flux_array[:,1])
    graphFH.set_data(time_axis, Global_flux_array[:,2])
    graphR .set_data(time_axis, Global_flux_array[:,3])
    graphHV.set_data(time_axis, Global_flux_array[:,4])
    graphT .set_data(time_axis, Global_flux_array[:,5])

    axF1.relim()  
    axF1.autoscale_view()
    axF1.legend()

    axF2.relim()  
    axF2.autoscale_view()
    axF2.legend()


    a = 'Total Time: ' + "%d"%((BASE_timestamp_ms-analysis_start_time)/1000)  + "s, Points: " + "%d"%(len(Global_flux_array))
    txtb.set_text(a)

    plt.pause(0.1)




DPP_Stop()

#np.savetxt(filenameEvents, dat, delimiter=',', fmt='%d')   # save all the events
#np.savetxt(filenameHist, frq, fmt='%d')   # x,y,z equal sized 1D arrays
np.savetxt(filenameFULL, Global_event_array, fmt='%d')   # x,y,z equal sized 1D arrays

print("\nDONE!")
