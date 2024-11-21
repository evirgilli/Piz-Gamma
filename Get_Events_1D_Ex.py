import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors 
import time
import keyboard
from datetime import datetime
from scipy.optimize import curve_fit
import math
# import threading as thrd


from DPP_comm import ser, FLAGS, DPP_PreConfig, DPP_Send, DPP_GetBG, DPP_Unpack_Board, DPP_Stop

DPP_Stop()  # Added this line to handle the fact that sometimes the board is not stopped properly and, in that case, doesn't initialize DPP_GetBG() correctly.
time.sleep(0.5)
if (ser.is_open == False):  ser.open()
ser.reset_input_buffer()
ser.reset_output_buffer()
time.sleep(0.5) #Give the board a moment to settle.

DPP_GetBG()

#filtered_region = DPP_PreConfig('sipm')
filtered_region = DPP_PreConfig('PMT')

max_duration = 1000

channel = FLAGS["FLAG_START_RAWeX_CHA"] # _RAW2_ # _RAW_# _CHB # _All
# channel = FLAGS["FLAG_START_RAW_CHA"] # _CHB # _All

# def UpdatePlots():
#     global do_replot
#     while (do_replot):
#         plt.pause(0.1)


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

    # if event.key == ' ':
    #     if (pause_plot == 0):
    #         pause_plot = 1
    #     else:
    #         pause_plot = 0

    if event.key == 'l':
        if (log_plot == 0):
            log_plot = 1
            axs[0].set_yscale('log')
            axs[1].set_yscale('linear')
        else:
            log_plot = 0
            axs[0].set_yscale('linear')
            axs[1].set_yscale('linear')
Events_full = []
Events_heights = []

Events_total = 0
Events_first_index = 0
Events_last_index = 0

Fluxes_heights = []
Fluxes_heights_H = []

# hist2ddata_x = []
events_parced_tot = []
# tot_registered_events = 0
fluxes = []
do_replot = 1
keep_updating_plot = 1
pause_plot = 0
log_plot = 0
plt.ion()  # turning interactive mode on

HIST_RESOLUTION=4096

HISTX = np.arange(HIST_RESOLUTION)
HIST = [0]*HIST_RESOLUTION
FULX = [0]*10
FULX_H = [0]*10


fig = plt.figure()
fig.canvas.manager.set_window_title('DPP Spectra')

gs = fig.add_gridspec( 1, 2, hspace=0, width_ratios = [1, 2] )
axs = gs.subplots(sharex=False, sharey=False)

fig.canvas.mpl_connect('key_press_event', press)
fig.canvas.mpl_connect('close_event', on_close)

plt.subplot(1, 2, 1)
graph1 = plt.plot(HISTX, HIST,ms=4,scalex=False)[0]
fr1a    = plt.axvspan(0, filtered_region[0]-1, alpha=0.2, color='r')
fr1b    = plt.axvspan(filtered_region[0], filtered_region[1], alpha=0.2, color='g')

txta = plt.text(HIST_RESOLUTION*0.75, 0, "a", ha='left')
plt.xlabel("Spectrum")

plt.subplot(1, 2, 2)
graph2 = plt.plot(FULX,ms=4,scalex=False)[0]
graph3 = plt.plot(FULX_H,ms=4,scalex=False)[0]

# txtb = plt.text(HIST_RESOLUTION*0.75, 0, "a", ha='left')
txtb = plt.text(0, 0, "a", ha='left')
plt.xlabel("Flux/200ms")

plt.pause(0.25)

coincidence_window = 2  # max peaks offset, points.


# Get the current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Construct the filename with the current time and date
filenameEvents = f"{current_time}_Events.txt"
filenameHist = f"{current_time}_hist.txt"
filenameFULL = f"{current_time}_Full.txt"


# thread = thrd.Thread(target=UpdatePlots, daemon=True)
# thread.start()

# start aquisition
DPP_Send(channel, max_duration, False)

# ser.write(bytearray([channel, 0x00, (max_duration>>8)&0xFF, (max_duration)&0xFF]))
timeold = time.time()



first_event_index = 0
last_event_index = 0
# untill ESC is pressed
# while not keyboard.is_pressed('esc') and (tot_registered_events<acquisition_tot):
while(keep_updating_plot>0):

    buf = ser.read(4096)
    timenew = time.time()

    # Parse incomming message
    board = DPP_Unpack_Board(buf[0:16])
    BASE_timestamp_ms = np.frombuffer(buf[0:4], np.uint32)[0]
    total_events = buf[16] + (buf[17]<<8)
    total_rejected = buf[18] + (buf[19]<<8)

    events = np.frombuffer(buf[20:20+800*5], dtype = np.uint8).reshape( 800, 5)

    flux_x200ms_F= np.frombuffer(buf[4020:4020+23*2], np.uint16) #.byteswap()
    flux_x200ms_H= np.frombuffer(buf[4020+23*2:4020+23*3], np.uint8) #.byteswap()
    # flux_x200ms_H = [0] * len(flux_x200ms_F)    

    flux_total = buf[4096-5] #+ (buf[4096-5]<<8)
    DR_timestamp_us = np.frombuffer(buf[4096-4:4096], np.uint32)[0]
    # parsing done

    flux = total_events/(timenew-timeold)
    timeold = timenew

    Fluxes_heights.extend(list(flux_x200ms_F[0:flux_total]))
    Fluxes_heights_H.extend(list(flux_x200ms_H[0:flux_total]))
    
    print(DR_timestamp_us, flux_total, flux_x200ms_H[0:flux_total], flux_x200ms_F[0:flux_total])
    Events_first_index = len(Events_full)

    fluxes.append(flux_x200ms_F[0:flux_total])
    first_event_index = len(events_parced_tot)

    for i in range(total_events):
        Events_last_index = len(Events_full)
        # timestamp 20bit, overflows at 1s
        bt02 = (events[i][0] << 16) + (events[i][1] << 8) + (events[i][2])
        bt34 = (events[i][3] <<  8) + (events[i][4] )

        ts_us = (bt02)
        ts_ns =  (bt34>>14) & 0b11
        ch    =  (bt34>>13) & 0b1
        h1    =  (bt34) & 0b1111111111111

        last_event_index = len(events_parced_tot)
        events_parced_tot.append([last_event_index, BASE_timestamp_ms, ts_us, ch, h1])
        Events_full.append([last_event_index, BASE_timestamp_ms, ts_us, ch, h1])
        Events_heights.append(h1)
        # hist2ddata_x.append(h1)

    # Printout some statistics, Optional
    b = "Tot: " + str(total_events) + " (" +str(total_rejected) + ") Load: " + str(board["cpu_load_ch1"]) + '/'+ str(board["cpu_load_ch2"]) + " Flux: " + str(round(flux,3)) 
    b+= "\t T=" + str(board["BME_Temperature"]/100) + "C, P=" + str(board["BME_Pressure"]) + "mB, RH=" + str(board["BME_Humidity"]) + "% "
    
    if total_events>0:
        c = Events_full[first_event_index]
        d = Events_full[last_event_index]
        b += "\t Event[" + str(last_event_index-first_event_index)+"] @ " + "8%d"%(d[1])  + "+"  +  "%.3f"%(d[2]/1000) + "ms, H= " +  "%4d"%(d[4]) + ";" 
        b += " TX=+" + "%.3f"%((DR_timestamp_us-d[2]*0)/1000) +"ms "
    else:
        b += "\t [NO EVENTS]"
    print(" ",len(Events_heights),'\t', b, end='\r')        
    # End of printing

    # Update the plot 
    # Update HIST
    dat_new = np.copy(Events_heights).astype(int)
    frq_new, edges_new = np.histogram(dat_new, range(0, HIST_RESOLUTION, 1))


    # thread.join()
    # do_replot = 0
    plt.subplot(1, 2, 1)
    graph1.remove()
    graph1 = plt.plot(edges_new[:-1], frq_new,ms=4,color = 'g')[0]
    # plot(HISTX, HIST,ms=4,scalex=False)[0]
    txta.remove()
    txta = plt.text(HIST_RESOLUTION*0.75, 0, "a", ha='left')
    
    plt.subplot(1, 2, 2)
    graph2.remove()
    graph3.remove()
    
    txtb.remove()
    graph2 = plt.plot(np.arange(len(Fluxes_heights))*0.2, Fluxes_heights,  '.-',color = 'g')[0]
    graph3 = plt.plot(np.arange(len(Fluxes_heights_H))*0.2, Fluxes_heights_H,  '.-',color = 'r')[0]
    a = 'Total Time: ' + "%.1f"%(len(Fluxes_heights)*0.2)  + "s, Points: " + "%d"%(len(Fluxes_heights))
    txtb = plt.text(0, 0, a, ha='left')

    # do_replot = 1

    plt.pause(0.1)




DPP_Stop()


# merge events to 1d histogram

dat = np.copy(Events_heights).astype(int)
frq, edges = np.histogram(dat, range(0, 4096, 1))

# np.savetxt(filenameEvents, dat, delimiter=',', fmt='%d')   # save all the events
# np.savetxt(filenameHist, frq, fmt='%d')   # x,y,z equal sized 1D arrays
np.savetxt(filenameFULL, Events_full, fmt='%d')   # x,y,z equal sized 1D arrays

# Get a gaussian fit for the peak
try:
    peak_C = np.argmax(frq)
    peak_w = 200
    popt0, pcov0 = curve_fit(Gauss, edges[peak_C-peak_w:peak_C+peak_w], frq[peak_C-peak_w:peak_C+peak_w],  p0=[np.max(frq)/2., peak_C, 20.])
    HIST_FIT = Gauss(edges[:-1], popt0[0],popt0[1],popt0[2])
    np.set_printoptions(precision=3)
    print("\nFit:", popt0, round(popt0[2]/popt0[1]*100,2), peak_C)
except (RuntimeError, ValueError):
    HIST_FIT = [0]*len(edges[:-1])
    print('\nFit failed')

fig = plt.figure(figsize=(10, 5))
ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=1)
plot_max =  pow(10,math.floor(math.log10(max(frq))+1))

fluxes = np.concatenate( fluxes, axis=0 )


ax1.plot(edges[:-1], frq,ms=4)
ax1.plot(edges[:-1], HIST_FIT,'-')
ax1.set_yscale('log')
ax1.set_xlim(0, 4000)
ax1.set_ylim(1, plot_max )  # Set only the lower limit
ax1.get_xaxis().set_visible(False)

ax2 = plt.subplot2grid((2, 2), (1, 0), sharex=ax1)
ax2.plot(edges[:-1], frq,ms=4)
ax2.plot(edges[:-1], HIST_FIT,'-')
ax2.set_ylim(bottom=0)  # Set only the lower limit

ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
ax3.plot(np.arange(len(fluxes))*0.2, fluxes,  '.-',color = 'g')
ax3.set_title('Total Time: ' + "%.1f"%(len(fluxes)*0.2)  + "s, Points: " + "%d"%(len(fluxes)))
ax3.set_ylim(bottom=0)  # Set only the lower limit

plt.tight_layout()
plt.subplots_adjust(hspace=0, wspace=0.1)  # Specifically remove vertical spacing
plt.show()

print("\nDONE!")
