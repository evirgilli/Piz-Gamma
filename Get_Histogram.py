import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime


from DPP_comm import DPP_PreConfig, DPP_GetBG, DPP_GetTao, DPP_GetHist, DPP_StartRepeatedHist, DPP_GetRepeatedHist, DPP_StopRepeatedHist


# DPP_StopRepeatedHist()

#filtered_region = DPP_PreConfig('sipm')
# filtered_region = DPP_PreConfig('sipm_plastic')
# filtered_region = DPP_PreConfig('sipm_gagg')
filtered_region = DPP_PreConfig('PMT')

step_duration = 1000


def on_close(event):
    global keep_updating_plot
    global pause_plot
    
    DPP_StopRepeatedHist()

    keep_updating_plot = 0
    pause_plot = 0

def press(event):
    global keep_updating_plot
    global pause_plot
    global HST1
    global HST2
    global log_plot
    global STATA, STATB, BOARD 
    # print('press', event.key)
    if event.key == 'escape':
        keep_updating_plot = 0
        pause_plot = 0
    if event.key == ' ':
        if (pause_plot == 0):
            pause_plot = 1
        else:
            pause_plot = 0
            # plt.xlim(0, 1000)
            # plt.ylim(0, 1000)  
    if event.key == 'a':
        ts = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        np.savetxt(ts+"_SPEC.txt", np.vstack((HST1, HST2)).T,fmt='%s', delimiter="\t")
        s = str(STATA)+ "\n" + str(STATB)+ "\n" + str(BOARD)+ "\n"
        
        with open(ts+"_SPEC.txt", "a") as myfile:
            myfile.write(s)

        print( "saved as "+ ts)

    if event.key == 'l':
        if (log_plot == 0):
            log_plot = 1
            axs[0].set_yscale('log')
            axs[1].set_yscale('linear')
            axs[2].set_yscale('log')
        else:
            log_plot = 0
            axs[0].set_yscale('linear')
            axs[1].set_yscale('linear')
            axs[2].set_yscale('linear')

def print_stat(stat, stat_tot, board):
    a = ""
    a+= "MCU Load: " + str(stat["proc_time"]/10) + "%, dead time" +  str(stat["dead_time"]/10) + "\n"
    a+="Total: time:" + str(stat_tot["actual_time_ms"]/1000) + "s, events:" + str(stat_tot["total_events"]/1000) + " (flux:" + str(round(stat_tot["total_events"]/stat_tot["actual_time_ms"]*1000.0,1)) +  "cnt/s)\n"
    a+="Last: time:" + str(stat["actual_time_ms"]/1000) + "s, events:" + str(stat["total_events"]) + " (flux:" + str(round(stat["total_events"]/stat["actual_time_ms"]*1000.0,1)) +  "cnt/s)\n"

    a+="Rejected total:" + str(sum(stat_tot["rejected"][0:8])) + " "
    a+="BG init/fin = " +  str(stat["BG_Initial"]) + "/" + str(stat["BG_Final"]) + "\n"
    a+="[ " + str(stat_tot["rejected"][0]) + "," + str(stat_tot["rejected"][1]) + "," + str(stat_tot["rejected"][2]) + "," + str(stat_tot["rejected"][3]) + "," 
    a+=str(stat_tot["rejected"][4]) + "," + str(stat_tot["rejected"][5]) + "," + str(stat_tot["rejected"][6]) + "," + str(stat_tot["rejected"][7]) + " ]\n" 
    
    b = ""
    b+= "T " + str(board["BME_Temperature"]/100) + "C, P " + str(board["BME_Pressure"]) + "mB, RH " + str(board["BME_Humidity"]) + "%\n"
    b+= "Vbias= " + str(board["ADC_HV_value"]/1000) + "V (DAC:" + str(board["dac_hv_setting"]) + ")\n"
    b+= "Rt= " + str(board["ADC_R_thermistor"]/100) + "\n"

    return a, b



keep_updating_plot = 1
pause_plot = 0
log_plot = 0
plt.ion()  # turning interactive mode on



HISTX = np.arange(2000)
HST1 = [0]*2000
HST2 = [0]*2000

FLX1 = [0]*2000
FLX2 = [0]*2000
FLX1f = [0]*2000
FLX2f = [0]*2000

# STT1 = [0]*16
# STT2 = [0]*16

fig = plt.figure()
fig.canvas.manager.set_window_title('DPP Spectra')

gs = fig.add_gridspec(3, hspace=0, height_ratios = [2, 1, 2] )
axs = gs.subplots(sharex=True, sharey=False)
fig.canvas.mpl_connect('key_press_event', press)
fig.canvas.mpl_connect('close_event', on_close)



plt.subplot(3, 1, 1)
graph1 = plt.plot(HISTX, HST1,ms=4,scalex=False)[0]
fr1a    = plt.axvspan(0, filtered_region[0]-1, alpha=0.2, color='r')
fr1b    = plt.axvspan(filtered_region[0], filtered_region[1], alpha=0.2, color='g')

txta = plt.text(1500, 0, "a", ha='left')
axs[0].tick_params('x', labelbottom=False)
plt.ylabel("Channel #1")


plt.subplot(3, 1, 2)
graphf1  = plt.plot(HISTX, FLX1,ms=4,scalex=False)[0]
graphf2  = plt.plot(HISTX, FLX2,ms=4,scalex=False)[0]
graphf1f = plt.plot(HISTX, FLX1,ms=4,scalex=False)[0]
graphf2f = plt.plot(HISTX, FLX2,ms=4,scalex=False)[0]

txt_board = plt.text(1500, 0, "b", ha='left')

plt.ylabel("FLUX")

plt.subplot(3, 1, 3)
graph2 = plt.plot(HISTX, HST2,ms=4,scalex=False)[0]
txtb = plt.text(1500, 0, "b", ha='left')
fr2a = plt.axvspan(0, filtered_region[0]-1, alpha=0.2, color='r')
fr2b = plt.axvspan(filtered_region[0], filtered_region[1], alpha=0.2, color='b')

plt.ylabel("Channel #2")

axs[1].xaxis.set_ticks(np.arange(0, 2001, 200))  

plt.xlim(0, 2000)
plt.xlabel("Height, bin")


plt.pause(0.25)


frames = 0
current_point = 0

stat_1 = { "total_events" : 0, "total_resets" : 0,  "actual_time_ms"    : 0,    "rejected" : [0,0,0,0,0,0,0,0]}
stat_2 = { "total_events" : 0, "total_resets" : 0,  "actual_time_ms"    : 0,    "rejected" : [0,0,0,0,0,0,0,0]}
STATA = {}
STATB = {}
BOARD = {}

# stat_2 = stat_1.copy
DPP_StartRepeatedHist(duration=step_duration)

while(keep_updating_plot>0):

    frames += 1
    print('\r',frames, end='', flush=True)
    # HISTA, HISTB, STATA, STATB, BOARD = DPP_GetHist(duration=step_duration)    #1000
    HISTA, HISTB, STATA, STATB, BOARD = DPP_GetRepeatedHist()
    stat_1['total_events']+=STATA['total_events']
    stat_1['total_resets']+=STATA['total_resets']
    stat_1['actual_time_ms']+=STATA['actual_time_ms']
    stat_1['rejected']= np.add(stat_1['rejected'], STATA['rejected'])

    stat_2['total_events']+=STATB['total_events']
    stat_2['total_resets']+=STATB['total_resets']
    stat_2['actual_time_ms']+=STATB['actual_time_ms']
    stat_2['rejected']= np.add(stat_2['rejected'], STATB['rejected'])

    if STATA['actual_time_ms']<1: STATA['actual_time_ms'] = 1
    if STATB['actual_time_ms']<1: STATB['actual_time_ms'] = 1
    
    flux_1 = STATA['total_events']/STATA['actual_time_ms']*1000
    flux_2 = STATB['total_events']/STATB['actual_time_ms']*1000

    
    # STT1[0] = STATA[8]*1000/STATA[1]
    # STT2[0] = STATB[8]*1000/STATB[1]
    
    FLX1[current_point] = flux_1 #STT1[0]
    FLX2[current_point] = flux_2 #STT2[0]
    FLX1f[current_point] = np.average(FLX1[current_point-10: current_point])
    FLX2f[current_point] = np.average(FLX2[current_point-10: current_point])
    FLX1f[current_point+1] = np.nan
    FLX2f[current_point+1] = np.nan

    current_point += 1
    if (current_point>1998) : current_point = 1


    ta, ra = print_stat(STATA, stat_1,BOARD)
    tb, rb = print_stat(STATB, stat_2,BOARD)


    HST1 = np.add(HST1, HISTA)
    HST2 = np.add(HST2, HISTB)
    HST1[-1]=0
    HST2[-1]=0
    

    plt.subplot(3, 1, 1)
    graph1.remove()
    graph1 = plt.plot(HISTX, HST1,ms=4,color = 'g',scaley=True)[0]
    txta.remove()
    txta = plt.text(2000, 1, ta, ha='right')
    # fr1.remove()
    # fr1  = plt.axvspan(filtered_region[0], filtered_region[1], alpha=0.5)

    plt.subplot(3, 1, 2)
    graphf1.remove()
    graphf2.remove()
    graphf1 = plt.plot(HISTX, FLX1, '.',ms=2,scalex=False,color = 'g')[0]
    graphf2 = plt.plot(HISTX, FLX2, '.',ms=2,scalex=False,color = 'b')[0]
    
    graphf1f.remove()
    graphf2f.remove()
    graphf1f = plt.plot(HISTX, FLX1f, linestyle='-',scalex=False,color = 'g')[0]
    graphf2f = plt.plot(HISTX, FLX2f, linestyle='-',scalex=False,color = 'b')[0]

    txt_board.remove()
    txt_board = plt.text(2000, 1, ra, ha='right')


    plt.subplot(3, 1, 3)
    graph2.remove()
    graph2 = plt.plot(HISTX, HST2,ms=4,color = 'b',scaley=True)[0]
    txtb.remove()
    txtb = plt.text(2000, 1, tb, ha='right')
    axs[2].sharey(axs[0])

    # axs[0].set_ylim(ymin=0)
    # axs[1].set_ylim(ymin=0)
    # axs[1].set_ylim(bottom=0, top = None)
    # axs[1].relim()
    # axs[1].set_ylim([0, None])
    # axs[2].set_ylim(ymin=0)


    plt.pause(0.5)
    while (pause_plot):
        plt.pause(0.1)


DPP_StopRepeatedHist()        


