import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from scipy.optimize import curve_fit

from DPP_comm import DPP_PreConfig, DPP_GetBG, DPP_GetTao, DPP_GetWaveform, FLAGS

# set default parameters
# HV level depends on offset level:
# offset 127 == HV 210
DPP_GetTao()
DPP_GetBG()

# DPP_PreConfig('sipm')
# DPP_PreConfig('sipm_plastic')
# DPP_PreConfig('sipm_gagg')
DPP_PreConfig('pmt')
#DPP_PreConfig('sdd')



def on_close(event):
    global keep_updating_plot
    global pause_plot
    keep_updating_plot = 0
    pause_plot = 0

def press(event):
    global event_catching_mode
    global keep_updating_plot
    global pause_plot
    # print('press', event.key)
    if event.key == 'escape':
        keep_updating_plot = 0
        pause_plot = 0
    if event.key == ' ':
        if (pause_plot == 0):
            pause_plot = 1
            label = fig._suptitle.get_text()
            if "(paused)" not in label: 
                fig.suptitle(label + " (paused)", fontsize=14, fontweight='bold')
        else: 
            pause_plot = 0
            label = fig._suptitle.get_text()
            fig.suptitle(label.replace(" (paused)", ''), fontsize=14, fontweight='bold')

            plt.xlim(0, 1000)
            # plt.ylim(-4000, 4000)
    if event.key == 'a':
        ts = strftime("%Y-%m-%d_%H-%M-%S_SIGNAL.txt", gmtime())
        np.savetxt(ts, np.vstack((HISTA,FILTA,HISTB,FILTB)).T,fmt='%s', delimiter="\t")
        print( "saved as "+ ts)
    if event.key == 'm' or event.key == 'M':
        match event_catching_mode:
            case 0:
                event_catching_mode = 1
                fig.suptitle('Channel A events mode', fontsize=14, fontweight='bold')
                # mode_txt.set_text("Channel A events mode")
            case 1:
                event_catching_mode = 2
                fig.suptitle('Channel B events mode', fontsize=14, fontweight='bold')
                # mode_txt.set_text("Channel B events mode")
            case 2:
                event_catching_mode = 3
                fig.suptitle('Simultanious events mode', fontsize=14, fontweight='bold')
                # mode_txt.set_text("Simultanious events mode")
            case 3:
                event_catching_mode = 0
                fig.suptitle('Raw signal mode', fontsize=14, fontweight='bold')
                # mode_txt.set_text("Raw signal mode")
                

keep_updating_plot = 1
pause_plot = 0
event_catching_mode = 0
plt.ion()  # turning interactive mode on





fig = plt.figure()
# mode_txt = plt.text(1, 5, "Raw signal mode")
fig.suptitle('Raw signal mode', fontsize=14, fontweight='bold')

fig.canvas.manager.set_window_title('DPP Signals')

gs = fig.add_gridspec(3, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
fig.canvas.mpl_connect('key_press_event', press)
fig.canvas.mpl_connect('close_event', on_close)

plt.subplot(3, 1, 1)
graph1 = plt.plot(np.arange(2000), [0]*2000, '.',ms=4)[0]
graph2 = plt.plot(np.arange(2000), [0]*2000, '.',ms=4)[0]
axs[0].tick_params('x', labelbottom=False)
plt.ylabel("Channel #1")

plt.subplot(3, 1, 2)
graph5 = plt.plot(np.arange(2000), [0]*2000, '.',ms=4)[0]
axs[0].tick_params('x', labelbottom=False)
plt.ylabel("Channel #1")


plt.subplot(3, 1, 3)
graph3 = plt.plot(np.arange(2000), [0]*2000, '.',ms=4)[0]
graph4 = plt.plot(np.arange(2000), [0]*2000, '.',ms=4)[0]
plt.xlim(0, 1000)
# plt.ylim(-0, 4000)

plt.xlabel("Time, us")
plt.ylabel("Channel #2")
axs[1].xaxis.set_ticks(np.arange(0, 1001, 100))

plt.pause(0.25)

frames = 0
while(keep_updating_plot>0):

    frames += 1
    print('\r',frames, end='', flush=True)


    #
    match event_catching_mode:
        case 0:
            HISTX, HISTA, FILTA,  HISTB, FILTB, HISTC = DPP_GetWaveform(channel=FLAGS["FLAG_START_WAVEFORM"], max_duration=1000)
            # HISTX, HISTA, FILTA,  HISTB, FILTB, HISTC = DPP_GetRaw()
        case 1:
            HISTX, HISTA, FILTA,  HISTB, FILTB, HISTC = DPP_GetWaveform(channel=FLAGS["FLAG_START_WAVEFORM_CHA"], max_duration=1000)
        case 2:
            HISTX, HISTA, FILTA,  HISTB, FILTB, HISTC = DPP_GetWaveform(channel=FLAGS["FLAG_START_WAVEFORM_CHB"], max_duration=1000)
        case 3:
            HISTX, HISTA, FILTA,  HISTB, FILTB, HISTC = DPP_GetWaveform(channel=FLAGS["FLAG_START_WAVEFORM_ALL"], max_duration=1000)

    plt.subplot(3, 1, 1) 
    graph1.remove()
    graph2.remove()
    graph1 = plt.plot(HISTX, HISTA, '.',ms=4,color = 'g')[0]
    graph2 = plt.plot(HISTX, FILTA, '-',ms=4,color = 'k', marker='.')[0]

    plt.subplot(3, 1, 2)
    graph5.remove()
    graph5 = plt.plot(HISTX, HISTC, '.',ms=4,color = 'r')[0]


    plt.subplot(3, 1, 3)
    graph3.remove()
    graph4.remove()
    graph3 = plt.plot(HISTX, HISTB, '.-',ms=4,color = 'b')[0]
    graph4 = plt.plot(HISTX, FILTB, '.-',ms=4,color = 'r')[0]
    
    plt.pause(0.9)
    while (pause_plot):
        plt.pause(0.9)

        
# else:
#     HISTX = np.arange(2000)
#     buf = ser.read(8000)
#     dat = np.frombuffer(buf, np.int16).byteswap()
#     HISTA= dat[50:2000] + [0]*50
#     HISTB= dat[2050:4000] + [0]*50

# ser.close

    
# axs["channel1"].plot(HISTX, HISTA, '.',ms=4)
# axs["channel2"].plot(HISTX, HISTB, '.',ms=4)

# if (use_raw==1):
#     axs["channel1"].plot(HISTX, FILTA, '.',ms=4)
#     axs["channel2"].plot(HISTX, FILTB, '.',ms=4)

# plt.show()



