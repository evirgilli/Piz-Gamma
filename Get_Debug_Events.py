import numpy as np
import matplotlib.pyplot as plt
from time import gmtime, strftime
from scipy.optimize import curve_fit

from DPP_comm import Dfilter, DPP_PreConfig, DPP_GetDebugEvents, FLAGS

channel = FLAGS["FLAG_START_DBG_CHA"]  # _CHB_GOOD / _CHA / _CHB

DPP_PreConfig('pmt')





def on_close(event):
    global keep_updating_plot
    global pause_plot
    keep_updating_plot = 0
    pause_plot = 0

def press(event):
    global event_catching_mode
    global keep_updating_plot
    global pause_plot
    global current_event
    global update_event

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

            # plt.xlim(0, 4000)
            # plt.ylim(-4000, 4000)
    if event.key == 'right':
        current_event += 1
        if (current_event>=125): current_event = 0
        update_event = 1
    if event.key == 'left':
        current_event -= 1
        if (current_event<0): current_event = 125-1
        update_event = 1

keep_updating_plot = 1
pause_plot = 0
event_catching_mode = 0
current_event = 0
update_event = 0
plt.ion()  # turning interactive mode on





fig = plt.figure()
# mode_txt = plt.text(1, 5, "Raw signal mode")
fig.suptitle('Raw events mode', fontsize=14, fontweight='bold')

fig.canvas.manager.set_window_title('DPP Signals')

gs = fig.add_gridspec(ncols=2, nrows=2, hspace=0,wspace=0,width_ratios = [3,1])
axs = gs.subplots(sharex='col', sharey=True)
fig.canvas.mpl_connect('key_press_event', press)
fig.canvas.mpl_connect('close_event', on_close)

plt.subplot(gs[0,0])
graph000 = plt.plot(np.arange(2000), [0]*2000, '.',ms=4)[0]
axs[0][0].tick_params('x', labelbottom=False)
plt.ylabel("Channel #1")

plt.subplot(gs[1,0])
graph100 = plt.plot(np.arange(2000), [0]*2000, '.',ms=4)[0]
plt.xlim(0, 4000)
plt.ylim(-4000, 4000)
plt.xlabel("point, us")
plt.ylabel("Channel #2")

plt.subplot(gs[0,1])
graph010 = plt.plot(np.arange(-14,14), [0]*28, '.',ms=4)[0]
graph011 = plt.plot([-14, 14], [0,0], 'g-',ms=4)[0]
graph012 = plt.plot([-14, 14], [0,0], 'r-',ms=4)[0]
vline1 = plt.axvline(x=0)
oline1 = plt.axvline(x=0, ls = ':')

txterr1 = plt.text(0, 0, "b", ha='left')
axs[0][1].yaxis.tick_right()
axs[0][1].yaxis.set_label_position("right")
axs[0][1].tick_params('x', labelright=True)

plt.subplot(gs[1,1])
graph110 = plt.plot(np.arange(-14,14), [0]*28, '.',ms=4)[0]
graph111 = plt.plot([-14, 14], [0,0], 'g-',ms=4)[0]
graph112 = plt.plot([-14, 14], [0,0], 'r-',ms=4)[0]
vline2 = plt.axvline(x=0)
pline2 = plt.axvline(x=0, ls = ':', color= 'r')
oline2 = plt.axvline(x=0, ls = ':')

txterr2 = plt.text(0, 0, "b", ha='left')
plt.xlim(-14, 14)
plt.ylim(-4000, 4000)
plt.xlabel("point, us")
axs[1][1].yaxis.tick_right()
axs[1][1].yaxis.set_label_position("right")



plt.pause(0.25)

frames = 0
while(keep_updating_plot>0):

    frames += 1      
    print('\r',frames, end='', flush=True)

    HISTX, CH_A, CH_B, raw_events, board = DPP_GetDebugEvents(channel , 5000, 0)

    b = ""
    b+= "T " + str(board["BME_Temperature"]/100) + "C, P " + str(board["BME_Pressure"]) + "mB, RH " + str(board["BME_Humidity"]) + "%\n"
    b+= "Vbias= " + str(board["ADC_HV_value"]/1000) + "V (DAC:" + str(board["dac_hv_setting"]) + ")\n"
    b+= "Rt= " + str(board["ADC_R_thermistor"]/100) + "\n"

    # check if any point beyond 1990
    a = [ x[0][0] for x in raw_events[:-1]]
    b = np.array(a)
    if (np.max(a) > 1994):
        colr = 'r' 
        print("__>1994:", np.argwhere(b > 1994))
    else:
        if (np.min(a) < 1):
            print("__<1:", np.argwhere(b < 1 ))
            colr = 'y'
        else:
            colr = 'g'

    plt.subplot(gs[0,0])
    graph000.remove()
    graph000 = plt.plot(HISTX, CH_A, '.',ms=4,color = colr)[0]


    plt.subplot(gs[1,0])
    graph100.remove()
    graph100 = plt.plot(HISTX, CH_B, '.',ms=4,color = 'b')[0]
    
    plt.pause(0.1)
    while (pause_plot):
        if (update_event != 0):
            update_event = 0
            x = np.arange(-14,14)



            pos1 = raw_events[current_event][0][0]
            h1 = raw_events[current_event][0][1]
            e1 = raw_events[current_event][0][2]
            o1 = raw_events[current_event][0][3]/100
            bg1 = raw_events[current_event][0][4]

            pos2 = raw_events[current_event][1][0]
            h2 = raw_events[current_event][1][1]
            e2 = raw_events[current_event][1][2]
            o2 = raw_events[current_event][1][3]/100
            bg2 = raw_events[current_event][1][4]


            y1 = raw_events[current_event][0][5]
            y2 = raw_events[current_event][1][5]

            plt.subplot(gs[0,1]) 
            axs[0][1].set_title("P " + str(current_event))

            graph010.remove()
            graph011.remove()
            graph012.remove()
            txterr1.remove()
            oline1.remove()
            
            graph010 = plt.plot(x, y1, '.',ms=4,color = 'b')[0]
            graph011 = plt.plot([-14, 14], [bg1,bg1], 'k:',ms=4)[0]
            oline1 = plt.axvline(x=-o1, ls = ':')


            labl = " pos: "+ str(pos1) + " offset:" + str(round(o1,2))
            if (e1==0):
                graph012 = plt.plot([-14, 14], [bg1+h1,bg1+h1], 'g-',ms=4)[0]
                txterr1 = plt.text(1, bg1,"\n"+ " H: " + str(h1) +labl, ha='left', va ='top', color="g")
            else:
                graph012 = plt.plot([-14, 14], [bg1+h1,bg1+h1], 'r-',ms=4)[0]
                txterr1 = plt.text(1, bg1,"\n"+ " Err: " + str(e1) +labl, ha='left', va ='top',color="r")


            plt.subplot(gs[1,1])
            graph110.remove()
            graph111.remove()
            graph112.remove()
            txterr2.remove()
            oline2.remove()
            pline2.remove()

            graph110 = plt.plot(x, y2, '.',ms=4,color = 'b')[0]
            graph111 = plt.plot([-14, 14], [bg2,bg2], 'k:',ms=4)[0]
            oline2 = plt.axvline(x=-o2+pos2-pos1, ls = ':')
            pline2 = plt.axvline(x=pos2-pos1, ls = ':', color= 'r')
            labl =" pos: "+ str(pos2) + " offset:" + str(round(o2,2))
            if (e2==0):
                graph112 = plt.plot([-14, 14], [bg2+h2,bg2+h2], 'g-',ms=4)[0]
                txterr2 = plt.text(1, bg2, "\n"+ " H: " + str(h2) + labl, ha='left', va ='top', color="g")
            else:
                graph112 = plt.plot([-14, 14], [bg2+h2,bg2+h2], 'r-',ms=4)[0]
                txterr2 = plt.text(1, bg2, "\n"+ " Err: " + str(e2) +labl, ha='left', va ='top',color="r")


        plt.pause(0.1)

        
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



