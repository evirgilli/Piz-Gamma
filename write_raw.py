"""
Documentazione per le funzioni principali di DPP_comm.py utilizzate nel codice:

Funzioni principali
1. Dfilter
Questa funzione imposta un filtro con parametri specifici e restituisce un dizionario di configurazione del filtro.

Parametri:
peak_threshold: soglia del picco per identificare eventi (default 50).
bg_duration_min: durata minima del background (default 25).
scan_step: passo di scansione per l'analisi del segnale (default 4).
peak_rising_time_max: tempo massimo di salita del picco per validare l'evento (default 10).
p2p_distance_min: distanza minima tra picchi (default 15).
use_syncrounous_events: utilizzo di eventi sincronizzati (default False).
offset_correct_mode: modalità di correzione dell'offset (default 0).
offset_min: valore minimo dell'offset (default 0).
offset_max: valore massimo dell'offset (default 10000).
2. DPP_Config
Questa funzione configura il sistema per l'acquisizione dei dati tramite una serie di parametri personalizzati.

Parametri:

duration: durata dell'acquisizione in ms (default 100).
run_type: tipo di esecuzione (default 0).
use_dummy: se impostato su True, utilizza valori fittizi (default False).
filter: filtro utilizzato, configurato con Dfilter.
GND_offset: offset di massa (default 31).
DAC_level: livello DAC (default 127).
ramp_direction: direzione della rampa (default 0).
generate_reset: abilita un reset della configurazione (default False).
use_manual_background: se True, usa un background manuale (default False).
adjust_background: se True, aggiusta automaticamente il background (default True).
FIR_BG: filtro FIR per il background (default [200, 200]).
tao: costante temporale del filtro (default 9.0).
output_port: porta di output, PORTS["USB"] di default.
Descrizione: Genera un array di byte con i parametri di configurazione e lo invia tramite porta seriale per configurare l'hardware.

3. DPP_GetBG
Questa funzione legge il background configurato dall'hardware.

Descrizione: Invia un comando tramite la seriale per ottenere i valori di background. Riceve e stampa i valori di background t1 e t2.
4. DPP_GetTao
Legge e restituisce la costante temporale (tao) dell'hardware.

Descrizione: Invia il comando per ottenere il tao, riceve la risposta dalla seriale, estrae i valori di t1 e t2 (in secondi) e li stampa.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import keyboard
import csv
from datetime import datetime, timedelta

import DPP_comm as dpp  # import DPP_GetRawEvents, Dfilter, DPP_Config, DPP_GetBG, DPP_GetTao, DPP_Set, DPP_GetRaw, DPP_GetEvent, DPP_Unpack_Board, FLAGS, ser
from DPP_comm import ser, FLAGS

# dpp.DPP_GetTao()
dpp.DPP_GetBG()

channel = FLAGS["FLAG_START_RAW_CH1"]  # _CH2 # _All
max_duration = 2000

_filterSiPM = dpp.Dfilter(peak_threshold=100, peak_rising_time_max=10, p2p_distance_min=15)
_filterPMT = dpp.Dfilter(peak_threshold=50, peak_rising_time_max=5, p2p_distance_min=7)

# SIPM main config:
# DPP_Config (duration=100,tao=8.5, filter = _filterSiPM, DAC_level=230, GND_offset=200, ramp_direction=1)

# PMT main config
dpp.DPP_Config(duration=100, tao=5.8, filter=_filterPMT, DAC_level=150, GND_offset=30, ramp_direction=0)

# dummy SDD config
# DPP_Config (FIR_BG_manual_A=170, FIR_BG_manual_B=170, use_manual_background=1, dynamic_background=0,duration=100, tao=11, peak_threshold=20, ramp_direction=0, use_dummy=1)


# Funzione per aprire un nuovo file CSV
def open_new_csv_file():
    start_time = datetime.now()
    filename = f"data_{start_time.strftime('%Y%m%d_%H%M%S')}.csv"
    file = open(filename, mode="w", newline='')
    writer = csv.writer(file)
    # Scrittura dell'intestazione
    writer.writerow(["Timestamp", "Total Events", "Rejected Events", "Flux", "Temperature (C)", 
                     "Pressure (mB)", "Humidity (%)", "Event TS", "Channel", "Height"])
    return file, writer, start_time

# Inizio acquisizione
if not ser.is_open:
    ser.open()

ser.write(bytearray([channel, 0x00, (max_duration >> 8) & 0xFF, (max_duration) & 0xFF]))

# Apertura del primo file CSV
csv_file, csv_writer, file_start_time = open_new_csv_file()
timeold = time.time()

# Loop di acquisizione
while not keyboard.is_pressed('esc'):
    buf = ser.read(20 + 1024 * 4)
    timenew = time.time()
    dat16 = np.frombuffer(buf[0:20], np.uint16)
    total_events = dat16[8]
    total_rejected = dat16[9]
    flux = total_events / (timenew - timeold)
    timeold = timenew

    events = np.frombuffer(buf[20:], np.uint32)
    board = dpp.DPP_Unpack_Board(buf[0:16])

    timestamp = datetime.now()
    temp = board["BME_Temperature"] / 100
    pressure = board["BME_Pressure"]
    humidity = board["BME_Humidity"]
    event_ts = board["BASE_timestamp_ms"] + (events[2] >> 12) / 1000
    channel = (events[2] >> 11) & 1
    height = events[2] & 0x7FF

    # Scrittura dei dati nel CSV
    csv_writer.writerow([timestamp.strftime("%Y-%m-%d %H:%M:%S"), total_events, total_rejected, round(flux, 3),
                         temp, pressure, humidity, event_ts, channel, height])

    # Verifica se è passato un'ora per chiudere e aprire un nuovo file CSV
    if datetime.now() >= file_start_time + timedelta(minutes=2):
        csv_file.close()
        csv_file, csv_writer, file_start_time = open_new_csv_file()

# Stop acquisizione e chiusura del file CSV finale
XRmeas = bytearray([FLAGS["FLAG_STOP"], 0x00, 0x01, 0x40])
ser.write(XRmeas)
ser.close()
csv_file.close()

print("Done")
