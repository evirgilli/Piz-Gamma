- 21/11/2024

Get_Events_1D_Ex.py: Ho decommentato fluxes (riga 87) e "fluxes.append(flux_x200ms_F[0:flux_total])" (riga 185), 
per evitare un errore dovuto alla riga 279 che tentava di inizializzare una variabile mai definita.
Cosa analoga con first_event_index & last_event_index (riga 212) e variabili collegate,
per ottenere il numero di evento corretto (da verificare se per i nostri scopi ci serve davvero, ma per il momento lo sto usando).
