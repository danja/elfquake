# ELFQuake

ELFQuake è un progetto di ricerca che esamina se osservazioni radio [Extremely](https://en.wikipedia.org/wiki/Extremely_low_frequency)/[Very Low Frequency](https://en.wikipedia.org/wiki/Very_low_frequency) possano integrare i dati sismici e astronomici abbastanza da supportare modelli predittivi utili per i terremoti.

L’ipotesi centrale è che anomalie radio ELF/VLF naturali possano contenere segnale che non è presente nella sola storia degli eventi sismici. Almeno [uno studio](https://pubs.geoscienceworld.org/ssa/bssa/article-abstract/113/6/2461/627949/Earthquake-Forecasting-Using-Big-Data-and)
suggerisce che questo possa essere un approccio praticabile.

Vogliamo sfruttare tecniche più moderne di machine learning/AI per creare un modello predittivo basato sull’architettura Transformer. Un punto chiave è la disponibilità di dati reali per l’addestramento. Il piano è costruire un sistema (basato su un modello ad avalanga) con caratteristiche simili al sistema geologico e usarlo per generare dati sintetici per l’addestramento di base. Dopo la validazione di questa fase, i dati reali saranno usati per il fine-tuning.

*Per essere utili, eventuali affermazioni devono essere dimostrate contro baseline sismiche riproducibili, periodi di validazione held-out e ablation multimodali.*

## Stato

Stato: il pretraining self-supervised su VLF reale è ora il percorso modello predefinito, mentre l’addestramento supervisionato sui target VLF reali resta bloccato da etichette di una sola classe; **non si rivendica alcuna capacità di previsione dei terremoti**.

In questo momento, in attesa di ulteriori etichette VLF allineate, il focus è sull’apprendimento di rappresentazioni VLF reali senza etichette, sulle baseline sismiche reali, sulla diagnostica dei regimi sintetici e sul mantenimento stabile dell’interfaccia multimodale del modello. La valutazione del modello corrente si trova in [report.md](docs/report.md), [model-comparison.md](docs/model-comparison.md) e [model-scaling-requirements.md](docs/model-scaling-requirements.md).

### Terremoti Simulati

![simulated earthquake map](docs/images/map_v1.1.0.png)

### Segnale VLF Simulato

![simulated VLF emissions](docs/images/vlf-simulated.png)

## Contesto

Questo lavoro è stato inizialmente ispirato dalla tragedia del terremoto dell’[Aquila del 2009](https://en.wikipedia.org/wiki/2009_L'Aquila_earthquake). Più o meno nello stesso periodo ero interessato agli sviluppi nel Deep Learning e, per coincidenza, avevo incontrato materiale relativo a segnali radio naturali che precedono eventi sismici (vedi [vlf.it](http://www.vlf.it/)). Ho iniziato la ricerca e ho aperto un blog su questo tema: [ELFQuake](https://elfquake.wordpress.com/) (che poi ho usato per blog generici). All’epoca sembrava possibile ma *molto difficile*. Da allora, però, i modelli predittivi sono migliorati molto, il che significa che la previsione ha maggiori possibilità di funzionare. Inoltre, la possibilità di delegare gran parte del lavoro di codice a un assistente intelligente ha ridotto in modo drastico la difficoltà di costruire il sistema.

## Lavoro Corrente

* Acquisizione, normalizzazione e finestre di training per eventi sismici INGV in Italia/Centro Italia, attualmente backfillati dal 2024-01-01 al 2026-07-07.
* Captura di spettrogrammi VLF live di Cumiana tramite systemd, con feature immagine derivate dai pixel.
* Connettori e normalizzazione per archivi astronomici e di space weather.
* Righe di feature VLF prospective ancorate al tempo, con target in attesa di etichettatura.
* Un autoencoder CPU PyTorch self-supervised su sequenze di immagini VLF reali di Cumiana.
* Controlli logistici leggeri e un MLP tabellare CPU PyTorch per righe sintetiche allineate.
* Un primo modello sequenziale GRU CPU PyTorch su tensori sintetici di avalanga e sensori piezo/VLF.
* Uno scaffold patch Transformer CPU PyTorch minimale per controlli di ingegneria sulle sequenze sintetiche.
* Script di confronto, sweep sulle sequenze e missing-modality per la diagnostica dei modelli.
* Un wrapper compatto per il confronto reale-vs-sintetico su baseline sismiche del Centro Italia e report di sequenza sintetica.
* Controlli di scala per modelli più grandi su numero di righe, bilanciamento delle classi, dimensione delle sequenze e limiti CPU.
* Feature VLF reali di Cumiana materializzate nella stessa forma sequenziale dei dati sintetici piezo/VLF.
* Input reali allineati per il modello scaffoldati per righe all-Italy e central-Italy; i wrapper per il fine-tuning di modelli profondi reali rifiutano l’addestramento finché entrambe le classi non esistono.
* Ruoli di feature VLF che permettono ai dati sintetici analoghi piezo/VLF di esercitare lo stesso percorso VLF del modello PyTorch prima che maturino le etichette reali.
* Diagnostica di transfer da inlier sintetici che addestra su descrittori piezo/VLF sintetici simili al reale e valuta la ricostruzione su descrittori VLF reali held-out.
* Allineamento di descrittori VLF reali/sintetici misti con loss CORAL e controlli sintetici centroid/random/full.
* Smoke forecast VLF reali senza etichette basati su anomalie, finché le etichette reali supervisionate restano bloccate.
* Simulazione sandpile con uscite separate simili a eventi sismici e uscite analoghe piezo/VLF.

Esegui il percorso predefinito di pretraining self-supervised su VLF reale con:

```sh
./pretrain-real-vlf-self-supervised.sh
```

Esegui lo smoke forecast VLF a 7 giorni, senza etichette, con:

```sh
./score-real-vlf-anomaly-forecast.sh
```

Confronta il dominio di embedding VLF reale con gli analoghi piezo/VLF sintetici con:

```sh
./compare-vlf-embedding-domains.sh
```

Esegui la diagnostica di transfer da inlier sintetici con:

```sh
./evaluate-vlf-synthetic-inlier-transfer.sh
```

Esegui la diagnostica di allineamento VLF mixed-domain con:

```sh
./evaluate-vlf-mixed-domain-alignment.sh
```

## Simulazione

La simulazione è una griglia artificiale simile a una montagna in cui un caricamento di fondo ampio è combinato con stress localizzato ripetuto in punti sorgente fissi. Quando le pendenze diventano instabili, piccole valanghe ridistribuiscono l’altezza tra celle vicine. L’obiettivo è generare sequenze sintetiche abbastanza simili, per forma, ai dati sismici reali da essere utili come training data per un sistema di deep learning, soprattutto prima che siano disponibili abbastanza dati sismici, VLF e astronomici corrispondenti.

Include anche sensori tipo piezo che osservano regioni suscettibili simili al quarzo prossime al cedimento e producono il canale analogico VLF/WAV. I dati diretti simili agli eventi sismici restano separati e derivano dal comportamento di avalanga/toppling.

Esegui il pipeline demo della simulazione locale con:

```sh
./run-all.sh
```

Gli output predefiniti usano `data/derived/sim/mountain_256x256_seed42_10000` come prefisso. L’immagine piezo normale è `*.piezo_vlf_summary.png` da `*.piezo.csv`; l’analogo diretto degli eventi sismici è `*.avalanche_events.csv`. La vecchia diagnostica FFT è opzionale con `RUN_FFT=1`.

La demo event-map proietta le location derivate dalle avalanghe su una fascia italiana in stile Appennini e usa la dimensione del punto per la magnitudo sintetica.

Per renderizzare una demo overlay degli eventi sintetici di avalanga e dei hit positivi previsti da PyTorch per la finestra target:

```sh
./prediction-event-map.sh
```

Per confrontare l’immagine VLF simulata con gli spettrogrammi Cumiana acquisiti:

```sh
./compare-piezo-vlf.sh
```

Questa è un’analogia semplificata di stress-and-release, non un modello geologico. Il suo valore dipende dal fatto che i dati generati abbiano una somiglianza strutturale utile con le osservazioni reali. Anche buone prestazioni sulle avalanghe simulate mostrerebbero solo che gli strumenti possono apprendere pattern sintetici; le affermazioni reali richiedono ancora dati sismici, VLF e astronomici held-out.

## Documentazione Chiave

* [Overview](docs/overview.md)
* [Documentation Index](docs/README.md)
* [Processing Graph](docs/processing-graph.md)
* [Next Actions](docs/next-actions.md)
* [Development Environment](docs/development-environment.md)
* [Source Inventory](docs/source-inventory.md)
* [Multimodal Feasibility](docs/multimodal-feasibility.md)
* [Systemd Service](docs/systemd-service.md)
* [Initial Model Trials](docs/initial-model-trials.md)
* [Model Comparison](docs/model-comparison.md)
* [Model Scaling Requirements](docs/model-scaling-requirements.md)
* [Sandpile Simulation](docs/sandpile-simulation.md)
