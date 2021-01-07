# AVG14
Repository per il progetto di Artificial Vision gruppo 14 AA 2020-2021

### Authors: Group 14
| Nome | Matricola |
|--------------|--------|
|Davide Della Monica | 0622701345|
|Vincenzo di Somma | 0622701283|
|Salvatore Gravina | 0622701063|
|Ferdinando Guarino | 0622701321|

# Setup
Le seguenti librerie sono necessarie per l'utilizzo di questa repository

```bash
keras_applications==1.0.8
tensorflow==2.4
keras==2.4.3
opencv-python==4.1.2.30
tqdm==4.46.1
numpy==1.19.3
dlib==19.18.0
```

Utilizzare il file <code>requirements.txt</code> per automatizzare tale processo

Inoltre è necessario scaricare il seguente [file](https://drive.google.com/file/d/12J_HeaLw4DXmDObXvZH5ljZ06xEKnX8T/view?usp=sharing) da estrarre nella root della repository, quest'operazione è necessaria per l'utilizzo degli script quindi non cambiare la struttura delle cartelle

# Utilizzo

## Creazione TFRecord training

Per creare nuovi TFRecord, necessari per l'addestramento, utilizzare lo script <code>TFRecord_training.py</code> messo a disposizione

```bash
python TFRecord_training.py --record_name 'tfrecord_file.tfrecord' --csv_name 'train.detected.csv' --face_align --record_dir 'shuffled/train'
```
Scegliere attentamente la cartella in cui salvare i TFRecord appena creati tramite <code>--record_dir</code>

Le righe del file csv devono presentare il seguente formato
```bash
...
n002309/0042_01.jpg,32
n002309/0195_02.jpg,25
n002309/0073_02.jpg,34
...
```
</br>

## Creazione TFRecord test

Per creare nuovi TFRecord, necessari per la predict, utilizzare lo script <code>TFRecord_training.py</code> messo a disposizione

```bash
python TFRecord_test.py --csv_path 'test.age_detected.csv' --record_name 'tfrecord_file.tfrecord'
```
Le righe del file csv devono presentare il seguente formato

```bash
...
n002309/0042_01.jpg
n002309/0195_02.jpg
n002309/0073_02.jpg
...
```
</br>

## Training

Per effettuare l'addestramento utilizzare il file <code>train.py</code>, per le possibili scelte dei parametri fare riferimento alle informazioni presenti nel codice

```bash
python train.py --net 'resnet50' --dataset 'balanced' --batch 128 --resume 'train_resnet50.h5' --pretraining 'resnet' --lr 0.005:0.2:20 --epoch 50 --training_mode 'fine_tuning' --momentum
```

## Evaluate

Per calcolare le prestazioni di un modello, dato un TFRecord (posizionato nella cartella */eval/[TFRecordname].tfrecord* ) creato tramite <code>TFRecord_training.py</code>. Utilizzare lo script <code>evaluate.py</code> specificando il modello su cui effettuare l'operazione

```bash
python evaluate.py --model 'train_resnet50.h5' 
```

## Prediction

Per generare uno file .csv con le prediction effettuate da un modello, utilizzare lo script <code>prediction.py</code> specificando il modello e il nome del csv da creare

```bash
python prediction.py --model 'resnet50.h5' --csv_path 'prediction_resnet50.csv'
```