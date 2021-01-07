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

## Creazione TFRecord

Per creare nuovi TFRecord, necessari per l'addestramento, utilizzare lo script <code>TFRecord.py</code> messo a disposizione

```bash
python TFRecord.py --record_name 'tfrecord_file.tfrecord' --csv_name 'train.detected.csv' --face_align True
```
## Training

Per effettuare l'addestramento utilizzare il file <code>train.py</code>, per le possibile scelte dei parametri fare riferimento alle informazioni presenti nel codice

```bash
python train.py --net 'resnet50' --dataset 'balanced' --batch 128 --resume 'train_resnet50.h5' --pretraining 'resnet' --lr 0.005:0.2:20 --epoch 50 --training_mode 'fine_tuning' --momentum
```

## Evaluate

Per calcolare le prestazioni di un modello, utilizzare lo script <code>evaluate.py</code> specificando il modello su cui effettuare l'operazione e la groundtruth.

```bash
python evaluate.py --model 'train_resnet50.h5' --groundtruth 'train.age_detected.csv'