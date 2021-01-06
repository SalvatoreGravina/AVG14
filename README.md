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
numpy==1.19.4
keras==2.4.3
opencv-python==4.1.2.30
tqdm==4.46.1
dlib==19.18.0
```

Utilizzare il file <code>requirements.txt</code> per automatizzare tale processo

# Utilizzo
## Creazione TFRecord
per creare nuovi TFRecord, necessari per l'addestramento, utilizzare lo script <code>TFRecord.py</code> messo a disposizione

```bash
python TFRecord.py --record_name 'tfrecord_file.tfrecord' --csv_name 'train.detected.csv' --face_align True
```
## Training

Per effettuare l'addestramento utilizzare il file <code>train.py</code>, per le possibile scelte dei parametri fare riferimento alle informazioni presenti nel codice

```bash
python train.py --net 'resnet50' --dataset 'balanced' --batch 128 --resume 'train_resnet50.h5' --pretraining 'resnet' --lr 0.005:0.2:20 --epoch 50 --training_mode 'fine_tuning' --momentum
```

## Evaluate



