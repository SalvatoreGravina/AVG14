import tensorflow as tf
from tqdm import tqdm
import argparse
import os
from TFRecord_tools import _readcsv, face_align, image_example

available_record_dir = ['shuffled/train', 'shuffled/val', 'balanced/train', 'balanced/val']

parser = argparse.ArgumentParser()

parser.add_argument('--record_name', type=str, dest='record_name', help='nome del TFRecord da creare seguire la seguente nomenclatura [*.tfrecord]')
parser.add_argument('--csv_name', type=str, dest='csv_name', help='nome del csv da usare per la creazione')
parser.add_argument('--face_align', type=bool, default=False, dest='facing', help='Esegue oppure no la face align con dlib')
parser.add_argument('--record_dir', type=str, dest='record_dir', help='cartella di destinazione dei TFRecord', choices=available_record_dir)

args = parser.parse_args()

path_dataset = os.getcwd() + '/data/dataset/train/'


path_tfrecord_root = os.getcwd() + '/data/TFRecord/'
path_tfrecord = os.path.join(path_tfrecord_root,args.record_dir)

#Creazione TFRECORD with face align (powered by dlib)
with tf.io.TFRecordWriter(os.path.join(path_tfrecord,args.record_name)) as writer:

  meta = _readcsv(args.csv_name)
  print("\nlettura csv finita")
  n_images_added=0

  for d in tqdm(meta):
    path_image = str(os.path.join(path_dataset,d[0]))

    if not os.path.exists(path_image): continue #se il path non esiste, non è stata caricata

    #carico l'immagine, faccio l'align e se ha problemi la scarta
    if args.facing:
        image = face_align(path_image)

    if image is None: continue
    n_images_added+=1

    label = int(d[1])
    tf_example = image_example(image, label)
    writer.write(tf_example.SerializeToString())

print("\nOperazione conclusa, sono state aggiunte " + str(n_images_added) + "immagini al TFRecord")