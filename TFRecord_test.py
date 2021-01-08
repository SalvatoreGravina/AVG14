
#import
import csv
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from functools import partial
import dlib
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', dest='csv_path', type=str, help='nome del .csv che contiene i path delle immagini')
parser.add_argument('--record_name', type=str, dest='record_name', help='nome del TFRecord da creare seguire la seguente nomenclatura [*.tfrecord]')
args = parser.parse_args()

csv_root = os.getcwd() + '/data/csv/'
csv_path = os.path.join(csv_root,args.csv_path)

path_dataset = os.getcwd() + '/data/dataset/test/'
path_predictor = os.getcwd() + '/data/shape_predictor_5_face_landmarks.dat'
path_tfrecord_root = os.getcwd() + '/data/TFRecord/predict/'
path_tfrecord = os.path.join(path_tfrecord_root, args.record_name)


def face_align(face_file_path, predictor_path=path_predictor):
  
  detector = dlib.get_frontal_face_detector()

  sp = dlib.shape_predictor(predictor_path)

  img = dlib.load_rgb_image(face_file_path)

  dets = detector(img, 1) #ci sono le informazioni delle bounding box
  num_faces = len(dets)
  if num_faces != 1: return None #prendiamo solo foto in cui troviamo esattamente 1 viso

  faces = dlib.full_object_detections()

  for detection in dets:
      faces.append(sp(img, detection))

  images = dlib.get_face_chips(img, faces, size=224)

  for image in images:
    _, image_numpy = cv2.imencode('.jpeg', cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
    image_bytes = image_numpy.tobytes()
    return image_bytes


#leggiamo il csv
def _readcsv(csvpath=csv_path):
    data = []
    with open(csvpath, newline='', encoding="utf8") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
          data.append(row[0])
    return np.array(data)

#Definizione delle feauture
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



#Crea dei tf.Example object
def image_example(image, name):
  feature = {
      'image_raw': _bytes_feature(image),
      'image_name': _bytes_feature(name)
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

with tf.io.TFRecordWriter(path_tfrecord) as writer:

  meta = _readcsv()
  print("\nlettura csv finita\n-----------------------")
  i=0

  for d in tqdm(meta):
    if not os.path.exists(str(os.path.join(path_dataset,d))): continue #se il path non esiste, non è stata caricata

    if i%10000==0 : print(i)

    #carico l'immagine, faccio l'align e se ha problemi la scarta
    load_path=str(os.path.join(path_dataset,d))
    image = face_align(load_path)
    if image is None: continue
    i+=1
    #tfrecordizza
    tf_example = image_example(image, str.encode(d))
    writer.write(tf_example.SerializeToString())

print("\nOperazione conclusa, sono state aggiunte " + str(i) + " immagini al TFRecord")
