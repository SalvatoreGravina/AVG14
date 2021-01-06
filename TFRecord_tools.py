import tensorflow as tf
from functools import partial
import csv
import numpy as np
import dlib
import cv2
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE
path_predictor = os.getcwd() + '/data/shape_predictor_5_face_landmarks'
"""Funzioni per scrittura TFRecord"""


def face_align(face_file_path, predictor_path=path_predictor):
  
  detector = dlib.get_frontal_face_detector()

  sp = dlib.shape_predictor(predictor_path)

  img = dlib.load_rgb_image(face_file_path)

  dets = detector(img, 1)

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

def _readcsv(csvpath):
    data = []
    with open(csvpath, newline='', encoding="utf8") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
          data.append((row[0],round(float(row[1]))))
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
def image_example(image, label):
  feature = {
      'label': _int64_feature(label),
      'image_raw': _bytes_feature(image),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

"""Funzioni per lettura TFRecord"""

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    return image

def read_tfrecord(example):
    image_feature_description ={
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(example, image_feature_description)
    image = decode_image(example["image_raw"])

    label = tf.cast(example["label"], tf.int32)
    label = tf.one_hot(label, 101, dtype=tf.int32)
    return image, label


def load_dataset(filenames):
  ignore_order = tf.data.Options()
  ignore_order.experimental_deterministic = False
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.with_options(ignore_order)
  dataset = dataset.map(partial(read_tfrecord), num_parallel_calls=AUTOTUNE)
  return dataset


def get_dataset(filenames, batch_size=128):
    dataset = load_dataset(filenames)
    dataset = dataset.batch(batch_size)
    return dataset


