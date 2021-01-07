import tensorflow as tf
import os
import keras
from functools import partial
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model_name', type=str, help="nome del modello su cui effettaure la predict")
parser.add_argument('--csv_path', dest='csv_path', type=str, help='nome del .csv che contiene le prediction ottenute')
args = parser.parse_args()

AUTOTUNE = tf.data.experimental.AUTOTUNE
FILENAMES_TEST = tf.io.gfile.glob(os.getcwd() + '/data/TFRecord/predict/*.tfrecord')

model_root = os.getcwd() + '/Checkpoints/'
model_path = os.path.join(model_root,str(args.model_name))

csv_root = os.getcwd() + '/data/csv/'
csv_path = os.path.join(csv_root,args.csv_path)

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    return image

def read_tfrecord(example):
    image_feature_description ={
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(example, image_feature_description)
    image = decode_image(example["image_raw"])

    name = example["image_name"]
    return image, name


def load_dataset(filenames):
  ignore_order = tf.data.Options()
  ignore_order.experimental_deterministic = False
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.with_options(ignore_order)
  dataset = dataset.map(partial(read_tfrecord), num_parallel_calls=AUTOTUNE)
  return dataset


def get_dataset(filenames, batch_size=1):
    dataset = load_dataset(filenames)
    dataset = dataset.batch(batch_size)
    return dataset

test_dataset = get_dataset(FILENAMES_TEST, 1)
model = keras.models.load_model(model_path)

lista = []
i=0
for image, name in test_dataset:
    prediction = model.predict(image)
    predicted_label = prediction.argmax()
    lista.append((name.numpy()[0].decode(),predicted_label))
    i += 1

print('number of predictions: ', i)


with open (csv_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    for i in lista:
      csv_writer.writerow([i[0],i[1]])