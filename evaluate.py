import tensorflow as tf
import os
import keras
import argparse
from TFRecord_tools import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model_name', type=str, help="path del modello su cui effettaure l'eval")
parser.add_argument('--groundtruth', dest='groundtruth', type=str, help='path del .csv contenente la groundtruth')
args = parser.parse_args()

FILENAMES_TEST = tf.io.gfile.glob(os.getcwd() + '/data/TFRecord/eval/*.tfrecord')

test_dataset = get_dataset(FILENAMES_TEST, 1)

model_root = os.getcwd() + '/Checkpoints/'
model_path = os.path.join(model_root,str(args.model_name))
model = keras.models.load_model(model_path)

mae = 0
i=0
for image, label in test_dataset:
    prediction = model.predict(image)
    predicted_label = prediction.argmax()
    true_label = label.numpy().argmax()
    mae += abs(predicted_label-true_label)
    i += 1

mae = mae/i

print('mae: ', mae)
