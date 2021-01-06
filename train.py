# -*- coding: utf-8 -*-
#ROBA INIZIALE
import os
import sys
import tensorflow as tf
import numpy as np
import keras
import datetime
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from ResNet50.src import model
from ResNet50.src import resnet
from keras_vggface.vggface import VGGFace
import argparse
from TFRecord_tools import get_dataset
from datetime import datetime

available_nets = ['resnet50', 'vgg16']
available_dataset = ['shuffled', 'balanced']
available_training_mode = ['full', 'fine_tuning']

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, choices=available_nets, dest='net', help='Rete da utilizzare')
parser.add_argument('--dataset', dest='dataset', type=str, choices=available_dataset, help='Dataset da usare per il training')
parser.add_argument('--batch', dest='batch_size', type=int, default=128, help='batch size')
parser.add_argument('--resume', type=str, default=None, dest='resume', help='checkpoint da dove ricominciare il training')
parser.add_argument('--pretraining', type=str, default=None, dest='pretraining', help="Usare pretraining per l'addestramento, per vgg16 usare 'vggface', per resnet50 'resnet'")
parser.add_argument('--momentum', type=bool, default=True, dest='momentum', help='Usare o no il momentum')
parser.add_argument('--lr', default='0.002', help='Initial learning rate or init:factor:epochs', type=str)
parser.add_argument('--epoch', dest='train_epoch', type=int, default=None, help='Numero di epochs per il training')
parser.add_argument('--training_mode', dest='training_mode', type=str, choices=available_training_mode, help='fine tuning o addestramento completo della rete')
args = parser.parse_args()


TFRecord_root = os.getcwd() + '/data/TFRecord/' + str(args.dataset) + '/'
FILENAMES_TRAIN = tf.io.gfile.glob(os.path.join(TFRecord_root,'train/*.tfrecord'))
FILENAMES_VAL = tf.io.gfile.glob(os.path.join(TFRecord_root,'val/*.tfrecord'))

Checkpoint_root = os.getcwd() + '/Checkpoints'
PATH_WEIGHTS = os.getcwd() + "/data/weights.h5"

# Learning Rate
lr = args.lr.split(':')
initial_learning_rate = float(lr[0])  # 0.002
learning_rate_decay_factor = float(lr[1]) if len(lr) > 1 else 0.5
learning_rate_decay_epochs = int(lr[2]) if len(lr) > 2 else 20


train_dataset = get_dataset(FILENAMES_TRAIN, args.batch_size)
val_dataset = get_dataset(FILENAMES_VAL, args.batch_size)


if args.resume:

    #CALLBACKS
    def step_decay_schedule(initial_lr, decay_factor, step_size):
        def schedule(epoch):
            return initial_lr * (decay_factor ** np.floor(epoch / step_size))

        return keras.callbacks.LearningRateScheduler(schedule, verbose=1)

    lr_schedule = step_decay_schedule(initial_learning_rate, learning_rate_decay_factor, learning_rate_decay_epochs)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        save_best_only=True, 
        filepath=os.path.join(Checkpoint_root,args.resume),  #RICORDA DI CAMBIARE NOME
        monitor='val_accuracy',
    )

    #LOAD AND FIT
    model = keras.models.load_model(str(args.resume))

    model.fit(train_dataset,validation_data=val_dataset, epochs=args.train_epoch, callbacks=[checkpoint_cb,lr_schedule])

elif args.net=='resnet50':

    checkpoint_name = 'train_' + str(args.net) + '_batch_' + str(args.batch_size) + '_epochs_' + str(args.train_epoch) + '_lr_' + str(args.lr) + '.h5'


    def step_decay_schedule(initial_lr, decay_factor, step_size):
        def schedule(epoch):
            return initial_lr * (decay_factor ** np.floor(epoch / step_size))

        return keras.callbacks.LearningRateScheduler(schedule, verbose=1)

    lr_schedule = step_decay_schedule(initial_learning_rate, learning_rate_decay_factor, learning_rate_decay_epochs)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        save_best_only=True, 
        filepath=os.path.join(Checkpoint_root,checkpoint_name),
        monitor='val_accuracy',
    )

    def compile(model): 
        #####################
        opt = SGD(momentum=0.9) if args.momentum else SGD()
        accuracy_metrics = ['accuracy']
        loss='categorical_crossentropy'
        ######################

        model.compile(
            loss=loss,
            optimizer = opt,
            metrics=accuracy_metrics
    )


    model = model.Vggface2_ResNet50()

    if args.pretraining : model.load_weights(PATH_WEIGHTS, by_name=True, skip_mismatch=True)

    if args.training_mode=='fine_tuning':
        for l in model.layers[:-3]:
            l.trainable=False

    compile(model)

    model.summary()

    model.fit(train_dataset,validation_data=val_dataset, epochs=args.train_epoch, callbacks=[checkpoint_cb,lr_schedule])

elif args.net=='vgg16':

    def step_decay_schedule(initial_lr, decay_factor, step_size):
        def schedule(epoch):
            return initial_lr * (decay_factor ** np.floor(epoch / step_size))

        return keras.callbacks.LearningRateScheduler(schedule, verbose=1)

    lr_schedule = step_decay_schedule(initial_learning_rate, learning_rate_decay_factor, learning_rate_decay_epochs)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        save_best_only=True, 
        filepath=os.path.join(Checkpoint_root,checkpoint_name),
        monitor='val_accuracy',
    )


    def model_create():
        #custom parameters
        nb_class = 101
        hidden_dim = 512


        # VGGFACE + ultimi livelli custom
        vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3), weights=args.pretraining)
        if args.training_mode=='fine_tuning': vgg_model.trainable = False 
        last_layer = vgg_model.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(hidden_dim, activation='relu', name='fc6')(x)
        x = Dense(hidden_dim, activation='relu', name='fc7')(x)
        out = Dense(nb_class, activation='softmax', name='fc8')(x)
        custom_vgg_model = Model(vgg_model.input, out)
        custom_vgg_model.summary()
        
        return custom_vgg_model

    def compile(model): 
        #####################
        opt = SGD(momentum=0.9) if args.momentum else SGD()
        accuracy_metrics = ['accuracy']
        loss='categorical_crossentropy'
        ######################

        model.compile(
            loss=loss,
            optimizer = opt,
            metrics=accuracy_metrics
    )

    model = model_create()
    compile(model)
    model.summary()
    model.fit(train_dataset, validation_data=val_dataset, epochs=args.train_epoch, callbacks = [checkpoint_cb,lr_schedule])