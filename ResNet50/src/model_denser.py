import keras
from . import resnet
import keras.backend as K
from keras.layers.experimental import preprocessing


def Vggface2_ResNet50(input_dim=(224, 224, 3), nb_classes=101, mode='train', weight_decay = 5e-4):
    # inputs are of size 224 x 224 x 3
    data_augmentation = keras.Sequential(
      [
        preprocessing.RandomFlip("horizontal", name="random_horizontal_flip"),
        preprocessing.RandomRotation(0.014, name="random_rotation"), #5gradi più o meno
        #preprocessing.RandomZoom(0.1, name="random_zoom"),
        #preprocessing.RandomCrop(200,200, name="random_crop"),
        preprocessing.RandomContrast(0.5, name="random_contrast"),
        preprocessing.Rescaling(1./255, name="rescaling")
      ]
    )
    inputs = keras.layers.Input(shape=input_dim, name='base_input')
    y = data_augmentation(inputs)
    x = resnet.resnet50_backend(y)

    # AvgPooling
    x = keras.layers.AveragePooling2D((7, 7), name='avg_pool')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu', name='dim_proj', kernel_regularizer=keras.regularizers.l2(weight_decay))(x) #aggiunto kernel_regularizer
    x = keras.layers.Dense(256, activation='relu', name='dim_proj2', kernel_regularizer=keras.regularizers.l2(weight_decay))(x)


    if mode == 'train':
        y = keras.layers.Dense(nb_classes, activation='softmax',
                               use_bias=False, trainable=True,
                               kernel_initializer='orthogonal',
                               kernel_regularizer=keras.regularizers.l2(weight_decay),
                               name='classifier_low_dim')(x)
    else:
        y = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, 1))(x)

    # Compile
    model = keras.models.Model(inputs=inputs, outputs=y)
    return model
    

