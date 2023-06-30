from __future__ import print_function
import os
import sys
import random
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.applications import DenseNet201
#from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import SGD

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import angle_error, RotNetDataGenerator

#train_filenames = os.listdir('./data/train')
#validation_filenames = os.listdir('./data/validate')
#prefix_train = './data/train/'
#prefix_validate = './data/validate/'
#train_filenames = [prefix_train + i for i in train_filenames]
#validation_filenames = [prefix_validate + i for i in validation_filenames]
from data.street_view import get_filenames as get_street_view_filenames


data_path = os.path.join('data', 'street_view')
train_filenames, validation_filenames = get_street_view_filenames(data_path)

print(len(train_filenames), 'train samples')
print(len(validation_filenames), 'validation samples')
from keras.applications.densenet import preprocess_input
from keras.models import load_model

model_name = 'DenseNet201_street_view_pi'
# training parameters
batch_size = 64
nb_epoch = 50
# input image shape
input_shape = (224, 224, 3)
output_folder = './models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# load base model
base_model = DenseNet201(weights='imagenet', include_top=False,
                      input_shape=input_shape)
# append classification layer
x = base_model.output
x = Flatten()(x)
final_output = Dense(360, activation='softmax', name='fc360')(x)

# create the new model
model = Model(inputs=base_model.input, outputs=final_output)

model.summary()
# model compilation
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(learning_rate=0.01, momentum=0.9),
              metrics=[angle_error])
# callbacks
monitor = 'val_angle_error'
checkpointer = ModelCheckpoint(
    filepath=os.path.join(output_folder, model_name + '.hdf5'),
    monitor=monitor,
    save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)
early_stopping = EarlyStopping(monitor=monitor, patience=5)
tensorboard = TensorBoard()

# training loop


model.fit(
    RotNetDataGenerator(
        train_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True,
        shuffle=True
    ),
    epochs=nb_epoch,
    validation_data=RotNetDataGenerator(
        validation_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True
    ),
    callbacks=[checkpointer, reduce_lr, early_stopping, tensorboard],
    workers=10
)
