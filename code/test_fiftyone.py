from __future__ import print_function
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import Dataset
#session = fo.launch_app(dataset.view())




import os
import sys
import random

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
#from keras.applications.resnet50 import ResNet50
from keras.applications.resnet import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import SGD

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import angle_error, RotNetDataGenerator

# load the new dataset from open images
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    max_samples=100000,
    seed=51,
    shuffle=True,
)


data_path = "C:\Users\jonas\fiftyone\open-images-v7\validation\data"

# Define the directory containing your images
source_dir = os.path.join('data', 'open_images')

# Define the percentage split for the train, validation, and test sets
train_pct = 0.8
val_pct = 0.2
#test_pct = 0.1

# Get a list of all the files in the source directory
files = os.listdir(source_dir)
num_files = len(files)

# Shuffle the list of files
random.shuffle(files)

# Split the files into the train, validation, and test sets
train_filenames = files[:int(num_files*train_pct)]
val_filenames = files[int(num_files*train_pct):int(num_files*(train_pct+val_pct))]
#test_filenames = files[int(num_files*(train_pct+val_pct)):]


print(len(train_filenames), 'train samples')
print(len(val_filenames), 'test samples')

model_name = 'rotnet_street_view_resnet50'

# number of classes
nb_classes = 360
# input image shape
input_shape = (224, 224, 3)

# load base model
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=input_shape)
# append classification layer
x = base_model.output
x = Flatten()(x)
final_output = Dense(nb_classes, activation='softmax', name='fc360')(x)

# create the new model
model = Model(inputs=base_model.input, outputs=final_output)

model.summary()

# model compilation
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, momentum=0.9),
              metrics=[angle_error])

# training parameters
batch_size = 64
nb_epoch = 50

output_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

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
model.fit_generator(
    RotNetDataGenerator(
        train_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True,
        shuffle=True
    ),
    steps_per_epoch=len(train_filenames) / batch_size,
    epochs=nb_epoch,
    validation_data=RotNetDataGenerator(
        val_filenames,
        input_shape=input_shape,
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True
    ),
    validation_steps=len(val_filenames) / batch_size,
    callbacks=[checkpointer, reduce_lr, early_stopping, tensorboard],
    workers=10
)
