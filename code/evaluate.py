from __future__ import print_function

import os
import sys
import matplotlib.pyplot as plt

from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model

from utils import display_examples, RotNetDataGenerator, angle_error

# Load filenames
testdata = os.path.join('data', 'images')
if os.path.exists(testdata):
    test_filenames = os.listdir(testdata)
else:
    test_examples_path = os.path.join('data', 'images')
    test_filenames = [
        os.path.join(testdata, f) for f in os.listdir(testdata)
    ]
test_filenames = ['./data/images/' + i for i in test_filenames]
print(test_filenames)

# Load model
model_location = os.path.join('models', 'rotnet_open_images_resnet50_TCML_2.hdf5')
model = load_model(model_location, custom_objects={'angle_error': angle_error})

# Evaluate model
batch_size = 64
out = model.evaluate(
    RotNetDataGenerator(
        test_filenames,
        input_shape=(224, 224, 3),
        batch_size=batch_size,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True,
        shuffle=True
    ),
    steps=len(test_filenames) / batch_size
)

print('Test loss:', out[0])
print('Test angle error:', out[1])
with open('evaluation_unsplashed.txt','w') as f:
    f.write('Test loss: {}'.format(out[0]))
    f.write('Test angle error: {}'.format(out[1]))

'''
predictions = model.predict(
        RotNetDataGenerator(
            image_paths,
            input_shape=(224, 224, 3),
            batch_size=64,
            one_hot=True,
            preprocess_func=preprocess_input,
            rotate=False,
            crop_largest_rect=True,
            crop_center=True
        )
    )
'''