from __future__ import print_function

import os
import sys
import matplotlib.pyplot as plt

import statistics
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from matplotlib import pyplot as plt

from utils import display_examples, RotNetDataGenerator, angle_error


# Load filenames
testdata = os.path.join('data','test','unsplashed')
if os.path.exists(testdata):
    test_filenames = os.listdir(testdata)
else:
    test_examples_path = os.path.join('data','test','unsplashed')
    test_filenames = [
        os.path.join(testdata, f) for f in os.listdir(testdata)
    ]
test_filenames = ['./data/test/unsplashed/' + i for i in test_filenames]

# Load model
model_location = os.path.join('models', 'rotnet_open_images_resnet50_TCML_2.hdf5')
model = load_model(model_location, custom_objects={'angle_error': angle_error})
angle_differences = []

for i in range(1,10):
    num_images = 5
    angle_difference = display_examples(
        model, 
        test_filenames,
        num_images=num_images,
        size=(224, 224),
        crop_center=True,
        crop_largest_rect=True,
        preprocess_func=preprocess_input,
        save_path='./visual_evaluation/' + str(i)
    )
    angle_differences.extend(angle_difference)
    plt.close()

# calculate the median error

print(angle_differences)
median = statistics.median(angle_differences)
print("The median is: {}".format(median))
plt.close()
plt.figure()
# Plot the histogram
plt.hist(angle_differences, bins=30)
plt.title('Histogram of error angles')
plt.ylabel("frequency")
plt.xlabel("error angle")
# Save the histogram
plt.savefig('error_hist.png')