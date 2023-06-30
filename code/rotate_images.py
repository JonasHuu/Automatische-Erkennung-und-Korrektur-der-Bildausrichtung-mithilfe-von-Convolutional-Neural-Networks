import utils
import os
import sys
import cv2
from PIL import Image

import matplotlib.pyplot as plt
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
from utils import display_examples, RotNetDataGenerator, angle_error, generate_rotated_image
# Load filenames
testdata = os.path.join('data','test','Pascal\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages')
#testdata = os.path.join('data','test','humancentric')
if os.path.exists(testdata):
    test_filenames = os.listdir(testdata)
else:
    test_examples_path = os.path.join('data','test','Pascal\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages')
    test_filenames = [
        os.path.join(testdata, f) for f in os.listdir(testdata)
    ]
test_filenames = ['./data/test/Pascal\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages/' + i for i in test_filenames]
#test_filenames = ['./data/test/humancentric/' + i for i in test_filenames]
# load files
images = []
filenames = test_filenames
N = len(filenames)
canonical_angles = np.array([0,90,180,270])
indexes = np.random.choice(N, 200)

#for i in indexes:
#    image = cv2.imread(filenames[i])
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    images.append(image)
#images = np.asarray(images)
print(filenames[1])
for i in indexes:
    image = cv2.imread(filenames[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rotation_angle = np.random.choice(canonical_angles)
    rot_im = generate_rotated_image(image,rotation_angle,crop_center=True,crop_largest_rect=True)
    im = Image.fromarray(rot_im)
    #im.save("./data/test/rotated_images_us/" + filenames[i].split('\\')[4].split('/')[1])
    im.save("./data/test/rotated_images_PC/" + filenames[i].split('/')[4])