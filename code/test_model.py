from __future__ import print_function
import os
import cv2
import numpy as np
import argparse
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from matplotlib import pyplot as plt
from utils import  angle_error, angle_difference, generate_rotated_image
from keras.applications.imagenet_utils import preprocess_input

def evaluate_predictions(model, input, num_images=5, size=(224, 224), crop_center=True,
                     crop_largest_rect=True, preprocess_func=preprocess_input, save_path=None):
    """
    Given a model that predicts the rotation angle of an image,
    and a NumPy array of images or a list of image paths, display
    the specified number of example images in three columns:
    Original, Rotated and Corrected.
    """
    
    if isinstance(input, (np.ndarray)):
        images = input
        N, h, w = images.shape[:3]
        if not size:
            size = (h, w)
        indexes = np.arange(0,N)
        images = images[indexes, ...]
    else:
        images = []
        filenames = input
        N = len(filenames)
        for i in range(0,N):
            image = cv2.imread(filenames[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        images = np.asarray(images)
    x = []
    y = []
    for image in images:
        rotation_angle = np.random.randint(360)
        rotated_image = generate_rotated_image(
            image,
            rotation_angle,
            size=size,
            crop_center=crop_center,
            crop_largest_rect=crop_largest_rect
        )
        x.append(rotated_image)
        y.append(rotation_angle)
    x = np.asarray(x, dtype='float32')
    y = np.asarray(y, dtype='float32')

    if x.ndim == 3:
        x = np.expand_dims(x, axis=3)

    y = to_categorical(y, 360)

    x_rot = np.copy(x)

    if preprocess_func:
        x = preprocess_func(x)

    y = np.argmax(y, axis=1)
    y_pred = np.argmax(model.predict(x), axis=1)
    corrected_angles = []
    for true_angle, predicted_angle in zip(y, y_pred):
        corrected_angle = angle_difference(predicted_angle, true_angle)
        corrected_angles.append(corrected_angle)
    # get the mean test angle error
    mean_error = np.mean(corrected_angles)
    # get the median test angle error
    median_error = np.median(corrected_angles)
    #print(corrected_angles)
    return [mean_error,median_error,corrected_angles]
# create a parser
def create_parser():
    '''
    Creates an argument parser to test different models

    '''
    # Make parser object
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('-m', '--model',
                   help="filepath to the model")
    p.add_argument('-t', '--test-folder',
                   help="the folder where the test images are located")
    p.add_argument('-n', '--num-test',
                   help='number of images to test')
    p.add_argument('-o', help='Output evaluation file name')
    return(p.parse_args())

def main():
    print(args)
    image_paths = os.listdir(args.test_folder)
    prefix = args.test_folder + '/'
    image_paths = [prefix + i for i in image_paths]
    # choose n images - not randomly for reproducability for testing different models
    # on the same data
    image_paths = image_paths[0:int(args.num_test)]
    model = load_model(args.model, custom_objects={'angle_error': angle_error})
    evaluation = evaluate_predictions(model=model, input=image_paths, num_images=args.num_test, size=(224,224),
                      preprocess_func=preprocess_input)
    with open(args.o + '_evaluation.txt', 'w') as f:
        f.write('The mean error is: {}\n'.format(evaluation[0]))
        f.write('Test median error is: {}'.format(evaluation[1]))
        
    
    plt.figure()
    # Plot the histogram of angle errors
    plt.hist(evaluation[2], bins=30)
    plt.title('Histogram of error angles')
    plt.ylabel("frequency")
    plt.xlabel("error angle")
    # Save the histogram
    plt.savefig(args.o + '_evaluation.png')
    
    return

if __name__ == "__main__":
    try:
        args = create_parser()
        main()
    except:
        print('Try:  python .\code\test_model.py -m ./models/rotnet_open_images_resnet50_TCML_2.hdf5 -t ./data/test/humancentric -n 100 -o ResNet50')