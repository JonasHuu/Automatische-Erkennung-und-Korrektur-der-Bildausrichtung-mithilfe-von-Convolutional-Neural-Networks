from __future__ import print_function
import os
import numpy as np
import argparse
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from matplotlib import pyplot as plt
from utils import  angle_error, angle_difference, generate_rotated_image
from keras.applications.imagenet_utils import preprocess_input
from correct_rotation import RotNetDataGenerator
def evaluate_predictions(model, input, num_images=5, size=(224, 224), crop_center=True,
                     crop_largest_rect=True, preprocess_func=preprocess_input, save_path=None):
    """
    Given a model that predicts the rotation angle of an image,
    and a NumPy array of images or a list of image paths, display
    the specified number of example images in three columns:
    Original, Rotated and Corrected.
    """
    predictions = model.predict(
        RotNetDataGenerator(
            input,
            input_shape=(224, 224, 3),
            batch_size=64,
            one_hot=True,
            preprocess_func=preprocess_input,
            rotate=False,
            crop_largest_rect=True,
            crop_center=True
        )
    )

    predicted_angles = np.argmax(predictions, axis=1)
    corrected_angles = []
    outliers = []
    for i in range(len(predicted_angles)):
        corrected_angle = angle_difference(predicted_angles[i], 0)
        corrected_angles.append(corrected_angle)
        print(corrected_angle)
        if corrected_angle > 10 and corrected_angle < 75 or corrected_angle > 100 and corrected_angle < 175:
            outliers.append(input[i])
    
    return [corrected_angles,outliers]
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
    image_paths = os.listdir('./data/train')
    prefix = args.test_folder + '/'
    image_paths = [prefix + i for i in image_paths]
    # choose n images - not randomly for reproducability for testing different models
    # on the same data
    image_paths = image_paths[0:int(args.num_test)]
    model = load_model('./models/efficientnetv2_sv_open_images.hdf5', custom_objects={'angle_error': angle_error})
    evaluation = evaluate_predictions(model=model, input=image_paths, size=(224,224),
                      preprocess_func=preprocess_input)
    with open('Cancel_images.txt', 'w') as f:
        for im in evaluation[1]:
            f.write('{}\n'.format(im))
    return

if __name__ == "__main__":
    #try:
    args = create_parser()
    main()
    #except:
    #    print('Try:  python .\code\check_quality.py -m './app/model/efficientnetv2_sv_open_images.hdf5' -t './data/unsplashed_images/' -n 1000 -o 'EfficientnetV2_sv_test'')