# Bachelorarbeit-Automatische Erkennung und Korrektur der Bildausrichtung mithilfe von Convolutional Neural Networks

Automatic detection and correction of image orientation using Convolutional Neural Networks (CNNs) <br>

The following description explains how to train the CNN EfficientnetV2, evaluate it and also how to use the implemented app.
There are more scripts that were created in order to e.g. convert .csv files to .txt files and download test files but those are not necessary for testing the code.
The usage of them will be explained below. <br>
Note: Some of the Code used for training the CNN architectures is taken partially from Daniel Saez GitHub repository https://github.com/d4nst/RotNet and was only slightly modified for my own purposes.
### Load required modules
* `pip install -r requirements.txt`

## The following commands can also be used to train ResNetV2 and DenseNet201
### Train EfficientnetV2 with google street view images
* `cd Train_street` <br>
* python ./code/train_EfficientnetV2_street_view.py
### Train EfficientnetV2 with Open Images
* `cd Train_OI`
#### Download test and train images using 50 processes -> lower/larger number may be used depending on your hardware
* `python ./code/downloader.py ./code/train.txt --download_folder=./data/train --num_processes=50`
* `python ./code/downloader.py ./code/validate.txt --download_folder=./data/validate --num_processes=50`
#### Start the training
* Copy efficientnetv2_street_view.hdf5 to ./models
* The model can be downloaded using this link: https://drive.google.com/file/d/1q_pfYORJG_jsozsPf9a-v0l74quHa6hN/view?usp=sharing
* `python ./code/train_efficientnetV2_open_images.py`

### Download pretrained EfficientnetV2
* https://drive.google.com/file/d/1VlRecLAzn3R9CUO3k2ArP_ERkihXuaOj/view?usp=sharing <br>
### Download pretrained DenseNet201
* https://drive.google.com/file/d/10VlI_1dShRIOY77EiWAVKMxX8QbrsBzM/view?usp=sharing
### Download pretrained ResNet50V2
* https://drive.google.com/file/d/1AY6WEFamnlGBoQBAwFwpjpdyxbtGSeh-/view?usp=sharing
### Evaluate EfficientnetV2
* copy efficientnetv2_sv_open_images.hdf5 to ./models
* Replace '/path/to/test/images' to actual path, test_num to the number of test images and 'EfficientnetV2_sv_oi_hc' to the name you want
your output files names to start with:
* `python .\code\test_model.py -m './models/efficientnetv2_sv_open_images.hdf5' -t '/path/to/test/images' -n test_num -o 'EfficientnetV2_sv_oi_hc'`

## Use the Image Editor app
* copy efficientnetv2_sv_open_images.hdf5 to ./ImageEditor/model
* `cd ./ImageEditor`
* `python ImageEditor.py`

## Download and use the Image Editor app as a .exe
* https://drive.google.com/file/d/1QwOQh38SKznwh9v2xghUYGyjUeNok952/view?usp=sharing <br>
After the download is done, one just needs to execute the installer and after it's done, the app can be run.
## Further scripts
#### csv_to_txt.py
* Given the file 'train-images-boxable-with-rotation.csv', it extracts the ImageID and Subset and writes those in a .txt file so the downloader.py script
can use it to download the images by ID. The paths are hardcoded.
#### visual_evaluation.py
* This script has been used to create visual evaluations of a model using testdata. In order to use it for own purposes one needs to modify the paths used in the script.
#### download_unsplashed.py
* The script downloads 1000 random unsplash images from the .tsv file using 32 parallel processes. It then writes them into ./data/unsplashed_images/ <br>
The images can be used for testing purposes
#### detect_oblique_images.py
* This script detects all images in the folder './data/train', that are oblique and writes their IDs into a text file.
#### remove_rotated_images.py
* This script prompts the user to enter the path to a text file containing image IDs and removes all of those images. 
