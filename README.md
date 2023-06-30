# Bachelorarbeit-Jonas-Huurdeman
Automatische Erkennung und Korrektur der Bildausrichtung mithilfe von Convolutional Neural Networks <br>
The following description explains how to train EfficientnetV2, evaluate it and also how to use the implemented app.
There are more scripts that were created in order to e.g. convert .csv files to .txt files and download test files but are not necessary for testing the code.
How they can be used will be explained below.
### Load required modules
pip install -r requirements.txt

## The following commands can also be used to train ResNetV2 and DenseNet201
### Train EfficientnetV2 with google street view images
cd Train_street <br>
python3 ./code/train_EfficientnetV2_street_view.py
### Train EfficientnetV2 with Open Images
cd Train_OI
#### Download test and train images using 25 processes -> lower/larger number may be used depending on your hardware
python3 ./code/downloader.py ./code/train.txt --download_folder=./data/train --num_processes=25
python3 ./code/downloader.py ./code/validate.txt --download_folder=./data/train --num_processes=25
#### Start the training
python3 ./code/train_efficientnetV2_open_images.py

### Download pretrained EfficientnetV2
https://drive.google.com/file/d/1VlRecLAzn3R9CUO3k2ArP_ERkihXuaOj/view?usp=sharing <br>

### Evaluate EfficientnetV2
copy efficientnetv2_sv_open_images.hdf5 to ./models
Replace '/path/to/test/images' to actual path, test_num to the number of test images and 'EfficientnetV2_sv_oi_hc' to the name you want
your output files names to start with:
python .\code\test_model.py -m './models/efficientnetv2_sv_open_images.hdf5' -t '/path/to/test/images' -n test_num -o 'EfficientnetV2_sv_oi_hc'

## Use the app 
copy efficientnetv2_sv_open_images.hdf5 to ./app/model
cd ./app
python3 ./GUI.py

## Further scripts
#### csv_to_txt.py
Given the file 'train-images-boxable-with-rotation.csv', it extracts the ImageID and Subset and writes those in a .txt file so the downloader.py script
can use it to download the images by ID. The paths are hardcoded.
#### create_tests.py
This script has been used to create visual evaluations of a model using testdata. In order to use it for own purposes one needs to modify the paths used in the script.
#### download_unsplashed.py
The script downloads 1000 random unsplash images from the .tsv file. It then writes them into ./data/unsplashed_images/ <br>
The images can be used for testing purposes
