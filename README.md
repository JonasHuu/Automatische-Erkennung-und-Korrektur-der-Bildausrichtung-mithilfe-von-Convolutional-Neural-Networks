# Bachelorarbeit-Jonas-Huurdeman
Automatische Erkennung und Korrektur der Bildausrichtung mithilfe von Convolutional Neural Networks

## Load required modules
pip install -r requirements.txt

# The following commands can also be used to train ResNetV2 and DenseNet201
## Train EfficientnetV2 with google street view images
cd Train_street
python3 ./code/train_EfficientnetV2_street_view.py
## Train EfficientnetV2 with Open Images
cd Train_OI
### Download test and train images using 25 processes -> lower/larger number may be used depending on your hardware
python3 ./code/downloader.py ./code/train.txt --download_folder=./data/train --num_processes=25
python3 ./code/downloader.py ./code/validate.txt --download_folder=./data/train --num_processes=25
### Start the training
python3 ./code/train_efficientnetV2_open_images.py

## Download completely trained EfficientnetV2 

# Use the app 
copy efficientnetv2_sv_open_images.hdf5 to ./app/model
cd ./app
python3 ./GUI.py
