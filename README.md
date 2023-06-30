# Bachelorarbeit-Jonas-Huurdeman
Automatische Erkennung und Korrektur der Bildausrichtung mithilfe von Convolutional Neural Networks

### Load required modules
pip install -r requirements.txt

## The following commands can also be used to train ResNetV2 and DenseNet201
### Train EfficientnetV2 with google street view images
cd Train_street

python3 ./code/train_EfficientnetV2_street_view.py
### Train EfficientnetV2 with Open Images
cd Train_OI
#### Download test and train images using 25 processes -> lower/larger number may be used depending on your hardware
python3 ./code/downloader.py ./code/train.txt --download_folder=./data/train --num_processes=25
python3 ./code/downloader.py ./code/validate.txt --download_folder=./data/train --num_processes=25
#### Start the training
python3 ./code/train_efficientnetV2_open_images.py

### Download completely trained EfficientnetV2 
https://drive.google.com/file/d/1VlRecLAzn3R9CUO3k2ArP_ERkihXuaOj/view?usp=sharing

### Evaluate EfficientnetV2
Replace '/path/to/test/images' to actual path, test_num to the number of test images and 'EfficientnetV2_sv_oi_hc' to the name you want
your output files names to start with:
python .\code\test_model.py -m './models/efficientnetv2_sv_open_images.hdf5' -t '/path/to/test/images' -n test_num -o 'EfficientnetV2_sv_oi_hc'

## Use the app 
copy efficientnetv2_sv_open_images.hdf5 to ./app/model
cd ./app
python3 ./GUI.py
