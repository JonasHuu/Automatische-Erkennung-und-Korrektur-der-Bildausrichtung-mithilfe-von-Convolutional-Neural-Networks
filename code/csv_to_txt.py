import pandas as pd
import random

df = pd.read_csv('train-images-boxable-with-rotation.csv')
ImageID = df["ImageID"]
Subset = df["Subset"]
file = open("train.txt", "w")
#ran_nums = random.sample(range(1,100000), 1500)

for i in range(1,1000000):
    file.write(Subset[i] + '/' +  ImageID[i] + "\n")

file.close()

