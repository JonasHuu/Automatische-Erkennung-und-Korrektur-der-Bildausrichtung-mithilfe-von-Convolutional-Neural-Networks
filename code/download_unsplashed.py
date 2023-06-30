import numpy as np
import pandas as pd

df = pd.read_csv('./code/photos.tsv000', sep='\t', header=0)
photos = df
urls = photos['photo_image_url']

import urllib.request
indexes = np.random.choice(24000, 1000)
for i in indexes:
    urllib.request.urlretrieve(urls[i] + "?ixid=2yJhcHBfaWQiOjEyMDd9&&fm=jpg&w=400&fit=max", "./data/unsplashed_images/" + str(i) + ".png")
  