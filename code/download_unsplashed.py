from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import numpy as np
import pandas as pd
import glob
import urllib.request 
# parse the .tsv file
path = './unsplashed/'
documents = ['photos', 'keywords', 'collections', 'conversions', 'colors']
datasets = {}

for doc in documents:
  files = glob.glob(path + doc + ".tsv*")

  subsets = []
  for filename in files:
    df = pd.read_csv(filename, sep='\t', header=0)
    subsets.append(df)

  datasets[doc] = pd.concat(subsets, axis=0, ignore_index=True)

# get the photo urls
photos = datasets['photos']
urls = photos['photo_image_url']
fns = ["./unsplashed/images/" + str(photo['photo_id']) + ".png" for id in photos['photo_id']]
inputs = zip(urls, fns)

def download_url(args):
    urllib.request.urlretrieve(args[0] + "?ixid=2yJhcHBfaWQiOjEyMDd9&&fm=jpg&w=400&fit=max", args[1])

def download_parallel(args):
    cpus = cpu_count()
    results = ThreadPool(cpus - 1).imap_unordered(download_url, args)
    for result in results:
        print('url:', result[0], 'time (s):', result[1])

download_parallel(inputs)