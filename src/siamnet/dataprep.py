"""Script to preprocess the omniglot dataset and pickle it into an array that's easy
    to index my character type"""

import numpy as np
import imageio
import os
from zipfile import ZipFile
import pickle


def loadimgs(path,file,n=0):
    #if data not already unzipped, unzip it.
    filepath = os.path.join(path,file)
    if not os.path.exists(filepath):
        print("unzipping")
        _unzip(path,filepath)
    imgs = []
    #we load every alphabet seperately so we can isolate them later
    for alphabet in os.listdir(filepath):
        if not alphabet[0] =='.':
            print("loading alphabet: " + alphabet)
            alphabet_path = os.path.join(filepath,alphabet)
            #every letter/category has it's own column in the array, so  load seperately
            for letter in os.listdir(alphabet_path):
                category_images=[]
                letter_path = os.path.join(alphabet_path, letter)
                for filename in os.listdir(letter_path):
                    image_path = os.path.join(letter_path, filename)
                    image = imageio.imread(image_path)
                    category_images.append(image)
                try:
                    imgs.append(np.stack(category_images))
                #edge case  - last one
                except ValueError as e:
                    print(e)
                    print("error - category_images:", category_images)
    imgs = np.stack(imgs)
    return imgs


def _unzip(path, filepath):
    os.system("unzip {}".format(path+".zip" ))
    with ZipFile('{}.zip'.format(filepath), 'r') as zip_ref:
        zip_ref.extractall('{}'.format(path))


def saveimgs(imgs,save_path,filename):
    with open(os.path.join(save_path,filename), "wb") as file:
        pickle.dump(imgs,file)
