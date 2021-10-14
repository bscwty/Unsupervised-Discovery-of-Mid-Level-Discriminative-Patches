import os
import cv2
import numpy as np
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

from feature import Dataset
from utils import load_models, gen_patches, hog
from models import Model

plt.cm.get_cmap('gist_rainbow')


class Display:
    def __init__(self, dataset:Dataset, model_path:str):

        self.dset = dataset.display_data()

        self.max_patches_per_layer = 150
        self.window_size = (80, 80)
        self.max_pyramid_level = 4
        self.overlap_threshold = 0.1

        self.hog = hog('hog.xml')
        self.model = Model('resnet18')

        self.C = load_models(model_path)

    def display_raw(self, img, output_file='./output.png'):

        img = T.Resize(256)(Image.open(img))
        im = np.array(img)
        plt.imshow(img)
        ax = plt.gca()
        window = self.window_size[0]
        for i in range(self.max_pyramid_level):
            if im.shape[0] < self.window_size[0] or im.shape[1] < self.window_size[1]:
                break
            else:
                _, index = gen_patches(im, window, self.max_patches_per_layer, self.overlap_threshold)
                if len(index) == 0:
                    break
                for idx in index:
                    idx = 2**i * idx
                    idx = [idx[1], idx[0]]
                    ax.add_patch(plt.Rectangle(idx, window*2**i, window*2**i, color="red", fill=False, linewidth=1.5))
                im = cv2.pyrDown(im)
        plt.savefig(output_file)

    def display_one_img(self, img, line, output_file='./output.png'):

        img = T.Resize(256)(Image.open(img))
        im = np.array(img)

        # plt.clf()
        # plt.imshow(img)
        # ax = plt.gca()

        C = self.C
        window = self.window_size[0]
        for i in range(self.max_pyramid_level):
            if im.shape[0] < self.window_size[0] or im.shape[1] < self.window_size[1]:
                break
            else:
                patches, index = gen_patches(im, window, self.max_patches_per_layer, self.overlap_threshold)
                if len(index) == 0:
                    break
                for pat, idx in zip(patches, index):
                    feat = self.model(pat).reshape(1, -1) # CNN
                    for j, c in enumerate(C):
                        if c.decision_function(feat) > -1:
                            line[j] += 1
                            idx = 2**i * idx
                            idx = [idx[1], idx[0]]
                            # ax.add_patch(plt.Rectangle(idx, window*2**i, window*2**i, color='red', fill=False, linewidth=1.5, label=str(j)))
                im = cv2.pyrDown(im)
        # plt.legend()
        # plt.savefig(output_file)
        return line

    def display_net_models(self, model_path, save_folder, id=0):
       
        # id is the index of attribute going to be displayed
        C = load_models(model_path)
        C = C[id: id + 1]

        flag = 0

        # img is copied to plot
        # im is transformed to calculate
        for img, _ in tqdm(self.dset):

            img = img[0].numpy()
            im = img.copy()

            window = self.window_size[0]

            for level in range(self.max_pyramid_level):
                if im.shape[0] < self.window_size[0] or im.shape[1] < self.window_size[1]:
                    break
                else:
                    patches, index = gen_patches(im, window, self.max_patches_per_layer, self.overlap_threshold)
                    if len(index) == 0:
                        break
                    for pat, idx in zip(patches, index):
                        feature = self.model(pat).reshape(1, -1)
                        idx = 2**level * idx
                        for j, c in enumerate(C):
                            cluster_id = j + id
                            if c.decision_function(feature) > -1:
                                img_tosave = Image.fromarray(img[idx[0]:idx[0]+window*2**level, idx[1]:idx[1]+window*2**level])
                                cluster_folder = save_folder + "cluster%d/"%cluster_id

                                if not os.path.exists(cluster_folder):
                                    os.makedirs(cluster_folder)

                                img_tosave.save(cluster_folder + "%d.png"%flag)                        
                        
                        flag += 1           
                    im = cv2.pyrDown(im)
