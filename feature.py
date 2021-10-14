import cv2
import numpy as np

import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder, VOCDetection

from tqdm import tqdm
from utils import hog, gen_patches, makedirs, exist
from models import Model

class ToArray:
    def __init__(self) -> None:
        return

    def __call__(self, img) -> np.array:
        im = np.array(img)
        return im

class Dataset:

    def __init__(self, d_folder, n_folder, batch_size=1):

        ######transofrm######
        self.d_folder = d_folder
        self.n_folder = n_folder

        self.transforms = T.Compose([T.Resize(256),
                                    ToArray()]) # hog
        self.batch_size = batch_size

        # {'d1': , 'd2', 'n1', 'n2'}
        self.dset_dict = self._load_data()
            
    def _load_data(self):

        # d_dset refers to the object dataset
        # n_dset refers to the natural world dataset

        d_dset = ImageFolder(self.d_folder, transform=self.transforms)
        #d_dset = VOCDetection(d_folder, transform=self.transforms, year='2007', image_set='train', ) # VOC
        n_dset = ImageFolder(self.n_folder, transform=self.transforms)

        # split the datasets for cross validation
        d1, d2 = random_split(d_dset, [int(0.5*len(d_dset)),len(d_dset)-int(0.5*len(d_dset))])
        n1, n2 = random_split(n_dset, [int(0.5*len(n_dset)),len(n_dset)-int(0.5*len(n_dset))])

        d1, d2 = DataLoader(d1, batch_size=self.batch_size), DataLoader(d2, batch_size=self.batch_size)
        n1, n2 = DataLoader(n1, batch_size=self.batch_size), DataLoader(n2, batch_size=self.batch_size)

        # apply dataloader
        # use dict
        dset_names = ['d1', 'd2', 'n1', 'n2'] 
        dset_dict = dict(zip(dset_names, [d1, d2, n1, n2]))

        return dset_dict

    def display_data(self):

        d_dset = ImageFolder(self.d_folder, transform=self.transforms)
        data_loader = DataLoader(d_dset, batch_size=self.batch_size)

        return data_loader
        
##############################
######feature extraction######
##############################
class FeatureExtraction:
    def __init__(self, dataset: Dataset, root: str, trained_domain: str):
        self.dset = dataset.dset_dict

        # hyper parameter setting
        self.max_patches_per_layer = 50
        self.window_size = (80, 80)
        self.max_pyramid_level = 4
        self.overlap_threshold = 0.1

        # path
        self.path = self._get_feature_path(root, trained_domain)

        # hog
        self.hog = hog('hog.xml')

        # CNN
        self.model = Model('resnet18')

    def _get_feature_path(self, root, trained_domain):

        d_folder = root + trained_domain + '/features/'
        n_folder = root

        makedirs(d_folder, n_folder)

        path = dict()
        path['d1'] = d_folder + 'd1_features.npy'
        path['d2'] = d_folder + 'd2_features.npy'
        path['n1'] = d_folder + 'd2_features.npy'
        path['n2'] = d_folder + 'd1_features.npy'
        return path

    def get_features(self):

        for dname, dset in self.dset.items():

            if exist(self.path[dname]):
                continue

            features = []
            for img, _ in tqdm(dset):
                feats = self._CNN_feature_extraction(img)
                if feats.shape[0] == 0:
                    continue
                features.append(feats)
            features = np.vstack(features)

            np.save(self.path[dname], features) 
            print('feature shape of %s: '%dname, features.shape)

    def _CNN_feature_extraction(self, img):

        im = img[0].numpy()
        features = []

        for i in range(self.max_pyramid_level):

            if im.shape[0] < self.window_size[0] or im.shape[1] < self.window_size[1]:
                break
            else:
                patches, _ = gen_patches(im, self.window_size[0], self.max_patches_per_layer, self.overlap_threshold)
                for pat in patches:
                    feat = self.model(pat)
                    features.append(feat)
                im = cv2.pyrDown(im)
        return np.array(features)
