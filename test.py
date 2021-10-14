import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from feature import Dataset
from display import Display
from utils import makedirs

root = r'/mnt/data1/output/midpatch/officehome/'

# d_folder = r'/mnt/data1/OSDADataset/Office-31/webcam/images' # office31

# d_folder = r'/mnt/data1/OSDADataset/OfficeHome/Real World' # officehome
# d_folder = r'/mnt/data1/OSDADataset/OfficeHome/Art' # officehome

def plot(display, trained_domain, test_class):
    
    domain = ['Art', 'Real World', 'Product', 'Clipart']

    for test_domain in domain:
        directory = '/mnt/data1/OSDADataset/OfficeHome/%s/%s/'%(test_domain, test_class) 
        img_names = os.listdir(directory)
        line = [0] * len(display.C)

        for img_name in tqdm(img_names):
            img_name = directory + img_name
            output_file = './output_%s/%s'%(test_class, img_name)
            line = display.display_one_img(img_name, line, output_file)

        threshold = 0# np.mean(line)
        line = [i if i > threshold else 0 for i in line]
        line = np.array(line)
        print(np.argsort(line)[::-1][:15])  # print top 15 attributes

        makedirs('./output_%s/%s'%(trained_domain, test_domain))

        plt.clf()
        plt.plot(line)
        plt.savefig('./output_%s/%s/%s.png'%(trained_domain, test_domain, test_class))
    
if __name__ == '__main__':   
 
    trained_domain = r'clipart' # the domain where svm trained
    test_class = 'Alarm_Clock' # output class of the line chart

    model_path = root + trained_domain + r'/models/'
    save_folder = root + trained_domain + r'/clusters/'

    # d_folder controls the cluster display in which domain
    # d_folder = r'/mnt/data1/OSDADataset/OfficeHome/Real World' # officehome
    d_folder = r'/mnt/data1/OSDADataset/OfficeHome/Clipart' # officehome
    n_folder = r'/mnt/data1/OSDADataset/Flicker8k' 
    dsets = Dataset(d_folder, n_folder)
    display = Display(dsets, model_path)

    # display.display_raw( r'/mnt/data1/OSDADataset/OfficeHome/Clipart/Alarm_Clock/00015.jpg')
    # line = [0]*888
    # _ = display.display_one_img(r'/mnt/data1/OSDADataset/OfficeHome/Clipart/Alarm_Clock/00015.jpg', line)
    # plot(display, trained_domain, test_class)

    display.display_net_models(model_path, save_folder, id=11)

    # display.display_raw( r'/mnt/data1/OSDADataset/OfficeHome/Real World/Alarm_Clock/00015.jpg')
