from feature import Dataset, FeatureExtraction
from clustersvm import DiscriPats, select_top

root = r'/mnt/data1/output/midpatch/officehome/'
trained_domain = r'clipart'

#d_folder = r'/mnt/data1/OSDADataset/Office-31/webcam/images' # office31

# d_folder = r'/mnt/data1/OSDADataset/OfficeHome/Real World' # officehome
# d_folder = r'/mnt/data1/OSDADataset/OfficeHome/Product' # officehome
d_folder = r'/mnt/data1/OSDADataset/OfficeHome/Clipart' # officehome
# d_folder = r'/mnt/data1/OSDADataset/OfficeHome/Art' # officehome

n_folder = r'/mnt/data1/OSDADataset/Flicker8k'


if __name__ == '__main__':

    ##############################
    ####  feature extraction  ####
    ##############################

    dsets = Dataset(d_folder, n_folder)

    feat = FeatureExtraction(dsets, root, trained_domain)
    feat.get_features()
    
    path = feat.path

    print('-- features extraction completed! --')

    ##############################
    #########  training  #########
    ##############################

    print('---------- start training! ----------')

    epochs = 6
    discri_pats = DiscriPats(path)

    for i in range(epochs):
        discri_pats.iter()

    discri_pats.save(root, trained_domain)

    print('-------- training completed! --------')

    ##############################
    ###########  eval  ###########
    ##############################

    print('---------- start selection! ----------')
    # sel_pats_save_folder = root + r'/select_patches/'
    # sel_models_save_folder = root + r'/select_models/'

    # top = 300
    # lamda = 10

    # select_top(top, lamda, path, models_save_folder, pats_save_folder, 
    #             sel_models_save_folder, sel_pats_save_folder)
