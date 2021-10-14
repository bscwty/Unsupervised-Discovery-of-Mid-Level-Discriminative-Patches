from genericpath import exists
import joblib
import numpy as np

from cuml.cluster import KMeans
from sklearn.svm import LinearSVC as SVC

from tqdm import tqdm
from utils import load_models, load_patches, makedirs

class DiscriPats:
    def __init__(self, path):

        self.m = 5
        self.min = 20
        self._load_data(path)  

        self.num_cluster = int(len(self.d1) / 20)
        self.kmeans = KMeans(n_clusters=self.num_cluster)

        self.K, self.C = self._init_patches_and_classifiers()

    def _load_data(self, path):

        self.d1, self.d2 = np.load(path['d1']), np.load(path['d2'])
        self.n1, self.n2 = np.load(path['n1']), np.load(path['n2'])

    def _init_patches_and_classifiers(self):

        K = []
        self.kmeans.fit(self.d1)
        centroids, labels = self.kmeans.cluster_centers_, self.kmeans.labels_
        for i, num in enumerate(np.bincount(labels)):
            if num > self.min:
                idx = np.argwhere(labels == i).reshape(-1)
                K.append(self.d1[idx, :])
        C = []
        print('------ kmeans intialization done! ------')
        return K, C

    def _detect_top(self, C, d, m):
        idx = np.argsort(C.decision_function(d))[::-1][:m]
        return d[idx, :]

    def get_KandC(self):
        return self.K, self.C

    def _detect_firings(self, C, d):

        dec = C.decision_function(d)
        num_firings = np.sum(dec > -1)
        if num_firings <= self.min:
            return None
        else:
            idx = np.argsort(dec)[::-1][:min(self.min, num_firings)]
            return d[idx, :]

    def _cosine_correlation(self, X_pos):

        X_neg = self.n1.copy()

        corr = np.matmul(X_pos, X_neg.T)
        norm1 = np.linalg.norm(X_pos, axis=1, keepdims=True)
        norm2 = np.linalg.norm(X_neg, axis=1, keepdims=True)

        norm = np.matmul(norm1, norm2.T)
        corr = corr / norm
        x = np.sum(corr > 0.8, axis=0)
        X_neg = X_neg[np.where(x==0)]      

        return X_neg        
        
    def iter(self):

        K_new = []
        C_new = []

        print('number of attributes: ', len(self.K))
        for X in tqdm(self.K):

            y_pos = np.ones(len(X), dtype=int)
            X_neg = self._cosine_correlation(X)
            y_neg = -1 * np.ones(len(X_neg), dtype=int)

            y = np.hstack((y_pos, y_neg))
            X = np.vstack((X, X_neg))

            svc = SVC(C=0.1)
            svc.fit(X, y)
            detect = self._detect_firings(svc, self.d2)

            if detect is None:
                continue
            else:
                K_new.append(detect)
                C_new.append(svc)

        self.K = K_new
        self.C = C_new

        self._swap()

    def _swap(self):
        self.d1, self.d2 = self.d2, self.d1
        self.n1, self.n2 = self.n2, self.n1

    def save(self, root, trained_domain):

        centroids_save_folder = root + trained_domain + '/'
        models_save_folder = root + trained_domain + '/models/'

        makedirs(centroids_save_folder, models_save_folder)

        centroids = []
        for i, k in enumerate(self.K):
            centroids.append(np.mean(k, axis=0))
        centroids = np.array(centroids)    
        np.save(centroids_save_folder + 'centroids.npy', centroids)

        # save model
        for i, c in enumerate(self.C):
            joblib.dump(c, models_save_folder + 'svc_%d.m'%i)

def select_top(top: int, lamda: int, path, models_save_folder, patches_save_folder,\
                select_models_save_folder, select_patches_save_folder):

    # select and save new models
    C = load_models(models_save_folder)
    K = load_patches(patches_save_folder)

    d1, d2 = np.load(path['d1']), np.load(path['d2'])
    n1, n2 = np.load(path['n1']), np.load(path['n2'])

    d = np.vstack((d1, d2))
    n = np.vstack((n1, n2))

    score = []

    for i in tqdm(range(len(C))):
        pur = C[i].decision_function(d) > -1
        pur = np.mean(C[i].decision_function(d)[pur])
        #pur = np.sum(np.sort(C[i].decision_function(d))[::-1][:10])

        dis = np.sum((C[i].decision_function(d) > -1))
        dis = dis / (dis + np.sum(C[i].decision_function(n) > -1))
        score.append(pur + dis)
        print(pur, dis, score[-1])

    idx = np.argsort(np.array(score))[::-1][:top]

    K_out = [K[i] for i in idx]
    C_out = [C[i] for i in idx]

    for i, k in enumerate(K_out):
        np.save(select_patches_save_folder + 'patches_%d.npy'%i, k)

    for i, c in enumerate(C_out):
        joblib.dump(c, select_models_save_folder + 'svc_%d.m'%i)

if __name__ == '__main__':

    # =====train svm and patches=====
    # epochs = 6
    # discri_pats = DiscriPats(path)

    # for i in range(epochs):
    #     discri_pats.iter()

    # discri_pats.save(patches_save_file, models_save_file)

    # =====select patches=====
    top = 300
    lamda = 10
    select_top(top, lamda)

