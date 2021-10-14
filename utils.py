from genericpath import exists
import os
import cv2
import joblib
import numpy as np

def hog(file):
    h = cv2.HOGDescriptor()
    h.load(file)
    return h

def load_models(folder):
    C = []
    files = os.listdir(folder)
    for f in files:
        svc = joblib.load(folder + f)
        C.append(svc)
    return C

def load_patches(folder):
    K = []
    files = os.listdir(folder)
    for f in files:
        pat = np.load(folder + f)
        K.append(pat)
    return K

def remove_low_energy(im, indices, window):

    im = cv2.cvtColor(im.copy(), cv2.COLOR_RGB2GRAY)
    gradient_Y, gradient_X = np.gradient(im)

    energy = []
    for y, x in indices:
        gradienty = gradient_Y[y: y+window, x: x+window]
        gradientx = gradient_X[y: y+window, x: x+window]
        energy.append(np.mean(np.sqrt(gradienty**2 + gradientx**2)))
    energy = np.array(energy)

    indices = indices[np.where(energy > np.mean(energy))]

    if len(indices) == 0:
        return np.array([[0, 0]])

    return indices

def remove_overlap(im, indices, window, overlap_threshold):


    indices = np.array(sorted(indices, key=lambda x:x[0]))
    H, W = im.shape[0], im.shape[1]

    i = 0
    while i <= len(indices)-1:
        y1, x1 = indices[i]
        y_limit = min(y1 + window, H)
        to_delete = []

        j = i
        while True:
            j = j + 1
            if j == len(indices):
                break

            y2, x2 = indices[j]
            iou = (y1+window-y2) * (x1+window-x2) if x1 < x2 \
                    else (y1+window-y2) * (x2+window-x1)
            iou = 1 if 2*window**2 - iou == 0 \
                    else iou / (2*window**2 - iou)

            if iou >= overlap_threshold:
                to_delete.append(j)

            if y2 >= y_limit:
                break

        if i+1 < j:
            indices = np.delete(indices, to_delete, 0)
        i += 1

    return np.array(indices)

def gen_patches(im, window, max_patches_per_layer, overlap_threshold):

    # generate sampled indices
    H, W = im.shape[0] - window, im.shape[1] - window
    if H == 0 or W == 0:
        return np.array([im[0: window, 0: window]]), np.array([[0, 0]])
    else:
        max_patches = min(max_patches_per_layer, H*W)
        Y, X = np.random.randint(low=0, high=H, size=max_patches), np.random.randint(low=0, high=W, size=max_patches)
        indices = np.array(list(zip(Y, X)))

    # remove low energy
    indices = remove_low_energy(im, indices, window)

    # remove overlap
    indices = remove_overlap(im, indices, window, overlap_threshold)

    # crop the patches according to indices
    patches = [im[y: y+window, x: x+window] for y, x in indices]

    return np.array(patches), indices

def makedirs(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def exist(*files):
    exists = True
    for f in files:
        exists = exists and os.path.exists(f)
    return exists
    