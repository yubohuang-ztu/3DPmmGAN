import numpy as np
import h5py
import tifffile
from skimage.filters import threshold_otsu
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hdf5', default='input', help='path to get .hdf5')
parser.add_argument('--target_dir', default='output', help='path to store .tif')
opt = parser.parse_args()
print(opt)

def play(file):
    file_in = opt.hdf5 +'/' + file + ".hdf5"
    f = h5py.File(file_in, 'r')
    img = f['data'][()]
    img = img[np.newaxis, np.newaxis, ...].astype(np.float32)
    file_mid = opt.target_dir +'/' + file + '.tif'
    tifffile.imsave(file_mid, img)

    img_in = tifffile.imread(file_mid)
    img_otsu = threshold_otsu(img_in)
    img_out = (img_in >= img_otsu).astype(np.float32)
    file_out = opt.target_dir +'/' + file + '.tif'
    tifffile.imsave(file_out, img_out.astype(np.float32))

if __name__ == '__main__':
    names = os.listdir(opt.hdf5)
    i = 1
    for name in names:
        index = name.rfind('.')
        name = name[:index]
        play(name)
        print(len(names), '/', i)
        i = i + 1