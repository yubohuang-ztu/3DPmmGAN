import tifffile
import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', help='path to image')
parser.add_argument('--name', default='', help='name of dataset')
parser.add_argument('--edgelength', type=int, default=96, help='input batch size')
parser.add_argument('--stride', type=int, default=16, help='the sampling interval of the original sample')
parser.add_argument('--target_dir', default='', help='path to store training images')
parser.add_argument('--enhance', default=False, help='flipping and rotating (Anisotropy needs to be considered)')
opt = parser.parse_args()
print(opt)

img0 = tifffile.imread(str(opt.image))
if opt.enhance==True:
	img1 = img0.transpose(1, 0, 2)
	img2 = img0.transpose(1, 2, 0)
	img3 = img0.transpose(2, 0, 1)
	img4 = img0.transpose(2, 1, 0)
	img5 = img0[::-1]
	img6 = img1[::-1]
	img7 = img2[::-1]
	img8 = img3[::-1]
	img9 = img4[::-1]
	all = [img0, img1, img2, img3, img4, img5, img6, img7, img8, img9]
else:
	all = [img0]

count = 0
for x in range(len(all)):
	image = all[x]
	for i in range(0, img0.shape[0], opt.stride):
		for j in range(0, img0.shape[1], opt.stride):
			for k in range(0, img0.shape[2], opt.stride):
				subset = image[i:i + opt.edgelength, j:j + opt.edgelength, k:k + opt.edgelength]
				if subset.shape == (opt.edgelength, opt.edgelength, opt.edgelength):
					f = h5py.File(str(opt.target_dir) + "/" + str(opt.name) + "_" + str(count) + ".hdf5", "w")
					f.create_dataset('data', data=subset, dtype="i8", compression="gzip")
					f.close()
					count += 1
print(count)
