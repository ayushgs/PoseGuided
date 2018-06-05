import numpy as np
from PIL import Image
import os
from skimage import data
from skimage.color import rgb2gray
from skimage.morphology import disk
from skimage.filters.rank import gradient


mask_path = '../test_results/mask_target/'
masks = []
for filename in os.listdir(mask_path):
	img = Image.open(mask_path+filename,'r')
	img = np.array(img).astype(np.float32)
	img /= 255.0
	masks.append(np.expand_dims(img,axis=0))
masks = np.array(masks)
print masks.shape

G1_path = '../test_results/G1/'
G1_images = []
for filename in os.listdir(G1_path):
	img = Image.open(G1_path+filename,'r')
	img = np.array(img).astype(np.float32) #HWC
	img = np.transpose(img,[2,0,1])
	img -= 127.5
	img /= 127.5
	G1_images.append(img)

G1_images = np.array(G1_images)
print G1_images.shape

G2_path = '../test_results/G2/'
G2_images = []
for filename in os.listdir(G2_path):
	img = Image.open(G2_path+filename,'r')
	img = np.array(img).astype(np.float32) #HWC
	img = np.transpose(img,[2,0,1])
	img -= 127.5
	img /= 127.5
	G2_images.append(img)

G2_images = np.array(G2_images)
print G2_images.shape

target_path = '../test_results/x_target/'
target_images = []
for filename in os.listdir(target_path):
	img = Image.open(target_path+filename,'r')
	img = np.array(img).astype(np.float32) #HWC
	img = np.transpose(img,[2,0,1])
	img -= 127.5
	img /= 127.5
	target_images.append(img)

target_images = np.array(target_images).astype(np.float32)
print target_images.shape

##calculate the pose mask loss
num_images = masks.shape[0]

G1_loss = np.mean(abs(G1_images-target_images)*(1+masks))

G2_loss = np.mean((G2_images-target_images)*(1+masks))

print G1_loss,G2_loss

G1s = 0.0
G2s = 0.0
tars = 0.0
for i in xrange(num_images):
	G1img = rgb2gray(G1_images[i].transpose(1,2,0))
	G2img = rgb2gray(G2_images[i].transpose(1,2,0))
	tarimg = rgb2gray(target_images[i].transpose(1,2,0))
	selection_element = disk(5)
	G1sharp = gradient(G1img,selection_element)
	G2sharp = gradient(G2img,selection_element)
	tarsharp = gradient(tarimg,selection_element)
	G1s += np.mean(abs(G1sharp))/num_images
	G2s += np.mean(abs(G2sharp))/num_images
	tars += np.mean(abs(tarsharp))/num_images

print "Mean absolute gradient of images: G1:", G1s,"G2:",G2s,"target:",tars

