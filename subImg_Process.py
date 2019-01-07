import os
from options.test_options import TestOptions
# from data import CreateDataLoader
from models import create_model
import os.path
import random
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
import cv2

TILE_SIZE = 256
PADDING_SIZE = 21
LEFT_EDGE = -2
TOP_EDGE = -1
MIDDLE = 0
RIGHT_EDGE = 1
BOTTOM_EDGE = 2
def subImgCrop(imgA,imgB):
    height, width, = TILE_SIZE, TILE_SIZE
    y_stride, x_stride, = TILE_SIZE - (2 * PADDING_SIZE), TILE_SIZE - (2 * PADDING_SIZE)
    if (height > imgA.size(1)) or (width > imgA.size(2))or(imgA.size()!=imgB.size()):
        # print("Invalid crop: crop dims larger than image (%r with %r)" % (imgA.shape, tokens))
        exit(1)
    imsA = list()
    imsB = list()
    bin_ims = list()
    locations = list()
    y = 0
    y_done = False
    num=0
    while y <= imgA.size(1) and not y_done:
        x = 0
        if y + height > imgA.size(1):
            y = imgA.size(1) - height
            y_done = True
        x_done = False
        while x <= imgA.size(2) and not x_done:
            if x + width > imgA.size(2):
                x = imgA.size(2) - width
                x_done = True
            locations.append(((y, x, y + height, x + width),
                              (y + PADDING_SIZE, x + PADDING_SIZE, y + y_stride, x + x_stride),
                              TOP_EDGE if y == 0 else (BOTTOM_EDGE if y == (imgA.size(1) - height) else MIDDLE),
                              LEFT_EDGE if x == 0 else (RIGHT_EDGE if x == (imgA.size(2) - width) else MIDDLE)
                              ))
            img_a=imgA[:, y:y + height, x:x + width].numpy()
            img_b = imgB[:, y:y + height, x:x + width].numpy()
            # if num==0:
            #     imsA=img_a
            #     imsB=img_b
            # else:
            #     imsA = np.concatenate([imsA[np.newaxis,:, :, :], img_a], axis=0)
            #     imsB=np.concatenate([imsB[np.newaxis,:, :, :], img_b], axis=0)
            # imsA.append(imgA[:,y:y + height,x:x + width])
            # imsB.append(imgB[:, y:y + height, x:x + width])
            imsA.append(img_a)
            imsB.append(img_b)
            x += x_stride
            num=num+1
        y += y_stride
    torch_a = torch.from_numpy(np.array(imsA))
    torch_b = torch.from_numpy(np.array(imsB))
    return locations, torch_a,torch_b

def stich_together(locations, subwindows, size):
	output = np.zeros(size, dtype=np.float32)
	for location, subwindow in zip(locations, subwindows):
		outer_bounding_box, inner_bounding_box, y_type, x_type = location
		y_paste, x_paste, y_cut, x_cut, height_paste, width_paste = -1, -1, -1, -1, -1, -1
		#print outer_bounding_box, inner_bounding_box, y_type, x_type

		if y_type == TOP_EDGE:
			y_cut = 0
			y_paste = 0
			height_paste = TILE_SIZE - PADDING_SIZE
		elif y_type == MIDDLE:
			y_cut = PADDING_SIZE
			y_paste = inner_bounding_box[0]
			height_paste = TILE_SIZE - 2 * PADDING_SIZE
		elif y_type == BOTTOM_EDGE:
			y_cut = PADDING_SIZE
			y_paste = inner_bounding_box[0]
			height_paste = TILE_SIZE - PADDING_SIZE

		if x_type == LEFT_EDGE:
			x_cut = 0
			x_paste = 0
			width_paste = TILE_SIZE - PADDING_SIZE
		elif x_type == MIDDLE:
			x_cut = PADDING_SIZE
			x_paste = inner_bounding_box[1]
			width_paste = TILE_SIZE - 2 * PADDING_SIZE
		elif x_type == RIGHT_EDGE:
			x_cut = PADDING_SIZE
			x_paste = inner_bounding_box[1]
			width_paste = TILE_SIZE - PADDING_SIZE

		#print (y_paste, x_paste), (height_paste, width_paste), (y_cut, x_cut)

		output[:,y_paste:y_paste+height_paste, x_paste:x_paste+width_paste] = subwindow[:,y_cut:y_cut+height_paste, x_cut:x_cut+width_paste]

	return output

def subImgCrop2(imgA,imgB):
    height, width, = TILE_SIZE, TILE_SIZE
    PADDING_SIZE=0
    y_stride, x_stride, = TILE_SIZE - (2 * PADDING_SIZE), TILE_SIZE - (2 * PADDING_SIZE)
    if (height > imgA.size(1)) or (width > imgA.size(2))or(imgA.size()!=imgB.size()):
        # print("Invalid crop: crop dims larger than image (%r with %r)" % (imgA.shape, tokens))
        exit(1)
    imsA = list()
    imsB = list()
    bin_ims = list()
    locations = list()
    y = 0
    y_done = False
    num=0
    while y <= imgA.size(1) and not y_done:
        x = 0
        if y + height > imgA.size(1):
            y = imgA.size(1) - height
            y_done = True
        x_done = False
        while x <= imgA.size(2) and not x_done:
            if x + width > imgA.size(2):
                x = imgA.size(2) - width
                x_done = True
            locations.append(((y, x, y + height, x + width),
                              (y + PADDING_SIZE, x + PADDING_SIZE, y + y_stride, x + x_stride),
                              TOP_EDGE if y == 0 else (BOTTOM_EDGE if y == (imgA.size(1) - height) else MIDDLE),
                              LEFT_EDGE if x == 0 else (RIGHT_EDGE if x == (imgA.size(2) - width) else MIDDLE)
                              ))
            img_a=imgA[:, y:y + height, x:x + width].numpy()
            img_b = imgB[:, y:y + height, x:x + width].numpy()
            # if num==0:
            #     imsA=img_a
            #     imsB=img_b
            # else:
            #     imsA = np.concatenate([imsA[np.newaxis,:, :, :], img_a], axis=0)
            #     imsB=np.concatenate([imsB[np.newaxis,:, :, :], img_b], axis=0)
            # imsA.append(imgA[:,y:y + height,x:x + width])
            # imsB.append(imgB[:, y:y + height, x:x + width])
            imsA.append(img_a)
            imsB.append(img_b)
            x += x_stride
            num=num+1
        y += y_stride
    torch_a = torch.from_numpy(np.array(imsA))
    torch_b = torch.from_numpy(np.array(imsB))
    return locations, torch_a,torch_b

def stich_together2(locations, subwindows, size):
	output = np.zeros(size, dtype=np.float32)
	for location, subwindow in zip(locations, subwindows):
		outer_bounding_box, inner_bounding_box, y_type, x_type = location
		y_paste, x_paste, y_cut, x_cut, height_paste, width_paste = -1, -1, -1, -1, -1, -1
		#print outer_bounding_box, inner_bounding_box, y_type, x_type

		if y_type == TOP_EDGE:
			y_cut = 0
			y_paste = 0
			height_paste = TILE_SIZE - 0
		elif y_type == MIDDLE:
			y_cut = 0
			y_paste = inner_bounding_box[0]
			height_paste = TILE_SIZE - 2 * 0
		elif y_type == BOTTOM_EDGE:
			y_cut = 0
			y_paste = inner_bounding_box[0]
			height_paste = TILE_SIZE - 0

		if x_type == LEFT_EDGE:
			x_cut = 0
			x_paste = 0
			width_paste = TILE_SIZE - 0
		elif x_type == MIDDLE:
			x_cut = 0
			x_paste = inner_bounding_box[1]
			width_paste = TILE_SIZE - 2 * 0
		elif x_type == RIGHT_EDGE:
			x_cut = 0
			x_paste = inner_bounding_box[1]
			width_paste = TILE_SIZE - 0

		#print (y_paste, x_paste), (height_paste, width_paste), (y_cut, x_cut)

		output[:,y_paste:y_paste+height_paste, x_paste:x_paste+width_paste] = subwindow[:,y_cut:y_cut+height_paste, x_cut:x_cut+width_paste]

	return output

