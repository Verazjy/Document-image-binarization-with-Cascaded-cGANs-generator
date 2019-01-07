import os
from options.test_options import TestOptions
from models import create_model
import os.path
import random
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
import cv2
import subImg_Process


def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def Generator_G1(model,AB_path,save_path):
    # opt.name = netName
    # model = create_model(opt)
    # model.setup(opt)
    # get_small_img_result
    AB = Image.open(AB_path).convert('RGB')
    w, h = AB.size
    w2 = int(w / 2)
    if opt.which_direction == 'BtoA':
        input_nc = opt.output_nc
        output_nc = opt.input_nc
    else:
        input_nc = opt.input_nc
        output_nc = opt.output_nc
    A = AB.crop((0, 0, w2, h))
    B = AB.crop((w2, 0, w, h))
    # A=AB.crop((0,0,w, h))
    # B=AB.crop((0, 0, w, h))
    A = transforms.ToTensor()(A)
    B = transforms.ToTensor()(B)
    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
    B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

    if input_nc == 1:  # RGB to gray
        tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        A = tmp.unsqueeze(0)

    if output_nc == 1:  # RGB to gray
        tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
        B = tmp.unsqueeze(0)
    # 20180713处理了全图的照片
    locations, subA, subB = subImg_Process.subImgCrop(A, B)
    ims = []
    realA = []
    fakeB = []
    realB = []
    for i, val in enumerate(locations):
        indices = torch.LongTensor([i])
        dataIn = {'A': torch.index_select(subA, 0, indices), 'B': torch.index_select(subB, 0, indices),
                  'A_paths': AB_path, 'B_paths': AB_path}
        if i >= opt.how_many:
            break
        model.set_input(dataIn)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        # if i % 5 == 0:
        #     print('processing (%04d)-th image... %s' % (i, img_path))
        for label, im_data in visuals.items():
            # im = tensor2im(im_data)
            a = im_data.cpu().numpy()
            im = a[0]
            # im=np.transpose(im, (1, 2, 0))
            if label == 'real_A':
                realA.append(im)
            if label == 'fake_B':
                fakeB.append(im)
            if label == 'real_B':
                realB.append(im)
    result = subImg_Process.stich_together(locations, fakeB, tuple(A.size()))
    result = np.transpose(result, (1, 2, 0))
    img_result0 = 255 * result
    img_result0 = np.maximum(img_result0, 0)
    # cv2.imshow('result',img_result0)
    # cv2.waitKey()

    # get_Resized_Img
    full_A2 = list()
    full_B2 = list()
    A_2 = AB.crop((0, 0, w2, h)).resize((256, 256), Image.BICUBIC)
    B_2 = AB.crop((w2, 0, w, h)).resize((256, 256), Image.BICUBIC)
    A_2 = transforms.ToTensor()(A_2)
    B_2 = transforms.ToTensor()(B_2)
    A_2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A_2)
    B_2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B_2)
    locations_2, subA_2, subB_2 = subImg_Process.subImgCrop2(A_2, B_2)
    ims_2 = []
    realA_2 = []
    fakeB_2 = []
    realB_2 = []
    for i, val in enumerate(locations_2):
        indices = torch.LongTensor([i])
        dataIn_2 = {'A': torch.index_select(subA_2, 0, indices), 'B': torch.index_select(subB_2, 0, indices),
                    'A_paths': AB_path, 'B_paths': AB_path}
        if i >= opt.how_many:
            break
        model.set_input(dataIn_2)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        # if i % 5 == 0:
        #     print('processing (%04d)-th image... %s' % (i, img_path))
        for label, im_data in visuals.items():
            # im = tensor2im(im_data)
            a = im_data.cpu().numpy()
            im = a[0]
            # im=np.transpose(im, (1, 2, 0))
            if label == 'real_A':
                realA_2.append(im)
            if label == 'fake_B':
                fakeB_2.append(im)
            if label == 'real_B':
                realB_2.append(im)
    result2 = subImg_Process.stich_together2(locations_2, fakeB_2, tuple(A_2.size()))
    result2 = np.transpose(result2, (1, 2, 0))
    img_result2 = 255 * result2
    img_result2 = np.maximum(img_result2, 0)

    # Get_input To Model2
    Full_img = cv2.resize(img_result2, (w2, h))
    # cv2.imshow('fakeb',Full_img)
    # cv2.waitKey()
    Full_img = cv2.cvtColor(Full_img, cv2.COLOR_RGB2GRAY)
    img_result0 = cv2.cvtColor(img_result0, cv2.COLOR_RGB2GRAY)
    img_result0 = img_result0.astype(np.uint8)
    Full_img = Full_img.astype(np.uint8)
    # cv2.imshow('img_result0',img_result0)
    # cv2.waitKey()
    # cv2.imshow('Full_img',Full_img)
    # cv2.waitKey()

    # Make_Input_To_Net2
    C = cv2.imread(AB_path, 0)
    C_1 = C[0:h, 0:w2]
    # cv2.imshow('C_1',C_1)
    # cv2.waitKey()
    Input_2 = cv2.merge([img_result0, Full_img, C_1])
    cv2.imwrite(save_path, Input_2)
    # print('processing (%04d)-th image... %s' % (i, img_path))
    return Input_2


def Generator_G2(model,input_path,save_path2):
    # opt.name = netName_2
    # model = create_model(opt)
    # model.setup(opt)
    AB = Image.open(input_path).convert('RGB')
    w, h = AB.size
    # w2 = int(w / 2)
    # A = AB.crop((0, 0, w2, h)).resize((opt.loadSize, opt.loadSize), Image.BICUBIC)
    # B = AB.crop((w2, 0, w, h)).resize((opt.loadSize, opt.loadSize), Image.BICUBIC)
    # A = transforms.ToTensor()(A)
    # B = transforms.ToTensor()(B)
    # w_offset = random.randint(0, max(0, opt.loadSize - opt.fineSize - 1))
    # h_offset = random.randint(0, max(0, opt.loadSize - opt.fineSize - 1))
    #
    # A = A[:, h_offset:h_offset + opt.fineSize, w_offset:w_offset + opt.fineSize]
    # B = B[:, h_offset:h_offset + opt.fineSize, w_offset:w_offset + opt.fineSize]
    #
    # A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
    # B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
    if opt.which_direction == 'BtoA':
        input_nc = opt.output_nc
        output_nc = opt.input_nc
    else:
        input_nc = opt.input_nc
        output_nc = opt.output_nc
    # A=AB.crop((0,0,w2, h))
    # B=AB.crop((w2, 0, w, h))
    A = AB.crop((0, 0, w, h))
    B = AB.crop((0, 0, w, h))
    A = transforms.ToTensor()(A)
    B = transforms.ToTensor()(B)
    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
    B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
    if input_nc == 1:  # RGB to gray
        tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        A = tmp.unsqueeze(0)

    if output_nc == 1:  # RGB to gray
        tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
        B = tmp.unsqueeze(0)
    # 20180713处理了全图的照片
    locations, subA, subB = subImg_Process.subImgCrop(A, B)
    ims = []
    realA = []
    fakeB = []
    realB = []
    for i, val in enumerate(locations):
        indices = torch.LongTensor([i])
        dataIn = {'A': torch.index_select(subA, 0, indices), 'B': torch.index_select(subB, 0, indices),
                  'A_paths': AB_path, 'B_paths': AB_path}
        if i >= opt.how_many:
            break
        model.set_input(dataIn)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        # if i % 5 == 0:
        #     print('processing (%04d)-th image... %s' % (i, img_path))
        for label, im_data in visuals.items():
            # im0 = tensor2im(im_data)
            a = im_data.cpu().numpy()
            im = a[0]
            # im=np.transpose(im, (1, 2, 0))

            if label == 'real_A':
                realA.append(im)
            if label == 'fake_B':
                fakeB.append(im)
            if label == 'real_B':
                realB.append(im)
    result = subImg_Process.stich_together(locations, fakeB, tuple(A.size()))
    result = np.transpose(result, (1, 2, 0))
    # tmp = result[0, ...] * 0.299 + result[1, ...] * 0.587 + result[2, ...] * 0.114
    cv2.imwrite(save_path2, 255 * result)
    # print('processing (%04d)-th image... %s' % (i, img_path))
    return 255 * result





if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    Netname1='facades_pix2pix'
    Netname2='G2_1222'
    AB_path = opt.Input_path
    save_path = opt.Output_path
    splits = os.listdir(AB_path)
    opt.name = Netname1
    model1 = create_model(opt)
    model1.setup(opt)
    opt.name = Netname2
    model2 = create_model(opt)
    model2.setup(opt)
    i=0
    for sp in splits:
        AB_path0 = os.path.join(AB_path, sp)
        save_path0 = os.path.join(save_path, sp)
        print('processing (%04d)-th image... %s' % (i, AB_path0))
        Generator_G1(model1, AB_path0, save_path0)
        Generator_G2(model2, save_path0, save_path0)
        i += 1
