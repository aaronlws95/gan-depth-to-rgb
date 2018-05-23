__author__ = 'QiYE'
import matplotlib
import sys
import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.morphology
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Model,model_from_json
from keras.optimizers import Adam
import os
import tensorflow as tf
import cv2
from skimage.transform import resize
import operator
sys.path.insert(0, '../utils')
import xyz_uvd
import math_utils
import loss
import show_blend_img   

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.5):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

## INTIALISATION
KTF.set_session(get_session())
rng = np.random.RandomState(0)

hand_img_size=256 # NxN size of final image
hand_size=300.0 # max hand size of a human being in millimetres  
centerU=315.944855 # width centre (~640/2)
padWidth= 100

def load_model(save_dir,version):
    print(version)
    # load json and create model
    json_file = open("%s/%s.json"%(save_dir,version), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("%s/weight_%s.h5"%(save_dir,version))
    loaded_model.compile(optimizer=Adam(lr=1e-5), loss=loss.cost_sigmoid)
    return loaded_model

def get_mask(depthimg,model):
    depth=np.zeros((1,480,640,1))
    depth[0,:,:,0] = depthimg/2000.0
    mask = model.predict(x=depth,batch_size=1)
    # out =np.zeros(mask.shape,dtype='uint8')
    mask = math_utils.sigmoid(mask[0,:,:,0])
    # out[np.where(mask>0.5)]=
    return mask

def prepare_data():
    # SET DIRECTORIES
    setname='mega'
    model_save_dir = '../../detector_data'
    hand_data_dir = 'D:/FYP-HPE-RGB/data/GAN-datasets/bighand/normal'
    ## INITIALIZE LIST
    rgb_list=  []
    depth_list = []
    ori_index_list = []
    num_files_hand_data_dir = (len([name for name in os.listdir(hand_data_dir) if os.path.isfile(os.path.join(hand_data_dir, name))]))
    num_images = int((num_files_hand_data_dir - 1))
    ## LOAD MODEL
    model= load_model(save_dir=model_save_dir,version='pixel_fullimg_ker32_lr0.001000')
    count = 1
    train_iterations = 2
    test_iterations = 2
    max_iterations = num_images
    index = 0

    # for i in np.random.randint(0, num_images, num_images):
    for i in range(0, num_images,1):

        print(count, '/', max_iterations, ':', i)

        # GET DEPTH IMAGE
        depth = Image.open('%s/image_D%08d.png' % (hand_data_dir, i))
        depth = np.asarray(depth, dtype='uint16')

        # # GET COLOUR IMAGE
        colour = Image.open('%s/image_C%08d.png' % (hand_data_dir, i))
        colour = np.asarray(colour, dtype='uint8')

        # CREATE MASK
        mask_threshold = 0.5
        mask = get_mask(depthimg=depth,model=model)
        mask[np.where(mask < mask_threshold)] = 0
        mask[np.where(mask >= mask_threshold)] = 1
        mask = scipy.ndimage.morphology.binary_erosion(mask)
        # CHECK IF THERE IS NO HAND
        loc = np.where(mask > mask_threshold)
        if  loc[0].shape[0]<30:
            print('no hand in the area')
            continue
        depth_value = depth[loc]
        # GET VALUES RELATIVE TO MASK
        U = np.mean(loc[1])
        U_min = np.min(loc[1])
        U_max = np.max(loc[1])
        V = np.mean(loc[0])
        V_min = np.min(loc[0])
        V_max = np.max(loc[0])
        D = np.mean(depth_value)
        if D<10:
            print('not valid hand area')
            continue
        bb = np.array([(hand_size,hand_size,np.mean(depth_value))])
        bbox_uvd = xyz_uvd.xyz2uvd(setname=setname,xyz=bb)
        margin = int(np.ceil(bbox_uvd[0,0] - centerU))

        ## DEPTH
        depth_w_hand_only = np.zeros(depth.shape)
        depth_w_hand_only[int(V_min):int(V_max), int(U_min):int(U_max)] = depth[int(V_min):int(V_max), int(U_min):int(U_max)]
        depth_w_hand_only[np.where(depth > D + hand_size/2)] = 0
        depth_w_hand_only[np.where(depth < D - hand_size/2)] = 0

        tmpDepth = np.zeros((depth.shape[0]+padWidth*2,depth.shape[1]+padWidth*2))
        tmpDepth[padWidth:padWidth+depth.shape[0],padWidth:padWidth+depth.shape[1]]= depth_w_hand_only
        if U-margin/2+padWidth<0 or U+margin/2+padWidth>tmpDepth.shape[1]-1 or V - margin/2+padWidth <0 or V+margin/2+padWidth>tmpDepth.shape[0]-1:
            print('hand part outside the image')
            continue
        crop = tmpDepth[int(V - margin / 2 + padWidth):int(V + margin / 2 + padWidth), int(U - margin / 2 + padWidth):int(U + margin / 2 + padWidth)]

        norm_hand_img=np.ones(crop.shape,dtype='float32')
        loc_hand=np.where(crop>0)
        norm_hand_img[loc_hand]=(crop[loc_hand]-D)/hand_size
        final_depth = resize(norm_hand_img, (hand_img_size,hand_img_size), order=3,preserve_range=True)

        ## COLOUR
        colour_w_hand_only = np.zeros(colour.shape, dtype='float32')
        colour_w_hand_only[int(V_min):int(V_max), int(U_min):int(U_max), :] = colour[int(V_min):int(V_max), int(U_min):int(U_max), :]
        colour_w_hand_only[np.where(depth > D + hand_size / 2)] = (0, 0, 0)
        colour_w_hand_only[np.where(depth < D - hand_size / 2)] = (0, 0, 0)

        tmpColour = np.zeros((colour.shape[0]+padWidth*2,colour.shape[1]+padWidth*2, 3), dtype='float32')
        tmpColour[padWidth:padWidth+colour.shape[0], padWidth:padWidth+colour.shape[1], :] = colour_w_hand_only
        crop_colour = tmpColour[int(V-margin/2+padWidth):int(V+margin/2+padWidth), int(U-margin/2+padWidth):int(U+margin/2+padWidth),:]
        
        final_colour = resize(crop_colour, (hand_img_size, hand_img_size), order=3, preserve_range=True)
        final_colour = final_colour.astype(np.float32)
        final_colour_uint8 = final_colour.astype(np.uint8)

        #remove blue
        # indices = np.where(np.all(final_colour_uint8[i] < (50,70,100), axis=-1))
        # final_colour_uint8[i][indices] = (0, 0, 0)
        # final_depth[i][indices] = 1
        # final_depth[i][np.where(final_depth[i] > 0.2)] = 1
        # final_colour_uint8[i][np.where(final_depth[i] > 0.2)] = (0,0,0)

        rgb_list.append(final_colour_uint8)
        depth_list.append(final_depth)
        ori_index_list.append(i)


    f = h5py.File(
        'D:/FYP-HPE-RGB-FRESH/data/GAN-datasets/hands/preprocessed/bighand_normal_final.h5',
        'w')
    f.create_dataset('depth', data=depth_list)
    f.create_dataset('rgb', data=rgb_list)
    f.create_dataset('ori_index', data=ori_index_list)
    f.close()



if __name__=='__main__':
    prepare_data()


