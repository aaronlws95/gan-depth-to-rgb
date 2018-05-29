__author__ = 'QiYE'

import numpy
import h5py
jnt_idx=[0,1,5,9,13,17,2,3,4,6,7,8,10,11,12,14,15,16,18,19,20]
to_ori_jnt_idx = [0,1,6,7,8,2,9,10,11,3,12,13,14,4,15,16,17,5,18,19,20]


def get_maxmodality_mdn_21jnt(multi_pred):
    num_img=multi_pred.shape[0]
    num_jnt=multi_pred.shape[1]
    num_gauss=multi_pred.shape[2]
    jnt_dim=multi_pred.shape[3]-2

    x=multi_pred[:,:,:,-1]
    e_x = numpy.exp(x - x.max(axis=1, keepdims=True))
    mixcoeff = e_x / e_x.sum(axis=1, keepdims=True)
    sigma_ori=multi_pred[:,:,:,-2]
    sigma = numpy.exp(sigma_ori)
    mixcoeff/=sigma

    mixcoeff.shape=(num_img*num_jnt,num_gauss)
    maxloc = numpy.argmax(mixcoeff,axis=-1)
    print(maxloc.shape)

    mu=multi_pred[:,:,:,:-2].reshape(num_img*num_jnt,num_gauss,jnt_dim)
    maxmu = mu[range(num_img*num_jnt),maxloc]
    maxmu.shape=(num_img,num_jnt,jnt_dim)
    return maxmu

def sigmoid(x):
    return  1 / (1 + numpy.exp(-x))
def softmax(x):
    # e_x = numpy.exp(x - x.max(axis=1, keepdims=True))
    e_x = numpy.exp(x)
    e_x = e_x / e_x.sum(axis=1, keepdims=True)
    return e_x


def get_mixmdn_err(source_dir,dataset,y_pred,y_true, num_gauss,jnt_dim,selected_idx):
    num_mulit_gauss=num_gauss-1
    mix_pred = numpy.reshape(y_pred[:y_true.shape[0],:-21],(-1,21,num_gauss,(jnt_dim+2)))

    vis_pred = sigmoid(y_pred[:y_true.shape[0],-21:])
    v_out =numpy.where(vis_pred>=0.5,1,0)
    # print('num visible joints by prediction', numpy.mean(v_out))

    loc_target = y_true[:,:-21].reshape(y_true.shape[0],21,3)
    single_pred = mix_pred[:,:,0,:]
    multi_pred = mix_pred[:,:,1:,:]

    mu = multi_pred[:,:,:,0:jnt_dim]

    sigma_ori = multi_pred[:,:,:,-2]
    sigma = numpy.exp(sigma_ori)
    alpha = softmax(multi_pred[:,:,:,-1])

    ratio = numpy.reshape(alpha/sigma,(-1,num_mulit_gauss))
    tmpmu = numpy.reshape(mu,(-1,num_mulit_gauss,jnt_dim))

    loc = numpy.argmax(ratio,axis=-1)
    maxmu = tmpmu[range(tmpmu.shape[0]),loc]
    maxmu =  numpy.reshape(maxmu,(-1,21,jnt_dim))

    singlemu = single_pred[:,:,:jnt_dim]
    mixmu=numpy.expand_dims(v_out,axis=-1)*singlemu+numpy.expand_dims((1-v_out),axis=-1)*maxmu

    square_root = numpy.sqrt(numpy.sum((loc_target-mixmu)**2,axis=-1))
    v_target = y_true[:,-21:]
    accu = numpy.where(v_out==v_target,1,0)
    print('metric accuracy, euclidean dist',numpy.mean(accu),numpy.mean(square_root))

    # xyz_pred, xyz_true, err = get_err_from_normuvd(base_dir=source_dir,
    #                                        normuvd=mixmu,dataset=dataset,
    #                                        jnt_idx=jnt_idx,selected_idx=selected_idx)
    # err_frame_jnt= numpy.sqrt(numpy.sum((xyz_pred-xyz_true)**2,axis=-1))


    return mixmu
