__author__ = 'QiYE'
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle
import numpy
import h5py
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from ..utils import math
def show_hand_skeleton(refSkeleton):
    linewidth=2
    markersize=2
    fig = plt.figure()

    #ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]], [axis_bounds[0], axis_bounds[1]], [axis_bounds[2], axis_bounds[3]])
    # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5,  marker='.')
    ax = fig.add_subplot(111, projection='3d')
    dot = refSkeleton*1000
    for k in [1,5,9,13,17]:
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        z=[dot[0,2],dot[k,2]]
        ax.plot(z,x,y,linewidth=linewidth,marker='o',c='b',markersize=markersize+3)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        z=[dot[k,2],dot[k+1,2],dot[k+2,2],dot[k+3,2]]
        ax.plot(z,x,y,c='b',linewidth=linewidth,marker='o',markersize=markersize+3)

def show_two_hand_skeleton(refSkeleton,Skeleton):
    linewidth=2
    markersize=2
    fig = plt.figure()

    #ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]], [axis_bounds[0], axis_bounds[1]], [axis_bounds[2], axis_bounds[3]])
    # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5,  marker='.')
    ax = fig.add_subplot(111, projection='3d')
    dot = refSkeleton*1000
    for k in [1,5,9,13,17]:
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        z=[dot[0,2],dot[k,2]]
        ax.plot(z,x,y,linewidth=linewidth,marker='o',c='b',markersize=markersize+3)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        z=[dot[k,2],dot[k+1,2],dot[k+2,2],dot[k+3,2]]
        ax.plot(z,x,y,c='b',linewidth=linewidth,marker='o',markersize=markersize+3)
    dot = Skeleton*1000
    for k in [1,5,9,13,17]:
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        z=[dot[0,2],dot[k,2]]
        ax.plot(z,x,y,linewidth=linewidth,marker='o',c='r',markersize=markersize+3)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        z=[dot[k,2],dot[k+1,2],dot[k+2,2],dot[k+3,2]]
        ax.plot(z,x,y,c='r',linewidth=linewidth,marker='o',markersize=markersize+3)
    plt.show()

def augment_data_3d_mega_rot_scale(r0,r1,r2,gr_uvd):
    new_r0=r0[:,:,:,0].copy()
    new_r1=r1[:,:,:,0].copy()
    new_r2=r2[:,:,:,0].copy()

    new_gr_uvd =gr_uvd.copy().reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]

    img_gr_uv =new_gr_uvd[:,:,:2]*96+48

    rot = numpy.random.uniform(low=-180,high=180,size=num_frame)
    scale_factor = numpy.random.normal(loc=1,scale=0.05,size=num_frame)



    # for i in range(0,gr_uvd.shape[0],1):
    for i in numpy.random.randint(0,num_frame,int(num_frame*0.5)):
        # print(center_x[i],center_y[i],center_z[i],rot[i],scale_factor[i])
        """depth translation"""

        """2d translation, rotation and scale"""
        # print(center_x[i],center_y[i],rot[i],scale_factor[i])
        M = cv2.getRotationMatrix2D((48,48),rot[i],scale_factor[i])
        new_r0[i] = cv2.warpAffine(new_r0[i],M,(96,96),borderValue=1)

        for j in range(num_jnt):
            tmp=numpy.dot(M,numpy.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
            new_gr_uvd[i,j,0:2] = (tmp-48)/96
        M = cv2.getRotationMatrix2D((24,24),rot[i],scale_factor[i])
        new_r1[i] = cv2.warpAffine(new_r1[i],M,(48,48),borderValue=1)
        M = cv2.getRotationMatrix2D((12,12),rot[i],scale_factor[i])
        new_r2[i] = cv2.warpAffine(new_r2[i],M,(24,24),borderValue=1)

        # fig = plt.figure()
        # ax =fig.add_subplot(121)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*96+48,new_gr_uvd[i,:,1]*96+48,c='r',s=10)
        #
        # tmp=gr_uvd[i].reshape(21,3)
        # # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b',s=5)
        # ax =fig.add_subplot(122)
        # ax.imshow(r0[i,:,:,0],'gray')
        # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b')
        # plt.show()
        # show_two_hand_skeleton(new_gr_uvd[i],tmp)


    return numpy.expand_dims(new_r0,axis=-1),numpy.expand_dims(new_r1,axis=-1),numpy.expand_dims(new_r2,axis=-1),new_gr_uvd.reshape(new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])



def augment_data_3d_mega_trans_rot_scale(r0,r1,r2,gr_uvd):
    new_r0=r0[:,:,:,0].copy()
    new_r1=r1[:,:,:,0].copy()
    new_r2=r2[:,:,:,0].copy()

    new_gr_uvd =gr_uvd.copy().reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]

    img_gr_uv =new_gr_uvd[:,:,:2]*96+48

    center_x = numpy.random.normal(loc=48,scale=3,size=num_frame)
    center_y = numpy.random.normal(loc=48,scale=3,size=num_frame)
    center_z = numpy.random.normal(loc=0,scale=0.05,size=num_frame)

    # center_x = numpy.random.randint(low=43,high=54,size=num_frame)
    # center_y = numpy.random.randint(low=43,high=54,size=num_frame)
    # center_z = numpy.random.uniform(low=-0.15,high=0.15,size=num_frame)
    rot = numpy.random.uniform(low=-180,high=180,size=num_frame)
    scale_factor = numpy.random.normal(loc=1,scale=0.05,size=num_frame)



    # for i in range(0,gr_uvd.shape[0],1):
    for i in numpy.random.randint(0,num_frame,int(num_frame*0.5)):
        # print(center_x[i],center_y[i],center_z[i],rot[i],scale_factor[i])
        """depth translation"""
        loc = numpy.where(new_r0[i]<1.0)
        new_r0[i][loc]+=center_z[i]
        loc = numpy.where(new_r0[i]>1.0)
        new_r0[i][loc]=1.0


        loc = numpy.where(new_r1[i]<1.0)
        new_r1[i][loc]+=center_z[i]
        loc = numpy.where(new_r1[i]>1.0)
        new_r1[i][loc]=1.0


        loc = numpy.where(new_r2[i]<1.0)
        new_r2[i][loc]+=center_z[i]
        loc = numpy.where(new_r2[i]>1.0)
        new_r2[i][loc]=1.0


        """2d translation, rotation and scale"""

        M = cv2.getRotationMatrix2D((center_x[i],center_y[i]),rot[i],scale_factor[i])
        new_r0[i] = cv2.warpAffine(new_r0[i],M,(96,96),borderValue=1)

        for j in range(num_jnt):
            tmp=numpy.dot(M,numpy.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
            new_gr_uvd[i,j,0:2] = (tmp-48)/96

        M = cv2.getRotationMatrix2D((center_x[i]/2,center_y[i]/2),rot[i],scale_factor[i])
        new_r1[i] = cv2.warpAffine(new_r1[i],M,(48,48),borderValue=1)
        M = cv2.getRotationMatrix2D((center_x[i]/4,center_y[i]/4),rot[i],scale_factor[i])
        new_r2[i] = cv2.warpAffine(new_r2[i],M,(24,24),borderValue=1)

        # print(center_x[i],center_y[i],center_z[i],rot[i],scale_factor[i])
        # fig = plt.figure()
        # ax =fig.add_subplot(221)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*96+48,new_gr_uvd[i,:,1]*96+48,c='r')
        # ax =fig.add_subplot(222)
        # ax.imshow(new_r1[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*48+24,new_gr_uvd[i,:,1]*48+24,c='r')
        # ax =fig.add_subplot(223)
        # ax.imshow(new_r2[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*24+12,new_gr_uvd[i,:,1]*24+12,c='r')
        #
        # tmp=gr_uvd[i].reshape(6,3)
        # # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b',s=5)
        # ax =fig.add_subplot(224)
        # ax.imshow(r0[i,:,:,0],'gray')
        # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b')
        # plt.show()
        # # show_two_hand_skeleton(new_gr_uvd[i],tmp)


    return numpy.expand_dims(new_r0,axis=-1),numpy.expand_dims(new_r1,axis=-1),numpy.expand_dims(new_r2,axis=-1),new_gr_uvd.reshape(new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])




def get_hand_part_for_palm(r0,r1,gr_uvd,pred_uvd):
    new_r0=r0[:,:,:,0].copy()
    new_r1=r1[:,:,:,0].copy()


    new_gr_uvd =gr_uvd.copy().reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]

    img_gr_uv =new_gr_uvd[:,:,:2]*96+48

    center_x = numpy.random.normal(loc=0,scale=0.005,size=num_frame)
    center_y = numpy.random.normal(loc=0,scale=0.005,size=num_frame)
    center_z = numpy.random.normal(loc=0,scale=0.005,size=num_frame)

    # center_x = numpy.random.randint(low=43,high=54,size=num_frame)
    # center_y = numpy.random.randint(low=43,high=54,size=num_frame)
    # center_z = numpy.random.uniform(low=-0.15,high=0.15,size=num_frame)
    rot = numpy.random.uniform(low=-30,high=30,size=num_frame)
    scale_factor = numpy.random.normal(loc=1,scale=0.05,size=num_frame)



    for i in range(0,gr_uvd.shape[0],1):
        loc = numpy.where(new_r0[i]<1.0)
        new_r0[i][loc]+=center_z[i]
        loc = numpy.where(new_r0[i]>1.0)
        new_r0[i][loc]=1.0


        loc = numpy.where(new_r1[i]<1.0)
        new_r1[i][loc]+=center_z[i]
        loc = numpy.where(new_r1[i]>1.0)
        new_r1[i][loc]=1.0
        """2d translation, rotation and scale"""

        M = cv2.getRotationMatrix2D((center_x[i]*96+48,center_y[i]*96+48),rot[i],scale_factor[i])
        new_r0[i] = cv2.warpAffine(new_r0[i],M,(96,96),borderValue=1)

        for j in range(num_jnt):
            tmp=numpy.dot(M,numpy.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
            new_gr_uvd[i,j,0:2] = (tmp-48)/96

        M = cv2.getRotationMatrix2D((center_x[i]/2,center_y[i]/2),rot[i],scale_factor[i])
        new_r1[i] = cv2.warpAffine(new_r1[i],M,(48,48),borderValue=1)



    # for i in numpy.random.randint(0,num_frame,int(num_frame*0.5)):
        rot_angle = math.get_angle_between_two_lines(line0=pred_uvd[i,3,:]-pred_uvd[i,0,:])
        # print(center_x[i],center_y[i],center_z[i],rot[i],scale_factor[i])
        """depth translation"""






        # print(center_x[i],center_y[i],center_z[i],rot[i],scale_factor[i])
        # fig = plt.figure()
        # ax =fig.add_subplot(221)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*96+48,new_gr_uvd[i,:,1]*96+48,c='r')
        # ax =fig.add_subplot(222)
        # ax.imshow(new_r1[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*48+24,new_gr_uvd[i,:,1]*48+24,c='r')
        # ax =fig.add_subplot(223)
        # ax.imshow(new_r2[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*24+12,new_gr_uvd[i,:,1]*24+12,c='r')
        #
        # tmp=gr_uvd[i].reshape(6,3)
        # # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b',s=5)
        # ax =fig.add_subplot(224)
        # ax.imshow(r0[i,:,:,0],'gray')
        # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b')
        # plt.show()
        # # show_two_hand_skeleton(new_gr_uvd[i],tmp)


    return numpy.expand_dims(new_r0,axis=-1),numpy.expand_dims(new_r1,axis=-1),numpy.expand_dims(new_r2,axis=-1),new_gr_uvd.reshape(new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])





def augment_data(r0,gr_uvd):


    new_r0=r0[:,:,:,0]
    new_gr_uvd =gr_uvd.reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]


    img_gr_uv =new_gr_uvd[:,:,:2]*96+48

    center_x = numpy.random.normal(loc=48,scale=2,size=num_frame)
    center_y = numpy.random.normal(loc=48,scale=2,size=num_frame)
    rot = numpy.random.uniform(low=-180,high=180,size=num_frame)
    scale_factor = numpy.random.normal(loc=1,scale=0.1,size=num_frame)

    # for i in range(0,gr_uvd.shape[0],1):
    for i in numpy.random.randint(0,num_frame,int(num_frame/2)):
        # print((center_x[i],center_y[i]),rot[i],scale_factor[i])
        M = cv2.getRotationMatrix2D((center_x[i],center_y[i]),rot[i],scale_factor[i])
        new_r0[i] = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)

        for j in range(num_jnt):
            tmp=numpy.dot(M,numpy.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
            new_gr_uvd[i,j,0:2] = (tmp-48)/96

        # fig = plt.figure()
        # ax =fig.add_subplot(121)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*96+48,new_gr_uvd[i,:,1]*96+48,c='r')
        # ax =fig.add_subplot(122)
        # ax.imshow(r0[i],'gray')
        # plt.scatter(gr_uvd[i,:,0]*96+48,gr_uvd[i,:,1]*96+48,c='r')
        # plt.show()
        # show_hand_skeleton(new_gr_uvd[i])


    return numpy.expand_dims(new_r0,axis=-1),new_gr_uvd.reshape(new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])
def augment_data_hmdn(r0,gr_uvd):


    new_r0=r0[:,:,:,0]
    new_gr_uvd =gr_uvd[:,:63].reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]

    img_gr_uv =new_gr_uvd[:,:,:2]*96+48

    center_x = numpy.random.normal(loc=48,scale=2,size=num_frame)
    center_y = numpy.random.normal(loc=48,scale=2,size=num_frame)
    rot = numpy.random.uniform(low=-10,high=10,size=num_frame)
    scale_factor = numpy.random.normal(loc=1,scale=0.1,size=num_frame)

    # for i in range(0,gr_uvd.shape[0],1):
    for i in numpy.random.randint(0,num_frame,int(num_frame/2)):
        # print((center_x[i],center_y[i]),rot[i],scale_factor[i])
        M = cv2.getRotationMatrix2D((center_x[i],center_y[i]),rot[i],scale_factor[i])
        new_r0[i] = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)

        for j in range(num_jnt):
            tmp=numpy.dot(M,numpy.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
            new_gr_uvd[i,j,0:2] = (tmp-48)/96

        # fig = plt.figure()
        # ax =fig.add_subplot(121)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*96+48,new_gr_uvd[i,:,1]*96+48,c='r')
        # ax =fig.add_subplot(122)
        # ax.imshow(r0[i],'gray')
        # plt.scatter(gr_uvd[i,:,0]*96+48,gr_uvd[i,:,1]*96+48,c='r')
        # plt.show()
        # show_hand_skeleton(new_gr_uvd[i])
    new_gr_uvd.shape = (new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])

    return numpy.expand_dims(new_r0,axis=-1),numpy.concatenate([new_gr_uvd,gr_uvd[:,63:]],axis=-1)

def augment_data_msrc_jntcenter_mdn(r0,gr_uvd):


    new_r0=r0[:,:,:,0]
    new_gr_uvd =gr_uvd.reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]


    img_gr_uv =new_gr_uvd[:,:,:2]*96+48

    center_x = numpy.random.normal(loc=48,scale=2,size=num_frame)
    center_y = numpy.random.normal(loc=48,scale=2,size=num_frame)
    rot = numpy.random.uniform(low=-10,high=10,size=num_frame)
    scale_factor = numpy.random.normal(loc=1,scale=0.1,size=num_frame)

    # for i in range(0,gr_uvd.shape[0],1):
    for i in numpy.random.randint(0,num_frame,int(num_frame/2)):
        # print((center_x[i],center_y[i]),rot[i],scale_factor[i])
        M = cv2.getRotationMatrix2D((center_x[i],center_y[i]),rot[i],scale_factor[i])
        new_r0[i] = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)

        for j in range(num_jnt):
            tmp=numpy.dot(M,numpy.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
            new_gr_uvd[i,j,0:2] = (tmp-48)/96

        # fig = plt.figure()
        # ax =fig.add_subplot(121)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*96+48,new_gr_uvd[i,:,1]*96+48,c='r')
        # ax =fig.add_subplot(122)
        # ax.imshow(r0[i],'gray')
        # plt.scatter(gr_uvd[i,:,0]*96+48,gr_uvd[i,:,1]*96+48,c='r')
        # plt.show()
        # show_hand_skeleton(new_gr_uvd[i])


    return numpy.expand_dims(new_r0,axis=-1),new_gr_uvd.reshape(new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])



def augment_data_nyu(r0,gr_uvd):

    new_r0=r0[:,:,:,0].copy()
    new_gr_uvd =gr_uvd.reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]


    img_gr_uv =new_gr_uvd[:,:,:2]*96+48

    center_x = numpy.random.normal(loc=48,scale=2,size=num_frame)
    center_y = numpy.random.normal(loc=48,scale=2,size=num_frame)
    rot = numpy.random.uniform(low=-180,high=180,size=num_frame)
    scale_factor=numpy.ones((num_frame))
    scale_factor[:int(num_frame/2)] = numpy.random.normal(loc=0.85,scale=0.05,size=int(num_frame/2))


    # for i in range(0,gr_uvd.shape[0],1):
    for i in numpy.random.randint(0,num_frame,int(num_frame/2)):
        # print((center_x[i],center_y[i]),rot[i],scale_factor[i])
        M = cv2.getRotationMatrix2D((center_x[i],center_y[i]),rot[i],scale_factor[i])
        new_r0[i] = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)

        for j in range(num_jnt):
            tmp=numpy.dot(M,numpy.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
            new_gr_uvd[i,j,0:2] = (tmp-48)/96

        # fig = plt.figure()
        # ax =fig.add_subplot(121)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*96+48,new_gr_uvd[i,:,1]*96+48,c='r')
        # ax =fig.add_subplot(122)
        # ax.imshow(r0[i,:,:,0],'gray')
        # # plt.scatter(gr_uvd[i,:,0]*96+48,gr_uvd[i,:,1]*96+48,c='r')
        # plt.show()
        # show_hand_skeleton(new_gr_uvd[i])


    return numpy.expand_dims(new_r0,axis=-1),new_gr_uvd.reshape(new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])

def augment_data_3d(r0,gr_uvd):


    new_r0=r0[:,:,:,0]
    new_gr_uvd =gr_uvd.reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]


    img_gr_uv =new_gr_uvd[:,:,:2]*96+48

    center_x = numpy.random.normal(loc=48,scale=2,size=num_frame)
    center_y = numpy.random.normal(loc=48,scale=2,size=num_frame)
    center_z = numpy.random.normal(loc=0,scale=0.1,size=num_frame)
    rot = numpy.random.uniform(low=-10,high=10,size=num_frame)
    scale_factor = numpy.random.normal(loc=1,scale=0.1,size=num_frame)

    # for i in range(0,gr_uvd.shape[0],1):
    for i in numpy.random.randint(0,num_frame,int(num_frame/2)):
        # print((center_x[i],center_y[i]),rot[i],scale_factor[i])
        print(center_z[i])

        loc = numpy.where(new_r0[i]<1.0)
        new_r0[i][loc]+=center_z[i]
        print(numpy.max(new_r0[i]))
        loc = numpy.where(new_r0[i]>1.0)
        new_r0[i][loc]=1.0


        fig = plt.figure()
        ax =fig.add_subplot(121)
        ax.imshow(new_r0[i],'gray')
        ax =fig.add_subplot(122)
        ax.imshow(r0[i,:,:,0],'gray')
        plt.show()
        # show_hand_skeleton(new_gr_uvd[i])


        # M = cv2.getRotationMatrix2D((center_x[i],center_y[i]),rot[i],scale_factor[i])
        # new_r0[i] = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)
        #
        # for j in range(num_jnt):
        #     tmp=numpy.dot(M,numpy.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
        #     new_gr_uvd[i,j,0:2] = (tmp-48)/96

        # fig = plt.figure()
        # ax =fig.add_subplot(121)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*96+48,new_gr_uvd[i,:,1]*96+48,c='r')
        # ax =fig.add_subplot(122)
        # ax.imshow(r0[i],'gray')
        # plt.scatter(gr_uvd[i,:,0]*96+48,gr_uvd[i,:,1]*96+48,c='r')
        # plt.show()
        # show_hand_skeleton(new_gr_uvd[i])


    return numpy.expand_dims(new_r0,axis=-1),new_gr_uvd.reshape(new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])

def augment_data_3d_nyu(r0,gr_uvd):
    new_r0=r0[:,:,:,0].copy()
    new_gr_uvd =gr_uvd.copy().reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]

    img_gr_uv =new_gr_uvd[:,:,:2]*96+48

    center_x = numpy.random.normal(loc=48,scale=3,size=num_frame)
    center_y = numpy.random.normal(loc=48,scale=3,size=num_frame)

    center_z = numpy.random.normal(loc=0,scale=0.05,size=num_frame)
    rot = numpy.random.uniform(low=-180,high=180,size=num_frame)
    scale_factor=numpy.ones((num_frame))
    scale_factor[:int(num_frame/2)] = numpy.random.normal(loc=0.85,scale=0.05,size=int(num_frame/2))


    # for i in range(0,gr_uvd.shape[0],1):
    for i in numpy.random.randint(0,num_frame,int(num_frame/2)):

        """depth translation"""
        loc = numpy.where(new_r0[i]<1.0)
        new_r0[i][loc]+=center_z[i]
        loc = numpy.where(new_r0[i]>1.0)
        new_r0[i][loc]=1.0

        new_gr_uvd[i,:,2]+=center_z[i]

        """2d translation, rotation and scale"""
        # print(center_x[i],center_y[i],rot[i],scale_factor[i])
        M = cv2.getRotationMatrix2D((center_x[i],center_y[i]),rot[i],scale_factor[i])
        new_r0[i] = cv2.warpAffine(new_r0[i],M,(96,96),borderValue=1)

        for j in range(num_jnt):
            tmp=numpy.dot(M,numpy.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
            new_gr_uvd[i,j,0:2] = (tmp-48)/96

        # fig = plt.figure()
        # ax =fig.add_subplot(121)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*96+48,new_gr_uvd[i,:,1]*96+48,c='r',s=10)
        #
        # tmp=gr_uvd[i].reshape(21,3)
        # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b',s=5)
        # ax =fig.add_subplot(122)
        # ax.imshow(r0[i,:,:,0],'gray')
        # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b')
        # plt.show()
        # show_two_hand_skeleton(new_gr_uvd[i],tmp)


    return numpy.expand_dims(new_r0,axis=-1),new_gr_uvd.reshape(new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])


def augment_data_3d_nyu_derot(r0,gr_uvd):
    new_r0=r0[:,:,:,0].copy()
    new_gr_uvd =gr_uvd.copy().reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]

    img_gr_uv =new_gr_uvd[:,:,:2]*96+48

    center_x = numpy.random.normal(loc=48,scale=3,size=num_frame)
    center_y = numpy.random.normal(loc=48,scale=3,size=num_frame)
    center_z = numpy.random.normal(loc=0,scale=0.1,size=num_frame)
    rot = numpy.random.normal(loc=0,scale=10,size=num_frame)
    scale_factor=numpy.ones((num_frame))
    scale_factor[:int(num_frame/2)] = numpy.random.normal(loc=0.85,scale=0.05,size=int(num_frame/2))


    # for i in range(0,gr_uvd.shape[0],1):
    for i in numpy.random.randint(0,num_frame,int(num_frame/2)):

        """depth translation"""
        loc = numpy.where(new_r0[i]<1.0)
        new_r0[i][loc]+=center_z[i]
        loc = numpy.where(new_r0[i]>1.0)
        new_r0[i][loc]=1.0

        new_gr_uvd[i,:,2]+=center_z[i]

        """2d translation, rotation and scale"""
        # print(center_x[i],center_y[i],rot[i],scale_factor[i])
        M = cv2.getRotationMatrix2D((center_x[i],center_y[i]),rot[i],scale_factor[i])
        new_r0[i] = cv2.warpAffine(new_r0[i],M,(96,96),borderValue=1)

        for j in range(num_jnt):
            tmp=numpy.dot(M,numpy.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
            new_gr_uvd[i,j,0:2] = (tmp-48)/96

        # fig = plt.figure()
        # ax =fig.add_subplot(121)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*96+48,new_gr_uvd[i,:,1]*96+48,c='r',s=10)
        #
        # tmp=gr_uvd[i].reshape(21,3)
        # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b',s=5)
        # ax =fig.add_subplot(122)
        # ax.imshow(r0[i,:,:,0],'gray')
        # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b')
        # plt.show()
        # show_two_hand_skeleton(new_gr_uvd[i],tmp)


    return numpy.expand_dims(new_r0,axis=-1),new_gr_uvd.reshape(new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])



def augment_data_3d_nyu_derot_v1(r0,gr_uvd):
    new_r0=r0[:,:,:,0].copy()
    new_gr_uvd =gr_uvd.copy().reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]

    img_gr_uv =new_gr_uvd[:,:,:2]*96+48

    center_x = numpy.random.normal(loc=48,scale=2,size=num_frame)
    center_y = numpy.random.normal(loc=48,scale=2,size=num_frame)
    center_z = numpy.random.normal(loc=0,scale=0.03,size=num_frame)
    rot = numpy.random.normal(loc=0,scale=10,size=num_frame)
    scale_factor=numpy.ones((num_frame))
    scale_factor[:int(num_frame/2)] = numpy.random.normal(loc=0.85,scale=0.05,size=int(num_frame/2))


    # for i in range(0,gr_uvd.shape[0],1):
    for i in numpy.random.randint(0,num_frame,int(num_frame/2)):

        """depth translation"""
        loc = numpy.where(new_r0[i]<1.0)
        new_r0[i][loc]+=center_z[i]
        loc = numpy.where(new_r0[i]>1.0)
        new_r0[i][loc]=1.0

        new_gr_uvd[i,:,2]+=center_z[i]

        """2d translation, rotation and scale"""
        # print(center_x[i],center_y[i],rot[i],scale_factor[i])
        M = cv2.getRotationMatrix2D((center_x[i],center_y[i]),rot[i],scale_factor[i])
        new_r0[i] = cv2.warpAffine(new_r0[i],M,(96,96),borderValue=1)

        for j in range(num_jnt):
            tmp=numpy.dot(M,numpy.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
            new_gr_uvd[i,j,0:2] = (tmp-48)/96

        # fig = plt.figure()
        # ax =fig.add_subplot(121)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*96+48,new_gr_uvd[i,:,1]*96+48,c='r',s=10)
        #
        # tmp=gr_uvd[i].reshape(21,3)
        # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b',s=5)
        # ax =fig.add_subplot(122)
        # ax.imshow(r0[i,:,:,0],'gray')
        # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b')
        # plt.show()
        # show_two_hand_skeleton(new_gr_uvd[i],tmp)


    return numpy.expand_dims(new_r0,axis=-1),new_gr_uvd.reshape(new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])


def augment_data_3d_nyu_derot_layer3(r0,gr_uvd):
    new_r0=r0[:,:,:,0].copy()
    new_gr_uvd =gr_uvd.copy().reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]

    img_gr_uv =new_gr_uvd[:,:,:2]*96+48

    center_x = numpy.random.normal(loc=48,scale=3,size=num_frame)
    center_y = numpy.random.normal(loc=48,scale=3,size=num_frame)
    center_z = numpy.random.normal(loc=0,scale=0.05,size=num_frame)
    rot = numpy.random.normal(loc=0,scale=10,size=num_frame)
    scale_factor=numpy.ones((num_frame))
    scale_factor[:int(num_frame/2)] = numpy.random.normal(loc=0.85,scale=0.05,size=int(num_frame/2))


    # for i in range(0,gr_uvd.shape[0],1):
    for i in numpy.random.randint(0,num_frame,int(num_frame/2)):

        """depth translation"""
        loc = numpy.where(new_r0[i]<1.0)
        new_r0[i][loc]+=center_z[i]
        loc = numpy.where(new_r0[i]>1.0)
        new_r0[i][loc]=1.0

        new_gr_uvd[i,:,2]+=center_z[i]

        """2d translation, rotation and scale"""
        # print(center_x[i],center_y[i],rot[i],scale_factor[i])
        M = cv2.getRotationMatrix2D((center_x[i],center_y[i]),rot[i],scale_factor[i])
        new_r0[i] = cv2.warpAffine(new_r0[i],M,(96,96),borderValue=1)

        for j in range(num_jnt):
            tmp=numpy.dot(M,numpy.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
            new_gr_uvd[i,j,0:2] = (tmp-48)/96

        # fig = plt.figure()
        # ax =fig.add_subplot(121)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*96+48,new_gr_uvd[i,:,1]*96+48,c='r',s=10)
        #
        # tmp=gr_uvd[i].reshape(21,3)
        # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b',s=5)
        # ax =fig.add_subplot(122)
        # ax.imshow(r0[i,:,:,0],'gray')
        # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b')
        # plt.show()
        # show_two_hand_skeleton(new_gr_uvd[i],tmp)


    return numpy.expand_dims(new_r0,axis=-1),new_gr_uvd.reshape(new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])


def augment_data_3d_msrc(r0,gr_uvd):
    new_r0=r0[:,:,:,0].copy()
    new_gr_uvd =gr_uvd.copy().reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]

    img_gr_uv =new_gr_uvd[:,:,:2]*96+48

    center_x = numpy.random.normal(loc=48,scale=3,size=num_frame)
    center_y = numpy.random.normal(loc=48,scale=3,size=num_frame)
    center_z = numpy.random.normal(loc=0,scale=0.1,size=num_frame)

    # center_x = numpy.random.randint(low=43,high=54,size=num_frame)
    # center_y = numpy.random.randint(low=43,high=54,size=num_frame)
    # center_z = numpy.random.uniform(low=-0.15,high=0.15,size=num_frame)
    rot = numpy.random.uniform(low=-180,high=180,size=num_frame)
    scale_factor = numpy.random.normal(loc=1,scale=0.05,size=num_frame)



    # for i in range(0,gr_uvd.shape[0],1):
    for i in numpy.random.randint(0,num_frame,int(num_frame*0.5)):
        # print(center_x[i],center_y[i],center_z[i],rot[i],scale_factor[i])
        """depth translation"""
        loc = numpy.where(new_r0[i]<1.0)
        new_r0[i][loc]+=center_z[i]
        loc = numpy.where(new_r0[i]>1.0)
        new_r0[i][loc]=1.0

        new_gr_uvd[i,:,2]+=center_z[i]

        """2d translation, rotation and scale"""
        # print(center_x[i],center_y[i],rot[i],scale_factor[i])
        M = cv2.getRotationMatrix2D((center_x[i],center_y[i]),rot[i],scale_factor[i])
        new_r0[i] = cv2.warpAffine(new_r0[i],M,(96,96),borderValue=1)

        for j in range(num_jnt):
            tmp=numpy.dot(M,numpy.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
            new_gr_uvd[i,j,0:2] = (tmp-48)/96
        # #
        # fig = plt.figure()
        # ax =fig.add_subplot(121)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*96+48,new_gr_uvd[i,:,1]*96+48,c='r',s=10)
        #
        # tmp=gr_uvd[i].reshape(21,3)
        # # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b',s=5)
        # ax =fig.add_subplot(122)
        # ax.imshow(r0[i,:,:,0],'gray')
        # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b')
        # plt.show()
        # show_two_hand_skeleton(new_gr_uvd[i],tmp)


    return numpy.expand_dims(new_r0,axis=-1),new_gr_uvd.reshape(new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])


def augment_data_3d_msrc_derot(r0,gr_uvd):
    new_r0=r0[:,:,:,0].copy()
    new_gr_uvd =gr_uvd.copy().reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]

    img_gr_uv =new_gr_uvd[:,:,:2]*96+48

    center_x = numpy.random.normal(loc=48,scale=3,size=num_frame)
    center_y = numpy.random.normal(loc=48,scale=3,size=num_frame)
    center_z = numpy.random.normal(loc=0,scale=0.1,size=num_frame)

    # center_x = numpy.random.randint(low=43,high=54,size=num_frame)
    # center_y = numpy.random.randint(low=43,high=54,size=num_frame)
    # center_z = numpy.random.uniform(low=-0.15,high=0.15,size=num_frame)
    rot = numpy.random.normal(loc=0,scale=15,size=num_frame)
    scale_factor = numpy.random.normal(loc=1,scale=0.05,size=num_frame)



    # for i in range(0,gr_uvd.shape[0],1):
    for i in numpy.random.randint(0,num_frame,int(num_frame*0.5)):
        # print(center_x[i],center_y[i],center_z[i],rot[i],scale_factor[i])
        """depth translation"""
        loc = numpy.where(new_r0[i]<1.0)
        new_r0[i][loc]+=center_z[i]
        loc = numpy.where(new_r0[i]>1.0)
        new_r0[i][loc]=1.0

        new_gr_uvd[i,:,2]+=center_z[i]

        """2d translation, rotation and scale"""
        # print(center_x[i],center_y[i],rot[i],scale_factor[i])
        M = cv2.getRotationMatrix2D((center_x[i],center_y[i]),rot[i],scale_factor[i])
        new_r0[i] = cv2.warpAffine(new_r0[i],M,(96,96),borderValue=1)

        for j in range(num_jnt):
            tmp=numpy.dot(M,numpy.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
            new_gr_uvd[i,j,0:2] = (tmp-48)/96
        # #
        # fig = plt.figure()
        # ax =fig.add_subplot(121)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*96+48,new_gr_uvd[i,:,1]*96+48,c='r',s=10)
        #
        # tmp=gr_uvd[i].reshape(21,3)
        # # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b',s=5)
        # ax =fig.add_subplot(122)
        # ax.imshow(r0[i,:,:,0],'gray')
        # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b')
        # plt.show()
        # show_two_hand_skeleton(new_gr_uvd[i],tmp)


    return numpy.expand_dims(new_r0,axis=-1),new_gr_uvd.reshape(new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])



def augment_data_3d_msrc_v2(r0,gr_uvd):
    new_r0=r0[:,:,:,0].copy()
    new_gr_uvd =gr_uvd.copy().reshape(gr_uvd.shape[0],-1,3)

    num_frame=gr_uvd.shape[0]
    num_jnt=new_gr_uvd.shape[1]

    img_gr_uv =new_gr_uvd[:,:,:2]*96+48


    rot = numpy.random.uniform(low=-180,high=180,size=num_frame)
    scale_factor = numpy.random.normal(loc=1,scale=0.01,size=num_frame)

    # for i in range(0,gr_uvd.shape[0],1):
    for i in numpy.random.randint(0,num_frame,int(num_frame/2)):
        """2d translation, rotation and scale"""
        # print(center_x[i],center_y[i],rot[i],scale_factor[i])
        M = cv2.getRotationMatrix2D((48,48),rot[i],scale_factor[i])
        new_r0[i] = cv2.warpAffine(new_r0[i],M,(96,96),borderValue=1)

        for j in range(num_jnt):
            tmp=numpy.dot(M,numpy.array([img_gr_uv[i,j,0],img_gr_uv[i,j,1],1]))
            new_gr_uvd[i,j,0:2] = (tmp-48)/96

        # fig = plt.figure()
        # ax =fig.add_subplot(121)
        # ax.imshow(new_r0[i],'gray')
        # ax.scatter(new_gr_uvd[i,:,0]*96+48,new_gr_uvd[i,:,1]*96+48,c='r',s=10)
        #
        # tmp=gr_uvd[i].reshape(21,3)
        # # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b',s=5)
        # ax =fig.add_subplot(122)
        # ax.imshow(r0[i,:,:,0],'gray')
        # plt.scatter(tmp[:,0]*96+48,tmp[:,1]*96+48,c='b')
        # plt.show()
        # show_two_hand_skeleton(new_gr_uvd[i],tmp)


    return numpy.expand_dims(new_r0,axis=-1),new_gr_uvd.reshape(new_gr_uvd.shape[0],new_gr_uvd.shape[1]*new_gr_uvd.shape[2])


if __name__=='__main__':
    base_dir = 'F:/cvpr2018'
    dataset='test'
    setname='msrc'
    path='%s/data/%s/source/%s_crop_norm.h5'%(base_dir,setname,dataset)
    print(path)
    f = h5py.File(path, 'r')
    r0=f['img'][...]
    gr_uvd=f['uvd_norm_gt'][...]
    f.close()
    augment_data_3d_msrc(numpy.expand_dims(r0,axis=-1),gr_uvd.reshape(-1,63))

    # base_dir = 'F:/cvpr2018'
    # dataset='test'
    # setname='nyu'
    # path='%s/data/%s/source/%s_crop_norm.h5'%(base_dir,setname,dataset)
    # print(path)
    # f = h5py.File(path, 'r')
    # r0=f['img'][...]
    # gr_uvd=f['uvd_norm_gt'][...]
    # f.close()
    # # print('num train,test',train_img.shape[0],test_img.shape[0])
    # # augment_data_nyu(numpy.expand_dims(r0,axis=-1),gr_uvd.reshape(-1,63))
    # augment_data_3d_nyu(numpy.expand_dims(r0,axis=-1),gr_uvd.reshape(-1,63))

    #