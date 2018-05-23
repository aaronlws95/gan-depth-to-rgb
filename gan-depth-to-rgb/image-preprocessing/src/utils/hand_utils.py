__author__ = 'QiYE'
from .transformations import  affine_matrix_from_points,decompose_matrix,euler_matrix,compose_matrix
import numpy
import numpy
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation  as interplt
import h5py
from src.utils import xyz_uvd
import csv
from skimage.transform import resize
from mpl_toolkits.mplot3d import Axes3D
def update_palm_jnts_by_kabsch(refPalmJoints,PalmJoints,shear=False, scale=False, usesvd=True):

    affinematrix = affine_matrix_from_points(v0=refPalmJoints, v1=PalmJoints, shear=shear, scale=scale, usesvd=usesvd)
    # scale0, shear0, angles0, trans0, persp0 = decompose_matrix(affinematrix)
    expSkeleton=numpy.ones((4,6))
    expSkeleton[:3]=refPalmJoints
    newpalm = numpy.dot(affinematrix,expSkeleton)[:3]

    return newpalm.T

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
    plt.show()



def show_muulti_hand_skeleton(refSkeleton):
    linewidth=2
    markersize=2

    fig = plt.figure()

    #ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]], [axis_bounds[0], axis_bounds[1]], [axis_bounds[2], axis_bounds[3]])
    # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5,  marker='.')
    ax = fig.add_subplot(111, projection='3d')
    for tmp in refSkeleton:
        dot = tmp*1000
        for k in [1,5,9,13,17]:
            x=[dot[0,0],dot[k,0]]
            y=[dot[0,1],dot[k,1]]
            z=[dot[0,2],dot[k,2]]
            ax.plot(z,x,y,linewidth=linewidth,marker='o',c='b',markersize=markersize+3)

            x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
            y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
            z=[dot[k,2],dot[k+1,2],dot[k+2,2],dot[k+3,2]]
            ax.plot(z,x,y,c='b',linewidth=linewidth,marker='o',markersize=markersize+3)
    plt.show()


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
        ax.plot(z,x,y,linewidth=linewidth,marker='o',c='r',markersize=markersize+3)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        z=[dot[k,2],dot[k+1,2],dot[k+2,2],dot[k+3,2]]
        ax.plot(z,x,y,c='r',linewidth=linewidth,marker='o',markersize=markersize+3)

    dot = Skeleton*1000
    for k in [1,5,9,13,17]:
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        z=[dot[0,2],dot[k,2]]
        ax.plot(z,x,y,linewidth=linewidth,marker='o',c='b',markersize=markersize+3)

        x=[dot[k,0],dot[k+1,0],dot[k+2,0],dot[k+3,0]]
        y=[dot[k,1],dot[k+1,1],dot[k+2,1],dot[k+3,1]]
        z=[dot[k,2],dot[k+1,2],dot[k+2,2],dot[k+3,2]]
        ax.plot(z,x,y,c='b',linewidth=linewidth,marker='o',markersize=markersize+3)

    plt.show()

def show_two_palm_skeleton(refSkeleton,Skeleton):
    linewidth=2
    markersize=2

    fig = plt.figure()

    #ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]], [axis_bounds[0], axis_bounds[1]], [axis_bounds[2], axis_bounds[3]])
    # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5,  marker='.')
    ax = fig.add_subplot(111, projection='3d')
    dot = refSkeleton*1000
    for k in range(1,6,1):
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        z=[dot[0,2],dot[k,2]]
        ax.plot(z,x,y,linewidth=linewidth,marker='o',c='r',markersize=markersize+3)
    dot = Skeleton*1000
    for k in range(1,6,1):
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        z=[dot[0,2],dot[k,2]]
        ax.plot(z,x,y,linewidth=linewidth,marker='o',c='b',markersize=markersize+3)

    plt.show()


def show_palm_skeleton(refSkeleton):
    linewidth=2
    markersize=2

    fig = plt.figure()

    #ax.auto_scale_xyz([axis_bounds[4], axis_bounds[5]], [axis_bounds[0], axis_bounds[1]], [axis_bounds[2], axis_bounds[3]])
    # ax.scatter3D(points0[:, 2], points0[:, 0], points0[:, 1], s=1.5,  marker='.')
    ax = fig.add_subplot(111, projection='3d')
    dot = refSkeleton*1000
    for k in range(1,6,1):
        x=[dot[0,0],dot[k,0]]
        y=[dot[0,1],dot[k,1]]
        z=[dot[0,2],dot[k,2]]
        ax.plot(z,x,y,linewidth=linewidth,marker='o',c='r',markersize=markersize+3)

    plt.show()


