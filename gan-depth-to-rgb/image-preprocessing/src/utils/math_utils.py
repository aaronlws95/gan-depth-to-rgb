__author__ = 'QiYE'
import numpy
from math import pi
_EPSILON = 10e-8

def sigmoid(x):
    return  1.0 / (1 + numpy.exp(-x+_EPSILON))

def get_angle_between_two_lines(line0,line1=(0,1)):
    rot =numpy.arccos(numpy.dot(line0,line1)/numpy.linalg.norm(line0,axis=1))
    loc_neg = numpy.where(line0[:,0]<0)
    rot[loc_neg] = -rot[loc_neg]
    rot = numpy.cast['float32'](rot/pi*180)
    # print numpy.where(rot==180)[0].shape[0]
    # rot[numpy.where(rot==180)] =179
    return rot