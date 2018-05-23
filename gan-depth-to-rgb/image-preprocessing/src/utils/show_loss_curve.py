__author__ = 'QiYE'

import numpy

import matplotlib.pyplot as plt

save_dir = 'F:/HuaweiProj/data/mega'
def show_loss_curve_singleloss2():

    loss = numpy.load('%s/hier/history_palm_s0_rot_scaleker32_lr0.000030.npy'%save_dir)
    train_loss = loss[0][1:]
    test_loss=loss[1][1:]
    x_axis=train_loss.shape[0]
    # x_axis=25
    print(numpy.min(test_loss))

    plt.figure()
    plt.xlim(xmin=0,xmax=x_axis)
    plt.plot(numpy.arange(0,x_axis,1),train_loss[0:x_axis,], 'blue')
    plt.plot(numpy.arange(0,x_axis,1),test_loss[0:x_axis,],  c='r')
    # plt.yscale('log')ocmin]]*x_axis, '--', c='r')
    # p
    plt.grid('on','minor')
    plt.tick_params(which='minor' )
    plt.show()



if __name__ == '__main__':
    show_loss_curve_singleloss2()

