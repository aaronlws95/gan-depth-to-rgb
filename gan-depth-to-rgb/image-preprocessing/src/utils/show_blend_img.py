# sphinx_gallery_thumbnail_number = 3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy
def show_two_imgs(backimg,topimg,alpha):

    # First we'll plot these blobs using only ``imshow``.
    vmax = topimg.max()
    vmin =  topimg.min()
    cmap = plt.cm.jet

    # Create an alpha channel of linearly increasing values moving to the right.
    alphas = np.ones(topimg.shape)*alpha
    # alphas[:, 30:] = np.linspace(1, 0, 70)

    # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
    colors = Normalize(vmin, vmax, clip=True)(topimg)
    colors = cmap(colors)

    # Now set the alpha channel to the one we created above
    colors[..., -1] = alphas
    return backimg,colors,cmap

    # Create the figure and image
    # Note that the absolute values may be slightly different

    # plt.show()


if __name__=='__main__':

    backimg=numpy.zeros((100,100))
    backimg[30:80,30:80]=1
    topimg = numpy.arange(0,10000,1).reshape(100,100)
    # topimg=numpy.zeros((100,100))
    # topimg[10:80,50:80]=1
    backimg,colors,cmap = show_two_imgs(backimg,topimg,0.1)
    fig, ax = plt.subplots()
    ax.imshow(backimg,'gray')
    ax.imshow(colors,cmap=cmap)
    plt.show()
