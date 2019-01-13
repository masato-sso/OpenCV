import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image

def imread(filename):
    '''
    img->np.array
    return np.array
    '''
    return Image.imread(filename)

def imshow(np_array,cmap="gray",vmin=0,vmax=255,interpolation=None):
    '''
    np.array->show
    '''
    
    img=np.clip(np_array,vmin,vmax).astype(np.uint8)
    plt.imshow(img,cmap=cmap,vmin=vmin,vmax=vmax,interpolation=interpolation)
    plt.show()
    plt.close()

def imwrite(np_array,output_filename):
    '''
    save image
    '''
    if np_array.ndim==3:
        plt.imshow(np_array)
    elif np_array.ndim==2:
        plt.imshow(np_array)
        plt.gray()
    plt.savefig(output_filename)

def convert_BGR(filename):
    '''
    RGB->BGR
    return numpy.array
    '''
    img=Image.imread(filename)

    tmp_img=img.copy()
    red=tmp_img[:,:,0]
    blue=tmp_img[:,:,2]
    
    tmp_img[:,:,0]=blue
    tmp_img[:,:,2]=red

    return tmp_img

def convert_GRAYSCALE(filename):
    '''
    input->GRAYSCALE
    using luminance signal
    return numpy.array
    '''
    
    img=Image.imread(filename).astype(np.float)

    tmp_img=img.copy()
    red=tmp_img[:,:,0]
    green=tmp_img[:,:,1]
    blue=tmp_img[:,:,2]

    tmp_img = red*299/1000 + green*587/1000 + blue*114/1000

    return tmp_img
