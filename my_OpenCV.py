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
    img=imread(filename)

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
    img=imread(filename).astype(np.float)

    tmp_img=img.copy()
    red=tmp_img[:,:,0]
    green=tmp_img[:,:,1]
    blue=tmp_img[:,:,2]

    tmp_img = red*299/1000 + green*587/1000 + blue*114/1000

    return tmp_img

def convert_Binarization(filename):
    '''
    input->Binarization
    return numpy.array
    '''
    img=convert_GRAYSCALE(filename).astype(np.uint8)

    tmp_img=img.copy()

    threshold=128
    tmp_img[tmp_img<threshold]=0
    tmp_img[tmp_img>=threshold]=255

    return tmp_img

def discriminant_Binarization(filename):
    '''
    input->Binarization (using discriminant analysis method)
    return numpy.array
    '''
    img=convert_GRAYSCALE(filename).astype(np.uint8)

    tmp_img=img.copy()
    H,W=tmp_img.shape

    Max_sigma=0
    Max_t=0

    for t in range(1,255):
        v0=tmp_img[np.where(tmp_img<t)]
        m0=np.mean(v0) if len(v0)>0 else 0
        w0=len(v0)/(H*W)
        v1=tmp_img[np.where(tmp_img>=t)]
        m1=np.mean(v1) if len(v1)>0 else 0
        w1=len(v1)/(H*W)
        sigma=w0*w1*((m0-m1)**2)
        if sigma>Max_sigma:
            Max_sigma=sigma
            Max_t=t
    
    threshold=Max_t

    tmp_img[tmp_img<threshold]=0
    tmp_img[tmp_img>=threshold]=255

    return tmp_img

def reverse_H(filename):
    '''
    input->HSV->reverse 
    return numpy.array
    '''
    img=imread(filename).astype(np.float32)/255

    tmp_img=np.zeros_like(img)
    Max=np.max(img,axis=2).copy()
    Min=np.min(img,axis=2).copy()

    H=np.zeros_like(Max)

    H[np.where(Max==Min)]=0
    red=img[:,:,0].copy()
    green=img[:,:,1].copy()
    blue=img[:,:,2].copy()

    idx=np.where(np.argmin(img,axis=2)==0)
    H[idx]=60*(blue[idx]-green[idx])/(Max[idx]-Min[idx]+1)+180

    idx=np.where(np.argmin(img,axis=2)==1)
    H[idx]=60*(red[idx]-blue[idx])/(Max[idx]-Min[idx]+1)+300

    idx=np.where(np.argmin(img,axis=2)==2)
    H[idx]=60*(green[idx]-red[idx])/(Max[idx]-Min[idx]+1)+60

    V=Max.copy()
    S=Max.copy()-Min.copy()
    H=(H+180)%360


    C=S
    Hd=H//60
    X=C*(1-np.abs(Hd%2-1))
    Z=np.zeros_like(H)
    vals=[[C,X,Z],[X,C,Z],[Z,C,X],[Z,X,C],[X,Z,C],[C,Z,X]]

    for i in range(len(vals)):
        idx=np.where((i<=Hd) & (Hd<(i+1)))
        tmp_img[:,:,0][idx]=(V-C)[idx]+vals[i][0][idx]
        tmp_img[:,:,1][idx]=(V-C)[idx]+vals[i][1][idx]
        tmp_img[:,:,2][idx]=(V-C)[idx]+vals[i][2][idx]

    tmp_img[np.where(Max==Min)]=0
    tmp_img=(tmp_img*255).astype(np.uint8)
    
    return tmp_img