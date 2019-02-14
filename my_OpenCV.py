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

def color_reduction(filename):
    '''
    RGB->32,96,160,224
    return numpy.array
    '''
    img=imread(filename)

    tmp_img=img.copy()

    red=tmp_img[:,:,0].copy()
    green=tmp_img[:,:,1].copy()
    blue=tmp_img[:,:,2].copy()

    red[np.where((0<=red) & (red<63))]=32
    red[np.where((63<=red) & (red<127))]=96
    red[np.where((127<=red) & (red<191))]=160
    red[np.where((191<=red) & (red<256))]=224

    green[np.where((0<=green) & (green<63))]=32
    green[np.where((63<=green) & (green<127))]=96
    green[np.where((127<=green) & (green<191))]=160
    green[np.where((191<=green) & (green<256))]=224

    blue[np.where((0<=blue) & (blue<63))]=32
    blue[np.where((63<=blue) & (blue<127))]=96
    blue[np.where((127<=blue) & (blue<191))]=160
    blue[np.where((191<=blue) & (blue<256))]=224

    tmp_img[:,:,0]=red
    tmp_img[:,:,1]=green
    tmp_img[:,:,2]=blue

    return tmp_img

def Average_Pooling(filename):
    '''
    128x128->8x8 Average Pooling
    return numpy.array
    '''
    img=imread(filename)
    
    tmp_img=img.copy()
    H,W,CHANNEL=tmp_img.shape
    BLOCK=8
    UNIT_H=int(H/BLOCK)
    UNIT_W=int(W/BLOCK)

    for h in range(UNIT_H):
        for w in range(UNIT_W):
            for c in range(CHANNEL):
                tmp_img[BLOCK*h:BLOCK*(h+1),BLOCK*w:BLOCK*(w+1),c]=np.mean(tmp_img[BLOCK*h:BLOCK*(h+1),BLOCK*w:BLOCK*(w+1),c]).astype(np.int)
    
    return tmp_img

def Max_Pooling(filename):
    '''
    128x128->8x8 Max Pooling
    return numpy.array
    '''
    img=imread(filename)

    tmp_img=img.copy()
    H,W,CHANNEL=tmp_img.shape
    BLOCK=8
    UNIT_H=int(H/BLOCK)
    UNIT_W=int(W/BLOCK)

    for h in range(UNIT_H):
        for w in range(UNIT_W):
            for c in range(CHANNEL):
                tmp_img[BLOCK*h:BLOCK*(h+1),BLOCK*w:BLOCK*(w+1),c]=np.max(tmp_img[BLOCK*h:BLOCK*(h+1),BLOCK*w:BLOCK*(w+1),c])
    
    return tmp_img

def Gaussian_Filter(filename,ksize=3,sigma=1.3):
    '''
    input->Gaussian Filter->result
    @default parameter
    ksize=3x3=9
    sigma=1.3
    using zero-padding
    return numpy.array
    '''
    img=imread(filename)

    H,W,CHANNEL=img.shape
    PAD=ksize//2

    tmp_img=np.zeros((H+PAD*2,W+PAD*2,CHANNEL),dtype=np.float)
    tmp_img[PAD:PAD+H,PAD:PAD+W]=img.copy().astype(np.float)

    kernel=np.zeros((ksize,ksize),dtype=np.float)
    for h in range(-PAD,-PAD+ksize):
        for w in range(-PAD,-PAD+ksize):
            kernel[h+PAD,w+PAD]=np.exp(-(w**2+h**2)/(2*(sigma**2)))
    kernel/=(sigma*np.sqrt(2*np.pi))
    kernel/=kernel.sum()

    for h in range(H):
        for w in range(W):
            for c in range(CHANNEL):
                tmp_img[PAD+h,PAD+w,c]=np.sum(kernel*tmp_img[h:h+ksize,w:w+ksize,c])
    
    tmp_img=tmp_img[PAD:PAD+h,PAD:PAD+w].astype(np.uint8)

    return tmp_img

def Median_Filter(filename,ksize=3):
    '''
    input->Median Filter->result
    @default parameter
    ksize=3x3
    using zero-padding
    return numpy.array
    '''
    img=imread(filename)

    H,W,CHANNEL=img.shape
    PAD=ksize//2

    tmp_img=np.zeros((H+PAD*2,W+PAD*2,CHANNEL),dtype=np.float)
    tmp_img[PAD:PAD+H,PAD:PAD+W]=img.copy().astype(np.float)

    for h in range(H):
        for w in range(W):
            for c in range(CHANNEL):
                tmp_img[PAD+h,PAD+w,c]=np.median(tmp_img[h:h+ksize,w:w+ksize,c])
    
    tmp_img=tmp_img[PAD:PAD+H,PAD:PAD+W].astype(np.uint8)

    return tmp_img

def Smoothing_Filter(filename,ksize=3):
    '''
    input->Smoothing Filter->result
    @default parameter
    ksize=3x3
    using zero-padding
    return numpy.array
    '''
    img=imread(filename)

    H,W,CHANNEL=img.shape
    PAD=ksize//2

    tmp_img=np.zeros((H+PAD*2,W+PAD*2,CHANNEL),dtype=np.float)
    tmp_img[PAD:PAD+H,PAD:PAD+W]=img.copy().astype(np.float)

    for h in range(H):
        for w in range(W):
            for c in range(CHANNEL):
                tmp_img[PAD+h,PAD+w,c]=np.median(tmp_img[h:h+ksize,w:w+ksize,c])
    
    tmp_img=tmp_img[PAD:PAD+H,PAD:PAD+W].astype(np.uint8)

    return tmp_img

def Motion_Filter(filename,ksize=3):
    '''
    input->Motion_Filter->result
    @default parameter
    ksize=3x3
    using zero-padding
    return numpy.array
    '''
    img=imread(filename)

    H,W,CHANNEL=img.shape
    K=np.diag([1]*ksize).astype(np.float)/ksize
    PAD=ksize//2

    tmp_img=np.zeros((H+PAD*2,W+PAD*2,CHANNEL),dtype=np.float)
    tmp_img[PAD:PAD+H,PAD:PAD+W]=img.copy().astype(np.float)
    tmp=tmp_img.copy()

    for h in range(H):
        for w in range(W):
            for c in range(CHANNEL):
                tmp_img[PAD+h,PAD+w,c]=np.sum(K*tmp[h:h+ksize,w:w+ksize,c])
    
    tmp_img=tmp_img[PAD:PAD+H,PAD:PAD+W].astype(np.uint8)
    
    return tmp_img

def MAX_MIN_Filter(filename,ksize=3):
    '''
    input->MAX-MIN Filter->result
    @default parameter
    ksize=3x3
    using zero-padding
    return numpy.array
    '''
    img=imread(filename)
    H,W,CHANNEL=img.shape

    gray_img=convert_GRAYSCALE(filename)

    PAD=ksize//2
    tmp_img=np.zeros((H+PAD*2,W+PAD*2),dtype=np.float)
    tmp_img[PAD:PAD+H,PAD:PAD+W]=gray_img.copy().astype(np.float)

    for h in range(H):
        for w in range(W):
            tmp_img[PAD+h,PAD+w]=np.max(tmp_img[h:h+ksize,w:w+ksize])-np.min(tmp_img[h:h+ksize,w:w+ksize])
    
    tmp_img=tmp_img[PAD:PAD+H,PAD:PAD+W].astype(np.uint8)
    
    return tmp_img

def Differential_Filter(filename,img_type="vertical"):
    '''
    input->Differential Filter->result
    @default parameter
    type=vertical image
    using zero-padding
    return numpy.array
    '''
    ksize=3
    img=imread(filename)
    H,W,CHANNEL=img.shape

    gray_img=convert_GRAYSCALE(filename).astype(np.uint8)

    PAD=ksize//2
    tmp_img=np.zeros((H+PAD*2,W+PAD*2),dtype=np.float)
    tmp_img[PAD:PAD+H,PAD:PAD+W]=gray_img.copy().astype(np.float)
    tmp=tmp_img.copy()


    for h in range(H):
        for w in range(W):
            if img_type=="vertical":
                K=[[0.,-1.,0.],
                   [0.,1.,0.],
                   [0.,0.,0.]]
                tmp_img[PAD+h,PAD+w]=np.mean(K*(tmp[h:h+ksize,w:w+ksize]))
            elif img_type=="horizontal":
                K=[[0.,0.,0.],
                  [-1.,1.,0.],
                  [0.,0.,0.]]
                tmp_img[PAD+h,PAD+w]=np.mean(K*(tmp[h:h+ksize,w:w+ksize]))
    
    tmp_img=tmp_img[PAD:PAD+H,PAD:PAD+W].astype(np.uint8)
    
    return tmp_img

def Sobel_Filter(filename,img_type="vertical"):
    '''
    input->Sobel Filter->result
    @default parameter
    type=vertical image
    using zero-padding
    return numpy.array
    '''
    ksize=3
    img=imread(filename)
    H,W,CHANNEL=img.shape

    gray_img=convert_GRAYSCALE(filename).astype(np.uint8)

    PAD=ksize//2
    tmp_img=np.zeros((H+PAD*2,W+PAD*2),dtype=np.float)
    tmp_img[PAD:PAD+H,PAD:PAD+W]=gray_img.copy().astype(np.float)
    tmp=tmp_img.copy()


    for h in range(H):
        for w in range(W):
            if img_type=="vertical":
                K=[[1.,0.,-1.],
                   [2.,0.,-2.],
                   [1.,0.,-1.]]
                tmp_img[PAD+h,PAD+w]=np.mean(K*(tmp[h:h+ksize,w:w+ksize]))
            elif img_type=="horizontal":
                K=[[1.,2.,1.],
                  [0.,0.,0.],
                  [-1.,-2.,-1.]]
                tmp_img[PAD+h,PAD+w]=np.mean(K*(tmp[h:h+ksize,w:w+ksize]))
    
    tmp_img=tmp_img[PAD:PAD+H,PAD:PAD+W].astype(np.uint8)
    
    return tmp_img

def Prewitt_Filter(filename,img_type="vertical"):
    '''
    input->Prewitt Filter->result
    @default parameter
    type=vertical image
    using zero-padding
    return numpy.array
    '''
    ksize=3
    img=imread(filename)
    H,W,CHANNEL=img.shape

    gray_img=convert_GRAYSCALE(filename).astype(np.uint8)

    PAD=ksize//2
    tmp_img=np.zeros((H+PAD*2,W+PAD*2),dtype=np.float)
    tmp_img[PAD:PAD+H,PAD:PAD+W]=gray_img.copy().astype(np.float)
    tmp=tmp_img.copy()


    for h in range(H):
        for w in range(W):
            if img_type=="vertical":
                K=[[-1.,-1.,-1.],
                   [0.,0.,0.],
                   [1.,1.,1.]]
                tmp_img[PAD+h,PAD+w]=np.mean(K*(tmp[h:h+ksize,w:w+ksize]))
            elif img_type=="horizontal":
                K=[[-1.,0.,1.],
                  [-1.,0.,1.],
                  [-1.,0.,1.]]
                tmp_img[PAD+h,PAD+w]=np.mean(K*(tmp[h:h+ksize,w:w+ksize]))
    
    tmp_img=tmp_img[PAD:PAD+H,PAD:PAD+W].astype(np.uint8)
    
    return tmp_img

def Laplacian_Filter(filename):
    '''
    input->Laplacian Filter->result
    using zero-padding
    return numpy.array
    '''
    ksize=3
    img=imread(filename)
    H,W,CHANNEL=img.shape

    gray_img=convert_GRAYSCALE(filename).astype(np.uint8)

    PAD=ksize//2
    tmp_img=np.zeros((H+PAD*2,W+PAD*2),dtype=np.float)
    tmp_img[PAD:PAD+H,PAD:PAD+W]=gray_img.copy().astype(np.float)
    tmp=tmp_img.copy()


    for h in range(H):
        for w in range(W):
            K=[[0.,1.,0.],
               [1.,-4.,1.],
               [0.,1.,0.]]
            tmp_img[PAD+h,PAD+w]=np.mean(K*(tmp[h:h+ksize,w:w+ksize]))
    
    tmp_img=tmp_img[PAD:PAD+H,PAD:PAD+W].astype(np.uint8)
    
    return tmp_img

def Emboss_Filter(filename):
    '''
    input->Emboss Filter->result
    using zero-padding
    return numpy.array
    '''
    ksize=3
    img=imread(filename)
    H,W,CHANNEL=img.shape

    gray_img=convert_GRAYSCALE(filename).astype(np.uint8)

    PAD=ksize//2
    tmp_img=np.zeros((H+PAD*2,W+PAD*2),dtype=np.float)
    tmp_img[PAD:PAD+H,PAD:PAD+W]=gray_img.copy().astype(np.float)
    tmp=tmp_img.copy()


    for h in range(H):
        for w in range(W):
            K=[[-2.,-1.,0.],
               [-1.,1.,1.],
               [0.,1.,2.]]
            tmp_img[PAD+h,PAD+w]=np.sum(K*(tmp[h:h+ksize,w:w+ksize]))
    
    tmp_img=tmp_img[PAD:PAD+H,PAD:PAD+W].astype(np.uint8)
    
    return tmp_img

def LoG_Filter(filename):
    '''
    input->LoG Filter->result
    using zero-padding
    return numpy.array
    '''
    ksize=5
    sigma=3
    img=imread(filename)
    H,W,CHANNEL=img.shape

    PAD=ksize//2
    gray_img=convert_GRAYSCALE(filename).astype(np.uint8)
    tmp_img=np.zeros((H+PAD*2,W+PAD*2),dtype=np.float)
    tmp_img[PAD:PAD+H,PAD:PAD+W]=gray_img.copy().astype(np.float)
    tmp=tmp_img.copy()

    kernel=np.zeros((ksize,ksize),dtype=np.float)
    for h in range(-PAD,-PAD+ksize):
        for w in range(-PAD,-PAD+ksize):
            kernel[h+PAD,w+PAD]=(w**2+h**2-sigma**2)*np.exp(-(w**2+h**2)/(2*(sigma**2)))
    kernel/=(2*np.pi*(sigma**6))
    kernel/=kernel.sum()

    for h in range(H):
        for w in range(W):
            tmp_img[PAD+h,PAD+w]=np.sum(kernel*tmp[h:h+ksize,w:w+ksize])
    
    tmp_img=tmp_img[PAD:PAD+H,PAD:PAD+W].astype(np.uint8)

    return tmp_img

def trans_GRAYSCALE(filename):
    '''
    input img->histogram nomalization->result img
    using gray-scale transformation
    return numpy.array
    '''
    img=imread(filename).astype(np.float)
    H,W,CHANNEL=img.shape

    MIN=0
    MAX=255
    IMG_MAX=max(max(i) for sub_img in img for i in sub_img)
    IMG_MIN=min(min(i) for sub_img in img for i in sub_img)

    tmp_img=img.copy()
    tmp_img[tmp_img<MIN]=MIN
    tmp_img[tmp_img>MAX]=MAX

    tmp_img=(MAX-MIN)/(IMG_MAX-IMG_MIN)*(tmp_img-IMG_MIN)+MIN
    tmp_img=tmp_img.astype(np.uint8)

    return tmp_img

def trans_NORMAL(filename,mean=128,std=52):
    '''
    input img->normal img
    not dynamic range
    return numpy.array
    '''
    img=imread(filename).astype(np.float)
    H,W,CHANNEL=img.shape

    m=np.mean(img)
    s=np.std(img)

    tmp_img=img.copy()
    tmp_img=std/s*(tmp_img-m)+mean
    tmp_img=tmp_img.astype(np.uint8)

    return tmp_img

def hist_flatten(filename):
    '''
    input img->flatten->result img
    using histogram
    return numpy.array
    '''
    img=imread(filename).astype(np.float)
    H,W,CHANNEL=img.shape

    T=float(H*W*CHANNEL)
    MAX=255

    tmp_img=img.copy()
    s=0
    for i in range(1,256):
        idx=np.where(img==i)
        s+=len(img[idx])
        tmp_img[idx]=MAX/T*s
    
    tmp_img=tmp_img.astype(np.uint8)

    return tmp_img

def gamma_correction(filename):
    '''
    input img->gamma correction->result img
    return numpy.array
    '''
    img=imread(filename).astype(np.float)

    tmp_img=img.copy()
    C=1.0
    GAMMA=2.2

    tmp_img/=255
    tmp_img=(1/C*tmp_img)**(1/GAMMA)
    tmp_img*=255

    tmp_img=tmp_img.astype(np.uint8)

    return tmp_img

def nearest_neighbor_interpolation(filename,expansion_rate=1.5):
    '''
    input img->nearest neighbor interpolation->result img
    return numpy.array
    '''
    img=imread(filename).astype(np.float)
    H,W,CHANNEL=img.shape
    
    expand_H=int(expansion_rate * H)
    expand_W=int(expansion_rate * W)
    
    y=np.arange(expand_H).repeat(expand_W).reshape(expand_W, -1)
    x=np.tile(np.arange(expand_W), (expand_H, 1))
    y=np.round(y/expansion_rate).astype(np.int)
    x=np.round(x/expansion_rate).astype(np.int)
    
    tmp_img=img[y,x]
    tmp_img=tmp_img.astype(np.uint8)

    return tmp_img

def Bi_linear_interpolation(filename,expansion_rate=1.5):
    '''
    input img->Bi-linear interpolation->result img
    return numpy.array
    '''
    img=imread(filename).astype(np.float)
    H,W,CHANNEL=img.shape

    expand_H=int(expansion_rate*H)
    expand_W=int(expansion_rate*W)

    y=np.arange(expand_H).repeat(expand_W).reshape(expand_W,-1)
    x=np.tile(np.arange(expand_W),(expand_H,1))
    y=y/expansion_rate
    x=x/expansion_rate

    y_idx=np.floor(y).astype(np.int)
    x_idx=np.floor(x).astype(np.int)
    y_idx=np.minimum(y_idx,H-2)
    x_idx=np.minimum(x_idx,W-2)

    dy=y-y_idx
    dx=x-x_idx
    dy=np.repeat(np.expand_dims(dy,axis=-1),3,axis=-1)
    dx=np.repeat(np.expand_dims(dx,axis=-1),3,axis=-1)

    tmp_img=(1-dx)*(1-dy)*img[y_idx,x_idx]+dx*(1-dy)*img[y_idx,x_idx+1]+(1-dx)*dy*img[y_idx+1,x_idx]+dx*dy*img[y_idx+1,x_idx+1]
    
    tmp_img[tmp_img>255]=255
    tmp_img=tmp_img.astype(np.uint8)

    return tmp_img

def Bi_cubic_interpolation(filename,expansion_rate=1.5):
    '''
    input img->Bi-cubic interpolation->result img
    return numpy.array
    '''
    img=imread(filename).astype(np.float)
    H,W,CHANNEL=img.shape

    expand_H=int(expansion_rate*H)
    expand_W=int(expansion_rate*W)

    y=np.arange(expand_H).repeat(expand_W).reshape(expand_W,-1)
    x=np.tile(np.arange(expand_W),(expand_H,1))
    y=y/expansion_rate
    x=x/expansion_rate

    y_idx=np.floor(y).astype(np.int)
    x_idx=np.floor(x).astype(np.int)
    y_idx=np.minimum(y_idx,H-1)
    x_idx=np.minimum(x_idx,W-1)

    dy2=y-y_idx
    dx2=x-x_idx
    dy1=dy2+1
    dx1=dx2+1
    dy3=1-dy2
    dx3=1-dx2
    dy4=1+dy3
    dx4=1+dx3

    dy=[dy1,dy2,dy3,dy4]
    dx=[dx1,dx2,dx3,dx4]

    W_sum=np.zeros((expand_H,expand_W,CHANNEL),dtype=np.float32)
    tmp_img=np.zeros((expand_H,expand_W,CHANNEL),dtype=np.float32)

    def calc_weight(t):
        a=-1
        at=np.abs(t)
        w=np.zeros_like(t)
        idx=np.where(at<=1)
        w[idx]=((a+2)*np.power(at,3)-(a+3)*np.power(at,2)+1)[idx]
        idx=np.where((at>1) & (at<=2))
        w[idx]=(a*np.power(at,3)-5*a*np.power(at,2)+8*a*at-4*a)[idx]
        return w


    for j in range(-1,3):
        for i in range(-1,3):
            y_img=np.minimum(np.maximum(y_idx+j,0),H-1)
            x_img=np.minimum(np.maximum(x_idx+i,0),W-1)
            y_weight=calc_weight(dy[j+1])
            x_weight=calc_weight(dx[i+1])
            y_weight=np.repeat(np.expand_dims(y_weight,axis=-1),3,axis=-1)
            x_weight=np.repeat(np.expand_dims(x_weight,axis=-1),3,axis=-1)
            W_sum+=x_weight*y_weight
            tmp_img+=x_weight*y_weight*img[y_img,x_img]
    
    tmp_img/=W_sum
    tmp_img[tmp_img>255]=255
    tmp_img=tmp_img.astype(np.uint8)

    return tmp_img

def Affine_trans(filename,x=0,y=0):
    '''
    input img->Affine->result img
    translation
    return numpy.array
    '''
    img=imread(filename).astype(np.float32)
    H,W,CHANNEL=img.shape

    #########
    ## matrix
    ## a  b
    ## c  d
    #########
    a=1.0
    b=0.0
    c=0.0
    d=1.0

    Y=np.arange(H).repeat(W).reshape(W,-1)
    X=np.tile(np.arange(W),(H,1))
    tmp_img=np.zeros((H+1,W+1,CHANNEL),dtype=np.float32)

    X_d=a*X+b*Y+x
    Y_d=c*X+d*Y+y
    X_d=np.minimum(np.maximum(X_d,0),W).astype(np.int)
    Y_d=np.minimum(np.maximum(Y_d,0),H).astype(np.int)

    tmp_img[Y_d,X_d]=img[Y,X]
    tmp_img=tmp_img[:H,:W]
    tmp_img=tmp_img.astype(np.uint8)

    return tmp_img

def Affine_resize(filename,w_ratio=1.0,h_ratio=1.0):
    '''
    input img->Affine->result img
    scaling
    return numpy.array
    '''
    img=imread(filename).astype(np.float)
    H,W,CHANNEL=img.shape

    tmp_img=np.zeros((H+2,W+2,CHANNEL),dtype=np.float32)
    tmp_img[1:H+1,1:W+1]=img

    H_d=np.round(H*h_ratio).astype(np.int)
    W_d=np.round(W*w_ratio).astype(np.int)
    result=np.zeros((H_d+1,W_d+1,CHANNEL),dtype=np.float32)

    X_d=np.tile(np.arange(W_d),(H_d,1))
    Y_d=np.arange(H_d).repeat(W_d).reshape(H_d,-1)
    deter=w_ratio*h_ratio
    X=np.round((h_ratio*X_d)/deter).astype(np.int)+1
    Y=np.round((w_ratio*Y_d)/deter).astype(np.int)+1
    X=np.minimum(np.maximum(X,0),W+1).astype(np.int)
    Y=np.minimum(np.maximum(Y,0),H+1).astype(np.int)

    result[Y_d,X_d]=tmp_img[Y,X]
    result=result[:H_d,:W_d]
    result=result.astype(np.uint8)

    return result

