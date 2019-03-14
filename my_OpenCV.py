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

def Affine_rotate(filename,rotate=0):
    '''
    input img->Affine->result img
    rotation
    return numpy.array
    '''
    img=imread(filename).astype(np.float)
    H,W,CHANNEL=img.shape

    theta=np.pi*rotate/180
    #########
    ## matrix
    ## a  b
    ## c  d
    #########
    a=np.cos(theta)
    b=-np.sin(theta)
    c=np.sin(theta)
    d=np.cos(theta)

    tmp_img=np.zeros((H+2,W+2,CHANNEL),dtype=np.float32)
    tmp_img[1:H+1,1:W+1]=img

    H_d=np.round(H).astype(np.int)
    W_d=np.round(W).astype(np.int)
    result=np.zeros((H_d,W_d,CHANNEL),dtype=np.float32)

    X_d=np.tile(np.arange(W_d),(H_d,1))
    Y_d=np.arange(H_d).repeat(W_d).reshape(H_d,-1)
    deter=a*d-b*c
    X=np.round((d*X_d-b*Y_d)/deter).astype(np.int)+1
    Y=np.round((-c*X_d+a*Y_d)/deter).astype(np.int)+1
    X=np.minimum(np.maximum(X,0),W+1).astype(np.int)
    Y=np.minimum(np.maximum(Y,0),H+1).astype(np.int)

    result[Y_d,X_d]=tmp_img[Y,X]
    result=result.astype(np.uint8)

    return result

def Affine_scue(filename,dx=0,dy=0):
    '''
    input img->Affine->result img
    scue
    return numpy.array
    '''
    img=imread(filename).astype(np.float)
    H,W,CHANNEL=img.shape

    #########
    ## matrix
    ## a  b
    ## c  d
    #########
    a=1.0
    b=dx/H
    c=dy/W
    d=1.0

    tmp_img=np.zeros((H+2,W+2,CHANNEL),dtype=np.float32)
    tmp_img[1:H+1,1:W+1]=img

    H_d=np.ceil(dy+H).astype(np.int)
    W_d=np.ceil(dx+W).astype(np.int)
    result=np.zeros((H_d,W_d,CHANNEL),dtype=np.float32)

    X_d=np.tile(np.arange(W_d),(H_d,1))
    Y_d=np.arange(H_d).repeat(W_d).reshape(H_d,-1)
    deter=a*d-b*c
    X=np.round((d*X_d-b*Y_d)/deter).astype(np.int)+1
    Y=np.round((-c*X_d+a*Y_d)/deter).astype(np.int)+1
    X=np.minimum(np.maximum(X,0),W+1).astype(np.int)
    Y=np.minimum(np.maximum(Y,0),H+1).astype(np.int)

    result[Y_d,X_d]=tmp_img[Y,X]
    result=result.astype(np.uint8)

    return result

def DFT(filename):
    '''
    Discrete Fourier Transformation
    '''
    img=imread(filename).astype(np.float)
    H,W,CHANNEL=img.shape

    gray_img=convert_GRAYSCALE(filename)

    K=W
    L=H
    M=W
    N=H
    G=np.zeros((L,K),dtype=np.complex)
    x=np.tile(np.arange(W),(H,1))
    y=np.arange(H).repeat(W).reshape(H,-1)

    for h in range(L):
        for w in range(K):
            G[h,w]=np.sum(gray_img*np.exp(-2j*np.pi*(x*w/M+y*h/N)))/np.sqrt(M*N)
    ps=(np.abs(G)/np.abs(G).max()*255).astype(np.uint8)

    return ps

def IDFT(filename):
    '''
    Inverse Discrete Fourier Transformation
    '''
    img=imread(filename).astype(np.float)
    H,W,CHANNEL=img.shape

    gray_img=convert_GRAYSCALE(filename)

    K=W
    L=H
    M=W
    N=H
    G=np.zeros((L,K),dtype=np.complex)
    x=np.tile(np.arange(W),(H,1))
    y=np.arange(H).repeat(W).reshape(H,-1)

    for h in range(L):
        for w in range(K):
            G[h,w]=np.sum(gray_img*np.exp(-2j*np.pi*(x*w/M+y*h/N)))/np.sqrt(M*N)
    
    result=np.zeros((H,W),dtype=np.float)

    for h in range(H):
        for w in range(W):
            result[h,w]=np.abs(np.sum(G*np.exp(2j*np.pi*(x*w/M+y*h/N))))/np.sqrt(M*N)
    
    result[result>255]=255
    result=result.astype(np.uint8)

    return result

def Low_Pass_Filter(filename):
    '''
    input img->Low-pass Filter->result img
    return numpy.array
    '''
    img=imread(filename).astype(np.float)
    H,W,CHANNEL=img.shape

    gray_img=convert_GRAYSCALE(filename)

    K=W
    L=H
    M=W
    N=H
    G=np.zeros((L,K),dtype=np.complex)
    x=np.tile(np.arange(W),(H,1))
    y=np.arange(H).repeat(W).reshape(H,-1)

    for h in range(L):
        for w in range(K):
            G[h,w]=np.sum(gray_img*np.exp(-2j*np.pi*(x*w/M+y*h/N)))/np.sqrt(M*N)
    
    low_pass_G=np.zeros_like(G)
    low_pass_G[:H//2,:W//2]=G[H//2:,W//2:]
    low_pass_G[:H//2,W//2:]=G[H//2,:W//2]
    low_pass_G[H//2:,:W//2]=G[:H//2,W//2:]
    low_pass_G[H//2:,W//2:]=G[:H//2,:W//2]
    p=0.5
    X=x-W//2
    Y=y-H//2
    r=np.sqrt(X**2+Y**2)
    mask=np.ones((H,W),dtype=np.float)
    mask[r>(W//2*p)]=0
    
    low_pass_G*=mask
    G[:H//2,:W//2]=low_pass_G[H//2:,W//2:]
    G[:H//2,W//2:]=low_pass_G[H//2:,:W//2]
    G[H//2:,:W//2]=low_pass_G[:H//2,W//2:]
    G[H//2:,W//2:]=low_pass_G[:H//2,:W//2]

    result=np.zeros((H,W),dtype=np.float)

    for h in range(N):
        for w in range(M):
            result[h,w]=np.abs(np.sum(G*np.exp(2j*np.pi*(x*w/M+y*h/N))))/np.sqrt(M*N)
    
    result[result>255]=255
    result=result.astype(np.uint8)

    return result

def High_Pass_Filter(filename):
    '''
    input img->High-pass Filter->result img
    return numpy.array
    '''
    img=imread(filename).astype(np.float)
    H,W,CHANNEL=img.shape

    gray_img=convert_GRAYSCALE(filename)

    K=W
    L=H
    M=W
    N=H
    G=np.zeros((L,K),dtype=np.complex)
    x=np.tile(np.arange(W),(H,1))
    y=np.arange(H).repeat(W).reshape(H,-1)

    for h in range(L):
        for w in range(K):
            G[h,w]=np.sum(gray_img*np.exp(-2j*np.pi*(x*w/M+y*h/N)))/np.sqrt(M*N)
    
    high_pass_G=np.zeros_like(G)
    high_pass_G[:H//2,:W//2]=G[H//2:,W//2:]
    high_pass_G[:H//2,W//2:]=G[H//2,:W//2]
    high_pass_G[H//2:,:W//2]=G[:H//2,W//2:]
    high_pass_G[H//2:,W//2:]=G[:H//2,:W//2]
    p=0.2
    X=x-W//2
    Y=y-H//2
    r=np.sqrt(X**2+Y**2)
    mask=np.ones((H,W),dtype=np.float)
    mask[r<(W//2*p)]=0
    
    high_pass_G*=mask
    G[:H//2,:W//2]=high_pass_G[H//2:,W//2:]
    G[:H//2,W//2:]=high_pass_G[H//2:,:W//2]
    G[H//2:,:W//2]=high_pass_G[:H//2,W//2:]
    G[H//2:,W//2:]=high_pass_G[:H//2,:W//2]

    result=np.zeros((H,W),dtype=np.float)

    for h in range(N):
        for w in range(M):
            result[h,w]=np.abs(np.sum(G*np.exp(2j*np.pi*(x*w/M+y*h/N))))/np.sqrt(M*N)
    
    result[result>255]=255
    result=result.astype(np.uint8)

    return result


def Band_Pass_Filter(filename):
    '''
    input img->Band-pass Filter->result img
    return numpy.array
    '''
    img=imread(filename).astype(np.float)
    H,W,CHANNEL=img.shape

    gray_img=convert_GRAYSCALE(filename)

    K=W
    L=H
    M=W
    N=H
    G=np.zeros((L,K),dtype=np.complex)
    x=np.tile(np.arange(W),(H,1))
    y=np.arange(H).repeat(W).reshape(H,-1)

    for h in range(L):
        for w in range(K):
            G[h,w]=np.sum(gray_img*np.exp(-2j*np.pi*(x*w/M+y*h/N)))/np.sqrt(M*N)
    
    band_pass_G=np.zeros_like(G)
    band_pass_G[:H//2,:W//2]=G[H//2:,W//2:]
    band_pass_G[:H//2,W//2:]=G[H//2,:W//2]
    band_pass_G[H//2:,:W//2]=G[:H//2,W//2:]
    band_pass_G[H//2:,W//2:]=G[:H//2,:W//2]
    p_low=0.1
    p_high=0.5
    X=x-W//2
    Y=y-H//2
    r=np.sqrt(X**2+Y**2)
    mask=np.ones((H,W),dtype=np.float)
    mask[np.where((r > (W//2*p_low)) & (r < (W//2*p_high)))] = 1
    
    band_pass_G*=mask
    G[:H//2,:W//2]=band_pass_G[H//2:,W//2:]
    G[:H//2,W//2:]=band_pass_G[H//2:,:W//2]
    G[H//2:,:W//2]=band_pass_G[:H//2,W//2:]
    G[H//2:,W//2:]=band_pass_G[:H//2,:W//2]

    result=np.zeros((H,W),dtype=np.float)

    for h in range(N):
        for w in range(M):
            result[h,w]=np.abs(np.sum(G*np.exp(2j*np.pi*(x*w/M+y*h/N))))/np.sqrt(M*N)
    
    result[result>255]=255
    result=result.astype(np.uint8)

    return result

def DCT(filename,ksize=8):
    '''
    input img->Discrete Cosine Transformation->result img
    return numpy.array
    '''
    img=imread(filename).astype(np.float32)
    H,W,CHANNEL=img.shape

    gray=convert_GRAYSCALE(filename).astype(np.float32)

    X=np.zeros((H,W),dtype=np.float32)

    def assign_weight(x,y,u,v):
        cu=1.0
        cv=1.0
        if u==0:
            cu/=np.sqrt(2)
        if v==0:
            cv/=np.sqrt(2)
        theta=np.pi/(2*ksize)
        return ((2*cu*cv/ksize)*np.cos((2*x+1)*u*theta)*np.cos((2*y+1)*v*theta))
    
    for yidx in range(0,H,ksize):
        for xidx in range(0,W,ksize):
            for v in range(ksize):
                for u in range(ksize):
                    for y in range(ksize):
                        for x in range(ksize):
                            X[v+yidx,u+xidx]+=gray[y+yidx,x+xidx]*assign_weight(x,y,u,v)
    
    return X

def IDCT(filename,dct_array,ksize=8):
    '''
    input img->Inverse Discrete Cosine Transformation->result img
    return numpy.array
    '''
    tmp_img=imread(filename).astype(np.float)
    H,W,CHANNEL=tmp_img.shape
    X=dct_array.copy()
    result=np.zeros((H,W),dtype=np.float)

    def assign_weight(x,y,u,v):
        cu=1.0
        cv=1.0
        if u==0:
            cu/=np.sqrt(2)
        if v==0:
            cv/=np.sqrt(2)
        theta=np.pi/(2*ksize)
        return ((2*cu*cv/ksize)*np.cos((2*x+1)*u*theta)*np.cos((2*y+1)*v*theta))

    for yidx in range(0,H,ksize):
        for xidx in range(0,W,ksize):
            for y in range(ksize):
                for x in range(ksize):
                    for v in range(ksize):
                        for u in range(ksize):
                            result[y+yidx,x+xidx]+=X[v+yidx,u+xidx]*assign_weight(x,y,u,v)
    
    result[result>255]=255
    result=np.round(result).astype(np.uint8)

    return result

def DCT_quantization(filename,ksize=8):
    '''
    input img->Discrete Cosine Transformation->result img
    return numpy.array
    '''
    img=imread(filename).astype(np.float32)
    H,W,CHANNEL=img.shape
    Q = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
              (12, 12, 14, 19, 26, 58, 60, 55),
              (14, 13, 16, 24, 40, 57, 69, 56),
              (14, 17, 22, 29, 51, 87, 80, 62),
              (18, 22, 37, 56, 68, 109, 103, 77),
              (24, 35, 55, 64, 81, 104, 113, 92),
              (49, 64, 78, 87, 103, 121, 120, 101),
              (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)

    gray=convert_GRAYSCALE(filename).astype(np.float32)

    X=np.zeros((H,W),dtype=np.float32)

    def assign_weight(x,y,u,v):
        cu=1.0
        cv=1.0
        if u==0:
            cu/=np.sqrt(2)
        if v==0:
            cv/=np.sqrt(2)
        theta=np.pi/(2*ksize)
        return ((2*cu*cv/ksize)*np.cos((2*x+1)*u*theta)*np.cos((2*y+1)*v*theta))
    
    for yidx in range(0,H,ksize):
        for xidx in range(0,W,ksize):
            for v in range(ksize):
                for u in range(ksize):
                    for y in range(ksize):
                        for x in range(ksize):
                            X[v+yidx,u+xidx]+=gray[y+yidx,x+xidx]*assign_weight(x,y,u,v)
            X[yidx:yidx+ksize,xidx:xidx+ksize]=np.round(X[yidx:yidx+ksize,xidx:xidx+ksize]/Q)*Q
    
    return X

def RGB2YCbCr(filename,Y_r=1.0,Cb_r=1.0,Cr_r=1.0):
    '''
    RGB->YCbCr
    return numpy.array
    '''
    img=imread(filename).astype(np.float)

    red=img[:,:,0]
    green=img[:,:,1]
    blue=img[:,:,2]

    Y =0.2990*red+0.5870*green+0.1140*blue
    Cb=-0.1687*red-0.3313*green+0.5000*blue+128
    Cr=0.5000*red-0.4817*green-0.0813*blue+128

    Y=Y*Y_r
    Cb=Cb*Cb_r
    Cr=Cr*Cr_r

    tmp_img=np.zeros_like(img,dtype=np.float)
    tmp_img[:,:,0]=Y+(Cr-128)*1.4020
    tmp_img[:,:,1]=Y-(Cb-128)*0.3441-(Cr-128)*0.7139
    tmp_img[:,:,2]=Y+(Cb-128)*1.7718

    tmp_img=tmp_img.astype(np.uint8)

    return tmp_img

def compress_JPEG(filename):
    '''
    JPEG->compress(YCbCr,DCT,IDCT)->result
    return numpy.array
    '''
    img=imread(filename).astype(np.float)
    H,W,CHANNEL=img.shape

    red=img[:,:,0]
    green=img[:,:,1]
    blue=img[:,:,2]

    Y =0.2990*red+0.5870*green+0.1140*blue
    Cb=-0.1687*red-0.3313*green+0.5000*blue+128
    Cr=0.5000*red-0.4817*green-0.0813*blue+128

    tmp_img=np.zeros_like(img,dtype=np.float)
    tmp_img[:,:,0]=Y
    tmp_img[:,:,1]=Cb
    tmp_img[:,:,2]=Cr

    ksize=8

    Q1=np.array(((16, 11, 10, 16, 24, 40, 51, 61),
              (12, 12, 14, 19, 26, 58, 60, 55),
              (14, 13, 16, 24, 40, 57, 69, 56),
              (14, 17, 22, 29, 51, 87, 80, 62),
              (18, 22, 37, 56, 68, 109, 103, 77),
              (24, 35, 55, 64, 81, 104, 113, 92),
              (49, 64, 78, 87, 103, 121, 120, 101),
              (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)
    
    Q2=np.array(((17, 18, 24, 47, 99, 99, 99, 99),
               (18, 21, 26, 66, 99, 99, 99, 99),
               (24, 26, 56, 99, 99, 99, 99, 99),
               (47, 66, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99)), dtype=np.float32)

    X=np.zeros((H,W,CHANNEL),dtype=np.float32)

    def assign_weight(x,y,u,v):
        cu=1.0
        cv=1.0
        if u==0:
            cu/=np.sqrt(2)
        if v==0:
            cv/=np.sqrt(2)
        theta=np.pi/(2*ksize)
        return ((2*cu*cv/ksize)*np.cos((2*x+1)*u*theta)*np.cos((2*y+1)*v*theta))
    
    for yidx in range(0,H,ksize):
        for xidx in range(0,W,ksize):
            for v in range(ksize):
                for u in range(ksize):
                    for y in range(ksize):
                        for x in range(ksize):
                            for c in range(CHANNEL):
                                X[v+yidx,u+xidx,c]+=tmp_img[y+yidx,x+xidx,c]*assign_weight(x,y,u,v)
            X[yidx:yidx+ksize,xidx:xidx+ksize,0]=np.round(X[yidx:yidx+ksize,xidx:xidx+ksize,0]/Q1)*Q1
            X[yidx:yidx+ksize,xidx:xidx+ksize,1]=np.round(X[yidx:yidx+ksize,xidx:xidx+ksize,1]/Q2)*Q2
            X[yidx:yidx+ksize,xidx:xidx+ksize,2]=np.round(X[yidx:yidx+ksize,xidx:xidx+ksize,2]/Q2)*Q2
    
    tmp=np.zeros((H,W,3),dtype=np.float)

    for yidx in range(0,H,ksize):
        for xidx in range(0,W,ksize):
            for y in range(ksize):
                for x in range(ksize):
                    for v in range(ksize):
                        for u in range(ksize):
                            tmp[y+yidx,x+xidx]+=X[v+yidx,u+xidx]*assign_weight(x,y,u,v)
    
    result=np.zeros_like(img,dtype=np.float)
    result[:,:,0]=tmp[:,:,0]+(tmp[:,:,2]-128)*1.4020
    result[:,:,1]=tmp[:,:,0]-(tmp[:,:,1]-128)*0.3441-(tmp[:,:,2]-128)*0.7139
    result[:,:,2]=tmp[:,:,0]+(tmp[:,:,1]-128)*1.7718

    result[result>255]=255
    result=result.astype(np.uint8)

    return result

def extract_edge(filename):
    '''
    input->Gaussian Filter
    ->Sobel Filter
    ->Non-maximum suppression
    ->Histeresis threshold
    ->extract edge
    ->result 
    return numpy.array(edge intensity),numpy.array(edge angle)
    '''
    img=imread(filename).astype(np.float)
    H,W,CHANNEL=img.shape

    gray=convert_GRAYSCALE(filename).astype(np.float)

    ksize=5
    sigma=1.4
    PAD=ksize//2
    G=np.zeros((H+PAD*2,W+PAD*2),dtype=np.float)
    G=np.pad(gray,(PAD,PAD),"edge")
    tmp_G=G.copy()

    # Gaussian Filter
    K=np.zeros((ksize,ksize),dtype=np.float)
    for x in range(-PAD,-PAD+ksize):
        for y in range(-PAD,-PAD+ksize):
            K[y+PAD,x+PAD]=np.exp(-(x^2+y^2)/(2*(sigma**2)))
    K/=(sigma*np.sqrt(2*np.pi))
    K/=K.sum()

    for y in range(H):
        for x in range(W):
            G[PAD+y,PAD+x]=np.sum(K*tmp_G[y:y+ksize,x:x+ksize])
    
    # Sobel Filter
    SV = np.array(((-1., -2., -1.), (0., 0., 0.), (1., 2., 1.)), dtype=np.float)
    SH = np.array(((-1., 0., 1.), (-2., 0., 2.), (-1., 0., 1.)), dtype=np.float)

    G=G[PAD-1:H+PAD+1,PAD-1:W+PAD+1]
    Y=np.zeros_like(G,dtype=np.float)
    X=np.zeros_like(G,dtype=np.float)
    ksize=3
    PAD=ksize//2

    for y in range(H):
        for x in range(W):
            Y[PAD+y,PAD+x]=np.sum(SV*G[y:y+ksize,x:x+ksize])
            X[PAD+y,PAD+x]=np.sum(SH*G[y:y+ksize,x:x+ksize])
    
    X=X[PAD:PAD+H,PAD:PAD+W]
    Y=Y[PAD:PAD+H,PAD:PAD+W]

    edge=np.sqrt(np.power(X,2)+np.power(Y,2))
    X[X==0]=1e-5
    theta=np.arctan(Y/X)

    angle=np.zeros_like(theta,dtype=np.uint8)
    angle[np.where((theta > -0.4142) & (theta <= 0.4142))] = 0
    angle[np.where((theta > 0.4142) & (theta < 2.4142))] = 45
    angle[np.where((theta >= 2.4142) | (theta <= -2.4142))] = 95
    angle[np.where((theta > -2.4142) & (theta <= -0.4142))] = 135

    # Non-Maximum Suppression
    for y in range(H):
        for x in range(W):
            if angle[y,x]==0:
                dx1,dy1,dx2,dy2=-1,0,1,0
            elif angle[y,x]==45:
                dx1,dy1,dx2,dy2=-1,1,1,-1
            elif angle[y,x]==90:
                dx1,dy1,dx2,dy2=0,-1,0,1
            elif angle[y,x]==135:
                dx1,dy1,dx2,dy2=-1,-1,1,1
            if x==0:
                dx1=max(dx1,0)
                dx2=max(dx2,0)
            if x==W-1:
                dx1=min(dx1,0)
                dx2=min(dx2,0)
            if y==0:
                dy1=max(dy1,0)
                dy2=max(dy2,0)
            if y==H-1:
                dy1=min(dy1,0)
                dy2=min(dy2,0)
            if max(max(edge[y,x],edge[y+dy1,x+dx1]),edge[y+dy2,x+dx2])!=edge[y,x]:
                edge[y,x]=0
    
    # Histeresis threshold
    HT=100
    LT=30
    edge[edge>=HT]=255
    edge[edge<=LT]=0

    tmp_edge=np.zeros((H+2,W+2),dtype=np.float)
    tmp_edge[1:H+1,1:W+1]=edge

    nn=np.array(((1.0,1.0,1.0),
                 (1.0,0,1.0),
                 (1.0,1.0,1.0)),dtype=np.float)
    
    for y in range(1,H+2):
        for x in range(1,W+2):
            if tmp_edge[y,x]<LT or tmp_edge[y,x]>HT:
                continue
            if np.max(tmp_edge[y-1:y+2,x-1:x+2]*nn)>=HT:
                tmp_edge[y,x]=255
            else:
                tmp_edge[y,x]=0
    
    edge=tmp_edge[1:H+1,1:W+1]

    edge=edge.astype(np.uint8)
    angle=angle.astype(np.uint8)

    return edge,angle

def Hough_transform(filename):
    '''
    input->Hough transform->result
    return numpy.array
    '''
    img=imread(filename).astype(np.float)
    H,W,CHANNEL=img.shape

    edge,angle=extract_edge(filename)

    rmax=np.ceil(np.sqrt(H**2+W**2)).astype(np.int)
    hough=np.zeros((rmax,180),dtype=np.int)
    idx=np.where(edge==255)

    for y,x in zip(idx[0],idx[1]):
        for theta in range(180):
            t=np.pi/180*theta
            r=int(x*np.cos(t)+y*np.sin(t))
            hough[r,theta]+=1
    
    hough=hough.astype(np.uint8)

    # Non-Maximum Suppression
    for y in range(rmax):
        for x in range(180):
            x1=max(x-1,0)
            x2=min(x+2,180)
            y1=max(y-1,0)
            y2=min(y+2,rmax)
            if np.max(hough[y1:y2,x1:x2])==hough[y,x] and hough[y,x]!=0:
                hough[y,x]=255
            else:
                hough[y,x]=0
    
    x_idx=np.argsort(hough.ravel())[::-1][:10]
    y_idx=x_idx.copy()
    ts=x_idx%180
    rs=y_idx//180

    # Inverse Hough Transform
    result=img.copy()
    for theta,rx in zip(ts,rs):
        t=np.pi/180*theta
        for x in range(W):
            if np.sin(t)!=0:
                y=-(np.cos(t)/np.sin(t))*x+rx/np.sin(t)
                y=int(y)
                if y>=H or y<0:
                    continue
                result[y,x]=[255,0,0]
        for y in range(H):
            if np.cos(t)!=0:
                x=-(np.sin(t)/np.cos(t))*y+rx/np.cos(t)
                x=int(x)
                if x>=W or x<0:
                    continue
                result[y,x]=[255,0,0]
    
    result=result.astype(np.uint8)

    return result

def Morphology_expand(filename,time=1):
    '''
    input->Binarization
    ->Morphology expand->result
    return numpy.array
    '''
    img=imread(filename)
    H,W,CHANNEL=img.shape

    gray=convert_Binarization(filename).astype(np.float)

    MF=np.array(((0,1,0),
                 (1,0,1),
                 (0,1,0)),dtype=np.float)
    
    for i in range(time):
        tmp_img=np.pad(gray,(1,1),'edge')
        for y in range(1,H+1):
            for x in range(1,W+1):
                if np.sum(MF*tmp_img[y-1:y+2,x-1:x+2])>=255:
                    gray[y-1,x-1]=255

    return gray

def Morphology_contract(filename,time=1):
    '''
    input->Binarization
    ->Morphology contract->result
    return numpy.array
    '''
    img=imread(filename)
    H,W,CHANNEL=img.shape

    gray=convert_Binarization(filename).astype(np.float)

    MF=np.array(((0,1,0),
                 (1,0,1),
                 (0,1,0)),dtype=np.float)
    
    for i in range(time):
        tmp_img=np.pad(gray,(1,1),'edge')
        for y in range(1,H+1):
            for x in range(1,W+1):
                if np.sum(MF*tmp_img[y-1:y+2,x-1:x+2])<255*4:
                    gray[y-1,x-1]=0

    return gray

