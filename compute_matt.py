'''

This module computes the closed form alpha image of an input image 
and returns the foreground image on a plain background.

To ascertain background and foreground regions, the user has to draw rough
scribbles around the image: white scribbles on the foreground region, and 
black scribbles on the background region. The paintbrush can be toggled 
between white and black colors by pressing the 'm' key on the keyboard.

'''


import scipy.misc
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy
import scipy.sparse
from numpy.lib.stride_tricks import as_strided



drawing = False # true if mouse is pressed
mode = True # if True, white paint. Press 'm' to toggle to black paint
ix,iy = -1,-1

def rolling_block(A, block=(3, 3)):
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)

# Returns sparse matting laplacian
def computeLaplacian(img, eps = 10**(-13), win_rad=1):
    win_size = (win_rad*2+1)**2
    h, w, d = img.shape
    # Number of window centre indices in h, w axes
    c_h, c_w = h - win_rad -1, w - win_rad -1
    win_diam = win_rad*2+1
    
    indsM = np.arange(h*w).reshape((h,w))
    ravelImg = img.reshape(h*w,d)
    win_inds = rolling_block(indsM, block=(win_diam,win_diam))
    
    win_inds = win_inds.reshape(c_h, c_w, win_size)
    winI = ravelImg[win_inds]
    
    win_mu = np.mean(winI, axis=2, keepdims=True)
    win_var = np.einsum('...ji,...jk ->...ik',winI,winI)/win_size - np.einsum('...ji,...jk ->...ik',win_mu,win_mu)
    
    inv = np.linalg.inv(win_var + (eps/win_size)*np.eye(3))
    
    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    vals = np.eye(win_size) - (1/win_size)*(1 + np.einsum('...ij,...kj->...ik',X , winI - win_mu))

    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h*w, h*w))
    return L



def closed_form_matte(img, scribbled_img, mylambda=1):
    h, w, c = img.shape
    scribbles_loc_img = rgb2gray(scribbled_img) - rgb2gray(img)
    bgInds = np.where(scribbles_loc_img.ravel() < 0 )[0]
    fgInds = np.where((scribbles_loc_img.ravel() > 0))[0]
    D_s = np.zeros(h*w)
    D_s[fgInds] = 1
    D_s[bgInds] = 1
    b_s = np.zeros(h*w)
    b_s[fgInds] = 1
    
    L = computeLaplacian(img/255)
    sD_s = scipy.sparse.diags(D_s)

    x = scipy.sparse.linalg.spsolve(L + mylambda*sD_s, mylambda*b_s)
    alpha = np.minimum(np.maximum(x.reshape(h,w),0),1)
    return alpha



def compute_matt(img_input):

    '''
	This function is responsible for drawing of scribbles on the img
	the alpha matt is computed using the closed_form_matte fn. 
	OpenCV functions are used to draw scribbles on the image.
	This function returns the output image which is the input
	image on a plain background.

    '''

    global ix,iy,drawing,mode



    def draw_circle(event,x,y,flags,param):
        global ix,iy,drawing,mode

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                if mode == True:
                    cv2.circle(img,(x,y),5,(255,255,255),-1)
                else:
                    cv2.circle(img,(x,y),5,(0,0,0),-1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if mode == True:
                cv2.circle(img,(x,y),5,(255,255,255),-1)
            else:
                cv2.circle(img,(x,y),5,(0,0,0),-1)


    img = img_input
    img_orig = copy.copy(img_input)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)

    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            scribbled_img = img
            break

    cv2.destroyAllWindows()

    alpha = closed_form_matte(img_orig, scribbled_img)
    scipy.misc.imsave('dandelion_clipped_alpha.bmp',alpha)    
#     plt.title("Alpha matte")
#     plt.imshow(alpha, cmap='gray')
#     plt.show()
    
    
    alpha = cv2.imread('dandelion_clipped_alpha.bmp')


    foreground = img_orig  #Foreground Image
    bg = cv2.resize(cv2.imread('background1.jpg'), (500,500))  #Background Image


    fg = foreground.astype(float)
    bg = bg.astype(float)
    # Normalizde the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255

    foreground = cv2.multiply(alpha, fg)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, bg)

    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)

    # Display image
    return outImage

