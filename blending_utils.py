'''
Implements laplacian pyramid blending functions.
Reference:
    Burt, Peter J., and Edward H. Adelson. "A multiresolution spline with application to image mosaics." ACM Transactions on Graphics (TOG) 2.4 (1983): 217-236.
'''
import os
import errno
from os import path
from glob import glob
import cv2
import numpy as np
import scipy as sp
import scipy.signal

def generating_kernel(a):

    kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(kernel, kernel)


def reduce_l(image, kernel=generating_kernel(0.4)):

    image = image.astype(np.float64)
    image = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return image[::2, ::2]


def expand_l(image, kernel=generating_kernel(0.4)):

    H, W = image.shape
    # create output image
    out_img = np.zeros((2*H, 2*W), dtype=np.float64)
    out_img[::2,::2] = image
    # convolve
    out_img = 4*cv2.filter2D(out_img, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return out_img


def gauss_pyr(image, levels):

    image = image.astype(np.float64)

    # level 0
    g_pyr = [image]

    # total level: levels+1
    for i in range(levels): # iterate from level 0 to levels-1
        g_pyr.append(reduce_l(g_pyr[i]))

    return g_pyr

def lapl_pyr(gaussPyr):

    # level 0 is same is the top level of gaussPyr
    l_pyr = [gaussPyr[-1]]

    # iterate in reverse from (top level - 1) to 0
    for i in range(len(gaussPyr)-1)[::-1]:

        # exapand the image from the level above current
        expand_image = expand_l(gaussPyr[i+1])
        # current level image
        g_pyr_img = gaussPyr[i]

        # check if these two images are aligned
        if g_pyr_img.shape != expand_image.shape:

            # NOTE: if misaligned then crop the residual rows and columns before taking
            # difference
            l_pyr.append(g_pyr_img-expand_image[:g_pyr_img.shape[0],:g_pyr_img.shape[1]])
        else:
            # compute difference: laplacian image at current scale
            l_pyr.append(g_pyr_img-expand_image)

    return l_pyr[::-1]


def blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):

    blend_pyr = []

    for lvl in range(len(lapl_pyr_white)):
        blend_pyr.append(gauss_pyr_mask[lvl]*lapl_pyr_white[lvl] + (1-gauss_pyr_mask[lvl])*lapl_pyr_black[lvl])

    return blend_pyr

def collapse(pyramid):

    prev_lvl_img = pyramid[-1]
    for curr_lvl in range(len(pyramid)-1)[::-1]:

        prev_lvl_img_expand = expand_l(prev_lvl_img)

        if pyramid[curr_lvl].shape != prev_lvl_img_expand.shape:
            prev_lvl_img = pyramid[curr_lvl] +\
                        prev_lvl_img_expand[:pyramid[curr_lvl].shape[0],:pyramid[curr_lvl].shape[1]]
        else:
            prev_lvl_img = pyramid[curr_lvl] + prev_lvl_img_expand

    return prev_lvl_img

def visualize_pyr(pyramid):

    """Create a single image by vertically stacking the levels of a pyramid."""
    shape = np.atleast_3d(pyramid[0]).shape[:-1]  # need num rows & cols only
    img_stack = [cv2.resize(layer, shape[::-1],
                            interpolation=3) for layer in pyramid]
    return np.vstack(img_stack).astype(np.uint8)

def blend_pipeline(black_image, white_image, mask, depth='auto'):

    # Automatically figure out the size; at least 16x16 at the highest level
    min_size = min(black_image.shape)
    if depth == 'auto':
        depth = int(np.log2(min_size)) - 4

    gauss_pyrmask = gauss_pyr(mask, depth)
    gauss_pyrblack = gauss_pyr(black_image, depth)
    gauss_pyrwhite = gauss_pyr(white_image, depth)

    lapl_pyr_black = lapl_pyr(gauss_pyrblack)
    lapl_pyr_white = lapl_pyr(gauss_pyrwhite)

    outpyr = blend(lapl_pyr_white, lapl_pyr_black, gauss_pyrmask)
    img = collapse(outpyr)

    return (gauss_pyrblack, gauss_pyrwhite, gauss_pyrmask,
            lapl_pyr_black, lapl_pyr_white, outpyr, [img])

def blend_images(black_image, white_image, mask, depth='auto'):
    b_img = np.atleast_3d(black_image).astype(np.float) / 255.
    w_img = np.atleast_3d(white_image).astype(np.float) / 255.
    m_img = np.atleast_3d(mask).astype(np.float) / 255.
    num_channels = b_img.shape[-1]

    imgs = []
    for channel in range(num_channels):
        imgs.append(blend_pipeline(b_img[:, :, channel],
                              w_img[:, :, channel],
                              m_img[:, :, channel], depth=depth)[-1])
    imgs = zip(*imgs)
    imgs = np.dstack(imgs).transpose(1,2,0)

    return cv2.normalize(imgs, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX)

def blend_and_store_images(black_image, white_image, mask, out_path, depth='auto'):

    """Apply pyramid blending to each color channel of the input images """

    # Convert to double and normalize the images to the range [0..1]
    # to avoid arithmetic overflow issues
    b_img = np.atleast_3d(black_image).astype(np.float) / 255.
    w_img = np.atleast_3d(white_image).astype(np.float) / 255.
    m_img = np.atleast_3d(mask).astype(np.float) / 255.
    num_channels = b_img.shape[-1]

    imgs = []
    for channel in range(num_channels):
        imgs.append(blend_pipeline(b_img[:, :, channel],
                              w_img[:, :, channel],
                              m_img[:, :, channel], depth=depth))

    try:
        os.makedirs(out_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    names = ['gauss_pyrblack', 'gauss_pyrwhite', 'gauss_pyrmask',
             'lapl_pyr_black', 'lapl_pyr_white', 'outpyr', 'outimg']

    for name, img_stack in zip(names, zip(*imgs)):
        imgs = map(np.dstack, zip(*img_stack))
        stack = [cv2.normalize(img, dst=None, alpha=0, beta=255,
                               norm_type=cv2.NORM_MINMAX)
                 for img in imgs]
        cv2.imwrite(path.join(out_path, name + '.png'), visualize_pyr(stack))
