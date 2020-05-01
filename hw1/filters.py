"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernel = np.flip(np.flip(kernel, axis = 0), axis = 1)
    delta_h = int((Hk-1)/2)
    delta_w = int((Wk-1)/2)
    
    temp = np.zeros((Hi+2*delta_h, Wi+2*delta_w))
    temp[delta_h:Hi+delta_h, delta_w:Wi+delta_w] = image
    for i in range(delta_h, Hi+delta_h):
        for j in range(delta_w, Wi+delta_w):
            out[i-delta_h, j-delta_w] = np.sum(temp[i-delta_h:i+delta_h+1,j-delta_w:j+delta_w+1]*kernel)
    ### END YOUR CODE
    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H + 2*pad_height, W + 2*pad_width))
    out[pad_height:H+pad_height, pad_width:W+pad_width] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernel = np.flip(np.flip(kernel, axis = 0), axis = 1)
    delta_h = int((Hk-1)/2)
    delta_w = int((Wk-1)/2)
    image = zero_pad(image, delta_h, delta_w)
    for i in range(delta_h, Hi+delta_h):
        for j in range(delta_w, Wi+delta_w):
            out[i-delta_h,j-delta_w] = np.sum(image[i-delta_h:i+delta_h+1,j-delta_w:j+delta_w+1]*kernel)
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = np.flip(np.flip(g, axis=0), axis=1)
    out = conv_fast(f, g[0:-1])
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g_mean = np.sum(g)/np.size(g)
    g = g - g_mean
    g = np.flip(np.flip(g, axis = 0), axis = 1)
    out = conv_fast(f, g[0:-1])
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    if g.shape[0]%2 == 0:
        g = g[0:-1]
    if g.shape[1]%2 == 0:
        g = g[:,0:-1]
    
    Hi, Wi = g.shape
    Hk, Wk = f.shape
    out = np.zeros((Hi, Wi))
    
    normalized_filter = (g - np.mean(g)) / np.std(g)
    

    delta_h = int((Hk - 1) / 2)
    delta_w = int((Wk - 1) / 2)
    
    for image_h in range(delta_h, Hi - delta_h):
        for image_w in range(delta_w, Hi - delta_w):
            image_patch = f[image_h - delta_h : image_h + delta_h + 1, image_w - delta_w : image_w + delta_w + 1]
            normalized_image_patch = (image_patch - np.mean(image_patch))/np.std(image_patch)
            out[image_h][image_w] = np.sum(normalized_image_patch * normalized_filter)
    ### END YOUR CODE

    return out
