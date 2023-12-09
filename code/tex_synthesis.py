import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.util import view_as_windows


C = 2 ## number of passes
S = 2 ## number of subpasses
DELTA = np.array([[0,0], [0,1], [1,0], [0,1]])

def gaussian_pyramid(img, depth=6):
    downsampled_img = [img]
    for i in range(depth):
        next_img = downsample(downsampled_img[-1])
        downsampled_img.append(next_img)

    downsampled_img.reverse()

    return downsampled_img

def downsample(im):
    blur = gaussian(im)
    return blur[::2,::2]

def build_gaussian(img):
    if with_pyramid:
        return gaussian_pyramid(img)
    else:
        pass

def gaussian_stack(img):
    img_shape = img.shape
    print(img.shape)
    augmented_image = np.pad(img, ((img_shape[0]//2, img_shape[0]//2), (img_shape[1]//2, img_shape[1]//2), (0, 0)), "reflect")
    print(augmented_image.shape)
    plt.imshow(augmented_image)
    plt.show()
    windows = view_as_windows(augmented_image, img_shape)
    a, b, c, d, e, f = windows.shape
    windows = np.reshape(windows, (a * b * c, d, e, f))
    all_gaussians = 
    for i in range(windows.shape[0]):
        current_window = windows[i]
        current_gaussian = gaussian_pyramid(current_window)

im = cv2.imread("data/texture2.jpg", cv2.COLOR_BGR2RGB)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
gaussian_stack(im)



def upsample(S, m, h, with_pyramid):
    # TODO: check if m should be the full-sized exemplar m or the appropriate pyramid/stack's m
    new_S = np.zeros(S.shape[0] * 2, S.shape[1] * 2, 2)

    if with_pyramid:
        new_S[2*S+DELTA] = np.mod(2*S + h*DELTA, m)
    else:
        rhs = np.floor(h * np.subtract(DELTA, 0.5))
        new_S[2*S+DELTA] = np.mod(S + rhs, m)
        
    # if we need loops
    # I, J = np.meshgrid(np.arange(S.shape[0]), np.arange(S.shape[1]))
    # for i, j in zip(I.flatten(), J.flatten()):
    #     for delta in np.array([[0,0], [0,1], [1,0], [0,1]]):
    #         p = S[i, j]
    #         new_S[2*p+delta] = np.mod(2*S[p]+h*delta, m)
    
    return new_S


def jitter(S):
    pass


def isometric_correction(S, E):
    pass


def anisometric_correction(S, E):
    pass


def synthesize_texture(E, E_prime, synth_mode="iso", is_toroidal=False):
    E_stack, l = build_gaussian(E)

    S_stack = []
    
    W = E_stack[0].shape(1) // 2
    H = E_stack[0].shape(0) // 2

    S_i = np.zeros((H, W, 2))

    for i in range(l+1):
        E_i = E_stack[i]
        S_i = upsample(S_i, is_toroidal)
        S_i = jitter(S_i)
        if l > 2:
            for _ in range(C):
                if synth_mode == "iso":
                    S_i = isometric_correction(S_i, E_i)
                elif synth_mode == "aniso":
                    S_i = anisometric_correction(S_i, E_i)

    return S_i