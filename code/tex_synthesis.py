import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.util import view_as_windows



C = 2 ## number of passes
S = 2 ## number of subpasses
DELTA = np.array([[0,0], [0,1], [1,0], [0,1]])
M = np.array([[[0,0],[0,0]], [[1,0],[0,0]], [[0,0],[0,1]]])
HASH_SEED = 1290 ## seed for jittering


def build_param_dict(m, l, with_pyramid):
    params = {}
    params['m'] = m
    params['l'] = l
    
    if with_pyramid:
        for i in range(l+1):
            params['h'][i] = 1
    else:
        for i in range(l+1):
            params['h'][i] = 2**(l-i)

    for i in range(l+1):
        params['r'][i] = (3/4)**(l-i)

    return params


def gaussian_pyramid(img, depth=2, downsample=True):
    downsampled_img = [img]
    for i in range(depth):
        next_img = blur_and_downsample(downsampled_img[-1], downsample=downsample)
        downsampled_img.append(next_img)

    downsampled_img.reverse()

    return downsampled_img


def blur_and_downsample(im, downsample=True):
    blur = gaussian(im)
    if downsample:
        return blur[::2,::2]
    else: 
        return blur


def build_gaussian(img, with_pyramid):
    img_size = img.shape[0]
    depth = int(np.log2(img_size))
    if with_pyramid or depth < 5:
        return gaussian_pyramid(img, depth - 1), depth
    else:
        return gaussian_stack(img, depth), depth

def gaussian_stack(img, depth=2):
    # augmented_image = np.pad(img, ((img.shape[0]//2, img.shape[0]//2), (img.shape[1]//2, img.shape[1]//2), (0, 0)), "reflect")
    # windows = view_as_windows(augmented_image, img.shape)
    # a, b, c, d, e, f = windows.shape
    # windows = np.reshape(windows, (a * b * c, d, e, f))
    # all_gaussians = np.zeros((windows.shape[0], depth + 1, img.shape[0], img.shape[1], img.shape[2]))
    # for i in range(windows.shape[0]):
    #     current_window = windows[i]
    #     current_gaussian = gaussian_pyramid(current_window, depth, downsample=False)
    #     current_gaussian = np.array(current_gaussian)
    #     all_gaussians[i, :, :, :, :] = current_gaussian
    # print(all_gaussians.shape)
    # final_gaussian = np.average(all_gaussians, axis=0)
    # print(final_gaussian.shape)
    # final_gaussian = gaussian_pyramid(img, depth, downsample=False)
    final_gaussian = []
    for i in range(depth):
        w = 2**(depth - i)
        s = depth - i
        t = (((w - 1)/2)-0.5)/s
        blur = gaussian(img, sigma=s, truncate=t)
        final_gaussian.append(blur)
    return final_gaussian

def upsample(S, m, h, with_pyramid, synth_mode="iso", J=None):
    # TODO: check if m should be the full-sized exemplar m or the appropriate pyramid/stack's m
    new_S = np.zeros(S.shape[0] * 2, S.shape[1] * 2, 2)

    if synth_mode == "aniso": # not sure if this is the correct way
        h = J

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


def jitter(S, h, r, m, l):
    J = np.zeros(S.shape[0], S.shape[1], 2)
    J = np.floor(h * hash_coords(S, m, l) * r + np.array[[0.5, 0.5]])
    return S + J


def hash_coords(S, m, l):
    '''
    Hash function that generates a subpixel shift for each pixel in a matrix
    '''
    np.random.seed(HASH_SEED)
    return np.random.rand(S.shape[0], S.shape[1], 2) * (m/(2**(l-1)) - m/(2**l))


def isometric_correction(S, E, E_prime):
    N_s_p = N_e_u = np.zeros(S.shape)

    N_s_p[S+DELTA] = sum(E_prime[S+DELTA+(M*DELTA)]-(M*DELTA)) / 3
    N_e_u[S+DELTA] = sum(E_prime[E+DELTA+(M*DELTA)]-(M*DELTA)) / 3
    C_p = [] # need to add C(p) stuff
    S = np.argmin(abs(N_s_p - N_e_u[C_p])) 

    return S


def anisometric_correction(S, E, E_prime, J):
    N_s_p = N_e_u = np.zeros(S.shape)
    J_inv = np.linalg.inv(J)
    j = (J_inv * DELTA) + (J_inv * (M * DELTA))

    N_s_p[S+DELTA] = sum(E_prime[S+j]-(J*j)+DELTA) / 3
    N_e_u[S+DELTA] = sum(E_prime[E+j]-(J*j)+DELTA) / 3
    C_p = [] # need to add C(p) stuff
    S = np.argmin(abs(N_s_p - N_e_u[C_p])) 

    return S


def synthesize_texture(E, E_prime, synth_mode="iso", with_pyramid=True):
    E_stack, l = build_gaussian(E, with_pyramid)

    S_stack = []
    
    W = E_stack[0].shape(1) // 2
    H = E_stack[0].shape(0) // 2

    S_i = np.zeros((H, W, 2))

    for i in range(l+1):
        E_i = E_stack[i]
        S_i = upsample(S_i, with_pyramid)
        S_i = jitter(S_i)
        if l > 2:
            for _ in range(C):
                if synth_mode == "iso":
                    S_i = isometric_correction(S_i, E_i)
                elif synth_mode == "aniso":
                    S_i = anisometric_correction(S_i, E_i)

    return S_i


if __name__ == "__main__":
    im = cv2.imread("data/texture3.png", cv2.COLOR_BGR2RGB)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    stack, l = build_gaussian(im, False)
    for i in range(l):
        plt.imshow(stack[i])
        plt.show()

