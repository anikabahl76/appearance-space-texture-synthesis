import math
import numpy as np
import hashlib


C = 2 ## number of passes
S = 2 ## number of subpasses
DELTA = np.array([[0,0], [0,1], [1,0], [1,1]]) ## 4 possible subpixel shifts IN (Y,X) coords

def build_gaussian(img, with_pyramid):
    pass


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


def jitter(S, h, r):
    J = np.zeros(S.shape[0], S.shape[1], 2)
    J = np.floor(h * hash_coords(S) * r + np.array[[0.5, 0.5]])
    return S + J


def hash_coords(S):
    '''
    Hash function that generates a subpixel shift for each pixel in a matrix
    '''
    y_hash = np.zeros(S.shape[0], S.shape[1])
    x_hash = np.zeros(S.shape[0], S.shape[1])

    y_hash = hash(S[:,:,0])
    x_hash = hash(S[:,:,1])

    stacked = np.stack((y_hash, x_hash), axis=2)
    hashed = stacked + 1 
    normed = hashed / np.power(10,len(str(hashed.max())))
    hashed = normed * np.power(-1,np.mod(hashed,2))

    return hashed


def isometric_correction(S, E):
    pass


def anisometric_correction(S, E):
    pass


def synthesize_texture(E, E_prime, synth_mode="iso", with_pyramid=False):
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