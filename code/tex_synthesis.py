import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.util import view_as_windows
from appearance_space import get_appearance_space_vector, get_neighborhoods, get_nearest_neighbors
from scipy.spatial import l2_distance 

C = 2 ## number of passes
S = 2 ## number of subpasses
UPSAMPLE_DELTA = np.expand_dims(np.array([[0,0], [0,1], [1,0], [0,1]]), (0,1))
CORR_DELTA = np.expand_dims(np.array([[1,1], [1,-1], [-1,1], [-1,-1]]), (0,1))
CORR_DELTA_PRIME = np.expand_dims(np.array([[[0,0], [1,0], [0,1]], [[0,0], [1,0], [0,-1]], [[0,0], [-1,0], [0,1]], [[0,0], [-1,0], [0,-1]]]), (0,1))
HASH_SEED = 1290 ## seed for jittering


def build_param_dict(E, with_pyramid):
    params = {}
    params['m'] = E.shape[0]
    l = int(np.log2(params['m']))
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
        return gaussian_pyramid(img, depth - 1)
    else:
        return gaussian_stack(img, depth)


def gaussian_stack(img, depth=2):
    final_gaussian = []
    for i in range(depth):
        w = 2**(depth - i)
        s = depth - i
        t = (((w - 1)/2)-0.5)/s
        blur = gaussian(img, sigma=s, truncate=t)
        final_gaussian.append(blur)
    return final_gaussian


def upsample(S, m, h, with_pyramid):
    new_S = np.zeros(S.shape[0] * 2, S.shape[1] * 2, 2)

    if with_pyramid:
        new_S[2*S+UPSAMPLE_DELTA] = np.mod(2*S + h*UPSAMPLE_DELTA, m)
    else:
        rhs = np.floor(h * np.subtract(UPSAMPLE_DELTA, 0.5))
        new_S[2*S+UPSAMPLE_DELTA] = np.mod(S + rhs, m)
        
    # if we need loops
    # I, J = np.meshgrid(np.arange(S.shape[0]), np.arange(S.shape[1]))
    # for i, j in zip(I.flatten(), J.flatten()):
    #     for delta in np.array([[0,0], [0,1], [1,0], [0,1]]):
    #         p = S[i, j]
    #         new_S[2*p+delta] = np.mod(2*S[p]+h*delta, m)
    
    return new_S


def jitter(S, m, h, r, l):
    J = np.zeros(S.shape[0], S.shape[1], 2)
    J = np.floor(h * hash_coords(S, m, l) * r + np.array[[0.5, 0.5]])
    return S + J


def hash_coords(S, m, l):
    '''
    Hash function that generates a subpixel shift for each pixel in a matrix
    '''
    np.random.seed(HASH_SEED)
    return np.random.rand(S.shape[0], S.shape[1], 2) * (m/(2**(l-1)) - m/(2**l))


def isometric_correction(S, Ept, Nt_Ept, pca, near_nbs):
    for i in range(4):
        # TODO: implement subpasses, possibly without for loop?
        pass

    Ns = np.zeros(S.shape[0], S.shape[1], 32)

    for i in range(4):
        Ns[..., 8*i:8*(i+1)] = np.sum(Ept[S + CORR_DELTA[..., i, :] + CORR_DELTA_PRIME[..., i, :, :]] - CORR_DELTA_PRIME[..., i, :], axis=2) / 3
    
    Ns = pca.transform(Ns)
    Ns = np.reshape(Ns, (S.shape[0], S.shape[1], 8))

    # find filled in neighbors in 3x3 window
    # compute delta from p to each filled neighbor
    # add delta to filled neighbors E-coordinates. this gets possible C1s.
    # for each C1, add the nearest neighbo to list as well. this gets possible C2s.
    # compute the distance between Ns[p] and Ne[C1]/Ne[C2]; find the candidate that minimizes this distance.
    # set s to be this candidate.

def anisometric_correction(S, E):
    pass


def synthesize_texture(E, synth_size=256, synth_mode="iso", with_pyramid=False):
    params = build_param_dict(E, with_pyramid)
    E_stack = build_gaussian(E, with_pyramid)
    ASV_stack = []
    for E_prime in E_stack:
        E_prime_tilde, _ = get_appearance_space_vector(E_prime, 2)
        ASV_stack.append(E_prime_tilde)
    neighbors_stack = []
    nearest_neighbors_stack = []
    for E_prime_tilde in ASV_stack:
        nbs, pca = get_neighborhoods(E_prime_tilde)
        neighbors_stack.append((nbs, pca))
        nearest_neighbors_stack.append(get_nearest_neighbors(nbs))

    S_i = np.zeros((synth_size, synth_size, 2))

    for i in range(params['l']+1):
        h = params['h'][i]
        r = params['r'][i]
        E_prime_tilde = ASV_stack[i]
        nbhds = neighbors_stack[i][0]
        pca = neighbors_stack[i][1]
        near_nbs = nearest_neighbors_stack[i][0]
        S_i = upsample(S_i, params['m'], h, with_pyramid)
        S_i = jitter(S_i, params['m'], h, r, i)
        if i > 2:
            for _ in range(C):
                if synth_mode == "iso":
                    S_i = isometric_correction(S_i, E_prime_tilde, nbhds, pca, near_nbs)
                elif synth_mode == "aniso":
                    S_i = anisometric_correction(S_i, E_prime_tilde)

    return S_i


if __name__ == "__main__":
    # im = cv2.imread("data/texture3.png", cv2.COLOR_BGR2RGB)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # stack, l = build_gaussian(im, False)
    # for i in range(l):
    #     plt.imshow(stack[i])
    #     plt.show()
    print(CORR_DELTA_PRIME[...,0,:,:].shape)
    print(CORR_DELTA[..., 0, ].shape)

