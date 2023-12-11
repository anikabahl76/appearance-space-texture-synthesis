import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.util import view_as_windows
from appearance_space import get_appearance_space_vector, get_neighborhoods, get_nearest_neighbors
from scipy.spatial.distance import euclidean

CORR_PASSES = 2
SQRT_S = 2
UPSAMPLE_DELTA = np.expand_dims(np.array([[0,0], [0,1], [1,0], [0,1]]), (0,1))
CORR_DELTA = np.expand_dims(np.array([[1,1], [1,-1], [-1,1], [-1,-1]]), (0,1))
CORR_DELTA_PRIME = np.expand_dims(np.array([[[0,0], [1,0], [0,1]], [[0,0], [1,0], [0,-1]], [[0,0], [-1,0], [0,1]], [[0,0], [-1,0], [0,-1]]]), (0,1))
SUBPASS_DELTA = np.expand_dims(np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0,0], [0,1], [1, -1], [1, 0], [1, 1]]), (0,1))
HASH_SEED = 1290 ## seed for jittering


def build_param_dict(E, with_pyramid):
    params = {}
    params['m'] = E.shape[0]
    l = int(np.log2(params['m']))
    params['l'] = l
    
    if with_pyramid:
        params['h'] = np.ones(l)
    else:
        params['h'] = np.power(2, np.arange(start=l, stop=-1, step=-1))

    params['r'] =  np.power(.75, np.arange(start=l, stop=-1, step=-1))

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
        with_pyramid = True
        return gaussian_pyramid(img, depth - 1), with_pyramid
    else:
        with_pyramid = False
        return gaussian_stack(img, depth), with_pyramid


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

    ## compute Ns
    Ns = np.zeros(S.shape[0], S.shape[1], 32)
    for i in range(4):
        Ns[..., 8*i:8*(i+1)] = np.sum(Ept[S + CORR_DELTA[..., i, :] + CORR_DELTA_PRIME[..., i, :, :]] - CORR_DELTA_PRIME[..., i, :], axis=2) / 3
    Ns = pca.transform(Ns)
    Ns = np.reshape(Ns, (S.shape[0], S.shape[1], 8))

    ## requisite variables
    corrS = np.ones_like(S) * np.inf
    corrS = np.pad(corrS, ((1,1), (1,1), (0,0)), mode='constant', constant_values=np.inf)

    ## subpass time!
    for i in range(SQRT_S):
        for j in range(SQRT_S):

            indices = np.stack(np.meshgrid(np.arange(i,S.shape[0],SQRT_S), np.arange(j,S.shape[1],SQRT_S), indexing='ij'), axis=2)

            # ## compute Ns
            # Ns = np.zeros(S.shape[0], S.shape[1], 32)
            # interS = np.where(corrS[indices[..., 0] + 1, indices[..., 1] + 1] != np.inf, corrS[indices[..., 0] + 1, indices[..., 1] + 1], S[indices[..., 0], indices[..., 1]])
            # for i in range(4):
            #     Ns[..., 8*i:8*(i+1)] = np.sum(Ept[S + CORR_DELTA[..., i, :] + CORR_DELTA_PRIME[..., i, :, :]] - CORR_DELTA_PRIME[..., i, :], axis=2) / 3
            # Ns = pca.transform(Ns)
            # Ns = np.reshape(Ns, (S.shape[0], S.shape[1], 8))

            indices = indices.reshape((-1, 2)) ## is this necessary?
            all_nbs = indices + SUBPASS_DELTA
            filled_nbs = np.where(corrS[all_nbs[..., 0] + 1, [..., 1] + 1] != np.inf, all_nbs, np.array([[np.inf,np.inf]])) ## +1 because of padding
            filled_nbs_coord = np.where(filled_nbs != np.inf, S[filled_nbs[..., 0], filled_nbs[..., 1]], np.array([[np.inf,np.inf]]))
            filled_near_nbs = np.where(filled_nbs_coord != np.inf, near_nbs[filled_nbs_coord[..., 0], filled_nbs_coord[..., 1]], np.array([[np.inf,np.inf]]))
            filled_near_nbs = filled_near_nbs - SUBPASS_DELTA

            candidates = np.stack([filled_nbs_coord, filled_near_nbs], axis=2)
            candidates = np.reshape(candidates, (-1, 2)) ## is this necessary?
            candidate_vecs = Nt_Ept[candidates[..., 0], candidates[..., 1]] ## what does this look like?

            distances = np.linalg.norm(candidate_vecs - Ns[indices[..., 0], indices[..., 1]], axis=1) ## check axis
            min_idx = np.argmin(distances, axis=0) ## check axis
        
            corrS[indices[..., 0] + 1, indices[..., 1] + 1] = candidates[min_idx]


def anisometric_correction(S, E):
    pass


def synthesize_texture(E, synth_size=256, synth_mode="iso", with_pyramid=False):
    E = E.astype(np.float32)
    E_stack, with_pyramid = build_gaussian(E, with_pyramid)
    params = build_param_dict(E, with_pyramid)
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
    print("reading image...")
    E = cv2.imread("../data/texture3.png")
    E = cv2.cvtColor(E, cv2.COLOR_BGR2RGB).astype(np.float32)
    with_pyramid = False
    print("building gaussian stack...")
    E_stack, with_pyramid = build_gaussian(E, with_pyramid)
    params = build_param_dict(E, with_pyramid)
    print("building appearance space vectors...")
    ASV_stack = []
    for E_prime in E_stack:
        E_prime_tilde, _ = get_appearance_space_vector(E_prime, 2)
        ASV_stack.append(E_prime_tilde)
    print("building neighborhoods...")
    neighbors_stack = []
    nearest_neighbors_stack = []
    for E_prime_tilde in ASV_stack:
        nbs, pca = get_neighborhoods(E_prime_tilde)
        neighbors_stack.append((nbs, pca))
        nearest_neighbors_stack.append(get_nearest_neighbors(nbs))
