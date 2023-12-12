import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.util import view_as_windows
from appearance_space import get_appearance_space_vector, get_neighborhoods, get_nearest_neighbors
from scipy.spatial.distance import euclidean
from skimage import io

CORR_PASSES = 2
SQRT_S = 2
UPSAMPLE_DELTA = np.array([[0,0], [0,1], [1,0], [0,1]])
CORR_DELTA = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
CORR_DELTA_PRIME = np.array([[[0,0], [1,0], [0,1]], [[0,0], [1,0], [0,-1]], [[0,0], [-1,0], [0,1]], [[0,0], [-1,0], [0,-1]]])
SUBPASS_DELTA = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0,0], [0,1], [1, -1], [1, 0], [1, 1]])
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

def upsample(S, m, h, with_pyramid, synth_mode="iso", J=None):
    # TODO: check if m should be the full-sized exemplar m or the appropriate pyramid/stack's m
    new_S = np.zeros((2*S.shape[0], 2*S.shape[1], 2))

    if synth_mode == "aniso": # not sure if this is the correct way
        h = J

    if with_pyramid:
        for delta in UPSAMPLE_DELTA:
            i, j = np.meshgrid(np.arange(S.shape[0]), np.arange(S.shape[1]), indexing='ij')
            new_S[2*i+delta[0],2*j+delta[1]] = np.mod(2*S + h*delta, m)
    else:
        for delta in UPSAMPLE_DELTA:
            i, j = np.meshgrid(np.arange(S.shape[0]), np.arange(S.shape[1]), indexing='ij')
            rhs = np.floor(h * np.subtract(delta, 0.5))
            new_S[2*i+delta[0], 2*j+delta[1]] = np.mod(S + rhs, m)
        
    return new_S


def jitter(S, m, h, r, l):
    J = np.zeros((S.shape[0], S.shape[1], 2))
    J = np.floor(h * hash_coords(S, m, l) * r + np.array([[0.5, 0.5]]))
    return S + J


def hash_coords(S, m, l):
    '''
    Hash function that generates a subpixel shift for each pixel in a matrix
    '''
    np.random.seed(HASH_SEED)
    return np.random.rand(S.shape[0], S.shape[1], 2) * (m/(2**(l-1)) - m/(2**l))


def isometric_correction(S, Ept, Nt_Ept, pca, near_nbs):
    ## compute Ns
    Ns = np.zeros((S.shape[0], S.shape[1], 32))

    i, j = np.meshgrid(np.arange(S.shape[0]), np.arange(S.shape[1]), indexing='ij')
    
    S = np.pad(S, ((1,1), (1,1), (0,0)), mode='constant', constant_values=0).astype(np.int32)
    
    for k in range(4):
        for l, corr_delta in enumerate(CORR_DELTA):
            for corr_delta_p in CORR_DELTA_PRIME[l]:
                print((S[i+1 + corr_delta[0] + corr_delta_p[0], j+1 + corr_delta[1] + corr_delta_p[1]]).dtype)
                Ns[i, j, 8*k:8*(k+1)] += Ept[S[i+1 + corr_delta[0] + corr_delta_p[0], 
                                               j+1 + corr_delta[1] + corr_delta_p[1]] - corr_delta_p[1]]
    Ns = Ns / 3
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


def anisometric_correction(S, E, E_prime, J):
    # N_s_p = N_e_u = np.zeros(S.shape)
    # J_inv = np.linalg.inv(J)
    # j = (J_inv * DELTA) + (J_inv * (M * DELTA))

    # N_s_p[S+DELTA] = sum(E_prime[S+j]-(J*j)+DELTA) / 3
    # N_e_u[S+DELTA] = sum(E_prime[E+j]-(J*j)+DELTA) / 3
    # C_p = [] # need to add C(p) stuff
    # S = np.argmin(abs(N_s_p - N_e_u[C_p])) 

    # return S
    pass


def synthesize_texture(E, synth_size=256, synth_mode="iso", with_pyramid=False):
    E = E.astype(np.float32)
    print("building gaussian stack... or pyramid... but pyramids are for losers...")
    E_stack, with_pyramid = build_gaussian(E, with_pyramid)
    params = build_param_dict(E, with_pyramid)
    ASV_stack = []
    print("building appearance space vectors...")
    for E_prime in E_stack:
        E_prime_tilde, _ = get_appearance_space_vector(E_prime, 2)
        ASV_stack.append(E_prime_tilde)
    neighbors_stack = []
    nearest_neighbors_stack = []
    print("building neighborhoods and computing nearest neighbors...")
    for E_prime_tilde in ASV_stack:
        nbs, pca = get_neighborhoods(E_prime_tilde)
        neighbors_stack.append((nbs, pca))
        nearest_neighbors_stack.append(get_nearest_neighbors(nbs))


    S_i = np.zeros((synth_size, synth_size, 2)).astype(np.int32)

    print("beginning coarse to fine synthesis...")
    for i in range(params['l']+1):
        h = params['h'][i]
        r = params['r'][i]
        E_prime_tilde = ASV_stack[i]
        nbhds = neighbors_stack[i][0]
        pca = neighbors_stack[i][1]
        near_nbs = nearest_neighbors_stack[i][0]
        print("synthesizing level {}...".format(i))
        print("upsampling...")
        S_i = upsample(S_i, params['m'], h, with_pyramid)
        print("jittering...")
        S_i = jitter(S_i, params['m'], h, r, i)
        if i > 2:
            print("correcting...")
            for _ in range(CORR_PASSES):
                if synth_mode == "iso":
                    S_i = isometric_correction(S_i, E_prime_tilde, nbhds, pca, near_nbs)
                elif synth_mode == "aniso":
                    S_i = anisometric_correction(S_i, E_prime_tilde)

    return S_i


if __name__ == "__main__":
    print("reading image...")
    E = cv2.imread("../data/texture2.jpg")
    E = cv2.cvtColor(E, cv2.COLOR_BGR2RGB)
    print("synthesizing texture...")
    S = synthesize_texture(E)
    print("done! u ate that up girlie pop!")
    plt.imshow(S)
    plt.show()
    print("saving...")
    io.imsave("../out/synth_texture2.jpg", S)


