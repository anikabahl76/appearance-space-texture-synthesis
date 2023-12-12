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
        params['h'] = np.power(2, np.arange(start=l-1, stop=-1, step=-1))

    params['r'] =  np.power(.75, np.arange(start=l-1, stop=-1, step=-1))

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


def isometric_correction(S, Ept, Nt_Ept, pca, near_nbs, m):
    ## compute Ns
    Ns = np.zeros((S.shape[0], S.shape[1], 32))

    i, j = np.meshgrid(np.arange(S.shape[0]), np.arange(S.shape[1]), indexing='ij')
    
    S = np.pad(S, ((2,2), (2,2), (0,0)), mode='reflect').astype(np.int32)
    Ept = np.pad(Ept, ((2,2), (2,2), (0,0)), mode='reflect').astype(np.int32)
    
    # TODO: figure out what the FAWK is going wrong with upsample/jitter that is breaking this shawty down so bad
    ## (it's probably the jitter)
    np.clip(S, 0, 31, out=S)

    for k in range(4):
        for l, corr_delta in enumerate(CORR_DELTA):
            for corr_delta_p in CORR_DELTA_PRIME[l]:
                Ept_idx = S[i+2 + corr_delta[0] + corr_delta_p[0], j+2 + corr_delta[1] + corr_delta_p[1]] - corr_delta_p[1]
                Ns[i, j, 8*k:8*(k+1)] += Ept[Ept_idx[..., 0] + 2, Ept_idx[..., 1] + 2]
    Ns = Ns / 3
    Ns = np.reshape(Ns, (-1, 32))
    Ns = pca.transform(Ns)
    S = S[2:-2, 2:-2]
    Ept = Ept[2:-2, 2:-2]
    Ns = np.reshape(Ns, (S.shape[0], S.shape[1], 8))

    print(S.shape)

    ## requisite variables
    ## subpass time!
    for i in range(SQRT_S):
        for j in range(SQRT_S):

            indices = np.stack(np.meshgrid(np.arange(i,S.shape[0],SQRT_S), np.arange(j,S.shape[1],SQRT_S), indexing='ij'), axis=2)
            indices_shape = indices.shape
            indices = indices.reshape((-1, 2)) ## is this necessary?

            # TODO: should we compare against S or corrS neighbors once first subpass is complete?
            candidates = []
            for delta in SUBPASS_DELTA:
                all_nbs = indices + delta
                np.clip(all_nbs, 0, S.shape[0]-1, out=all_nbs)
                filled_nbs_coord = S[all_nbs[..., 0], all_nbs[..., 1]]
                filled_near_nbs = near_nbs[filled_nbs_coord[..., 0], filled_nbs_coord[..., 1]]
                filled_near_nbs = filled_near_nbs - delta
                filled_nbs_coord = filled_nbs_coord - delta
                np.clip(filled_near_nbs, 0, m-1, out=filled_near_nbs)
                np.clip(filled_nbs_coord, 0, m-1, out=filled_nbs_coord)
                candidates.append(np.stack([filled_nbs_coord, filled_near_nbs], axis=2))
            
            candidates = np.array(candidates)
            candidates = np.swapaxes(candidates, 0, 1)
            candidates = np.reshape(candidates, (candidates.shape[0], 18, 2))
            candidate_vecs = Nt_Ept[candidates[..., 0], candidates[..., 1]] ## what does this look like?

            differences = candidate_vecs - Ns[indices[..., np.newaxis, 0], indices[..., np.newaxis, 1]]
            distances = np.linalg.norm(differences, axis=2) ## check axis
            min_idx = np.argmin(distances, axis=1) ## check axis

            selections = candidates[np.arange(distances.shape[0]), min_idx]
        
            indices = indices.reshape(indices_shape)
            S[indices[..., 0], indices[..., 1]] = np.reshape(selections, (indices.shape[0], indices.shape[1], 2))

    return S


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


def synthesize_texture(E, synth_size=32, synth_mode="iso", with_pyramid=False):
    E = E.astype(np.float32)
    print("building gaussian stack... or pyramid... but pyramids are for losers...")
    E_stack, with_pyramid = build_gaussian(E, with_pyramid)
    params = build_param_dict(E, with_pyramid)
    ASV_stack = []
    print("building appearance space vectors...")
    for E_prime in E_stack:
        E_prime_tilde, _ = get_appearance_space_vector(E_prime, 2)
        E_prime_tilde = E_prime_tilde.astype(np.float32)
        ASV_stack.append(E_prime_tilde)
    neighbors_stack = []
    nearest_neighbors_stack = []
    print("building neighborhoods and computing nearest neighbors...")
    for E_prime_tilde in ASV_stack:
        nbs, pca = get_neighborhoods(E_prime_tilde)
        nbs = nbs.astype(np.float32)
        neighbors_stack.append((nbs, pca))
        nearest_neighbors_stack.append(get_nearest_neighbors(nbs))

    S_i = np.zeros((synth_size, synth_size, 2)).astype(np.int32)

    print("beginning coarse to fine synthesis...")
    for i in range(params['l']):
        h = params['h'][i]
        r = params['r'][i]
        E_prime_tilde = ASV_stack[i]
        nbhds = neighbors_stack[i][0]
        pca = neighbors_stack[i][1]
        near_nbs = nearest_neighbors_stack[i]
        print("synthesizing level {}...".format(i))
        print("upsampling...")
        S_i = upsample(S_i, params['m'], h, with_pyramid)
        print("jittering...")
        S_i = jitter(S_i, params['m'], h, r, i)
        if i > 2:
            print("correcting...")
            for _ in range(CORR_PASSES):
                if synth_mode == "iso":
                    S_i = isometric_correction(S_i, E_prime_tilde, nbhds, pca, near_nbs, params['m'])
                elif synth_mode == "aniso":
                    S_i = anisometric_correction(S_i, E_prime_tilde)

    return S_i


if __name__ == "__main__":
    print("reading image...")
    E = cv2.imread("../data/texture2.jpg")
    E = cv2.cvtColor(E, cv2.COLOR_BGR2RGB)
    print("read image of size: ", E.shape)
    print("synthesizing texture...")
    S = synthesize_texture(E)
    print("done! u ate that up girlie pop!")
    plt.imshow(S)
    plt.show()
    print("saving...")
    io.imsave("../out/synth_texture2.jpg", S)


