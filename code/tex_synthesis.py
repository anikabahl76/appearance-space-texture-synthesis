import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from appearance_space import get_appearance_space_vector, get_neighborhoods, get_nearest_neighbors
from skimage import io

CORR_PASSES = 2
SQRT_S = 2
CORR_DELTA = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
CORR_DELTA_PRIME = np.array([[[0,0], [1,0], [0,1]], [[0,0], [1,0], [0,-1]], [[0,0], [-1,0], [0,1]], [[0,0], [-1,0], [0,-1]]])
SUBPASS_DELTA = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0,1], [1, -1], [1, 0], [1, 1]])
HASH_SEED = 2 ## seed for jittering


def build_param_dict(E):
    params = {}

    params['m'] = E.shape[0]
    l = int(np.log2(params['m']))
    params['l'] = l
    
    params['h'] = np.power(2, np.arange(start=l-1, stop=-1, step=-1))
    params['r'] = np.repeat(0.05, l)

    return params


def build_gaussian_stack(img):
    img_size = img.shape[0]
    depth = int(np.log2(img_size))

    final_gaussian = []
    for i in range(depth):
        w = 2**(depth - i)
        s = depth - i
        t = (((w - 1)/2)-0.5)/s
        blur = gaussian(img, sigma=s, truncate=t)
        final_gaussian.append(blur)
    return final_gaussian


def upsample(S, m, h, synth_mode="iso", J=None):
    new_S = np.zeros((2*S.shape[0], 2*S.shape[1], 2))

    if synth_mode == "aniso": # not sure if this is the correct way
        h = J

    i, j = np.meshgrid(np.arange(S.shape[0]), np.arange(S.shape[1]), indexing='ij')
    new_S[::2, ::2] = S
    new_S[1::2, ::2,0] = S[i,j,0] + (m/h)
    new_S[1::2, ::2,1] = S[i,j,1]
    new_S[::2, 1::2,0] = S[i,j,0]
    new_S[::2, 1::2,1] = S[i,j,1] + (m/h)
    new_S[1::2, 1::2] = S + (m/h)
    new_S = np.mod(new_S, m)

    return new_S


def jitter(S, m, l, h, r):
    J = np.floor(hash_coords(S) * r * h + np.array([[0.5, 0.5]]))
    S = S+J
    S = np.mod(S, m)
    return S


def hash_coords(S):
    '''
    Hash function that generates a subpixel shift for each pixel in a matrix
    '''
    np.random.seed(HASH_SEED)
    return 2 * np.random.rand(S.shape[0], S.shape[1], 2) - 1


def isometric_correction(S, Ept, Nt_Ept, pca, near_nbs, m):
    # TODO: move this inside of subpass loop?
    # compute Ns
    Ns = np.zeros((S.shape[0], S.shape[1], 32))

    y, x = np.meshgrid(np.arange(S.shape[0]), np.arange(S.shape[1]), indexing='ij')
    
    S = np.pad(S, ((2,2), (2,2), (0,0)), mode='reflect').astype(np.int32)
    Ept = np.pad(Ept, ((2,2), (2,2), (0,0)), mode='reflect').astype(np.int32)
    
    # TODO: figure out what is going wrong here that is breaking this shawty down so bad
    # np.clip(S, 0, 31, out=S)

    for k in range(4):
        for l, corr_delta in enumerate(CORR_DELTA):
            for corr_delta_p in CORR_DELTA_PRIME[l]:
                Ept_idx = S[y+2 + corr_delta[0] + corr_delta_p[0], x+2 + corr_delta[1] + corr_delta_p[1]] - corr_delta_p[1]
                Ns[y, x, 8*k:8*(k+1)] += Ept[Ept_idx[y,x,0] + 2, Ept_idx[y,x,1] + 2]
    Ns = Ns / 3
    Ns = np.reshape(Ns, (-1, 32))
    Ns = pca.transform(Ns)
    S = S[2:-2, 2:-2]
    Ept = Ept[2:-2, 2:-2]
    Ns = np.reshape(Ns, (S.shape[0], S.shape[1], 8))

    for i in range(SQRT_S):
        for j in range(SQRT_S):

            indices = np.stack(np.meshgrid(np.arange(i,S.shape[0],SQRT_S), np.arange(j,S.shape[1],SQRT_S), indexing='ij'), axis=2)
            indices_shape = indices.shape

            # Ns = np.zeros((S.shape[0], S.shape[1], 32))
            # y, x = np.meshgrid(np.arange(S.shape[0]), np.arange(S.shape[1]), indexing='ij')
            
            # S = np.pad(S, ((2,2), (2,2), (0,0)), mode='reflect').astype(np.int32)
            # Ept = np.pad(Ept, ((2,2), (2,2), (0,0)), mode='reflect').astype(np.int32)
            
            # # TODO: figure out what is going wrong here that is breaking this shawty down so bad
            # np.clip(S, 0, 31, out=S)

            # for k in range(4):
            #     for l, corr_delta in enumerate(CORR_DELTA):
            #         for corr_delta_p in CORR_DELTA_PRIME[l]:
            #             Ept_idx = S[y+2 + corr_delta[0] + corr_delta_p[0], x+2 + corr_delta[1] + corr_delta_p[1]] - corr_delta_p[1]
            #             Ns[y, x, 8*k:8*(k+1)] += Ept[Ept_idx[..., 0] + 2, Ept_idx[..., 1] + 2]
            # Ns = Ns / 3
            # Ns = np.reshape(Ns, (-1, 32))
            # Ns = pca.transform(Ns)
            # S = S[2:-2, 2:-2]
            # Ept = Ept[2:-2, 2:-2]
            # Ns = np.reshape(Ns, (S.shape[0], S.shape[1], 8))

            indices = indices.reshape((-1, 2)) ## is this necessary?

            # TODO: should we compare against S or corrS neighbors once first subpass is complete?
            candidates = []
            for delta in SUBPASS_DELTA:
                all_nbs = indices + delta
                np.clip(all_nbs, 0, S.shape[0]-1, out=all_nbs)
                filled_nbs_coord = S[all_nbs[..., 0], all_nbs[..., 1]]
                filled_nbs_coord = filled_nbs_coord - delta
                np.clip(filled_nbs_coord, 0, m-1, out=filled_nbs_coord)
                filled_near_nbs = near_nbs[filled_nbs_coord[..., 0], filled_nbs_coord[..., 1]]
                np.clip(filled_near_nbs, 0, m-1, out=filled_near_nbs)
                
                candidates.append(np.stack([filled_nbs_coord, filled_near_nbs], axis=2))
            
            candidates = np.array(candidates)
            candidates = np.swapaxes(candidates, 0, 1)
            candidates = np.reshape(candidates, (candidates.shape[0], 2 * SUBPASS_DELTA.shape[0], 2))
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


def synthesize_texture(E, synth_size=128, synth_mode="iso"):
    E = E.astype(np.float32)
    print("building gaussian stack... or pyramid... but pyramids are for losers...")
    E_stack = build_gaussian_stack(E)
    params = build_param_dict(E)
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

    S_i = np.stack(np.meshgrid(np.arange(synth_size), np.arange(synth_size), indexing='ij'), axis=2)
    S_i = np.mod(S_i, params['m'])

    E_Si = convert_coords_to_image(E, S_i)
    io.imshow(E_Si)
    io.show()

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
        S_i = upsample(S_i, params['m'], h)
        
        E_Si = convert_coords_to_image(E, S_i)
        io.imshow(E_Si)
        io.show()
        
        print("jittering...")
        S_i = jitter(S_i, params['m'], i, h, r)
        
        E_Si = convert_coords_to_image(E, S_i)
        io.imshow(E_Si)
        io.show()
        
        if i >= 2:
            print("correcting...")
            for _ in range(CORR_PASSES):
                if synth_mode == "iso":
                    S_i = isometric_correction(S_i, E_prime_tilde, nbhds, pca, near_nbs, params['m'])
                elif synth_mode == "aniso":
                    S_i = anisometric_correction(S_i, E_prime_tilde)

        E_Si = convert_coords_to_image(E, S_i)
        io.imshow(E_Si)
        io.show()
        

    return S_i.astype(np.int32)


def convert_coords_to_image(E, S):
    S = S.astype(np.int32)
    E_S = np.zeros((S.shape[0], S.shape[1], 3))
    E_S = E[S[..., 0], S[..., 1]]
    return E_S.astype(np.int32)


if __name__ == "__main__":
    print("reading image...")
    E = cv2.imread("../data/texture3.png")
    E = cv2.cvtColor(E, cv2.COLOR_BGR2RGB)
    print("read image of size: ", E.shape)
    print("synthesizing texture...")
    S = synthesize_texture(E)
    print("done! u ate that up girlie pop!")
    E_S = convert_coords_to_image(E, S)
    plt.imshow(E_S)
    plt.show()
    print("saving...")
    io.imsave("../out/synth_texture2.png", E_S)