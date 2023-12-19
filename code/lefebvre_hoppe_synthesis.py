import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from appearance_space import get_appearance_space_vector, get_neighborhoods, get_nearest_neighbors
from skimage import io
from perlin2d import generate_perlin_noise_2d
import argparse


CORR_PASSES = 2
SQRT_S = 2
UPSAMPLE_DELTA = np.array([[0,0], [0,1], [1,0], [1,1]])
CORR_DELTA = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
CORR_DELTA_PRIME = np.array([[[0,0], [1,0], [0,1]], [[0,0], [1,0], [0,-1]], [[0,0], [-1,0], [0,1]], [[0,0], [-1,0], [0,-1]]])
SUBPASS_DELTA = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0,1], [1, -1], [1, 0], [1, 1]])
HASH_SEED = 1290 ## seed for jittering



def build_param_dict(E):
    params = {}

    params['m'] = E.shape[0]
    l = int(np.log2(params['m'])) -1
    params['l'] = l
    
    params['h'] = np.power(2, np.arange(start=l, stop=-1, step=-1))
    params['r'] = np.power(.75, .5 * np.arange(start=l, stop=-1, step=-1))

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


def upsample(S, m, h):
    S_up = np.zeros((2*S.shape[0], 2*S.shape[1], 2))

    i, j = np.meshgrid(np.arange(S.shape[0]), np.arange(S.shape[1]), indexing='ij')
    for delta in UPSAMPLE_DELTA:
        S_up[2*i + delta[0], 2*j + delta[1]] = np.mod(S[i, j] + np.floor(h * (delta - np.array([[0.5, 0.5]]))), m)

    return S_up


def jitter(S, m, h, r):
    H = hash_coords(S, h, r)
    J = np.floor(H[..., np.newaxis] * h * r + np.array([[0.5, 0.5]]))
    J = np.mod(J, h)

    S = S + J
    S = np.mod(S, m)
    return S


def hash_coords(S, h, r):
    '''
    Hash function that generates a subpixel shift for each pixel in a matrix
    '''
    np.random.seed(HASH_SEED + h)
    length = S.shape[0]
    noise = generate_perlin_noise_2d((length//2, length//2), (length//16,length//16), tileable=(True, True))
    noise = cv2.resize(noise, (length, length), interpolation=cv2.INTER_CUBIC)
    noise = np.where(np.abs(noise) > 1-r, np.sign(noise), 0)
    return noise


def isometric_correction(S, Ept, Nt_Ept, pca, near_nbs, m, iter_n):
    
    Ept = Ept.astype(np.float32)
    S_corr = np.zeros(S.shape, dtype=np.int32)

    ## shuffle order of correction depending on if we are on first or second pass
    iter_n = iter_n % 2
    step = 1 if iter_n == 0 else -1
    stop = SQRT_S if iter_n == 0 else -1
    start = 0 if iter_n == 0 else SQRT_S-1

    for i in range(start, stop, step):
        for j in range(start, stop, step):

            indices = np.stack(np.meshgrid(np.arange(i,S.shape[0],SQRT_S), np.arange(j,S.shape[1],SQRT_S), indexing='ij'), axis=2)
            y = indices[..., 0]
            x = indices[..., 1]
            Ns = np.zeros((y.shape[0], y.shape[0], 32), dtype=np.float32)
                        
            for k, corr_delta in enumerate(CORR_DELTA):
                for corr_delta_p in CORR_DELTA_PRIME[k]:
                    idx = np.stack([y + corr_delta[0] + corr_delta_p[0], x + corr_delta[1] + corr_delta_p[1]], axis=2)
                    idx = np.clip(idx, 0, S.shape[0]-1).astype(np.int32)
                    idx = S[idx[..., 0], idx[..., 1]] - corr_delta_p
                    idx = np.clip(idx, 0, m-1).astype(np.int32)
                    Ns[..., 8*k:8*(k+1)] += Ept[idx[..., 0], idx[..., 1]]

            Ns = Ns / 3
            Ns = np.reshape(Ns, (-1, 32))
            Ns = pca.transform(Ns)

            indices_shape = indices.shape
            indices = indices.reshape((-1, 2))

            candidates = []
            for delta in SUBPASS_DELTA:
                all_nbs = indices + delta
                np.clip(all_nbs, 0, S.shape[0]-1, out=all_nbs)
                filled_nbs_coord = S[all_nbs[..., 0], all_nbs[..., 1]].astype(np.int32)
                filled_near_nbs = near_nbs[filled_nbs_coord[..., 0], filled_nbs_coord[..., 1]]
                
                filled_near_nbs = filled_near_nbs - delta
                filled_nbs_coord = filled_nbs_coord - delta
                candidates.append(np.stack([filled_nbs_coord, filled_near_nbs], axis=2))
            
            candidates = np.array(candidates)
            np.clip(candidates, 0, m-1, out=candidates)
            candidates = np.transpose(candidates, axes=(1,0,2,3))
            candidates = np.reshape(candidates, (candidates.shape[0], 2 * SUBPASS_DELTA.shape[0], 2))
            candidates = np.mod(candidates, m) 
            candidate_vecs = Nt_Ept[candidates[..., 0], candidates[..., 1]]

            differences = candidate_vecs - Ns[:, np.newaxis, :]
            distances = np.linalg.norm(differences, axis=2) 
            min_idx = np.argmin(distances, axis=1)
            selections = candidates[np.arange(distances.shape[0]), min_idx]
        
            indices = indices.reshape(indices_shape)
            S_corr[indices[..., 0], indices[..., 1]] = np.reshape(selections, indices.shape)
            

    return S_corr


def convert_coords_to_image(E, S):
    S = S.astype(np.int32)
    E_S = np.zeros((S.shape[0], S.shape[1], 3))
    E_S = E[S[..., 0], S[..., 1]]
    return E_S


def convert_coords_to_rg(S, m):
    scale = 255 / m
    S = S.astype(np.int32)
    S_rg = np.zeros((S.shape[0], S.shape[1], 3))

    S_rg[..., 0] = S[..., 0] * scale
    S_rg[..., 1] = S[..., 1] * scale
    S_rg[..., 2] = 0
    
    np.clip(S_rg, 0, 255, out=S_rg)
    return S_rg.astype(np.int32)


def view(E, S, m):
    E_Si = convert_coords_to_image(E, S)
    plt.imshow(E_Si)
    plt.show()

    S_rgi = convert_coords_to_rg(S, m)
    plt.imshow(S_rgi)
    plt.show()


def synthesize_texture(E, synth_size=32):
    E = E.astype(np.float32) / 255.0
    print("building gaussian stack...")
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
    S_i = np.mod(S_i, np.min([params['m'], synth_size]))

    print("beginning coarse to fine synthesis...")
    for i in range(params['l']):
        h = params['h'][i]
        r = params['r'][i]
        E_prime_tilde = ASV_stack[i]
        nbhds = neighbors_stack[i][0]
        pca = neighbors_stack[i][1]
        near_nbs = nearest_neighbors_stack[-1]
        print("\tsynthesizing level {}...".format(i))
        
        print("\t\tupsampling...")
        S_i = upsample(S_i, params['m'], h)
        view(E, S_i, params['m'])
        
        print("\t\tjittering...")
        S_i = jitter(S_i, params['m'], h, r)
        view(E, S_i, params['m'])
        
        if i > 2:
            print("\t\tcorrecting...")
            for _ in range(CORR_PASSES):
                S_i = isometric_correction(S_i, E_prime_tilde, nbhds, pca, near_nbs, params['m'], i)
        
            view(E, S_i, params['m'])
        
    return S_i.astype(np.int32)


def main(args):
    print("reading image...")
    E = cv2.imread(f"{args.data}/{args.image_path}")
    E = cv2.cvtColor(E, cv2.COLOR_BGR2RGB)
    print("read image of size: ", E.shape)
    print("synthesizing texture...")
    S = synthesize_texture(E, args.scale)
    print("done! u ate that up girlie pop!")
    E_S = convert_coords_to_image(E, S)
    plt.imshow(E_S)
    plt.show()
    print("saving...")
    io.imsave(f"../out/lefebvre_hoppe_{args.image_path}", E_S)


def parse_args():
    parser = argparse.ArgumentParser(description='Synthesize a texture.')
    parser.add_argument('--data', type=str, default="../data", help='path to data directory')
    parser.add_argument('--image_path', type=str, default="texture3_64.png", help='path to image to synthesize (must be in data directory)')
    parser.add_argument('--scale', type=int, default=32, help='scale of synthesized texture relative to original image in each dimensions (must be a power of 2)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)