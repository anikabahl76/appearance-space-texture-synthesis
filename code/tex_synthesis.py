import numpy as np


C = 2 ## number of passes
S = 2 ## number of subpasses


def build_gaussian_stack():
    pass


def upsample(S, is_toroidal):
    pass


def jitter(S):
    pass


def isometric_correction(S, E):
    pass


def anisometric_correction(S, E):
    pass


def synthesize_texture(E, E_prime, synth_mode="iso", is_toroidal=False):
    E_stack, l = build_gaussian_stack(E)

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