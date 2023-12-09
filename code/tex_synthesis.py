import math
import numpy as np


C = 2 ## number of passes
S = 2 ## number of subpasses
DELTA = np.array([[0,0], [0,1], [1,0], [0,1]])
M = np.array([[[0,0],[0,0]], [[1,0],[0,0]], [[0,0],[0,1]]])


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


def jitter(S):
    pass


def isometric_correction(S, E): #E should be ~E'
    # N_s_p = N_e_u = np.zeros(S.shape)
    # N_s_p[S+DELTA] = sum(E_prime[S+DELTA+(M*DELTA)]-(M*DELTA)) / 3
    # N_e_u[S+DELTA] = sum(E_prime[E+DELTA+(M*DELTA)]-(M*DELTA)) / 3
    # S = np.argmin(N_s_p - N_e_u) # need to add C(p) stuff

    # return N_s_p
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