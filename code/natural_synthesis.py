import os
import numpy as np
import math
import cv2
import argparse
import matplotlib.pyplot as plt
from appearance_space import get_appearance_space_vector, get_adj_neighborhoods

def ssd(a, b):
    return np.average(np.square(a - b))

def convert_coords_to_image(E, S):
    S = S.astype(np.int32)
    E_S = np.zeros((S.shape[0], S.shape[1], 3))
    E = (E * 255).astype(np.int32)
    E_S = E[S[..., 0], S[..., 1]]
    return E_S.astype(np.int32)


def natural_synthesis(subpasses = 5):
    img = cv2.imread("../data/texture9.png", )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    if img.shape[0] != img.shape[1]:
        img = img[:min(img.shape[0], img.shape[1]), :min(img.shape[0], img.shape[1]), :]

    insize = img.shape[0]
    outsize = 300
    
    left, up, right = 4, 4, 4
    as_image, _ = get_appearance_space_vector(img, 4)
    neighbors_image, pca = get_adj_neighborhoods(as_image)
    # print(img.shape)
    # print(as_image.shape)
    # print(neighbors_image.shape)

    indices = np.stack(np.meshgrid(np.arange(insize), np.arange(insize), indexing='ij'), axis=2)
    new_img = np.stack(np.meshgrid(np.arange(outsize), np.arange(outsize), indexing='ij'), axis=2)
    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            new_img[i, j, :] = indices[np.random.randint(0, insize), np.random.randint(0, insize), :]
    
    # plt.imshow(convert_coords_to_image(img, new_img))
    # plt.show()
    for num in range(5):
        for i in range(1, new_img.shape[0]):
            for j in range(1, new_img.shape[1]):
                coord_candidates = []
                for k in range(1, left + 1):
                    if j - k < 0:
                        break
                    coord = new_img[i, j - k, :]
                    coord = [coord[0], min(img.shape[1] - 1, coord[1] + k)]
                    coord_candidates.append(coord)
                for m in range(1, up + 1):
                    if i - m < 0:
                        break
                    for n in range(-left, right + 1):
                        if j + n < 0:
                            continue
                        if j + n > new_img.shape[1] - 1:
                            break
                        coord = new_img[i - m, j + n, :]
                        coord = [min(img.shape[0] - 1, coord[0] + m), min(max(0, coord[1] - n), img.shape[1] - 1)]
                        coord_candidates.append(coord)

                current_features = []
                if i > 0:
                    current_features.append(as_image[new_img[i - 1, j][0], new_img[i - 1, j][1]])
                    if j > 0:
                        current_features.append(as_image[new_img[i - 1, j-1][0], new_img[i - 1, j-1][1]])
                if j > 0:
                    current_features.append(as_image[new_img[i, j - 1][0], new_img[i, j - 1][1]])

                while len(current_features) != 3:
                    current_features.append(as_image[new_img[i, j][0], new_img[i, j][1]])
                current_features = np.array(current_features)
                current_features = np.reshape(current_features, (-1, 8 * 3))
                current_features = pca.transform(current_features)
                min_error, coord = (None, None)
                for candidate in coord_candidates:
                    neighbor_features = neighbors_image[candidate[0], candidate[1]]
                    neighbor_features = np.reshape(neighbor_features, (-1, neighbor_features.shape[0]))
                    err = ssd(current_features, neighbor_features)
                    if min_error is None or min_error > err:
                        min_error = err
                        coord = candidate
                new_img[i, j] = coord
    
    plt.imshow(convert_coords_to_image(img, new_img))
    plt.show()

            
natural_synthesis()