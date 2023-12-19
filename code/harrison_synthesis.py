import os
import cv2
import numpy as np
from random import shuffle
import time
from appearance_space import get_appearance_space_vector
import matplotlib.pyplot as plt

SOURCE_PATH = '../data/'
OUTPUT_PATH = '../out/'

N = 30
M = 80
DISPERSION = 30
STEPS = 8

def calculate_error(feature_vectors, target_features, dispersion):
    err = 0
    for i in range(len(feature_vectors)):
        cur_feature = feature_vectors[i]
        target_feature = target_features[i]
        cur_feature /= dispersion
        target_feature /= dispersion
        features = np.square(cur_feature-target_feature) + 1
        product = 1
        for f in features:
            product *= f
        err += np.log(product)
    return err


def best_neighbor(n_candidates, all_candidates, position, texture, original_mappings, as_image, dispersion):
    y, x = position
    original_height, original_width, _ = texture.shape
    feature_vectors = []
    offsets = list((candidate_y - y, candidate_x - x) for candidate_y, candidate_x in n_candidates)
    for candidate in all_candidates:
        candidate_y, candidate_x = candidate
        vec = []
        for offset_y, offset_x in offsets:
            feature_y = (candidate_y + offset_y) % original_height
            feature_x = (candidate_x + offset_x) % original_width
            vec.append(as_image[feature_y, feature_x])
        feature_vectors.append(vec)
    target_feature = list(as_image[original_mappings[candidate][0], original_mappings[candidate][1]] for candidate in n_candidates)
    
    min_error, index = None, None
    for i in range(len(feature_vectors)):
        vec = feature_vectors[i]
        cur_error = calculate_error(vec, target_feature, dispersion)
        if min_error == None or cur_error < min_error:
            index = i
            min_error = cur_error
    return all_candidates[index]


def n_nearest_neighbors(coord, original_mappings, mask, N, original_shape):
    y, x = coord
    original_height, original_width = original_shape
    n_candidates = [] 
    ys, xs = np.nonzero(mask)
    indices = zip(ys, xs)
    distances = dict()
    for cur_y, cur_x in indices:
        dist = int(np.sqrt((cur_y - y)**2 + (cur_x - x)**2))
        if dist in distances.keys():
            distances[dist].append((cur_y, cur_x))
        else:
            distances[dist] = [(cur_y, cur_x)]
    dist_keys = list(distances.keys())
    dist_keys = sorted(dist_keys)
    for dist in dist_keys:
        for coord in distances[dist]:
            if len(n_candidates) < N:
                n_candidates.append(coord)
            else:
                break
        if not(len(n_candidates) < N):
                break

    n_candidates = n_candidates[:N]
    n_offset_candidates = []
    for i in range(N):
        candidate_y, candidate_x = n_candidates[i]
        original_y, original_x = original_mappings[candidate_y, candidate_x]
        offset_candidate_y = (original_y + (y - candidate_y)) % original_height 
        offset_candidate_x = (original_x + (x - candidate_x)) % original_width
        offset_candidate = (offset_candidate_y,offset_candidate_x)
        n_offset_candidates.append(offset_candidate)

    return n_candidates, n_offset_candidates

def count_true_vals(arr):
    return len(np.nonzero(arr)[0])

def update(texture, new_img, mappings_to_texture, mask, new_coordinates, original_coordinates):
    new_img[new_coordinates] = texture[original_coordinates]
    mappings_to_texture[new_coordinates] = original_coordinates
    mask[new_coordinates] = True

def synthesize_texture(texture, N, M, dispersion, steps, scale=2, p=0.8):
    height, width, channels = texture.shape
    new_height, new_width = scale*height, scale*width
    new_img = np.zeros((new_height, new_width, channels))
    mappings_to_texture = np.zeros((new_height, new_width, 2)).astype(np.uint8)
    mask = np.zeros((new_height, new_width)).astype(bool)
    as_image, _ = get_appearance_space_vector(texture, 2)
    all_new_coordinates = list((y, x) for y in range(new_height) for x in range(new_width))

    for step in range(0, steps + 1):
        shuffle(all_new_coordinates) 
        for i in range(int(new_height * new_width * (p ** (steps - step)))):
            new_coordinates = new_y, new_x = all_new_coordinates[i]
            if count_true_vals(mask) < N:
                original_coordinates = original_y, original_x = np.random.randint(0, height), np.random.randint(0, width)
                update(texture, new_img, mappings_to_texture, mask, new_coordinates, original_coordinates)
            else:
                n_candidates, n_offset_candidates = n_nearest_neighbors(new_coordinates, mappings_to_texture, mask, N, (height, width))
                m_offset_candidates = [(np.random.randint(0, height), np.random.randint(0, width)) for _ in range(M)]
                all_candidates = n_offset_candidates + m_offset_candidates
                best_candidate = best_neighbor(n_candidates, all_candidates, new_coordinates, texture, mappings_to_texture, as_image, dispersion)
                update(texture, new_img, mappings_to_texture, mask, new_coordinates, best_candidate)
    return new_img

def synthesis():
    for i in range(0, 16):
        path = SOURCE_PATH + "texture" + str(i) + ".png"
        texture = cv2.imread(path)
        if texture.shape[0] > 100:
            texture = cv2.resize(texture, (100, 100))
            cv2.imwrite(path, texture)
        texture = cv2.cvtColor(texture, cv2.COLOR_RGB2BGR)
        texture = texture.astype(np.float32) / 255.0
        print("synthesizing image", str(i))

        start = time.time()
        output = synthesize_texture(texture, N, M, DISPERSION, STEPS)
        end = time.time()

        output = (output * 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUTPUT_PATH + "texture" + str(i) + "result.png"), output)
        print("wrote image", i, "in", start-end, "seconds")
