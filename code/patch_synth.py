import os
import numpy as np
import math
import cv2
import argparse
import matplotlib.pyplot as plt
from appearance_space import get_appearance_space_vector, get_neighborhoods, get_nearest_neighbors


SOURCE_PATH = '../data'
OUTPUT_PATH = '../output/'
SYNTHESIS_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'synthesis')
TRANSFER_OUTPUT_PATH = os.path.join(OUTPUT_PATH, 'transfer')
    

def image_synthesis(sample, outsize, tilesize, overlapsize):
    '''
    Takes in an sample image, and outputs an image with dimensions that are specified
    by outsize (plus channels) that has texture synthesized from the sample

    Arguments:
    sample-       sample texure image
    outsize-      size of result image
    tilesize-     size of the tiles to use (patch size)
    overlapsize-  size of the overlap region (same in each direction -> left
                  and above)
    texture-      texure image
    method-       1 = random, 2 = best ssd, 3 = best ssd + minimum error boundary cut
    quiet-        no images shown if true

    Output:
        synthesized image of size outsize[1] x outsize[2] x channels 
    '''

    # The amount of additional space each tile takes up (accounting for overlap)
    adjsize = tilesize - overlapsize

    as_img, _ = get_appearance_space_vector(sample, 3)

    # imout is the array we will fill. The size is slightly larger than outsize to start
    # since it will be filled with an integer number of tiles
    real_imout = np.zeros((int(math.ceil(outsize[0] / adjsize) * adjsize + overlapsize), int(math.ceil(outsize[1]/adjsize) * adjsize + overlapsize), sample.shape[2]))
    real_imout_mask = np.zeros((int(math.ceil(outsize[0] / adjsize) * adjsize + overlapsize), int(math.ceil(outsize[1]/adjsize) * adjsize + overlapsize)), dtype=bool)
    
    as_imout = np.zeros((int(math.ceil(outsize[0] / adjsize) * adjsize + overlapsize), int(math.ceil(outsize[1]/adjsize) * adjsize + overlapsize), as_img.shape[2]))
    as_imout_mask = np.zeros((int(math.ceil(outsize[0] / adjsize) * adjsize + overlapsize), int(math.ceil(outsize[1]/adjsize) * adjsize + overlapsize)), dtype=bool)
    

    # iterate over each tile
    for y in range(0, outsize[0], adjsize):
        for x in range(0, outsize[1], adjsize):
        
            # the patch of imout that we want to fill
            to_fill = as_imout[y:y + tilesize, x:x + tilesize, :]

            # the mask for this patch (includes left and above overlap)
            to_fill_mask = as_imout_mask[y:y + tilesize, x:x + tilesize]

            to_fill_texture = real_imout[y:y + tilesize, x:x + tilesize, :]
            to_fill_mask_texture = real_imout_mask[y:y + tilesize, x:x + tilesize]

            _, as_tile, texture_tile = get_min_cut_patch(sample, as_img, tilesize, overlapsize, to_fill, to_fill_mask, to_fill_texture)

            # _, as_tile, texture_tile = get_ssd_patch(sample, as_img, tilesize, to_fill, to_fill_mask)
            
            # texture_tile = get_random_patch(sample, tilesize)
            # as_tile = get_random_patch(as_imout, tilesize)

            # update result image and mask
            real_imout[y:y + tilesize, x:x + tilesize,:] = texture_tile
            real_imout_mask[y:y + tilesize, x:x + tilesize] = 1
            as_imout[y:y + tilesize, x:x + tilesize,:] = as_tile
            as_imout_mask[y:y + tilesize, x:x + tilesize] = 1

    real_imout = real_imout[:outsize[0],:outsize[1]]
    
    return real_imout

def get_random_patch_coordinates(texture, tilesize):
    x = np.random.randint(0, texture.shape[1] - tilesize)
    y = np.random.randint(0, texture.shape[0] - tilesize)

    return x, y

def get_random_patch(texture, tilesize):
    x = np.random.randint(0, texture.shape[1] - tilesize)
    y = np.random.randint(0, texture.shape[0] - tilesize)
    print(x, y)

    return texture[y:y+tilesize, x:x+tilesize]


def ssd(a, b):
    return np.average(np.square(a - b))

def get_ssd_patch(texture, as_image, tilesize, to_fill, to_fill_mask):
    def ssd_overlap(a, b):
        stacked_mask_list = [to_fill_mask for i in range(8)]
        stacked_mask = np.stack(stacked_mask_list, axis=2)
        return ssd(a * stacked_mask, b * stacked_mask)

    return get_min_error_patch(texture, as_image, tilesize, to_fill, ssd_overlap)

def get_min_error_patch(texture_image, as_image, tilesize, to_fill, error_func):
    min_error_pair = (None, None, None)
    for i in range(2500):
        min_error, _, _ = min_error_pair
        x, y = get_random_patch_coordinates(texture_image, tilesize)
        appreance_based_tile =  as_image[y:y+tilesize, x:x+tilesize]
        texture_tile = texture_image[y:y+tilesize, x:x+tilesize]
        cur_error = error_func(appreance_based_tile, to_fill)
        if min_error == None or min_error > cur_error:
            min_error_pair = (cur_error, appreance_based_tile, texture_tile)
    return min_error_pair

def carve_seam(patch_overlap, fill_overlap):
    ssd_overlap = np.square(np.sum(patch_overlap, axis=2) - np.sum(fill_overlap, axis=2))
    seams = np.zeros(ssd_overlap.shape)
    seams[0, :] = ssd_overlap[0, :]
    for i in range(1, seams.shape[0]):
        for j in range(0, seams.shape[1]):
            mins = [seams[i - 1, j]]
            if j > 0:
                mins.append(seams[i - 1, j - 1])
            if j < seams.shape[1] - 1:
                mins.append(seams[i - 1, j + 1])
            seams[i, j] = min(mins) + ssd_overlap[i, j]
    seam_mask = np.zeros(seams.shape)
    j = np.argmin(seams[seams.shape[0] - 1, :])
    for i in range(0, seams.shape[0]):
        i_actual = seams.shape[0] - 1 - i
        seam_mask[i_actual, j:] = 1
        if i_actual == 0:
            break
        mid_val = seams[i_actual - 1, j]
        left_val = mid_val + 1 if j == 0 else seams[i_actual - 1, j - 1]
        right_val = mid_val + 1 if j == seams.shape[1] - 1 else seams[i_actual - 1, j + 1]
        if left_val < mid_val and left_val < right_val:
            j -= 1
        if right_val < mid_val and right_val < left_val:
            j += 1
    return seam_mask       

def get_min_cut_patch(texture, as_image, tilesize, overlapsize, to_fill, to_fill_mask, to_fill_texture, patch=None):
    if patch is None:
        _, as_patch, texture_patch  = get_ssd_patch(texture, as_image, tilesize, to_fill, to_fill_mask)
    vertical_patch_overlap = as_patch[:, :overlapsize, :]
    vertical_fill_overlap = to_fill[:, :overlapsize, :]
    horizontal_patch_overlap = as_patch[:overlapsize, :, :]
    horizontal_fill_overlap = to_fill[:overlapsize, :, :]
    vertical_seam_mask = np.ones(to_fill_mask.shape)
    horizontal_seam_mask = np.ones(to_fill_mask.shape)
    if to_fill_mask[to_fill_mask.shape[0] - 1, overlapsize - 1]:
        vertical_seam = carve_seam(
            vertical_patch_overlap,
            vertical_fill_overlap 
        )
        vertical_seam_mask[:, :overlapsize] = vertical_seam
    if to_fill_mask[overlapsize - 1, to_fill_mask.shape[1] - 1]:
        horizontal_seam = carve_seam(
            np.transpose(horizontal_patch_overlap, (1, 0, 2)),
            np.transpose(horizontal_fill_overlap, (1, 0, 2)),
        )
        horizontal_seam = np.transpose(horizontal_seam)
        horizontal_seam_mask[:overlapsize, :] = horizontal_seam
    
    seam_mask = np.floor((vertical_seam_mask + horizontal_seam_mask)/2.0)
    stacked_seam_mask_list_texture = [seam_mask for i in range(3)]
    stacked_seam_mask_list_as = [seam_mask for i in range(8)]
    stacked_seam_mask_texture = np.stack(stacked_seam_mask_list_texture, axis=2)
    stacked_seam_mask_as = np.stack(stacked_seam_mask_list_as, axis=2)
    new_patch_texture = texture_patch * stacked_seam_mask_texture + to_fill_texture * (1 - stacked_seam_mask_texture)
    new_patch_as = as_patch * stacked_seam_mask_as + to_fill * (1 - stacked_seam_mask_as)
    return None, new_patch_as, new_patch_texture


def synthesis():
    img = cv2.imread("../data/texture12.png", )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.

    # specify output size and algorithm parameters
    # outsize = (128, 128)
    # tilesize = 15
    # overlapsize = 8



    outsize = (300, 300)
    tilesize = (int((img.shape[0] * img.shape[1] / 2000)) // 10) * 10
    overlapsize = tilesize // 3
    print(img.shape)
    print(tilesize)
    print(overlapsize)

    out = image_synthesis(img, outsize, tilesize, overlapsize)
    plt.imshow(out)
    plt.show()

synthesis()
