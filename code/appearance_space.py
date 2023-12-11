import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from skimage.filters import sobel
import matplotlib.pyplot as plt

def get_appearance_space_vector(im, surrounding_size, feature_distance=True):
    dims = 4 if feature_distance else 3
    grayscale_im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    edges = sobel(grayscale_im)
    useable_im = im[surrounding_size:-surrounding_size, surrounding_size:-surrounding_size, :]
    vector_im = np.zeros((useable_im.shape[0], useable_im.shape[1], dims * (2 * surrounding_size + 1)**2))
    for i in range(surrounding_size, im.shape[0] - surrounding_size):
        for j in range(surrounding_size, im.shape[1] - surrounding_size):
            patch = im[i - surrounding_size:i + surrounding_size + 1, j - surrounding_size:j + surrounding_size + 1, :]
            if feature_distance:
                patch_edges = edges[i - surrounding_size:i + surrounding_size + 1, j - surrounding_size:j + surrounding_size + 1]
                patch_edges = np.expand_dims(patch_edges, axis=2)
                patch = np.concatenate([patch, patch_edges], axis=2)
            patch = np.reshape(patch, (dims * (2 * surrounding_size + 1)**2,))
            vector_im[i - surrounding_size, j - surrounding_size] = patch
    return conduct_pca(vector_im)


def conduct_pca(image_with_features, desired_dimensions=8):
    original_shape = image_with_features.shape
    image_with_features = np.reshape(image_with_features, (image_with_features.shape[0] * image_with_features.shape[1], image_with_features.shape[2]))
    pca = PCA(n_components=desired_dimensions)
    new_features = pca.fit_transform(image_with_features)
    new_features = np.reshape(new_features, (original_shape[0], original_shape[1], desired_dimensions))
    return new_features, pca


def get_neighborhoods(image_with_features, desired_dimensions=8):
    height, width, channels = image_with_features.shape
    neighbor_features = np.zeros((height, width, desired_dimensions * 4))
    for i in range(height):
        for j in range(width):
            dimensions = []
            if i > 0:
                if j > 0:
                    dimensions.append(image_with_features[i - 1, j - 1])
                if j < width - 1:
                    dimensions.append(image_with_features[i - 1, j + 1])
            elif i < height - 1:
                if j > 0:
                    dimensions.append(image_with_features[i + 1, j - 1])
                if j < width - 1:
                    dimensions.append(image_with_features[i + 1, j + 1])
            while len(dimensions) != 4:
                dimensions.append(image_with_features[i, j])
            dimensions = np.array(dimensions)
            neighbor_features[i, j] = np.reshape(dimensions, (desired_dimensions * 4,))
    return conduct_pca(neighbor_features)


def get_nearest_neighbors(nb_vectors):
    # TODO: sanity check that linear indices (0-3600) are being mapped to the correct (y,x) coordinates
    shape = (nb_vectors.shape[0], nb_vectors.shape[1], 2)
    nb_vectors = np.reshape(nb_vectors, (-1, nb_vectors.shape[2]))
    tree = NearestNeighbors(n_neighbors=2, algorithm='kd_tree', metric='l2', n_jobs=-1).fit(nb_vectors)
    nearest_nbs = tree.kneighbors(nb_vectors, return_distance=False)
    I, J = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.stack([I, J], axis=2)
    indices = indices.reshape((-1, 2))
    nearest_nbs = indices[nearest_nbs[..., 1]]
    nearest_nbs = np.reshape(nearest_nbs, shape)
    return nearest_nbs


if __name__ == '__main__':
    im = cv2.imread("data/texture2.jpg", cv2.COLOR_BGR2RGB)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    og, vec = get_appearance_space_vector(im, 2)
    new_features, m = conduct_pca(vec)