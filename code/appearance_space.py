import numpy as np
from sklearn.decomposition import PCA
from PIL import Image

def get_appearance_space_vector(im, surrounding_size):
    useable_im = im[surrounding_size:-surrounding_size, surrounding_size:-surrounding_size, :]
    vector_im = np.zeros((useable_im.shape[0], useable_im.shape[1], 3 * (2 * surrounding_size + 1)**2))
    for i in range(surrounding_size, im.shape[0] - surrounding_size):
        for j in range(surrounding_size, im.shape[1] - surrounding_size):
            patch = im[i - surrounding_size:i + surrounding_size + 1, j - surrounding_size:j + surrounding_size + 1, :]
            patch = np.reshape(patch, (3 * (2 * surrounding_size + 1)**2,))
            vector_im[i - surrounding_size, j - surrounding_size] = patch
    return useable_im, vector_im


def conduct_pca(image_with_features, desired_dimensions=8):
    original_shape = image_with_features.shape
    image_with_features = np.reshape(image_with_features, (image_with_features.shape[0] * image_with_features.shape[1], image_with_features.shape[2]))
    pca = PCA(n_components=desired_dimensions)
    new_features = pca.fit_transform(image_with_features)
    new_features = np.reshape(new_features, (original_shape[0], original_shape[1], desired_dimensions))
    return new_features


    
# im = np.array(Image.open("data/1.3.1.png"), dtype=np.float64)[:, :, :3]/255.0
# og, vec = get_appearance_space_vector(im, 2)
# new_features = conduct_pca(vec)
