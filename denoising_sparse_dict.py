from time import time
import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pdb
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.datasets import make_sparse_coded_signal
from sklearn.decomposition import MiniBatchDictionaryLearning
from matplotlib import pyplot as plt
from skimage.exposure import rescale_intensity


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-n_comp", "--n_components", type=int, default=100, help="number of componets in the dictionary")
ap.add_argument("-iter", "--n_iter", type=int, default=500, help="number of iterations to learn dictionnary")
ap.add_argument("-sigma","--std_noise",type=float,default=0.1,help="standard deviation of noise")
ap.add_argument("-coeff", "--non_zero_coeff", type=int,default=1, help="number of non zero coefficients in sparse representation of the image")
args = vars(ap.parse_args())

n_comp = args['n_components']
n_iter = args['n_iter']
non_zero_coeff = args['non_zero_coeff'] 
sigma=args['std_noise']

def get_noisy_patches(image,sigma, dict_learning=False, channel=None):
    """Return the patches of the the noisy image
    In:
    :image : considered image
    :sigma (float): standard deviation of the noise added on the image
    :dict_learning(bool):???
    :channel(bool): indicates the presence of several channels
    Out:
    :data: noisy patches
    :distorted: noisy image
    :mean: mean patch
    :std:  std patch
    """

    image = image / 255
    print("dict learning",dict_learning)
    print("TEST")
    #downsample for higher speed
    if dict_learning==True :
        image = image[::2, ::2] + image[1::2, ::2] + image[::2, 1::2] + image[1::2, 1::2]
        image /= 4.0
    
	
    print('Distorting image...')
    noisy_image= image.copy()

    if channel :
        height, width, channel = image.shape
        noisy_image += sigma * np.random.randn(height, width, channel)
    else:
        height, width = image.shape
        noisy_image += sigma * np.random.randn(height, width)
        cv2.imwrite('noisy.jpg', (noisy_image*255))
    print(noisy_image.shape)

    print('Extracting reference patches...')
    t0 = time()
    patch_size = (7, 7)
    #Reshaping distorted image in a collection of patches, collected in dedicated array
    patches = extract_patches_2d(noisy_image, patch_size)
    patches = patches.reshape(patches.shape[0], -1)
    mean = np.mean(patches, axis=0)
    std = np.std(patches, axis=0)
    patches -= mean
    patches /= std
    print('done in %.2fs.' % (time() - t0)) 
	
    return (patches, noisy_image,mean, std)

def ksvd(noisy_patches):
    
    print('Updating Dictionary')
    t0 = time()
    #Mini-batch dictionnary Learning
    dico = MiniBatchDictionaryLearning(n_components=n_comp, 
                                        alpha=2, 
                                        n_iter=n_iter,)
                                        #dict_init=D)
    print('done in %.2fs.' % (time() - t0))
    V = dico.fit(noisy_patches).components_
    return V, dico


if __name__ == '__main__':

    image = cv2.imread(args['image'])
    channel=None
    if len(image.shape) >2:
        channel = image.shape[2]
    noisy_patches,noisy_image, _, _ = get_noisy_patches(image,sigma, dict_learning=True, channel=channel)
    dict_final, dico = ksvd(noisy_patches)
    n0_data,noisy_image, mean, std= get_noisy_patches(image,sigma,channel=channel)
    dico.set_params(transform_algorithm='omp',transform_n_nonzero_coefs = non_zero_coeff )
    code = dico.transform(n0_data)
    denoised_patches = np.dot(code,dict_final)
    denoised_patches*= std
    denoised_patches += mean
    denoised_patches = (denoised_patches.reshape(n0_data.shape[0], 7, 7, channel))
    print('Reconstructing...')
    reconstruction = reconstruct_from_patches_2d(denoised_patches, (image.shape[0], image.shape[1], channel))
    reconstruction*=255
    difference = image - reconstruction
    error = np.sqrt(np.sum(difference ** 2))
    print('Difference (norm: %.2f)' %error)
    print('Finished reconstruction..')
    cv2.imshow('reconstructed', reconstruction.astype('uint8'))
    cv2.imshow('orignal', image)
    cv2.imshow('noisy image',noisy_image)
    cv2.imwrite('orignal.jpg', image)
    cv2.imwrite('reconstructed.jpg', reconstruction.astype('uint8'))
    cv2.waitKey(0)