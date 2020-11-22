import os
import time
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def one_hot_encode(kmeans_result):
    """
    One hot encodes an image;

    Parameters
    ----------
    kmeans_result : ndarray
        Image after applying clusterization

    Returns
    -------
    encoded_image  : ndarray
        Array with k masks: one per each binary one-hot-encode mask
    """
    unique = np.unique(kmeans_result)
    colormap = {i:unique[i] for i in range(len(unique))}
    encoded_image = np.zeros(kmeans_result.shape[:2] + (len(colormap), ), dtype=np.float32)
    for i in range(encoded_image.shape[2]):
        encoded_image[:, :, i] = np.all(kmeans_result.reshape((-1, 1)) == colormap[i],
                                        axis=1).reshape(kmeans_result.shape[:2])
    return encoded_image

def binarize_data(img=None, k=None, filters=None):
    """
    Apply clusterization to segment data;

    Parameters
    ----------
    img : ndarray
        Image to segment
    k : int
        Num of clusters
    filters : dict
        whether to apply filters
        possible key arguments 'bilateral', 'nlm', 'sharpen'

    Returns
    -------
    segmented_image, labels, porosity  : ndarray
        results of prediction
    """
    if filters != None:
        for fil in filters:
            if fil == 'bilateral':
                d = filters[fil][0]
                sigmaColor = filters[fil][1]
                sigmaSpace = filters[fil][2]
                img = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
            if fil == 'nlm':
                h = filters[fil][0]
                templateWindowSize = filters[fil][1]
                searchWindowSize = filters[fil][2]
                img = cv2.fastNlMeansDenoising(img, h, templateWindowSize, searchWindowSize)
            if fil == 'sharpen':
                kernel = np.array([[-1,-1,-1], 
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
                img = cv2.filter2D(img, -1, kernel)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.99) 
    
    #reshape to appropriate for KMeans shape
    pixel_vals = img.reshape((-1,1)) 
    
    # Convert to float type 
    pixel_vals = np.float32(pixel_vals)
    
    # KMEans
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers) 
    
    segmented_data = centers[labels.flatten()] 
    segmented_image = one_hot_encode(segmented_data.reshape((img.shape)))[:,:, 1]
    
    # Watershed
    foot = 20
    distance = ndi.distance_transform_edt(segmented_image)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((foot, foot)),
                                labels=segmented_image)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=segmented_image)
    
    porosity = np.sum(segmented_image==0) / np.prod(segmented_image.shape)

    return segmented_image, labels, porosity

def binarize_3d(img_3d, k = 2, bilateral_filter = None,
               d=None, sigmaColor=None, sigmaSpace=None):
    """
    Apply clusterization to segment 3D image;

    Parameters
    ----------
    img_3d : ndarray
        3D image to segment
    k : int
        Num of clusters
    bilateral_filter : boolean
        whether to apply bilateral filter
    d, sigmaColor, sigmaSpace : int
        bilateral filter params

    Returns
    -------
    result_3d, porosity_arr  : ndarray
        results of prediction
    """   
    x_dim, y_dim, z_dim = img_3d.shape
    result_3d = np.zeros((x_dim, y_dim, z_dim))
    porosity_arr = np.zeros((x_dim,))
    for x in range(x_dim):
        result_3d[x, :, :], porosity_arr[x] = binarize_data(img = img_3d[x, :, :],
                                                            k = 2,
                                                            bilateral_filter = bilateral_filter,
                                                            d=d,
                                                            sigmaColor=sigmaColor,
                                                            sigmaSpace=sigmaSpace)
    return result_3d, porosity_arr

def plot_central_planes(image):
    """
    Plot central planes of volumetric data

    Parameters
    ----------
    img : ndarray
        3D image
    img_title : string
        plot title

    """  
    n_x, n_y, n_z = image.shape
    fig, axs = plt.subplots(1,3, figsize = (15, 10))
    axs[0].imshow(image[n_x//2, :, :], cmap = 'gray'), axs[0].set_title('X central plane')
    axs[1].imshow(image[:, n_y//2, :], cmap = 'gray'), axs[1].set_title('Y central plane')
    axs[2].imshow(image[:, :, n_z//2], cmap = 'gray'), axs[2].set_title('Z central plane')
    plt.show()

def plot_3d(image, img_title=''):
    """
    Plot boundary planes of volumetric data

    Parameters
    ----------
    img : ndarray
        3D image
    img_title : string
        plot title

    """  
    n_x, n_y, n_z = image.shape 
    yy, zz = np.mgrid[0:n_y, 0:n_z]
    xx, zz = np.mgrid[0:n_x, 0:n_z]
    xx, yy = np.mgrid[0:n_x, 0:n_y]

    x_center_loc, y_center_loc, z_center_loc = n_x-1, 0, n_z-1

    # plot 3 orthogonal slices
    X, Y, Z = image[x_center_loc, :, :], image[:, y_center_loc, :], image[:, :, z_center_loc]

    fig = plt.figure(figsize = (25,10))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(img_title, fontsize = 17)
    ax1.contourf(X, xx, zz,  zdir='x', offset=n_x-1, cmap='gray')
    ax1.contourf(xx, Y, zz,  zdir='y', offset=y_center_loc, cmap='gray')
    ax1.contourf(xx, yy, Z,  zdir='z', offset=n_z-1, cmap='gray')
    plt.show()
    
def create_dir(dir_to_save):
    """
    creates folder in defined directory

    Parameters
    ----------
    dir_to_save : str
        path

    """ 
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)