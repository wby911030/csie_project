import numpy as np
import lpips
from scipy import linalg
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

def get_features(images): 
    model = InceptionV3(include_top= False, pooling= 'avg') # weights= 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    images = preprocess_input(images)
    features = model.predict(images)
    return features

def calculate_FID(features_r, features_g):
    mean_r = np.mean(features_r, axis=0)
    mean_g = np.mean(features_g, axis=0)
    square_distance = np.sum((mean_r - mean_g) ** 2)
    
    sigma_r = np.cov(features_r, rowvar=False) #shape=(2048, 2048)
    sigma_g = np.cov(features_g, rowvar=False)
    covariance_mean = linalg.sqrtm(np.dot(sigma_g, sigma_r), disp= False)
    
    # if covariance mean is complex, take its real part 
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    fid = square_distance + np.trace(sigma_r + sigma_g - covariance_mean * 2)
    return fid

def calculate_KID(features_r, features_g, num_subsets=100, max_subset_size=1000):
    n = features_r.shape[1]
    m = min(min(features_r.shape[0], features_g.shape[0]), max_subset_size)
    t = 0
    for subset_id in range(num_subsets):
        x = features_g[np.random.choice(features_g.shape[0], m, replace= False)]
        y = features_r[np.random.choice(features_r.shape[0], m, replace= False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return kid

def fid5k_full(img_real, img_gen, batch_size):
    features_r = get_features(img_real)
    features_g = get_features(img_gen)
    fid = calculate_FID(features_r, features_g)
    #fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=5000)
    return dict(fid5k_full=fid)

def kid5k_full(img_real, img_gen, batch_size):
    features_r = get_features(img_real)
    features_g = get_features(img_gen)
    kid = calculate_KID(features_r, features_g)
    #kid = kernel_inception_distance.compute_kid(opts, max_real=1000000, num_gen=5000, num_subsets=100, max_subset_size=1000)
    return dict(kid5k_full=kid)

def clpips1k(opts):
    clpips1k, clpips1k_rz = lpips.compute_clpips(opts, num_gen = 1000)
    return dict(clpips1k = clpips1k, clpips1k_rz = clpips1k_rz)