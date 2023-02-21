#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Perform Rotated Complex Kernel PCA (ROCK-PCA).

This code is a translation of the original Matlab code into Python.

Original reference:
Bueso, D., Piles, M. & Camps-Valls, G. Nonlinear PCA for Spatio-Temporal
Analysis of Earth Observation Data. IEEE Trans. Geosci. Remote Sensing 1–12
(2020) doi:10.1109/TGRS.2020.2969813.

You can find this code at https://github.com/DiegoBueso/ROCK-PCA.

Author:         Niclas Rieger
Affilation:     Mathematical Research Centre (CRM), Barcelona, Spain
Contact:        nrieger@crm.cat
GitHub:         https://github.com/NiclasRieger

"""
# =============================================================================
# Imports
# =============================================================================
import numpy as np
from scipy.signal import hilbert
from numba import jit, complex128
from tqdm import tqdm
from rotation import promax

# =============================================================================
# Functions
# =============================================================================

@jit(complex128[:,:](complex128[:,:],complex128[:,:]), fastmath=True, parallel=True,boundscheck=False, nopython=True)
def distance_matrix(A,B):
    """
    Calculate distance matrix d_ij = ||a_i-b_i||^2 in space, that is distance
    between stations (columns) over time.

    Parameters
    ----------
    A : array
        First matrix
    B : array
        Second matrix

    Returns
    -------
    D : array
        Distance matrix

    """
    assert A.shape[0] == B.shape[0], 'Distance matrix could not be calculated.'
    ' First dimension of input matrices need to be same.'
    n = A.shape[0]
    p1 = A.shape[1]
    p2 = B.shape[1]
    D = np.zeros((p1, p2),dtype=np.complex128)
    for i in range(p1):
        for j in range(p2):
            s = 0
            for k in range(n):
                s += (A[k, i] - B[k, j])*(A[k, i] - B[k, j]).conjugate()
            D[i, j] = s
    return D


def rock(data, n_rot, n_pow, n_sig):
    """Perform ROCK-PCA.

    Parameters
    ----------
    data : ndarray
        Data with dimensions (temporal, spatial x, spatial y).
    n_rot : int
        Number of PCs to check for optimal rotation.
    n_pow :
        Number of powers to check for optimal rotation.
    n_sig : type
        Number of sigmas to check for optimal kernel matrix estimation.

    Returns
    -------
    projected_data
        Projections of principcal components on data.
    pcs
        Principal components.
    expvar
        Explained variances of PCs.
    sigma
        Optimal sigma.
    kernel
        Estimated kernel matrix.

    """
    nt = data.shape[0]
    nx = data.shape[1]
    ny = data.shape[2]
    ns = np.product((nx, ny))

    data = data.reshape(nt,ns)

    # center data
    center_matrix = np.identity(nt) - (1. / nt) * np.ones(nt)
    data = data - data.mean(axis=0) # computationally more efficient

    # remove NaN in data field (such as continents/ocean)
    no_nan_index = np.where(~(np.isnan(data[0])))[0]
    no_nan_data  = data[:,no_nan_index]

    # complexify data via Hilbert transform
    no_nan_data = hilbert(no_nan_data, axis=0)
    no_nan_data = no_nan_data - no_nan_data.mean(axis=0)


    # Sigma estimation from median distance
    print('Build distance matrix ... ', flush=True)
    distance = distance_matrix(no_nan_data.T, no_nan_data.T)
    median_distance = np.median(abs(distance))
    max_distance    = np.max(abs(distance))
    sigmas          = np.linspace(.1 * median_distance, 10 * max_distance,n_sig)

    # Kernel matrix RBF
    kernels = np.zeros((n_sig, nt, nt), dtype=complex)
    pcs     = np.zeros((n_sig, nt, n_rot), dtype=complex)

    description = 'Build kernel matrices'
    for s, sigma in tqdm(enumerate(sigmas), total=len(sigmas), desc=description):
        K = np.exp( -distance / (2 * (sigma**2)) )
        K = center_matrix @ K @ center_matrix
        U, eigenvalues, _ = np.linalg.svd(K, full_matrices=False)

        kernels[s,:,:] = K
        pcs[s,:,:] = U[:,:n_rot]

    # Kurtosis optimization criteria
    powers      = range(0,n_pow+1) # power = 0 means no rotation
    rotations   = range(2,n_rot+1)

    kurtosis = np.zeros((len(sigmas),len(rotations)))

    # optimize sigma & number of PCs
    description = 'Optimize SIGMA and ROTATION'
    for s,sig in tqdm(enumerate(sigmas), total=len(sigmas), desc=description):
        for r,rot in enumerate(rotations):
            temp_pcs = pcs[s,:,:(rot)]
            temp_pcs = temp_pcs / np.linalg.norm(temp_pcs, axis=0)**2
            kurtosis[s,r] = (nt/rot)*np.sum(np.sum(np.real(temp_pcs)**4,axis=0)/(np.sum(np.real(temp_pcs**2),axis=0)**2))

    # find maximum kurtosis
    idx_optimal_rot     = np.argmax(np.max(kurtosis, axis=0))
    idx_optimal_sig   = np.argmax(np.max(kurtosis, axis=1))

    optimal_rot     = rotations[idx_optimal_rot]
    optimal_sig     = sigmas[idx_optimal_sig]
    optimal_kernel  = kernels[idx_optimal_sig,:,:]

    # optimize power of Promax
    power_pcs = np.zeros((len(powers),nt,optimal_rot), dtype=complex)
    kurtosis = np.zeros((len(powers)))

    description = 'Optimize POWER'

    for i,pow in tqdm(enumerate(powers), total=len(powers), desc=description):

        temp_pcs = pcs[idx_optimal_sig,:,:optimal_rot]

        if ( pow > 0):
            fft_pcs         = np.fft.fft(temp_pcs, axis=0)
            temp_pcs,_,_    = promax(fft_pcs, power=pow)
            temp_pcs        = abs(temp_pcs) / abs(fft_pcs) * fft_pcs
            temp_pcs        = np.real(np.fft.ifft(temp_pcs, axis=0))
            temp_pcs        = hilbert(temp_pcs, axis=0)
            temp_pcs        = center_matrix @ temp_pcs

        power_pcs[i,:,:] = temp_pcs

        temp_pcs = temp_pcs / np.linalg.norm(temp_pcs,axis=0)**2
        kurtosis[i] = (nt/optimal_rot)*np.sum(np.sum(np.real(temp_pcs)**4,axis=0)/(np.sum(np.real(temp_pcs**2),axis=0)**2))


    # find maximum kurtosis
    idx_optimal_pow = np.argmax(kurtosis)
    optimal_pow = powers[idx_optimal_pow]

    # select PCs accordingly to optimized power
    pcs = power_pcs[idx_optimal_pow,:,:]

    # reconstuction Eigenvalues
    expvar = np.zeros(optimal_rot)
    for i, pc in enumerate(pcs.T):
        expvar[i] = np.linalg.norm(pc.conjugate().T @ optimal_kernel) \
        / np.linalg.norm(pc)

    idx_eig = np.argsort(expvar)
    expvar  = expvar[idx_eig][::-1]
    pcs     = pcs[:,idx_eig][:,::-1]

    # recreate original data dimensions with nan values (continents, oceans)
    data_proj = np.zeros((optimal_rot,ns), dtype=complex) * np.nan

    # spatial projection
    data_proj[:,no_nan_index] = (pcs.conjugate().T @ no_nan_data)
    data_proj = data_proj.reshape(optimal_rot, nx, ny)

    print(optimal_sig)
    print(optimal_rot)
    print(optimal_pow)
    print('Finished')

    return data_proj.real, pcs, expvar, optimal_sig, optimal_kernel
