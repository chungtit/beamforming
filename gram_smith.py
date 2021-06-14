import numpy as np
from numpy import linalg as LA

import scipy as sp
from numpy.distutils.system_info import x11_info
from scipy import linalg as sp_LA
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
from scipy.linalg import null_space
from sympy import Matrix


import math

def proj(u,v):
    return LA.norm((u.T * v) / (u.T * u))

def gram_smith(X):
    X1 = X[:, 0]
    row , col = X.shape
    col1 = 1
    for i in range(1,col):
        tmp = X[: ,i]
        for j in range(col1):
            tmp = tmp - proj(X1[:, j],X[:, i]) * X1[:, j]
        X1 = np.c_[X1, np.matrix(tmp)]
        col1 = col1 +1
    for i in range(col):
        X1[:, i] = normalize(X1[:, i])
    return X1

def isHermitian(a, tol=1e-8):
    return np.allclose(a, a.getH(), atol=tol)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def genComplexMatrix(num_row, num_col):
    # calculate normal distribution
    mu, sigma = 0, 1.0
    norm_dist_real = np.random.normal(mu, sigma, num_row * num_col)
    norm_dist_imag = np.random.normal(mu, sigma, num_row * num_col)
    # generate complex random array based on normal distribution
    complex_norm = (1 / np.sqrt(2)) * (norm_dist_real + 1j * norm_dist_imag)

    # reshape the complex array to a num_row-row and num_col-col array
    complex_norm.shape = (num_row, num_col)

    # get the matrix form of complex_norm array
    return np.matrix(complex_norm)

def powerConsume(Xs,P1_hat,P2_hat,Q1_hat,Q2_hat):
    tmp =LA.norm(Xs.getH() * (np.kron(P1_hat.getT(),P2_hat)) * Xs)
    tmp1 =LA.norm( Xs.getH() * (np.kron(Q1_hat.getT(), Q2_hat)) * Xs)
    #print(np.kron(P1_hat.getT(),P2_hat).shape)
    return (tmp/tmp1)

def createG(N,V):
    G = np.random.rand(N, 0)
    for i in range(2, N):
        tmp = V[i, :]
        G = np.c_[ G, tmp.getT()]
    return G

def de_Vec(M, N):
    aList = []
    for i in range(N-1):
        tmp = M[:, i]
        aList.append(tmp)
    return aList

def normalize(vectorA):
    sumA = np.sum(np.multiply(vectorA,np.conjugate(vectorA)),axis=0)
    sumA = np.sqrt(sumA)
    vectorA = np.matrix(vectorA/sumA)
    return vectorA

def sqr_absolute_val(x):
    return np.real(x) * np.real(x) + np.imag(x) * np.imag(x)

def solving(N,Pt,i):
    path = 'data/Testdata_CHUNG_' + str(int(N)) + '_' + str(int(Pt)) + '_' + str(i) + '.npz'
    data = np.load(path)

    N = np.int(data['N'])
    f1 = np.matrix(data['f1'])
    f2 = np.matrix(data['f2'])
    sigmaR = np.int(data['sigmaR'])
    sigma1 = np.int(data['sigma1'])
    sigma2 = np.int(data['sigma2'])
    L = np.matrix(data['L'])
    sigma1E = np.int(data['sigma1E'])
    sigma2E = np.int(data['sigma2E'])
    F1 = np.matrix(data['F1'])
    F2 = np.matrix(data['F2'])
    g1 = np.complex128(data['g1'])
    g2 = np.complex128(data['g2'])
    D1 = np.matrix(data['D1'])
    D2 = np.matrix(data['D2'])
    Z = np.matrix(data['Z'])
    PowerT = Pt * 1

    U,S,V =LA.svd(Z.getH())
    #V = V.T
    #G is column orthogonal matrix corresponding to zero singular of
    # => column[i] of G is row[i+2] of V

    G = np.random.rand(N, 0)
    for i in range(2, N):
        tmp = np.matrix(V[i, :])
        G = np.c_[G, tmp.getT()]

    #Power consume at two Sources
    null =np. matrix(null_space(Z.H))
    G = gram_smith(null)
    Power1 = PowerT/4
    Power2 = PowerT/4

    #variance
    variance2_R = 1
    variance2_k = 1
    variance2_E_1 = 1

    #(28a) , (28b)
    R1 = F1*f2*f2.getH()*F1.getH()  #(13)
    R2 = R1 #(13)

    P1 = G.H * (variance2_R*D1 + Power2*R1)* G
    P2 = G.H * (variance2_R*D2 + Power1*R2)* G
    Q1 = variance2_R * G.getH() * D1 * G
    Q2 = variance2_R * G.getH() * D2 * G
    #A0
    A0 = G.getH() * (Power1*D1 + Power2*D2 + variance2_R*np.identity(N)) *G  #(12)

    try:
        Ac = LA.inv(np.matrix(sp_LA.sqrtm(A0))) #ok
    except np.linalg.LinAlgError:
        print("Input Matrix Is Not Invertible")
        pass

    #(29)
    P1_nga = Ac.getH() * P1 * Ac
    P2_nga = Ac.getH() * P2 * Ac
    ##
    Q1_nga = Ac.getH() * Q1 * Ac
    Q2_nga = Ac.getH() * Q2 * Ac

    #(31) & (32)

    tmp = variance2_k / (PowerT-Power1-Power2)

    P1_hat = P1_nga + tmp * np.identity(N-2)
    P2_hat = P2_nga + tmp * np.identity(N-2)
    Q1_hat = Q1_nga + tmp * np.identity(N-2)
    Q2_hat = Q2_nga + tmp * np.identity(N-2)

    Aopt  = LA.inv(np.kron(Q1_hat.getT(), Q2_hat)) * (np.kron(P1_hat.getT(),P2_hat))

    maxveigenvalues , xOpt = eigs(Aopt,1,which='LM')

    xOpt =np.matrix(xOpt.reshape(N-2, N-2))

    if (isHermitian(xOpt)==True):
        eigenvalues, c_hat = eigs(xOpt,1,which='LM')
        w = math.sqrt(PowerT-Power1-Power2) * G * Ac * c_hat
        result = powerConsume(xOpt,P1_hat,P2_hat,Q1_hat,Q2_hat)
        return result
    else:
        result =0
        w = np.random.rand(N).reshape(N,1)
        for i in range(61):
            phi_l = genComplexMatrix(N-2,N-2)
            X_l = xOpt * phi_l
            ##find the eigenvectors corresponding to largest eigenvalue
            ##subAlgorithm B
            if (N>4):
                values, xl_nga = eigs(X_l,1,which='LM')
                xl_nga = np.matrix(xl_nga)
            else:
                values,vectors = eig(X_l)
                values = np.matrix(values).getT()
                ##
                largest = 0
                for i in range(1,(N-2)):
                    if (LA.norm(values[i])> LA.norm(largest)):
                        largest = i;
                xl_nga = np.matrix(vectors[largest]).getT()
                xl_nga = normalize(xl_nga)

            Xl_nga = xl_nga*xl_nga.getH()
            xl_hat = Xl_nga.reshape((N-2)* (N-2), 1)
            cur_value = powerConsume(xl_hat,P1_hat,P2_hat,Q1_hat,Q2_hat)
            if (LA.norm(cur_value)> 0):
                result = LA.norm(cur_value)
                w = G * A0 * xl_nga
        print(N," ",PowerT)

        result = result / (1 + (Power1 * sqr_absolute_val(g1) + Power2 * sqr_absolute_val(g2)) /variance2_E_1)
        result = (0+ math.log2(result) /2)
        if (result < 0) :
            result = 0
        print(result)
        print("------------------------------")
        return result
