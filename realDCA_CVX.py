import numpy as np
from scipy.sparse.linalg import eigs
import cvxpy as cvx
import matplotlib.pyplot as plt
from numpy.ma import size, shape

def make_hat_matrix(Matrix):
    Mat_hat = np.vstack((np.hstack((Matrix.real, Matrix.imag)),
                         np.hstack((-Matrix.imag, Matrix.real))))
    return Mat_hat

def get_largest_eigenvalue(aMatrix):
    values, vector = eigs(aMatrix,1,which="LM")
    return values[0]

def G_result(xk,qmat, C):
    xk = np.matrix(xk)
    return xk.T*qmat*xk + np.log(C)

def H(xk,qmat,A1_hat,A2_hat,B1_hat,B2_hat,sigma1,sigma2):
    xk = np.matrix(xk)
    return xk.T*qmat*xk + np.log(sigma1*sigma1 + xk.T*A1_hat*xk) + np.log(sigma2*sigma2 + xk.T*A2_hat*xk)\
            - np.log(sigma1*sigma1 + xk.T*B1_hat*xk) - np.log(sigma2*sigma2 + xk.T*B2_hat*xk)

def Hbar(xk,ro,A1_hat,A2_hat,B1_hat,B2_hat,sigma1,sigma2):
    return (ro*xk + 2*A1_hat*xk/(sigma1*sigma1+xk.T*A1_hat*xk) + 2*A2_hat*xk/(sigma2*sigma2+xk.T*A2_hat*xk) - 2*B1_hat*xk/(sigma1*sigma1+xk.T*B1_hat*xk) - 2*B2_hat*xk/(sigma1*sigma1+xk.T*B2_hat*xk)).T

def F(xk,qmat,A1_hat,A2_hat,B1_hat,B2_hat,C,sigma1,sigma2):
    return 1/(2*np.log(2))*(G_result(xk,qmat, C) - H(xk,qmat,A1_hat,A2_hat,B1_hat,B2_hat,sigma1,sigma2))

def solvePro(N, Pt,i):
    # =======================================READ DATA===============================================
    # path = 'data/Testdata_Exp1_' + str(int(N)) + '_' + str(int(Pt)) + '.npz'
    path = 'data/Testdata_CHUNG_' + str(int(N)) + '_' + str(int(Pt)) + '_' + str(i) +'.npz'
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
    Pt = np.int(data['Pt'])

    # ==============================VARIABLES=========================================
    w0 = (np.matrix(np.random.rand(N, 1)) + np.matrix(np.random.rand(N, 1) * 1j))
    x0 = np.vstack((w0.real, w0.imag))

    Pt = Pt * 1
    P0 = P1 = P2 = Pt / 4
    e = 0.0001

    I = np.identity(2 * N)  # I is unit matrix
    R1 = F1 * f2 * f2.H * F1.H
    R2 = F2 * f1 * f1.H * F2.H
    A1 = P2 * R1 + sigmaR * sigmaR * D1
    A2 = P1 * R2 + sigmaR * sigmaR * D2
    B1 = sigmaR * sigmaR * D1
    B2 = sigmaR * sigmaR * D2
    C = 1 + (P1 * pow(np.abs(g1), 2) + P2 * pow(np.abs(g2), 2)) / pow(sigma1E, 2)

    A1_hat = make_hat_matrix(A1)
    A2_hat = make_hat_matrix(A2)
    B1_hat = make_hat_matrix(B1)
    B2_hat = make_hat_matrix(B2)
    D1_hat = make_hat_matrix(D1)
    D2_hat = make_hat_matrix(D2)
    Z_hat = make_hat_matrix(Z)

    ro = np.real(get_largest_eigenvalue(A1_hat / (2 * sigma1 * sigma1) + A2_hat / (2 * sigma2 * sigma2) \
                                        + 2 * (B1_hat / (sigma1 * sigma1)) + 2 * (B2_hat / (sigma1 * sigma1))))
    qmat = 1 / 2 * np.real(ro * I)
   
    # =================================== BUILDING MODEL ==============================================
    x = cvx.Variable(shape=(2 * N, 1))
    constraints = [P1*cvx.quad_form(x,D1_hat) + P2*cvx.quad_form(x,D2_hat) + sigmaR*sigmaR*cvx.quad_form(x,I) <= Pt-P1-P2,
                   Z_hat.T*x == 0]
    pre_xk = np.real(x0)

    while(1):
        obj = cvx.Minimize(cvx.quad_form(x,qmat) +np.log(C) -(H(pre_xk,qmat,A1_hat,A2_hat,B1_hat,B2_hat,sigma1,sigma2) + Hbar(pre_xk,ro,A1_hat,A2_hat,B1_hat,B2_hat,sigma1,sigma2)*(x-pre_xk)))
        prob = cvx.Problem(obj, constraints)
        prob.solve()
        if(np.linalg.norm(x.value-pre_xk)/(1+pow(np.linalg.norm(pre_xk),2)) < e):
            if (F(x.value, qmat, A1_hat, A2_hat, B1_hat, B2_hat, C, sigma1, sigma2) < 0):
                print(-F(x.value, qmat, A1_hat, A2_hat, B1_hat, B2_hat, C, sigma1, sigma2).item())
                return -F(x.value, qmat, A1_hat, A2_hat, B1_hat, B2_hat, C, sigma1, sigma2).item()
            else:
                return 0
        if(abs(F(x.value, qmat, A1_hat, A2_hat, B1_hat, B2_hat, C,sigma1,sigma2) - F(pre_xk, qmat, A1_hat, A2_hat, B1_hat, B2_hat, C,sigma1,sigma2)) / (1 + abs(F(pre_xk, qmat, A1_hat, A2_hat, B1_hat, B2_hat, C,sigma1,sigma2))) < e):
            if(F(x.value, qmat, A1_hat, A2_hat, B1_hat, B2_hat, C,sigma1,sigma2) < 0):
                print(-F(x.value, qmat, A1_hat, A2_hat, B1_hat, B2_hat, C, sigma1, sigma2).item())
                return -F(x.value, qmat, A1_hat, A2_hat, B1_hat, B2_hat, C,sigma1,sigma2).item()
            else:
                return 0
        pre_xk = x.value
