import numpy as np
import math

from numpy import linalg as LA
from scipy import linalg as sp_LA
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
from scipy.linalg import null_space

from read_data import get_data


def secrecy_sum_rate_maximization(number_layers, power):
    """
    Secrecy Sum Rate Maximization algorithm
    in paper "Algorithms for Secrecy Guarantee With Null Space Beamforming in Two-Way Relay Networks"
    """
    # get data from dataset
    number_of_layers, coefficient_from_source_1, coefficient_from_source_2, sigma_relay, sigma_from_source_1, \
    sigma_from_source_2, diag_coefficient_from_source_1, diag_coefficient_from_source_2, sigma_source_1_eavesdropper, \
    coefficient_source_1_eavesdropper, coefficient_source_2_eavesdropper, \
    D1, D2, Z_matrix, total_power = get_data(number_layers, power)

    U, S, V = LA.svd(Z_matrix.getH())

    # G is column orthogonal matrix corresponding to zero singular value
    # of matrix Z and a combination vector with dimension (N-2)x1
    G = np.random.rand(number_of_layers, 0)
    for i in range(2, number_of_layers):
        tmp = np.matrix(V[i, :])
        G = np.c_[G, tmp.getT()]

    # Power consume at two Sources
    null = np.matrix(null_space(Z_matrix.H))
    G = gram_schmidt_process(null)
    power_source_1 = power_source_2 = total_power / 4

    # variance
    variance2_R = 1
    variance2_k = 1
    variance2_E_1 = 1

    # equation (28a) , (28b) in paper
    R1 = diag_coefficient_from_source_1 * coefficient_from_source_2 * coefficient_from_source_2.getH() * diag_coefficient_from_source_1.getH()  # (13)
    R2 = R1  # equation (13)
    P1 = G.H * (variance2_R * D1 + power_source_2 * R1) * G
    P2 = G.H * (variance2_R * D2 + power_source_1 * R2) * G
    Q1 = variance2_R * G.getH() * D1 * G
    Q2 = variance2_R * G.getH() * D2 * G

    # equation (12) on paper, calculate A0
    A0 = G.getH() * (power_source_1 * D1 + power_source_2 * D2 + variance2_R * np.identity(number_of_layers)) * G

    try:
        Ac = LA.inv(np.matrix(sp_LA.sqrtm(A0)))
    except np.linalg.LinAlgError:
        print("Input Matrix Is Not Invertible")
        pass

    # equation (29) in paper
    P1_nga = Ac.getH() * P1 * Ac
    P2_nga = Ac.getH() * P2 * Ac
    Q1_nga = Ac.getH() * Q1 * Ac
    Q2_nga = Ac.getH() * Q2 * Ac

    # equation (31) & (32) in paper
    tmp = variance2_k / (total_power - power_source_1 - power_source_2)
    P1_hat = P1_nga + tmp * np.identity(number_of_layers - 2)
    P2_hat = P2_nga + tmp * np.identity(number_of_layers - 2)
    Q1_hat = Q1_nga + tmp * np.identity(number_of_layers - 2)
    Q2_hat = Q2_nga + tmp * np.identity(number_of_layers - 2)

    A_opt = LA.inv(np.kron(Q1_hat.getT(), Q2_hat)) * (np.kron(P1_hat.getT(), P2_hat))

    maxveigenvalues, x_opt = eigs(A_opt, 1, which='LM')

    x_opt = np.matrix(x_opt.reshape(number_of_layers - 2, number_of_layers - 2))

    if is_vectorization_of_hermitian_matrix(x_opt):
        eigenvalues, c_hat = eigs(x_opt, 1, which='LM')
        w = math.sqrt(total_power - power_source_1 - power_source_2) * G * Ac * c_hat
        result = power_consumption_on_relays(x_opt, P1_hat, P2_hat, Q1_hat, Q2_hat)
        return result
    else:
        result = 0
        for i in range(61):
            phi_l = generate_complex_norm_matrix(number_of_layers - 2, number_of_layers - 2)
            X_l = x_opt * phi_l

            # Find the eigenvectors corresponding to largest eigenvalue (subAlgorithm B)
            if number_of_layers > 4:
                values, xl_nga = eigs(X_l, 1, which='LM')
                xl_nga = np.matrix(xl_nga)
            else:
                values, vectors = eig(X_l)
                values = np.matrix(values).getT()
                largest = 0
                for i in range(1, (number_of_layers - 2)):
                    if LA.norm(values[i]) > LA.norm(largest):
                        largest = i
                xl_nga = np.matrix(vectors[largest]).getT()
                xl_nga = normalize(xl_nga)

            Xl_nga = xl_nga * xl_nga.getH()
            xl_hat = Xl_nga.reshape((number_of_layers - 2) * (number_of_layers - 2), 1)
            cur_value = power_consumption_on_relays(xl_hat, P1_hat, P2_hat, Q1_hat, Q2_hat)
            if LA.norm(cur_value) > 0:
                result = LA.norm(cur_value)
        print(number_of_layers, " ", total_power)

        result = result / (1 + (power_source_1 * sqr_absolute_val(
            coefficient_source_1_eavesdropper) + power_source_2 * sqr_absolute_val(
            coefficient_source_2_eavesdropper)) / variance2_E_1)
        result = (0 + math.log2(result) / 2)
        if result < 0:
            result = 0
        print(result)
        print("------------------------------")
        return result


def projection_operator(vector1, vector2):
    """
    Calculate projection of 2 vectors
    Args:
        vector1 : a vector
        vector2 : a vector
    Returns:
        projection of 2 vectors
    """
    return LA.norm((vector1.T * vector2) / (vector1.T * vector2))


def gram_schmidt_process(matrix_X):
    """
    Orthonormalize a set of vectors in an inner product space

    Args:
        matrix_X    : a matrix
    Returns:
        a column orthogonal matrix
    """
    col_X = matrix_X[:, 0]
    row, col = matrix_X.shape
    col1 = 1
    for i in range(1, col):
        tmp = matrix_X[:, i]
        for j in range(col1):
            tmp = tmp - projection_operator(col_X[:, j], matrix_X[:, i]) * col_X[:, j]
        col_X = np.c_[col_X, np.matrix(tmp)]
        col1 = col1 + 1
    for i in range(col):
        col_X[:, i] = normalize(col_X[:, i])
    return col_X


def is_vectorization_of_hermitian_matrix(vector_x, tol=1e-8):
    """
    Check whether a vector is a vectorization of an Hermitian matrix
    Args:
        vector_x    : a vector
        tol         : tolerance
    Returns:
        True if the vector is a vectorization of an Hermitian matrix,
        False otherwise
    """
    return np.allclose(vector_x, vector_x.getH(), atol=tol)


def generate_complex_norm_matrix(number_of_row, number_of_col):
    """
    Generate a matrix form of complex norm array
    Args:
        number_of_row : number of rows
        number_of_col : number of columns
    Returns:
        a matrix form of complex norm array
    """
    # calculate normal distribution
    mu, sigma = 0, 1.0
    norm_dist_real = np.random.normal(mu, sigma, number_of_row * number_of_col)
    norm_dist_imag = np.random.normal(mu, sigma, number_of_row * number_of_col)

    # generate complex random array based on normal distribution
    complex_norm = (1 / np.sqrt(2)) * (norm_dist_real + 1j * norm_dist_imag)

    # reshape the complex array to a num_row-row and num_col-col array
    complex_norm.shape = (number_of_row, number_of_col)

    # get the matrix form of complex_norm array
    return np.matrix(complex_norm)


def power_consumption_on_relays(vectorization, power_source_1, power_source_2, orthogonal_matrix_1,
                                orthogonal_matrix_2):
    """
    Calculate the power consumption on relays.
    p/s: equation (30) on paper
    Args:
        vectorization       : a column vector.
        power_source_1      : power of the first source
        power_source_2      : power of the second source
        orthogonal_matrix_1 : one of two subspaces of the unitary space
        orthogonal_matrix_2 : one of two subspaces of the unitary space
    Returns:
        the power consumption on relays
    """
    tmp1 = LA.norm(vectorization.getH() * (np.kron(power_source_1.getT(), power_source_2)) * vectorization)
    tmp2 = LA.norm(vectorization.getH() * (np.kron(orthogonal_matrix_1.getT(), orthogonal_matrix_2)) * vectorization)
    return tmp1 / tmp2


def normalize(vector_x):
    """
    Normalize a vector by dividing each element by the square root of the sum of the squares
    Args:
        vector_x : a vector
    Return:
        the normalized vector
    """
    sum_x = np.sqrt(np.sum(np.multiply(vector_x, np.conjugate(vector_x)), axis=0))
    vector_x = np.matrix(vector_x / sum_x)
    return vector_x


def sqr_absolute_val(x):
    return np.real(x) * np.real(x) + np.imag(x) * np.imag(x)


