import numpy as np
from scipy.sparse.linalg import eigs
import cvxpy as cvx

from read_data import get_data


def dca_scheme(number_layers, power):
    """
    Different of Convex functions Algorithm (DCA) for achieving maximum secrecy rate with the appearance of an eavesdropper

    Args:
        number_layers   : Number of layers of relays
        power           : total energy used to operate the network
    """
    # get data from dataset
    number_of_layers, coefficient_from_source_1, coefficient_from_source_2, sigma_relay, sigma_from_source_1, \
    sigma_from_source_2, diag_coefficient_from_source_1, diag_coefficient_from_source_2, sigma_source_1_eavesdropper, \
    coefficient_source_1_eavesdropper, coefficient_source_2_eavesdropper, \
    D1, D2, Z_matrix, total_power = get_data(number_layers, power)

    # setup initial variables
    w0 = (np.matrix(np.random.rand(number_of_layers, 1)) + np.matrix(np.random.rand(number_of_layers, 1) * 1j))
    x0 = np.vstack((w0.real, w0.imag))

    power_source_1 = power_source_2 = total_power / 4
    epsilon = 0.0001

    # Prepare intermediate variables (some of them are complex numbers) for calculating in the DCA algorithm
    unit_matrix = np.identity(2 * number_of_layers)
    R1, R2, A1, A2, B1, B2, C = intermediate_variables(diag_coefficient_from_source_1, diag_coefficient_from_source_2,
                                                       coefficient_from_source_1, coefficient_from_source_2,
                                                       power_source_1, power_source_2, sigma_relay,
                                                       sigma_source_1_eavesdropper, D1, D2,
                                                       coefficient_source_1_eavesdropper,
                                                       coefficient_source_2_eavesdropper)
    # convert intermediate variables (complex numbers) to a matrix with 2 components (real and imaginary)
    A1_real_matrix, A2_real_matrix, B1_real_matrix, B2_real_matrix, D1_real_matrix, D2_real_matrix, \
    Z_real_matrix = convert_intermediate_variables_to_real_form(A1, A2, B1, B2, D1, D2, Z_matrix)

    # rho value is chosen as the largest eigenvalue for computing the second convex function in DCA algorithm
    rho = np.real(get_largest_eigenvalue(
        A1_real_matrix / (2 * sigma_from_source_1 * sigma_from_source_1) + A2_real_matrix / (
                2 * sigma_from_source_2 * sigma_from_source_2) + 2 * (
                B1_real_matrix / (sigma_from_source_1 * sigma_from_source_1)) + 2 * (
                B2_real_matrix / (sigma_from_source_1 * sigma_from_source_1))))
    rho_matrix = 1 / 2 * np.real(rho * unit_matrix)

    # ============================BUILDING MODEL=====================================
    x = cvx.Variable(shape=(2 * number_of_layers, 1))
    previous_x = np.real(x0)

    power_on_relays = power_source_1 * cvx.quad_form(x, D1_real_matrix) + \
                      power_source_2 * cvx.quad_form(x, D2_real_matrix) + \
                      sigma_relay * sigma_relay * cvx.quad_form(x, unit_matrix)

    # constrains of the optimization problem
    constraints = [power_on_relays <= total_power - power_source_1 - power_source_2,
                   Z_real_matrix.T * x == 0]

    while 1:
        linear_approximation_second_function = linear_approximation_of_second_function(x, previous_x, rho, rho_matrix,
                                                                                       A1_real_matrix, A2_real_matrix,
                                                                                       B1_real_matrix, B2_real_matrix,
                                                                                       sigma_from_source_1,
                                                                                       sigma_from_source_2)

        # Objective function for the optimization problem
        obj = cvx.Minimize(cvx.quad_form(x, rho_matrix) + np.log(C) - linear_approximation_second_function)
        prob = cvx.Problem(obj, constraints)
        prob.solve()

        current_dca_value = dca_function(x.value, rho_matrix, A1_real_matrix, A2_real_matrix, B1_real_matrix,
                                         B2_real_matrix, C, sigma_from_source_1, sigma_from_source_2)
        previous_dca_value = dca_function(previous_x, rho_matrix, A1_real_matrix, A2_real_matrix, B1_real_matrix,
                                          B2_real_matrix, C, sigma_from_source_1, sigma_from_source_2)

        # 2 conditions for stopping the DCA algorithm
        stopping_condition_1 = np.linalg.norm(x.value - previous_x) / (1 + pow(np.linalg.norm(previous_x), 2))
        stopping_condition_2 = abs(current_dca_value - previous_dca_value) / (1 + abs(previous_dca_value))

        if stopping_condition_1 < epsilon or stopping_condition_2 < epsilon:
            if current_dca_value < 0:
                print(-current_dca_value.item())
                return -current_dca_value.item()
            else:
                return 0

        previous_x = x.value


def dca_function(x_value, rho_matrix, A1_matrix, A2_matrix, B1_matrix, B2_matrix, C, sigma_from_source_1,
                 sigma_from_source_2):
    """
    Calculate an exact value of DCA function when knowing a value of x variable.
    DCA is an approximation method to split a non-convex function into the difference of two convex functions.
    DCA_function = G(.) - H(.), where G and H are convex
    For more information, read this link: http://www.lita.univ-lorraine.fr/~lethi/index.php/dca.html

    Args:
        x_value             : value of x variable
        rho_matrix          : the product of the largest eigenvalue and a unit matrix
        A1_matrix, A2_matrix, B1_matrix, B2_matrix    :intermediate variables
        sigma_from_source_1 : sigma coefficient from the first source to relays
        sigma_from_source_2 : sigma coefficient from the second source to relays
    Returns:
        a value of DCA function
    """
    return 1 / (2 * np.log(2)) * (first_convex_function_of_dca(x_value, rho_matrix, C) -
                                  second_convex_function_of_dca(x_value, rho_matrix, A1_matrix, A2_matrix, B1_matrix,
                                                                B2_matrix, sigma_from_source_1, sigma_from_source_2))


def first_convex_function_of_dca(x_value, rho_matrix, C_matrix):
    """
     Calculate the value of the first convex function of DCA algorithms

     Args:
         x_value             : value of x variable
         rho_matrix          : product of the largest eigenvalue and the unit matrix
         C_matrix            : an intermediate matrix with real numbers
     Returns:
          The value of the first convex function of DCA algorithms (G function)
     """
    x_value = np.matrix(x_value)
    return x_value.T * rho_matrix * x_value + np.log(C_matrix)


def second_convex_function_of_dca(x_value, rho_matrix, A1_matrix, A2_matrix, B1_matrix, B2_matrix, sigma_from_source_1,
                                  sigma_from_source_2):
    """
    Calculate the value of the second convex function of DCA (H function)

    Args:
        x_value             : value of x variable
        rho_matrix          : the largest eigenvalue (Rho in paper)
        A1_matrix, A2_matrix, B1_matrix, B2_matrix    :intermediate variables
        sigma_from_source_1 : sigma coefficient from the first source to relays
        sigma_from_source_2 : sigma coefficient from the second source to relays
    Returns:
        the value of the second convex function of DCA algorithm (H function)
    """
    x_value = np.matrix(x_value)
    return x_value.T * rho_matrix * x_value + np.log(
        pow(sigma_from_source_1, 2) + x_value.T * A1_matrix * x_value) + np.log(
        pow(sigma_from_source_2, 2) + x_value.T * A2_matrix * x_value) - np.log(
        pow(sigma_from_source_1, 2) + x_value.T * B1_matrix * x_value) - np.log(
        pow(sigma_from_source_2, 2) + x_value.T * B2_matrix * x_value)


def linear_approximation_of_second_function(x_value, previous_x, rho_value, rho_matrix, A1_matrix, A2_matrix,
                                            B1_matrix, B2_matrix, sigma_from_source_1, sigma_from_source_2):
    """
    Find the linear approximation of the second convex function (H function) of DCA algorithm.
    Because, To find directly an exact value of H function is not easy.

    Args:
        x_value             : value of x variable
        previous_x          : value of predicted variable in the previous step
        rho_value           : rho is chosen as the largest eigenvalue
        rho_matrix          : the product of the largest eigenvalue and a unit matrix
        A1_matrix, A2_matrix, B1_matrix, B2_matrix    :intermediate variables
        sigma_from_source_1 : sigma coefficient from the first source to relays
        sigma_from_source_2 : sigma coefficient from the second source to relays
    Returns:
        the linear approximation of the second convex function H.
    """
    second_function = second_convex_function_of_dca(previous_x, rho_matrix, A1_matrix, A2_matrix, B1_matrix, B2_matrix,
                                                    sigma_from_source_1, sigma_from_source_2)
    gradient_second_function = gradient_of_second_function(previous_x, rho_value, A1_matrix, A2_matrix, B1_matrix,
                                                           B2_matrix, sigma_from_source_1, sigma_from_source_2)
    return second_function + gradient_second_function * (x_value - previous_x)


def gradient_of_second_function(x_value, rho_value, A1_matrix, A2_matrix, B1_matrix, B2_matrix, sigma_from_source_1,
                                sigma_from_source_2):
    """
    Calculate gradients of the second convex function in order to find the linear approximation

    Args:
        x_value             : value of x variable
        rho_value           : rho is chosen as the largest eigenvalue
        A1_matrix, A2_matrix, B1_matrix, B2_matrix    :intermediate variables
        sigma_from_source_1 : sigma coefficient from the first source to relays
        sigma_from_source_2 : sigma coefficient from the second source to relays
    Returns:
        gradients of the second convex function
    """
    return (rho_value * x_value + 2 * A1_matrix * x_value / (
            pow(sigma_from_source_1, 2) + x_value.T * A1_matrix * x_value) +
            2 * A2_matrix * x_value / (pow(sigma_from_source_2, 2) + x_value.T * A2_matrix * x_value) -
            2 * B1_matrix * x_value / (pow(sigma_from_source_1, 2) + x_value.T * B1_matrix * x_value) -
            2 * B2_matrix * x_value / (pow(sigma_from_source_1, 2) + x_value.T * B2_matrix * x_value)).T


def get_largest_eigenvalue(matrix):
    """
    Choose the maximum eigenvalue of a matrix

    Args:
        matrix: a matrix with real numbers
    Returns:
        the largest eigenvalue
    """
    values, vector = eigs(matrix, 1, which="LM")
    return values[0]


def intermediate_variables(diag_coefficient_from_source_1, diag_coefficient_from_source_2, coefficient_from_source_1,
                           coefficient_from_source_2, power_source_1, power_source_2, sigma_relay,
                           sigma_source_1_eavesdropper, D1, D2, coefficient_source_1_eavesdropper,
                           coefficient_source_2_eavesdropper):
    """
    Collect all intermediate variables.
    There are a lot of intermediate variables used in DCA algorithm.
    """
    R1 = diag_coefficient_from_source_1 * coefficient_from_source_2 * coefficient_from_source_2.H * diag_coefficient_from_source_1.H
    R2 = diag_coefficient_from_source_2 * coefficient_from_source_1 * coefficient_from_source_1.H * diag_coefficient_from_source_2.H
    A1 = power_source_2 * R1 + sigma_relay * sigma_relay * D1
    A2 = power_source_1 * R2 + sigma_relay * sigma_relay * D2
    B1 = pow(sigma_relay, 2) * D1
    B2 = pow(sigma_relay, 2) * D2
    C = 1 + (power_source_1 * pow(np.abs(coefficient_source_1_eavesdropper), 2) + power_source_2 * pow(
        np.abs(coefficient_source_2_eavesdropper), 2)) / pow(sigma_source_1_eavesdropper, 2)
    return R1, R2, A1, A2, B1, B2, C


def convert_intermediate_variables_to_real_form(A1, A2, B1, B2, D1, D2, Z_matrix):
    """
    Convert all intermediate variables to the real form to avoid complex numbers in computing.
    """
    return real_imaginary_matrix(A1), real_imaginary_matrix(A2), real_imaginary_matrix(B1), real_imaginary_matrix(B2), \
           real_imaginary_matrix(D1), real_imaginary_matrix(D2), real_imaginary_matrix(Z_matrix)


def real_imaginary_matrix(matrix):
    """
    Create a matrix with 2 components (real and imaginary) to avoid working with complex numbers in computation

    Args:
        matrix: original matrix with complex numbers
    Returns:
        a matrix with 2 new components
    """
    return np.vstack((np.hstack((matrix.real, matrix.imag)), np.hstack((-matrix.imag, matrix.real))))
