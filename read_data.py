import numpy as np


def get_data(n_th_layer, the_total_power):
    path = 'data/Testdata_Exp1_' + str(int(n_th_layer)) + '_' + str(int(the_total_power)) + '.npz'
    data = np.load(path)

    number_of_layers = np.int(data['N'])
    coefficient_from_source_1 = np.matrix(data['f1'])
    coefficient_from_source_2 = np.matrix(data['f2'])

    sigma_relay = np.int(data['sigmaR'])
    sigma_from_source_1 = np.int(data['sigma1'])
    sigma_from_source_2 = np.int(data['sigma2'])

    diag_coefficient_from_source_1 = np.matrix(data['F1'])
    diag_coefficient_from_source_2 = np.matrix(data['F2'])

    sigma_source_1_eavesdropper = np.int(data['sigma1E'])
    coefficient_source_1_eavesdropper = np.complex128(data['g1'])
    coefficient_source_2_eavesdropper = np.complex128(data['g2'])

    D1 = np.matrix(data['D1'])
    D2 = np.matrix(data['D2'])
    Z_matrix = np.matrix(data['Z'])
    total_power = np.int(data['Pt'])

    return number_of_layers, \
           coefficient_from_source_1, coefficient_from_source_2, \
           sigma_relay, sigma_from_source_1, sigma_from_source_2, \
           diag_coefficient_from_source_1, diag_coefficient_from_source_2, \
           sigma_source_1_eavesdropper, \
           coefficient_source_1_eavesdropper, coefficient_source_2_eavesdropper, \
           D1, D2, Z_matrix, total_power


