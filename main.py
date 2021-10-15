from realDCA_CVX import *
from gram_smith import *

import matplotlib.pyplot as plt


def main():
    list_total_power = []
    list_rate_n4_dca, list_rate_n6_dca, list_rate_n8_dca = [], [], []
    list_rate_n4_ssrm, list_rate_n6_ssrm, list_rate_n8_ssrm = [], [], []
    for Pt in range(30, 41, 1):
        print("Pt = ", Pt)
        list_total_power.append(Pt)
        for N in range(4, 9, 2):
            print("N = ", N)
            dca_result = 0
            ssrm_result = 0
            j = 0
            for i in range(1, 11, 1):
                dca = dca_scheme(N, Pt)  # dca
                ssrm = secrecy_sum_rate_maximization(N, Pt)
                dca_result = dca_result + dca
                ssrm_result = ssrm_result + ssrm
            print("dca_result = ", dca_result)
            print("ssrm_result = ", ssrm_result)
            print("j = ", j)
            average_dca = dca_result / 10.0
            average_ssrm = ssrm_result / 10.0
            print("avarage_c = ", average_dca)
            print("avarage_d = ", average_ssrm)
            if N == 4:
                list_rate_n4_dca.append(average_dca)
                list_rate_n4_ssrm.append(average_ssrm)
            if N == 6:
                list_rate_n4_dca.append(average_dca)
                list_rate_n6_ssrm.append(average_ssrm)
            if N == 8:
                list_rate_n8_dca.append(average_dca)
                list_rate_n8_ssrm.append(average_ssrm)

    print("list_rate_N4:", list_rate_n4_dca)
    print("list_rate_N6:", list_rate_n6_dca)
    print("list_rate_N8:", list_rate_n8_dca)
    print("list_rate_N4_d:", list_rate_n4_ssrm)
    print("list_rate_N6_d:", list_rate_n6_ssrm)
    print("list_rate_N8_d:", list_rate_n8_ssrm)
    plt.plot(list_total_power, list_rate_n4_dca, color='red', label='N=4: DCA')
    plt.plot(list_total_power, list_rate_n4_ssrm, '--r', label='N=4: OBV')
    plt.plot(list_total_power, list_rate_n6_dca, color='green', label='N=6: DCA')
    plt.plot(list_total_power, list_rate_n6_ssrm, '--g', label='N=6: OBV')
    plt.plot(list_total_power, list_rate_n8_dca, color='blue', label='N=8: DCA')
    plt.plot(list_total_power, list_rate_n8_ssrm, '--b', label='N=8: OBV')
    plt.xlabel("Pt(dBW)")
    plt.ylabel("Sum Secrecy Rate (bít/s/Hz)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
