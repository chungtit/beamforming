from realDCA_CVX import *
from gram_smith import * 
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from numpy.ma import size, shape
from scipy.sparse.linalg import eigs

if __name__ == '__main__':

    list_Pt = []
    list_rate_N4 = []
    list_rate_N6 = []
    list_rate_N8 = []
    list_rate_N4_d = []
    list_rate_N6_d = []
    list_rate_N8_d = []
    for Pt in range(30, 41, 1):
        print("Pt = ", Pt)
        list_Pt.append(Pt)
        for N in range(4, 9, 2):
            print("N = ", N)
            sum_c = 0
            sum_d = 0
            j = 0
            for i in range(1, 11, 1):
                s_c = solvePro(N, Pt,i) #dca
                s_d = solving(N,Pt,i)
                sum_c = sum_c + s_c
                sum_d = sum_d + s_d
            print("sum_c = ", sum_c)
            print("sum_d = ", sum_d)
            print("j = ", j)
            average_c = sum_c/10.0
            average_d = sum_d/10.0
            print("avarage_c = ", average_c)
            print("avarage_d = ", average_d)
            if(N==4):
                list_rate_N4.append(average_c)
                list_rate_N4_d.append(average_d)
            if(N==6):
                list_rate_N6.append(average_c)
                list_rate_N6_d.append(average_d)
            if(N==8):
                list_rate_N8.append(average_c)
                list_rate_N8_d.append(average_d)

    print("list_rate_N4:", list_rate_N4)
    print("list_rate_N6:", list_rate_N6)
    print("list_rate_N8:", list_rate_N8)
    print("list_rate_N4_d:", list_rate_N4_d)
    print("list_rate_N6_d:", list_rate_N6_d)
    print("list_rate_N8_d:", list_rate_N8_d)
    plt.plot(list_Pt, list_rate_N4,color='red', label='N=4: DCA')
    plt.plot(list_Pt, list_rate_N4_d, '--r', label='N=4: OBV')
    plt.plot(list_Pt, list_rate_N6, color='green', label='N=6: DCA')
    plt.plot(list_Pt, list_rate_N6_d, '--g',label='N=6: OBV')
    plt.plot(list_Pt, list_rate_N8, color='blue',label='N=8: DCA')
    plt.plot(list_Pt, list_rate_N8_d, '--b',label='N=8: OBV')
    plt.xlabel("Pt(dBW)")
    plt.ylabel("Sum Secrecy Rate (bít/s/Hz)")
    plt.legend()
    plt.show()