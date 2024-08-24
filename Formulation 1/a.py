import numpy as np
import matplotlib.pyplot as plt
def plot():
    xs = [0.225,0.29,0.3,0.31]
    xs2 = np.linspace(0.21,0.32,100)
    ys = [1.509,1.439,1.435,1.421]
    ys_err = [0.007,0.005,0.015,0.009]
    axis = plt.subplot(111)
    axis.errorbar(xs,ys,yerr=ys_err,fmt='.',label='Data',color='blue',elinewidth=0.5)
    ys_analytical = [np.sqrt(2) for x in xs2]
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.plot(xs2,ys_analytical,label='Continuum Prediction',linestyle='dashed',color = 'red',linewidth=0.75)
    plt.legend()
    plt.yticks([1.3,1.35,1.4,1.45,1.5])
    plt.xlabel('$\\beta$',rotation=0)
    plt.ylabel('$\\frac{M_1}{M_0}$',rotation=0)
    plt.xlim(0.22,0.32)
    plt.ylim(1.32,1.52)
    plt.show()
plot()