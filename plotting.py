import matplotlib.pyplot as plt
import numpy as np
def plot_difference(N,lambda_,N_measurements,N_thermalization):
    observable_name = "Difference"  
    file_name = "Results/Processed/"+observable_name+"/"+observable_name+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measurements)+" N Thermal = "  + str(N_thermalization)+'.npy'

    vals = np.load(file_name)
    results = vals[0]/np.abs(lambda_)
    errs = vals[1]/np.abs(lambda_)
    axis = plt.subplot(111)

    print(results)
    x = np.arange(1,int(N/2)+1,1)
    print(x)
    axis.errorbar(x,results,yerr=errs,fmt='.',capsize=2,label = "Data")
    axis.plot(x,results,linestyle = 'dashed',color = 'blue',linewidth = 0.75)
    plt.xlabel("$|x-y|$")
    plt.ylabel("$\langle() \phi(x) - \phi (y))^2\\rangle$")
    axis.spines[['right', 'top']].set_visible(False)
    
    plt.title("$\lambda = $ "+str(lambda_)+ " N = "+str(N))

    plt.show()