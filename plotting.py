import matplotlib.pyplot as plt
import numpy as np
def plot_difference(N,lambda_,N_measurements,N_thermalization):
    observable_name = "Difference"  
    file_name = "Results/Processed/"+observable_name+"/"+observable_name+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measurements)+" N Thermal = "  + str(N_thermalization)+'.npy'

    vals = np.load(file_name)
    results = vals[0]#/np.abs(lambda_)
    errs = vals[1]#/np.abs(lambda_)
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
def plot_difference_lambdas(N,lambdas,N_measurements,N_thermalization):
    axis = plt.subplot(111)
    colors = ['red','blue','green','orange','purple','black']

    for i,lambda_ in enumerate(lambdas):
        observable_name = "Difference"  
        file_name = "Results/Processed/"+observable_name+"/"+observable_name+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measurements)+" N Thermal = "  + str(N_thermalization)+'.npy'
        color = colors[i]
        vals = np.load(file_name)
        results = vals[0]#/np.abs(lambda_)
        errs = vals[1]#/np.abs(lambda_)

        print(results)
        x = np.arange(1,int(N/2)+1,1)
        print(x)
        axis.errorbar(x,results,yerr=errs,fmt='.',capsize=2,label = str(lambda_),color = color)
        axis.plot(x,results,linestyle = 'dashed',color = color,linewidth = 0.75)
        plt.xlabel("$|x-y|$")
        plt.ylabel("$\langle() \phi(x) - \phi (y))^2\\rangle$")
        axis.spines[['right', 'top']].set_visible(False)
        
        plt.title(" N = "+str(N))
    analytic =[ 0.0381944,0.0575829,0.0661302, 0.0686041]

    x = np.arange(1,int(N/2)+1,1)
    axis.plot(x,analytic,linestyle = 'dashed',color = 'black',linewidth = 0.75)
    axis.plot(x,analytic,'o',color = 'black',markersize = 3,label = "Analytic")
    plt.legend()

    plt.ylim(0,0.1)
    plt.show()