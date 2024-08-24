import matplotlib.pyplot as plt
import numpy as np
def plot_difference(N,lambda_,N_measurements,N_thermalization,HMC=False):
    observable_name = "Difference"  
    file_name = "Results/Processed/"+observable_name+"/"+observable_name+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measurements)+" N Thermal = "  + str(N_thermalization)+'.npy'
    if HMC:
        file_name = "Results/Processed/"+observable_name+"/"+observable_name+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measurements)+" N Thermal = "  + str(N_thermalization)+" HMC.npy"
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
    
    plt.title("$\lambda = $ "+str(lambda_)+ " N = "+str(N) + " HMC = "+str(HMC))

    plt.show()
def plot_difference_lambdas(N,lambdas,N_measurements,N_thermalization,HMC=False):
    axis = plt.subplot(111)
    colors = ['red','blue','green','orange','purple'] + ['cyan','magenta']

    for i,lambda_ in enumerate(lambdas):
        observable_name = "Difference"  
        file_name = "Results/Processed/"+observable_name+"/"+observable_name+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measurements)+" N Thermal = "  + str(N_thermalization)+'.npy'
        if HMC:
            file_name = "Results/Processed/"+observable_name+"/"+observable_name+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measurements)+" N Thermal = "  + str(N_thermalization)+" HMC.npy"       
        color = colors[i]
        vals = np.load(file_name)
        results = vals[0]#/np.abs(lambda_)
        errs = vals[1]#/np.abs(lambda_)
        results = list(results)
        errs = list(errs)
        results = [0] + results
        errs = [0] + errs

        #print(results)
        x = np.arange(0,int(N/2)+1,1)
        #print(x)
        axis.errorbar(x,results,yerr=errs,fmt='.',capsize=2,label = "$\lambda = $"+str(lambda_),color = color)
        axis.plot(x,results,linestyle = 'dashed',color = color,linewidth = 0.75)
        plt.xlabel("$|x-y|$")
        plt.ylabel("$\langle( \phi(x) - \phi (y))^2\\rangle$")
        
        axis.spines[['right', 'top']].set_visible(False)
        
        plt.title(" N = "+str(N) + " HMC = "+str(HMC))
        analytic =np.array([ 0,0.0381944,0.0575829,0.0661302, 0.0686041])
        #print(lambda_,np.abs(results-analytic)*100/analytic)
        print(lambda_,results)
    analytic =np.array([0, 0.0381944,0.0575829,0.0661302, 0.0686041])

    x = np.arange(0,int(N/2)+1,1)
    axis.plot(x,analytic,linestyle = 'dashed',color = 'black',linewidth = 0.75)
    axis.plot(x,analytic,'o',color = 'black',markersize = 3,label = "Analytic")
    observable_name = "Difference multiple Lambda"
    if HMC:
        observable_name = "Difference multiple Lambda HMC"
    
    file_name = "Results/Processed/Plots/"+observable_name+".svg"
    plt.legend(loc='lower right')
    #plt.legend()

    plt.xlim(0,6)
    plt.savefig(file_name)
    plt.show()