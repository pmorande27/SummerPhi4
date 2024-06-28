from Stats import Stats
import numpy as np
def process_difference(N,lambda_,N_measurements,N_thermalization):
    observable_name = "Difference"  
    file_name = "Results/"+observable_name+"/"+observable_name+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measurements)+" N Thermal = "  + str(N_thermalization)+'.npy'

    vals = np.load(file_name)
    results = np.zeros((int(N/2)))
    errs = np.zeros((int(N/2)))
    for i in range(0,int(N/2)):
        
        result ,err,_,_= Stats(vals[i]).estimate()
        results[i] = result
        errs[i] = err
    file_name = "Results/Processed/"+observable_name+"/"+observable_name+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measurements)+" N Thermal = "  + str(N_thermalization)+'.npy'
    np.save(file_name,(results,errs))

    return results,errs