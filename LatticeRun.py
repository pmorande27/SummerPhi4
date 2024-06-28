import numpy as np
from lattice import Lattice


def calibration( N, lambda_, width_guess):

    accel = False
    print('Calibration with lambda = ' + str(lambda_) + " N = " +str(N) )
    up = 0.6
    low = 0.2
    max_count = 10
    results = [0 for i in range(max_count)]
    width = width_guess

    for i in range(max_count):
        calibration_runs = 10**2
        N_measurements = 0
        N_thermalization = 0
        lat = Lattice( N, lambda_, N_measurements, N_thermalization,width)
        rate = lat.calibration_runs(calibration_runs, 100)
        
        
        results[i] = (rate-up,width)
        print(rate,width)
        

        if rate <=up and rate >= low:
            if accel == False:
                file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N)
            else:
                file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N) + " Accel"
            print("-----------------")
            np.save(file_name, [width])
            return width
        
        else:
            new_width = width
            if rate > up:
                new_width *= 2
            else:
                new_width *=    0.5
        
        width = new_width
       

    print("-----------------")
    print("Calibration Unsucessful, better run:")
    results_abs = [(abs(x),y) for (x,y) in results]
    d_rate, width = min(results_abs)
    d_rate_2 = lookup(d_rate,width,results)
    rate = (d_rate_2+up)*100
    print(rate,width)
    if accel == False:
        file_name = "ChiralParams/Chiral Calibration parameters lambda = " + str(lambda_) + " N = " + str(N) 
    else:
        file_name = "ChiralParams/Chiral Calibration parameters lambda = " + str(lambda_) + " N = " + str(N)  + " Accel"
    np.save(file_name, [width])
    return width
def load_calibration( N, lambda_ ,accel = False):
    if accel == False:
        file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N) + ".npy"
    else:
        file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N) + " Accel.npy"
    values = np.load(    file_name)
    return values[0]


def lookup(d_rate,N_tau,results):
    for (x,y) in results:
        if abs(x) == d_rate and y == N_tau:
            return x
        
def measure( N,lambda_, N_measure,N_thermal, observable, observable_name,accel =False, mass = 0.1):
    count = 0
    while True:
        try:
            if count == 10:    
                count = 0
                print('Recalibration')
                calibration(N,lambda_,1)
            if accel == False:
                file_name = "Results/"+observable_name+"/"+observable_name+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+'.npy'
            else:
                file_name = "ChiralResults/"+observable_name+"/"+observable_name+" beta = " + str(lambda_) + " N = " + str(N)  + " N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+" Accel.npy"
            width = load_calibration(N,lambda_,accel)
            model = Lattice(N,lambda_,N_measure,N_thermal,width) 
            results = model.generate_measurements(observable)

        except (ValueError):
            count+= 1
            continue
        break
    #print(Stats(vals).estimate())
    np.save(file_name,results)



def measure_func_1D( N,lambda_,N_measure,N_thermal, observable, observable_name, accel =False):
    count = 0
    while True:
        try:
            if count == 10:    
                count = 0
                print('Recalibration')
                calibration(N,lambda_,1)
            if accel == False:
                file_name = "Results/"+observable_name+"/"+observable_name+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+'.npy'
            else:
                file_name = "ChiralResults/"+observable_name+"/"+observable_name+" beta = " + str(lambda_) + " N = " + str(N)  + " N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+" Accel.npy"
            width = load_calibration(N,lambda_,accel)
            model = Lattice(N,lambda_,N_measure,N_thermal,width) 
            results = model.generate_measurements(observable)

        except (ValueError):
            count+= 1
            continue
        break

    results = np.array(results)
    vals = results.swapaxes(0,1)
    np.save(file_name,vals)
