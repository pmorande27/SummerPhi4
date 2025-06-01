import numpy as np
from lattice import Lattice
from alive_progress import alive_bar
import os

def calibration( N, lambda_, width_guess,HMC,  N_steps_guess=0,mode=0,msq = 0):


    accel = False
    print('Calibration with lambda = ' + str(lambda_) + " N = " +str(N) )
    up = 0.8
    low = 0.4
    max_count = 10
    results = [0 for i in range(max_count)]
    width = width_guess
    accel = False
    if not HMC:
        for i in range(max_count):
            calibration_runs = 10**2
            N_measurements = 0
            N_thermalization = 0
            lat = Lattice( N, lambda_, N_measurements, N_thermalization,width,False,mode = 0)
            rate = lat.calibration_runs(calibration_runs, 100)
            
            
            results[i] = (rate-up,width)
            print(rate,width)
            

            if rate <=up and rate >= low:
                if accel == False:
                    if msq == 0:
                        file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N)
                    else:
                        file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N) + f" msq = {msq}"
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
    else:
        minimum = 10
        N_tau = N_steps_guess
        epsilon = 1/(N_tau)
        width = 0
        print('Calibration with beta = ' + str(lambda_) + " N = " +str(N)+ " N_tau = " + str(N_tau))
        up = 0.95
        low = 0.75
        max_count = 10
        results = [0 for i in range(max_count)]
        for i in range(max_count):
            if N_tau < minimum and i != 0:
                N_tau = minimum
                epsilon = 1/(N_tau)
                if accel == False:
                    if msq == 0:
                        file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N) + " HMC"
                    else:
                        file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N) + f" HMC msq = {msq}"
                else:
                    file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N) + " HMC Accel"
                print(file_name)
                np.save(file_name, [N_tau,1/N_tau])
                print("-----------------")
                print(rate, N_tau)
                return N_tau
            epsilon = 1/(N_tau)
            calibration_runs = 10**3
            lat = Lattice(N, lambda_,0,0,width, HMC, epsilon,N_tau)
            lat.calibration_runs(1000, 1000)
            rate = lat.accepted/calibration_runs
            d_rate = 0.85-rate
            results[i] = (rate-up,N_tau)
            print(rate*100,N_tau)
            

            new_N = int(np.rint(N_tau*(1+d_rate)))
            if rate <=up and rate >= low:
                if accel == False:
                    if msq == 0:
                        file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N) + " HMC"
                    else:
                        file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N) + f" HMC msq = {msq}"
                    
                    
                else:
                    file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N) + " HMC Accel"
                np.save(file_name, [N_tau,1/N_tau])
                print("-----------------")
                print(file_name,rate, N_tau)
                return N_tau
            if new_N == N_tau:
                if d_rate <0:
                    new_N -= 1
                else:
                    new_N +=1
            
            N_tau = new_N

       

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
def load_calibration( N, lambda_ ,HMC = False,accel = False,msq = 0):
    if accel == False:

        file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N) + ".npy"
    else:
        file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N) + " Accel.npy"
    if HMC:
        if accel == False:
            if msq == 0:
                file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N) + " HMC.npy"
            else:
                file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N) + f" HMC msq = {msq}.npy"
        else:
            file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N) + " HMC Accel.npy"
    values = np.load(    file_name)
    if HMC:
        return values[0],values[1]
    else:  

        return values[0]


def lookup(d_rate,N_tau,results):
    for (x,y) in results:
        if abs(x) == d_rate and y == N_tau:
            return x
        
def measure( N,lambda_, N_measure,N_thermal, observable, observable_name,HMC = False,dim=4,accel =False, mass = 0.1):
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
            if HMC:
                N_tau,epsilon = load_calibration(N,lambda_,HMC,accel)
                N_tau = int(N_tau)
                model = Lattice(N,lambda_,N_measure,N_thermal,1,HMC,epsilon,N_tau,dim)
            
            else:
                width = load_calibration(N,lambda_,accel)
                model = Lattice(N,lambda_,N_measure,N_thermal,width,False,0,0,dim) 

            results = model.generate_measurements(observable)

        except (ValueError) as e:
            print(e)
            count+= 1
            continue
        break
    #print(Stats(vals).estimate())
    np.save(file_name,results)
def generate_phis( N,lambda_, N_measure,N_thermal,HMC = False,dim=4,accel =False, mass = 0.1,guess = 40,mode=1,msq=0):
    if accel == False:
        if msq == 0:
            file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N)+' HMC.npy'
        else:  
            file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N)+f' HMC msq = {msq}.npy'
    else:
        file_name = "Parameters/Calibration parameters lambda = " + str(lambda_) + " N = " + str(N) + " Accel"+'.npy'
    if os.path.exists(file_name):
        print('Calibration already done',file_name)
        pass
    else:
        print(file_name)
        calibration(N,lambda_,1,HMC,guess,mode=1,msq =msq)

    observable_name = 'phi'
    count = 0
    while True:
        try:
            if count == 10:    
                count = 0
                print('Recalibration')
                calibration(N,lambda_,1)
            if accel == False:
                if msq == 0:
                    file_name = "Results/"+observable_name+"/"+observable_name+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+'.npy'
                else:
                    file_name = "Results/"+observable_name+"/"+observable_name+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+f' msq = {msq}.npy'
                #file_name = "Results/"+observable_name+"/"+observable_name+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+'.npy'
            else:
                file_name = "ChiralResults/"+observable_name+"/"+observable_name+" beta = " + str(lambda_) + " N = " + str(N)  + " N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+" Accel.npy"
            if HMC:
                N_tau,epsilon = load_calibration(N,lambda_,HMC,accel,msq=msq)
                N_tau = int(N_tau)
                model = Lattice(N,lambda_,N_measure,N_thermal,1,HMC,epsilon,N_tau,dim,mode=mode,msq=msq)
            
            else:
                width = load_calibration(N,lambda_,accel)
                model = Lattice(N,lambda_,N_measure,N_thermal,width,False,0,0,dim,msq =msq) 

            results = model.generate_phis()

        except (ValueError) as e:
            print(e)
            count+= 1
            continue
        break
    #print(Stats(vals).estimate())
    np.save(file_name,results)
    return N_tau


def measure_func_1D( N,lambda_,N_measure,N_thermal, observable, observable_name, HMC=False,dim=4,accel =False):
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
            if HMC:
                N_tau,epsilon = load_calibration(N,lambda_,HMC,accel)
                N_tau = int(N_tau)
                model = Lattice(N,lambda_,N_measure,N_thermal,1,HMC,epsilon,N_tau,dim)
            else:    
                width = load_calibration(N,lambda_,accel)
                model = Lattice(N,lambda_,N_measure,N_thermal,width,False,0,0,dim) 
            results = model.generate_measurements(observable)

        except (ValueError) as e:
            print(e)
            count+= 1
            continue
        break

    results = np.array(results)
    vals = results.swapaxes(0,1)
    np.save(file_name,vals)
    return N_tau

def measure_func_1D_3( N,lambda_,N_measure,N_thermal, observable_one, observable_name_one, observable_two,observable_name_two,observable_three,observable_name_three,HMC=False,dim=4,accel =False,mode = 0):
    count = 0
    while True:
        try:
            if count == 10:    
                count = 0
                print('Recalibration')
                calibration(N,lambda_,1)
            if accel == False:
                file_name_one = "Results/"+observable_name_one+"/"+observable_name_one+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+'.npy'
                file_name_two = "Results/"+observable_name_two+"/"+observable_name_two+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+'.npy'
                file_name_three = "Results/"+observable_name_three+"/"+observable_name_three+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+'.npy'
            if HMC:
                N_tau,epsilon = load_calibration(N,lambda_,HMC,accel)
                N_tau = int(N_tau)
                model = Lattice(N,lambda_,N_measure,N_thermal,1,HMC,epsilon,N_tau,dim,mode=mode)
            else:    
                width = load_calibration(N,lambda_,accel)
                model = Lattice(N,lambda_,N_measure,N_thermal,width,False,0,0,dim,mode) 
            results_one,results_two,results_three = model.generate_measurements_3(observable_one,observable_two,observable_three)


        except (ValueError) as e:
            print(e)
            count+= 1
            continue
        break
    results_one = np.array(results_one)
    vals = results_one.swapaxes(0,1)
    np.save(file_name_one,vals)
    results_two = np.array(results_two)
    vals = results_two.swapaxes(0,1)
    np.save(file_name_two,vals)
    results_three = np.array(results_three)
    vals = results_three.swapaxes(0,1)
    np.save(file_name_three,vals)

def turn_phis_to_measurements( N,lambda_, N_measure,N_thermal, observable, observable_name,HMC = False,dim=4,accel =False, mass = 0.1):
    if accel == False:
                file_name = "Results/"+"phi"+"/"+"phi"+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+'.npy'
    else:
        file_name = "ChiralResults/"+"phi"+"/"+"phi"+" beta = " + str(lambda_) + " N = " + str(N)  + " N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+" Accel.npy"
    configurations = np.load(file_name)
    measurements = []
    with alive_bar(len(configurations)) as bar:
        for config in configurations:
            measurements.append(observable(config))
            bar()
    measurements = np.array(measurements)
    file_name = "Results/"+observable_name+"/"+observable_name+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+'.npy'
    np.save(file_name,measurements)
def turn_phis_to_measurements_1D( N,lambda_, N_measure,N_thermal, observable, observable_name,HMC = False,dim=4,accel =False, mass = 0.1,msq = 0):
    
    if accel == False:
        if msq == 0:
                
            file_name = "Results/"+"phi"+"/"+"phi"+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+'.npy'
        else:
            file_name = "Results/"+"phi"+"/"+"phi"+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+f' msq = {msq}.npy'
    else:
        file_name = "ChiralResults/"+"phi"+"/"+"phi"+" beta = " + str(lambda_) + " N = " + str(N)  + " N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+" Accel.npy"
    configurations = np.load(file_name)
    measurements = []
    with alive_bar(len(configurations)) as bar:

        for config in configurations:
                measurements.append(observable(config))
                bar()
    measurements = np.array(measurements)
    if msq == 0:
        file_name = "Results/"+observable_name+"/"+observable_name+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+'.npy'
    else:
        file_name = "Results/"+observable_name+"/"+observable_name+" lambda = " + str(lambda_) + " N = " + str(N) +" N measurements = "  + str(N_measure)+" N Thermal = "  + str(N_thermal)+f' msq = {msq}.npy'
    vals = measurements.swapaxes(0,1)
    np.save(file_name,vals)