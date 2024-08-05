from lattice import Lattice
import LatticeRun
import numpy as np
import plotting
import processing
def main():
    N = 8
    lambda_ = -1
    N_measurements = 1000
    N_thermalization = 1000
    lambdas = [-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2]
    guess = 50

    for lambda_ in lambdas:
        pass
        #guess = LatticeRun.calibration(N,lambda_,0.25,True,guess)
        #LatticeRun.measure(N,lambda_,N_measurements,N_thermalization, Lattice.measure_field, "Field",True,2)

        #LatticeRun.measure_func_1D(N,lambda_,N_measurements,N_thermalization,lambda l: Lattice.measure_difference(2,l), "Difference",True,2)
        #processing.process_difference(N,lambda_,N_measurements,N_thermalization,True)
    
    #L = Lattice(N,-1,10,100,10,True,epsilon,N_tau)4
    #processing.process_difference(N,-1,N_measurements,N_thermalization,False)


    #print(L.lattice)
    #LatticeRun.measure(N,lambda_,N_measurements,N_thermalization, Lattice.measure_field, "Field",True)
    #print(processing.process_field(N,lambda_,N_measurements,N_thermalization,True))
    #plotting.plot_difference(N,lambda_,N_measurements,N_thermalization,True)
    plotting.plot_difference_lambdas(N,[-1,-0.1,-0.01,-0.9,-0.8,-0.7,-0.6],N_measurements,N_thermalization,True)
    #lattice = Lattice(N,lambda_,N_measurements,N_thermalization)
    #print(np.average(lattice.generate_measurements(Lattice.measure_difference))
    #)

    #print(processing.process_difference(N,lambda_,N_measurements,N_thermalization))

    #lattice = np.random.normal(0,10,(N, N, N, N))

    #lattice[0,0,0,0] = 1

    #print(lattice)

# Call the function with the test array
    #result = print(Lattice.measure_difference(lattice))
main()
        