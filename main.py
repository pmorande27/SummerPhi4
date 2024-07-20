from lattice import Lattice
import LatticeRun
import numpy as np
import plotting
import processing
def main():
    N = 8
    lambda_ = -0.1
    N_measurements = 10000
    N_thermalization = 10000
    #LatticeRun.calibration(N,lambda_,1,True,100)
    #L = Lattice(N,-1,10,100,10,True,epsilon,N_tau)

    #print(L.lattice)
    #LatticeRun.measure(N,lambda_,N_measurements,N_thermalization, Lattice.measure_field, "Field")
    LatticeRun.measure_func_1D(N,lambda_,N_measurements,N_thermalization, Lattice.measure_difference, "Difference",True)
    processing.process_difference(N,lambda_,N_measurements,N_thermalization)
    #plotting.plot_difference(N,lambda_,N_measurements,N_thermalization)
    #plotting.plot_difference_lambdas(N,[-1,-0.1,-0.01,-0.001],N_measurements,N_thermalization)
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
        