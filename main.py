from lattice import Lattice
import LatticeRun
import numpy as np
import plotting
import processing
def main():
    N = 8
    lambda_ = -0.1
    N_measurements = 100
    N_thermalization = 100
    #LatticeRun.calibration(N,lambda_,0.5,False)
    #L = Lattice(N,-1,10000,10000,1,True,1,10)
    #print(L.lattice)
    #LatticeRun.measure(N,lambda_,N_measurements,N_thermalization, Lattice.measure_field, "Field")
    #LatticeRun.measure_func_1D(N,lambda_,N_measurements,N_thermalization, Lattice.measure_difference, "Difference")
    #processing.process_difference(N,lambda_,N_measurements,N_thermalization)
    plotting.plot_difference(N,lambda_,N_measurements,N_thermalization)
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
        