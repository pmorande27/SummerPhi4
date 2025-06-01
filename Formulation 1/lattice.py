import numpy as np
from alive_progress import alive_bar
import matplotlib.pyplot as plt

class Lattice(object):
    """
    Class to represent a lattice and perform Monte Carlo simulations on it.
    This can be used in the following way:
    1. Create an instance of the class with the desired parameters. This will laso thermalize the lattice.
    2. Generate measurements of an observable using the generate_measurements method.
    3. Analyse the results of the measurements, using the Stats class (and the processing.py file)
    """


    def __init__(self, N, lambda_, N_measurements, N_thermalization, width, HMC = False, epsilon = 0, N_steps = 0, dim = 4,mode = 0, msq =0):
        """
        Initializes the lattice with the given parameters.

        Parameters:
        N: int, number of sites in each dimension
        lambda_: float, coupling constant
        N_measurements: int, number of measurements to be taken
        N_thermalization: int, number of thermalization sweeps
        width: float, width of the gaussian used in the metropolis algorithm
        HMC: bool, whether to use the Hamiltonian Monte Carlo algorithm or metropolis
        epsilon: float, step size for HMC
        N_steps: int, number of steps for HMC
        dim: int, number of dimensions of the lattice
        """
        self.width = width
        self.lambda_ = lambda_
        self.N = N
        self.N_thermalization = N_thermalization
        self.N_measurements = N_measurements
        self.dim = dim
        self.shape = [N for i in range(dim)] # shape of the lattice in numpy (library) format
        self.lattice = np.zeros(shape = self.shape) # initialize the lattice
        self.dhs = np.zeros(N_measurements) # array to store the exponential of the change in Hamiltonian for each measurement (test purposes)
        self.HMCs = HMC
        self.mode = mode
        self.msq = 0
        if HMC: # if HMC is True, initialize the HMC parameters
            self.epsilon = epsilon
            self.N_steps = int(N_steps)
        
        self.accepted = 0 # number of accepted configurations accepted
        
        # Hot start of the lattice
        #self.randomize()

       # Thermalize the lattice
        if self.N_thermalization > 0:
            self.thermalize()

    def action(self):
        """
        Computes the action of the lattice.
        """
        return 1/(2*np.abs(self.lambda_))*np.sum(self.I(self.lattice)**2)
    
    def metropolis(self, i, j, k, l):
        """
        Performs a metropolis update at site (i,j,k,l).
        """
        x = i
        y = j
        z = k
        t = l

        old = self.lattice[x, y, z, t]
        old_action = self.action() #store old action
        self.lattice[x, y, z, t] += np.random.normal(0, self.width) # propose a new configuration
        delta = self.action() - old_action # compute the change in action
        
        # accept or reject the new configuration
        if np.random.rand() > np.exp(-delta):
            self.lattice[x, y, z, t] = old
        else:
            self.accepted += 1
      
   
    def sweep(self):
        """
        Performs a sweep of the lattice using the metropolis algorithm, inefficent due to the loop over all sites.
        """
        for i in range(self.N):

            for j in range(self.N):

                for k in range(self.N):

                    for l in range(self.N):

                        self.metropolis(i, j, k, l)
    def thermalize(self):
        """
        Thermalizes the lattice.
        It uses the HMC algorithm if HMC is True, otherwise it uses the metropolis algorithm.
        """
        print("Thermalizing")
        with alive_bar(self.N_thermalization) as bar: # this line just creates the progress bar, not important
            for i in range(self.N_thermalization):
                
                if self.HMCs:
                
                    self.HMC(self.N_steps, self.epsilon,mode=self.mode) # call HMC
                
                else:
                
                    self.sweep()
                
                bar() # this line updates the progress bar, not important
        
        print("Thermalization Complete----------------")
      
    def generate_measurements(self, observable):
        """
        Generates the measurements of the observable.
        It uses the HMC algorithm if HMC is True, otherwise it uses the metropolis
        algorithm.
        attribute:
        observable: function, the observable to be measured
        """
        results = [0 for i in range(self.N_measurements)] # Initialize the array to store the results of the measurements
        print("Generating Measurements")
        
        with alive_bar(self.N_measurements) as bar: # this line just creates the progress bar, not important
            for i in range(self.N_measurements):
                
                if self.HMCs:

                    self.dhs[i] = np.exp(-self.HMC(self.N_steps, self.epsilon,mode=self.mode)) # store the exponential of the change in Hamiltonian (testing purposes), it call HMC
                else:

                    self.sweep() # perform a sweep if metropolis is used
                
                results[i] = observable(self.lattice) # store the result of the observable
                
                bar() # this line updates the progress bar, not important
        print("Measurements Complete----------------", np.average(self.dhs)) # print the average of the exponential of the change in Hamiltonian, should be close to 1
        
        return results
    def generate_phis(self):
        """
        Generates the measurements of the observable.
        It uses the HMC algorithm if HMC is True, otherwise it uses the metropolis
        algorithm.
        attribute:
        observable: function, the observable to be measured
        """
        results = [0 for i in range(self.N_measurements)] # Initialize the array to store the results of the measurements
        print("Generating Measurements")
        
        with alive_bar(self.N_measurements) as bar: # this line just creates the progress bar, not important
            for i in range(self.N_measurements):
                
                if self.HMCs:

                    self.dhs[i] = np.exp(-self.HMC(self.N_steps, self.epsilon,mode=self.mode)) # store the exponential of the change in Hamiltonian (testing purposes), it call HMC
                else:

                    self.sweep() # perform a sweep if metropolis is used
                
                results[i] = self.lattice.copy() # store the result of the observable
                
                bar() # this line updates the progress bar, not important
        print("Measurements Complete----------------", np.average(self.dhs)) # print the average of the exponential of the change in Hamiltonian, should be close to 1
        
        return results
    def generate_measurements_3(self, observable_one, observable_two, observable_three):
        """
        Generates the measurements of the observable.
        It uses the HMC algorithm if HMC is True, otherwise it uses the metropolis
        algorithm.
        attribute:
        observable: function, the observable to be measured
        """
        results_one = [0 for i in range(self.N_measurements)] # Initialize the array to store the results of the measurements
        results_two = [0 for i in range(self.N_measurements)] # Initialize the array to store the results of the measurements
        results_three = [0 for i in range(self.N_measurements)] # Initialize the array to store the results of the measurements
        print("Generating Measurements")
        
        with alive_bar(self.N_measurements) as bar: # this line just creates the progress bar, not important
            for i in range(self.N_measurements):
                
                if self.HMCs:

                    self.dhs[i] = np.exp(-self.HMC(self.N_steps, self.epsilon,mode = self.mode)) # store the exponential of the change in Hamiltonian (testing purposes), it call HMC
                else:

                    self.sweep() # perform a sweep if metropolis is used
                
                results_one[i] = observable_one(self.lattice) # store the result of the observable
                results_two[i] = observable_two(self.lattice) # store the result of the observable
                results_three[i] = observable_three(self.lattice) # store the result of the observable
                bar() # this line updates the progress bar, not important
        print("Measurements Complete----------------", np.average(self.dhs)) # print the average of the exponential of the change in Hamiltonian, should be close to 1
        
        return results_one, results_two, results_three
    
    
    def randomize(self):
        """
        Randomizes the lattice.
        """
        
        self.lattice = np.random.normal(size = self.shape)
    
    def calibration_runs(self,calibration_runs, thermal_runs):
        
        """ Performs calibration runs to determine the acceptance rate of the HMC algorithm.
        Used to tune the algorithms parameters
        """

        dHs = np.zeros(calibration_runs)
        with alive_bar(thermal_runs) as bar:
            for i in range(thermal_runs):
                if self.HMCs:
                    self.HMC(self.N_steps, self.epsilon,mode=self.mode)
                else:
                    self.sweep()
                bar()
        self.accepted = 0
        with alive_bar(calibration_runs) as bar:

            for i in range(calibration_runs):
                if self.HMCs:
                    dHs[i] = np.exp(-self.HMC(self.N_steps, self.epsilon,self.mode))
                else:
                    self.sweep()
                bar()
        if self.HMCs:
            return self.accepted/calibration_runs
        else:
            return self.accepted/(calibration_runs*self.N**4)
    def measure_derivative_correlator(dim,lattice):
        N = len(lattice)
        shape = [N+1,dim] + [N for i in range(dim)]
        result = np.zeros(shape)
        derivative_0=  (np.roll(lattice, 1, axis = 0) -lattice)
        derivative_1= (np.roll(lattice, 1, axis = 1) - lattice)
        derivative_2=  (np.roll(lattice, 1, axis = 2) - lattice)
        derivative_3=  (np.roll(lattice, 1, axis = 3) - lattice)

        

        #print(result.shape)

        corr = derivative_0*derivative_0[0,0,0,0] + derivative_1*derivative_1[0,0,0,0] + derivative_2*derivative_2[0,0,0,0] + derivative_3*derivative_3[0,0,0,0]

        final = np.sum(corr)
        #print(result)
        #print(result)
       
        return final
        
    def measure_difference(dim, lattice):
        """
        Measures (phi(x)-phi(y))^2
        """
        #Inefficient code first, then efficient code
        """ N = len(lattice)
        result = np.zeros((int(N/2),4,N,N,N,N))
        final = np.zeros((int(N/2)))

        for x in range(N):
            for y in range(N):
                for z in range(N):
                    for t in range(N):
                        for i in range(0,int(N/2)):
                            
                            result[i,0,x,y,z,t] = (lattice[x,y,z,t]-lattice[(x+1+i)%N,y,z,t])**2
                            result[i,1,x,y,z,t] = (lattice[x,y,z,t]-lattice[x,(y+1+i)%N,z,t])**2
                            result[i,2,x,y,z,t] = (lattice[x,y,z,t]-lattice[x,y,(z+1+i)%N,t])**2
                            result[i,3,x,y,z,t] = (lattice[x,y,z,t]-lattice[x,y,z,(t+1+i)%N])**2
        for i in range(0,int(N/2)):
            for j in range(4):

                final[i] += np.sum(result[i,j])/N**4
        final = final/4
        return final"""
       
        N = len(lattice)
        shape = [int(N/2),dim] + [N for i in range(dim)]
        result = np.zeros(shape)

        #print(result.shape)

        for i in range(int(N/2)):
            for j in range(dim):
                shift = np.roll(lattice, i+1, axis = j)
                result[i, j] =(lattice - shift)**2
        #print(result)
        #print(result)
        axis = tuple([i for i in range(1, dim+2)])
        final = np.sum(result, axis = axis) / (dim * N**dim)
        return final
    def measure_correlation(dim, lattice):
        N = len(lattice)
        shape = [N+1,dim] + [N for i in range(dim)]
        result = np.zeros(shape)

        #print(result.shape)

        for i in range(N+1):
            for j in range(dim):
                shift = np.roll(lattice, i, axis = j)
                result[i, j] =(lattice * shift)
        #print(result)
        #print(result)
        axis = tuple([i for i in range(1, dim+2)])
        final = np.sum(result, axis = axis) / (dim * N**dim)
        return final
    def measure_correlation_deri(dim, lattice,lambda_):
        N = len(lattice)
        shape = [N+1,dim] + [N for i in range(dim)]
        result = np.zeros(shape)
        der_0 = -lattice + np.roll(lattice, -1, axis = 0)
        der_1 = -lattice + np.roll(lattice, -1, axis = 1)
        der_2 = -lattice + np.roll(lattice, -1, axis = 2)
        der_3 = -lattice + np.roll(lattice, -1, axis = 3)

        #print(result.shape)

        for i in range(N+1):
            for j in range(dim):
                shift = np.roll(lattice, i, axis = j)
                result[i, j] = der_0*np.roll(der_0,i,axis=j) + der_1*np.roll(der_1,i,axis=j) + der_2*np.roll(der_2,i,axis=j) + der_3*np.roll(der_3,i,axis=j)
        #print(result)
        #print(result)
        axis = tuple([i for i in range(1, dim+2)])
        final = 1/np.abs(lambda_)*np.sum(result, axis = axis) / (dim * N**dim)*(4*np.pi)**2
        print(final)
        return final
    def measure_correlation_box(dim, lattice,lambda_):
        N = len(lattice)
        shape = [N+1,dim] + [N for i in range(dim)]
        result = np.zeros(shape)
        der_0 = -2*lattice + np.roll(lattice, -1, axis = 0) + np.roll(lattice, 1, axis = 0)
        der_1 = -2*lattice + np.roll(lattice, -1, axis = 1) + np.roll(lattice, 1, axis = 1)
        der_2 = -2*lattice + np.roll(lattice, -1, axis = 2) + np.roll(lattice, 1, axis = 2)
        der_3 = -2*lattice + np.roll(lattice, -1, axis = 3) + np.roll(lattice, 1, axis = 3)
        box = der_0 + der_1 + der_2 + der_3

        #print(result.shape)

        for i in range(N+1):
            for j in range(dim):
                shift = np.roll(lattice, i, axis = j)
                result[i, j] = box*np.roll(box,i,axis=j)
        #print(result)
        axis = tuple([i for i in range(1, dim+2)])
        final = 1/np.abs(lambda_)*np.sum(result, axis = axis) / (dim * N**dim)
        print(final)
        return final
    def measure_exponential_f(dim, lattice,c,lambda_):
        N = len(lattice)
        shape = [N+1,dim] + [N for i in range(dim)]
        result = np.zeros(shape, dtype = np.complex128)
        exponnential = np.exp(1j*c*lattice/np.sqrt(abs(lambda_)))
        spatial_average = np.sum(exponnential)/N**dim
        substracted = exponnential - spatial_average


        #print(result.shape)

        for i in range(N+1):
            for j in range(dim):
                shift = np.roll(substracted, i, axis = j)
                result[i, j] = substracted*np.conjugate(shift)
        #print(result)
        axis = tuple([i for i in range(1, dim+2)])
        
                
        final = np.sum(result, axis = axis) / (dim * N**dim)
        #plt.plot(final)
        #plt.show()
        #print(final)
        return final
    def measure_exponential(dim, lattice,c):
        N = len(lattice)
        shape = [N+1,dim] + [N for i in range(dim)]
        result = np.zeros(shape)

        #print(result.shape)

        for i in range(N+1):
            for j in range(dim):
                shift = np.roll(lattice, i, axis = j)
                result[i, j] =(np.exp(1j*c*lattice)) * np.exp(-1j*c*shift)
        #print(result)
        #print(result)
        axis = tuple([i for i in range(1, dim+2)])
        final = np.sum(result, axis = axis) / (dim * N**dim)
        return final
    def measure_exponential_average_f(dim, lattice,c,lambda_):
        N = len(lattice)
        lat_exp = np.exp(1j*c*lattice/np.sqrt(abs(lambda_)))
        avg = np.sum(lat_exp)/N**dim
        sq = np.real(avg*np.conj(avg))

        
        return sq
    def measure_sq_array(dim, lattice):
        N = len(lattice)
        shape = [N+1,dim] + [N for i in range(dim)]
        result = np.zeros(shape)

        #print(result.shape)

        for i in range(N+1):
            for j in range(dim):
                shift = np.roll(lattice, i, axis = j)
                result[i, j] =(lattice * shift)
        #print(result)
        #print(result)
        axis = tuple([i for i in range(1, dim+2)])
        final = np.sum(result, axis = axis) / (dim * N**dim)
        final = np.array([final[0] for i in range(len(final))])
        return final

    def measure_field(dim, lattice):
        """
        Measures phi(x)
        """
        return np.sum(lattice)/(len(lattice)**dim)
    
    def backwards(self, lattice,axis):
        """
        Computes backward coefficient for the molecular dynamics evolution.
        """
        result = (1 + 2*lattice - 2*np.roll(lattice, 1, axis)) #np.roll(lattice, 1, axis) is phi(x-mu) in the notes, axis plays the role of mu in the notes
        return result
    
    def forwards(self, lattice, axis):
        """
        Computes forward coefficient for the molecular dynamics evolution.
        """
        result = 1
        return result
    
    def centre(self, lattice, axis):
        """
        Computes centre coefficient for the molecular dynamics evolution.
        """
        return 2*lattice - 2 - 2*np.roll(lattice, -1, axis) #np.roll(lattice, -1, axis) is phi(x+mu) in the notes, axis plays the role of mu in the notes
    
    def molecular_dynamics(self, N_steps, epsilon, p_0, phi_0):
        """
        
        Performs the molecular dynamics evolution of the Hamiltonian Monte Carlo algorithm.
        
        Parameters:
        N_steps: int, number of steps in the molecular dynamics evolution
        epsilon: float, step size for the molecular dynamics evolution
        p_0: numpy array, the momentum at the beginning of the evolution
        phi_0: numpy array, the lattice at the beginning of the evolution

        Returns:
        numpy array, the momentum at the end of the evolution
        numpy array, the lattice at the end of the evolution
        """

        
        p = p_0 + epsilon/2*self.dot_p(phi_0) # half step for the momentum
        phi = phi_0.copy() # copy the lattice
        
        for i in range(N_steps):
            phi +=  epsilon*p # full step for the lattice
            
            if i == N_steps-1:
                p +=  epsilon/2*self.dot_p(phi) # half step for the momentum in the last step
            
            else:
                p += epsilon*self.dot_p(phi) # full step for the momentum in the other steps
        
        return p,phi
    def molecular_dynamics2(self, N_steps, epsilon, p_0, phi_0):
        """
        
        Performs the molecular dynamics evolution of the Hamiltonian Monte Carlo algorithm.
        
        Parameters:
        N_steps: int, number of steps in the molecular dynamics evolution
        epsilon: float, step size for the molecular dynamics evolution
        p_0: numpy array, the momentum at the beginning of the evolution
        phi_0: numpy array, the lattice at the beginning of the evolution

        Returns:
        numpy array, the momentum at the end of the evolution
        numpy array, the lattice at the end of the evolution
        """

        
        p = p_0 + epsilon/2*self.dphimin_2(phi_0) # half step for the momentum
        phi = phi_0.copy() # copy the lattice
        
        for i in range(N_steps):
            phi +=  epsilon*p # full step for the lattice
            
            if i == N_steps-1:
                p +=  epsilon/2*self.dphimin_2(phi) # half step for the momentum in the last step
            
            else:
                p += epsilon*self.dphimin_2(phi) # full step for the momentum in the other steps
        
        return p,phi        
    
    def HMC(self, N_steps, epsilon,flag = False,mode = 0):
        """
        Performs a Hamiltonian Monte Carlo update.

        Parameters:
        N_steps: int, number of steps in the molecular dynamics evolution
        epsilon: float, step size for the molecular dynamics evolution
        flag: bool, test purposes used to always accept the new lattice if needed

        Returns:
        float, the change in Hamiltonian
        """
        if mode == 0:
            p = np.random.normal(size=self.shape)
            H = self.action() + np.sum(p**2)/2 # compute the Hamiltonian
            p_new, lattice_new = self.molecular_dynamics(N_steps, epsilon, p.copy(), self.lattice.copy()) # perform the molecular dynamics evolution
            H_new = self.actions(lattice_new) + np.sum(p_new**2)/2 # compute the new Hamiltonian
            delta_H = H_new - H # compute the change in Hamiltonian

            # Metropolis acceptance step
            if delta_H <0 or np.exp(-delta_H) > np.random.random():
            
                self.lattice = lattice_new.copy()
                self.accepted += 1
            if flag:
                self.lattice = lattice_new.copy()
        else:
            p = np.random.normal(size=self.shape)
            H = self.action2() + np.sum(p**2)/2 # compute the Hamiltonian
            p_new, lattice_new = self.molecular_dynamics2(N_steps, epsilon, p.copy(), self.lattice.copy()) # perform the molecular dynamics evolution
            H_new = self.actions2(lattice_new) + np.sum(p_new**2)/2 # compute the new Hamiltonian
            delta_H = H_new - H # compute the change in Hamiltonian

            # Metropolis acceptance step
            if delta_H <0 or np.exp(-delta_H) > np.random.random():
            
                self.lattice = lattice_new.copy()
                self.accepted += 1
            if flag:
                self.lattice = lattice_new.copy()
            


        return delta_H
    def I(self, lattice):
        """
        Calculates the factor I(x) for the action and the dot_p function.

        Parameters:
        lattice: numpy array, the lattice

        Returns:
        numpy array, the factor I(x)
        """
        result = 0
        for i in range(self.dim):
            forward = np.roll(lattice, -1, axis = i)
            backward = np.roll(lattice, 1, axis = i)
            result += (forward + backward - 2*lattice + (forward-lattice)**2+self.msq)
        return result 
    def I2(self, lattice):
        """
        Calculates the factor I(x) for the action and the dot_p function.

        Parameters:
        lattice: numpy array, the lattice

        Returns:
        numpy array, the factor I(x)
        """
        result = 0
        for i in range(self.dim):
            forward = np.roll(lattice, -1, axis = i)
            backward = np.roll(lattice, 1, axis = i)
            result += forward + backward - 2*lattice + 0.5*(forward-lattice)**2+0.5*(backward-lattice)**2+self.msq
        return result 
    def action2(self,):
        """
        Compute the action of the lattice

        Parameters:
        lattice: numpy array, the lattice

        Returns:
        float, the action of the lattice
        """
        return 1/(2*np.abs(self.lambda_))*np.sum(self.I2(self.lattice)**2)
    def actions2(self, lattice):
        """
        Compute the action of the lattice

        Parameters:
        lattice: numpy array, the lattice

        Returns:
        float, the action of the lattice
        """
        return 1/(2*np.abs(self.lambda_))*np.sum(self.I2(lattice)**2)
    def actions(self, lattice):
        """
        Compute the action of the lattice

        Parameters:
        lattice: numpy array, the lattice

        Returns:
        float, the action of the lattice
        """
        return 1/(2*np.abs(self.lambda_))*np.sum(self.I(lattice)**2)
    
    
    
    def dot_p(self, lattice):
        """
        Computes the evolution of the momentum in the Hamiltonian Monte Carlo algorithm. dot p = -dS/dphi

        Parameters:
        lattice: numpy array, the lattice

        Returns:
        numpy array, the evolution of the momentum
        """

        I = self.I(lattice)
        result = 0
        for i in range(self.dim):
            Foward = np.roll(I,-1,axis = i)
            Backward = np.roll(I,1,axis = i)*(1 + 2*lattice - 2*np.roll(lattice, 1, i))
            Centre = I*(2*lattice - 2 - 2*np.roll(lattice, -1, i))
            result += Foward + Backward + Centre
        return -1/np.abs(self.lambda_)*result # return the evolution of the momentum, don't forget the minus sign
    def return_slice_lattice_x3(self,y):
        """
        Returns a slice of the lattice along the axis axis.
        """
        slice_ = [self.lattice[i][y][0][0] for i in range(self.N)]
        return slice_
    def return_slice_lattice_x(self):
        """
        Returns a slice of the lattice along the axis axis.
        """
        slice_ = [self.lattice[i][0][0][0] for i in range(self.N)]
        return slice_
    def return_slice_lattice_x2(self,lattice):
        """
        Returns a slice of the lattice along the axis axis.
        """
        slice_ = [lattice[i][0][0][0] for i in range(self.N)]
        return slice_

    def two_vortices_smoothed_initial_configuration(self,d,shift,Ls):
        x = (0.5+shift,.5,0.5,0.5)
        y = (0.5+shift+2*d,.5,0.5,0.5)
        
        x_loc_i = (x[0])
        x_loc_j = (x[1])
        x_loc_k = (x[2])
        x_loc_l = (x[3])

        y_loc_i = (y[0])
        y_loc_j = (y[1])
        y_loc_k = (y[2])
        y_loc_l = (y[3])

        z_loc_i = (x[0]+y[0])/2
        z_loc_j = (x[1]+y[1])/2
        z_loc_k = (x[2]+y[2])/2
        z_loc_l = (x[3]+y[3])/2
        print(z_loc_i,z_loc_j,z_loc_k,z_loc_l)

        
        lat = np.zeros((self.N, self.N, self.N, self.N))

        i, j, k, l = np.indices((self.N, self.N, self.N, self.N))

        # Calculate distances with periodic boundary conditions
        #dist_x_loc = np.sqrt((x_loc - i)**2 +(j)**2 + (k)**2 + (l)**2)
        #dist_y_loc = np.sqrt((y_loc - i)**2 +(j)**2 + (k)**2 + (l)**2)
        #print(dist_x_loc[0][0][0][0])
        #for i in range(10):
        #   print(dist_x_loc[i][0][0][0],-np.log(dist_x_loc[i][0][0][0])-np.log(dist_y_loc[i][0][0][0]))
        
        
        dist_x_loc = np.sqrt(np.minimum(abs(x_loc_i - i) % self.N,abs(self.N-abs(x_loc_i -i)%self.N))**2 + 
                            np.minimum(abs(x_loc_j-j) % self.N,abs(self.N-abs(x_loc_j-j)) % self.N)**2 + 
                            np.minimum(abs(x_loc_k-k) % self.N,abs(self.N-abs(x_loc_k-k)) % self.N)**2 + 
                            np.minimum(abs(x_loc_l-l) % self.N,abs(self.N-abs(x_loc_l-l)) % self.N)**2)

        

        
        dist_y_loc = np.sqrt(np.minimum(abs(y_loc_i- i) % self.N,abs(self.N-abs(y_loc_i -i)%self.N))**2 + 
                            np.minimum(abs(y_loc_j-j) % self.N,abs(self.N-abs(y_loc_j-j)) % self.N)**2 + 
                            np.minimum(abs(y_loc_k-k) % self.N,abs(self.N-abs(y_loc_k-k)) % self.N)**2 + 
                            np.minimum(abs(y_loc_l-l) % self.N,abs(self.N-abs(y_loc_l-l)) % self.N)**2)

        
        dist_z_loc = np.sqrt(np.minimum(abs(z_loc_i- i) % self.N,abs(self.N-abs(z_loc_i -i)%self.N))**2 + 
                            np.minimum(abs(z_loc_j-j) % self.N,abs(self.N-abs(z_loc_j-j)) % self.N)**2 + 
                            np.minimum(abs(z_loc_k-k) % self.N,abs(self.N-abs(z_loc_k-k)) % self.N)**2 + 
                            np.minimum(abs(z_loc_l-l) % self.N,abs(self.N-abs(z_loc_l-l)) % self.N)**2)
        eps = 1
        lat = -np.log(dist_x_loc)-np.log(dist_y_loc)+np.log(np.sqrt(dist_z_loc**2+Ls**2))
        return lat
    
    def define_potential(self,x,y,a = 1,b=1,L = 1):
        x_loc_i = (x[0])
        x_loc_j = (x[1])
        x_loc_k = (x[2])
        x_loc_l = (x[3])

        y_loc_i = (y[0])
        y_loc_j = (y[1])
        y_loc_k = (y[2])
        y_loc_l = (y[3])

        
        lat = np.zeros((self.N, self.N, self.N, self.N))

        i, j, k, l = np.indices((self.N, self.N, self.N, self.N))

        # Calculate distances with periodic boundary conditions
        #dist_x_loc = np.sqrt((x_loc - i)**2 +(j)**2 + (k)**2 + (l)**2)
        #dist_y_loc = np.sqrt((y_loc - i)**2 +(j)**2 + (k)**2 + (l)**2)
        #print(dist_x_loc[0][0][0][0])
        #for i in range(10):
        #   print(dist_x_loc[i][0][0][0],-np.log(dist_x_loc[i][0][0][0])-np.log(dist_y_loc[i][0][0][0]))
        
        
        dist_x_loc = np.sqrt(np.minimum(abs(x_loc_i - i) % self.N,abs(self.N-abs(x_loc_i -i)%self.N))**2 + 
                            np.minimum(abs(x_loc_j-j) % self.N,abs(self.N-abs(x_loc_j-j)) % self.N)**2 + 
                            np.minimum(abs(x_loc_k-k) % self.N,abs(self.N-abs(x_loc_k-k)) % self.N)**2 + 
                            np.minimum(abs(x_loc_l-l) % self.N,abs(self.N-abs(x_loc_l-l)) % self.N)**2)

        

        
        dist_y_loc = np.sqrt(np.minimum(abs(y_loc_i- i) % self.N,abs(self.N-abs(y_loc_i -i)%self.N))**2 + 
                            np.minimum(abs(y_loc_j-j) % self.N,abs(self.N-abs(y_loc_j-j)) % self.N)**2 + 
                            np.minimum(abs(y_loc_k-k) % self.N,abs(self.N-abs(y_loc_k-k)) % self.N)**2 + 
                            np.minimum(abs(y_loc_l-l) % self.N,abs(self.N-abs(y_loc_l)) % self.N)**2)

        
        
        eps = 1
        lat = -a*np.log(np.sqrt(eps**2+dist_x_loc**2)/np.sqrt(L**2+dist_x_loc)) -b*np.log(np.sqrt(dist_y_loc**2+eps**2)/np.sqrt(L**2+dist_y_loc))
        return lat
    
    def dphimin_2(self,lattice):
        I = self.I2(lattice)
        result = 0
        for i in range(self.dim):
            Foward = np.roll(I,-1,axis = i)*(1 + lattice - np.roll(lattice, -1, i))
            Backward = np.roll(I,1,axis = i)*(1 + lattice - np.roll(lattice, 1, i))
            Centre = I*(2*lattice - 2 - np.roll(lattice, -1, i)-np.roll(lattice, 1, i))
            result += Foward + Backward + Centre
        return -1/np.abs(self.lambda_)*result


    def dphimin(self,lattice):
        I = self.I(lattice)
        result = 0
        for i in range(self.dim):
            Foward = np.roll(I,-1,axis = i)
            Backward = np.roll(I,1,axis = i)*(1 + 2*lattice - 2*np.roll(lattice, 1, i))
            Centre = I*(2*lattice - 2 - 2*np.roll(lattice, -1, i))
            result += Foward + Backward + Centre
        return -1/np.abs(self.lambda_)*result
    def update_dissipative_phi(self,lattice,N_steps,epsilon):
        latt = lattice.copy()
        for i in range(N_steps):
            latt += epsilon*self.dphimin(latt)
            #print(np.max([self.dphimin(latt)[i][0][0][0] for i in range(self.N)]))
        return latt
    def update_dissipative_phi2(self,lattice,N_steps,epsilon):
        latt = lattice.copy()
        for i in range(N_steps):
            latt += epsilon*self.dphimin_2(latt)
            #print(np.max([self.dphimin(latt)[i][0][0][0] for i in range(self.N)]))
        return latt
    
    def update_dissipative_phi2_force(self,lattice,N_steps,epsilon,C,x,y,z,t):
        latt = lattice.copy()
        latti = np.zeros((self.N,self.N,self.N,self.N))
        latti[x][y][z][t] = 1
        for i in range(N_steps):
            latt += epsilon*self.dphimin_2(latt) + epsilon*C*latti
            #print(np.max([self.dphimin(latt)[i][0][0][0] for i in range(self.N)]))
        return latt
    def update_dissipative_phi2_force_two(self,lattice,N_steps,epsilon,C_1,C_2,x,y,z,t,xp,yp,zp,tp):
        latt = lattice.copy()
        latti = np.zeros((self.N,self.N,self.N,self.N))
        latti[x][y][z][t] = C_1
        latti[xp][yp][zp][tp] = C_2
        for i in range(N_steps):
            latt += epsilon*self.dphimin_2(latt) + epsilon*latti
            #print(np.max([self.dphimin(latt)[i][0][0][0] for i in range(self.N)]))
        return latt
    def run_dissipative2_force(self,N_steps,epsilon,total_steps,c,x,y,z,t):
        #self.lattice = self.define_potential(0,4)
        for i in range(total_steps):
            self.lattice = self.update_dissipative_phi2_force(self.lattice,N_steps,epsilon,c,x,y,z,t)
            #print(self.return_slice_lattice_x2(self.dphimin_2(self.lattice)))
               
        return self.action2()
    def run_dissipative2_force_two(self,N_steps,epsilon,total_steps,c,c_2,x,y,z,t,xp,yp,zp,tp):
        #self.lattice = self.define_potential(0,4)
        for i in range(total_steps):
            self.lattice = self.update_dissipative_phi2_force_two(self.lattice,N_steps,epsilon,c,c_2,x,y,z,t,xp,yp,zp,tp)
            #print(self.return_slice_lattice_x2(self.dphimin_2(self.lattice)))
               
        return self.action2()
    
    def run_dissipative(self,N_steps,epsilon,total_steps):
        #self.lattice = self.define_potential(0,4)
        with alive_bar(total_steps) as bar:
            for i in range(total_steps):
                self.lattice = self.update_dissipative_phi(self.lattice,N_steps,epsilon)
                bar()
                #print(self.action())
        return self.action()
    def run_dissipative2(self,N_steps,epsilon,total_steps):
        #self.lattice = self.define_potential(0,4)
        for i in range(total_steps):
            self.lattice = self.update_dissipative_phi2(self.lattice,N_steps,epsilon)
            #print(self.return_slice_lattice_x2(self.dphimin_2(self.lattice)))
               
        return self.action2()
        
            
    