import numpy as np
from alive_progress import alive_bar

class Lattice(object):
    """
    Class to represent a lattice and perform Monte Carlo simulations on it.
    This can be used in the following way:
    1. Create an instance of the class with the desired parameters. This will laso thermalize the lattice.
    2. Generate measurements of an observable using the generate_measurements method.
    3. Analyse the results of the measurements, using the Stats class (and the processing.py file)
    """


    def __init__(self, N, lambda_, N_measurements, N_thermalization, width, HMC = False, epsilon = 0, N_steps = 0, dim = 4):
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

        if HMC: # if HMC is True, initialize the HMC parameters
            self.epsilon = epsilon
            self.N_steps = int(N_steps)
        
        self.accepted = 0 # number of accepted configurations accepted
        
        # Hot start of the lattice
        self.randomize()

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
                
                    self.HMC(self.N_steps, self.epsilon)
                
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

                    self.dhs[i] = np.exp(-self.HMC(self.N_steps, self.epsilon)) # store the exponential of the change in Hamiltonian (testing purposes), it call HMC
                else:

                    self.sweep() # perform a sweep if metropolis is used
                
                results[i] = observable(self.lattice) # store the result of the observable
                
                bar() # this line updates the progress bar, not important
        print("Measurements Complete----------------", np.average(self.dhs)) # print the average of the exponential of the change in Hamiltonian, should be close to 1
        
        return results
    
    
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
                    self.HMC(self.N_steps, self.epsilon)
                else:
                    self.sweep()
                bar()
        self.accepted = 0
        with alive_bar(calibration_runs) as bar:

            for i in range(calibration_runs):
                if self.HMCs:
                    dHs[i] = np.exp(-self.HMC(self.N_steps, self.epsilon))
                else:
                    self.sweep()
                bar()
        if self.HMCs:
            return self.accepted/calibration_runs
        else:
            return self.accepted/(calibration_runs*self.N**4)

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
        axis = tuple([i for i in range(2, dim+2)])
        final = np.sum(result, axis = axis) / (dim * N**dim)
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
    
    def HMC(self, N_steps, epsilon,flag = False):
        """
        Performs a Hamiltonian Monte Carlo update.

        Parameters:
        N_steps: int, number of steps in the molecular dynamics evolution
        epsilon: float, step size for the molecular dynamics evolution
        flag: bool, test purposes used to always accept the new lattice if needed

        Returns:
        float, the change in Hamiltonian
        """
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
            result += forward + backward - 2*lattice + (forward-lattice)**2
        return result 
     
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
           
   
    
    
    