import numpy as np
from alive_progress import alive_bar

class Lattice(object):
    def __init__(self, N, lambda_, N_measurements, N_thermalization, width, HMC= False, epsilon=0, N_steps=0,dim =4):
        self.width = width
        
        self.lambda_ = lambda_
        self.N = N
        self.N_thermalization = N_thermalization
        self.N_measurements = N_measurements
        self.dim = dim
        self.shape = [N for i in range(dim)]
        self.lattice = np.zeros(shape=self.shape)
        self.dhs = np.zeros(N_measurements)


        self.HMCs = HMC
        if HMC:
            self.epsilon = epsilon
            self.N_steps = N_steps
        self.accepted = 0
        self.randomize()
        #print(self.action())
        #for i in range(100):
            #self.HMC(10,0.0001,True)
        #self.lattice = np.ones(shape=self.shape)
        #self.lattice = np.array([[1,2],[3,4]])
        if self.N_thermalization > 0:
            self.thermalize()
        #print(self.lattice)

   
    def action(self):
        return 1/(2*np.abs(self.lambda_))*np.sum(self.I(self.lattice)**2 )
    
    def metropolis(self,i,j,k,l):
        x = i
        y = j
        z = k
        t = l
        old = self.lattice[x,y,z,t]
        old_action = self.action()
        self.lattice[x,y,z,t] += np.random.normal(0,self.width)
        delta = self.action()-old_action
        #print(delta)
        if np.random.rand() > np.exp(-delta):
            self.lattice[x,y,z,t] = old
        else:
            self.accepted += 1
      
   
    def sweep(self):
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.N):
                    for l in range(self.N):

                        self.metropolis(i,j,k,l)
    def thermalize(self):
        print("Thermalizing")
        with alive_bar(self.N_thermalization) as bar:
            for i in range(self.N_thermalization):
                if self.HMCs:
                    self.HMC(self.N_steps, self.epsilon)
                else:
                    self.sweep()
                bar()
        print("Thermalization Complete----------------")
      
    def generate_measurements(self,observable):
        results = [0 for i in range(self.N_measurements)]
        print("Generating Measurements")
        with alive_bar(self.N_measurements) as bar:
            for i in range(self.N_measurements):
                if self.HMCs:
                    self.dhs[i] = np.exp(-self.HMC(self.N_steps, self.epsilon))
                else:
                    self.sweep()
                results[i] = observable(self.lattice)
                bar()
        print("Measurements Complete----------------",np.average(self.dhs))
        
        return results
    
    
    def randomize(self):
        self.lattice = np.random.normal(size=(self.N,self.N,self.N,self.N))
    def calibration_runs(self,calibration_runs, thermal_runs):
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

    def measure_difference(lattice):
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
        result = np.zeros((int(N/2), 4, N, N, N, N))

        for i in range(int(N/2)):
            for j in range(4):
                shift = np.roll(lattice, i+1, axis=j)
                result[i, j] =( lattice - shift)**2
        #print(result)
        #print(result)

        final = np.sum(result, axis=(1,2, 3, 4, 5)) / (4 * N**4)
        return final
    def measure_field(lattice):
        return np.sum(lattice)/(len(lattice)**4)
    
    def backwards(self,lattice,axis):
        result = (1+2*lattice-2*np.roll(lattice,1,axis))
        return result
    
    def forwards(self,lattice,axis):
        result = 1
        return result
    def centre(self,lattice,axis):
        return 2*lattice-2-2*np.roll(lattice,-1,axis)
    def molecular_dynamics(self, N_steps, epsilon,p_0,phi_0):
        
        p = p_0 + epsilon/2*self.dot_p(phi_0)
        max_p = np.max(p)
        phi = phi_0.copy()
        for i in range(N_steps):
            phi = phi + epsilon*p
            if i == N_steps-1:
                p = p + epsilon/2*self.dot_p(phi)
            else:
                p = p + epsilon*self.dot_p(phi)
            max_p = np.max(p)
        return p,phi        
    
    def HMC(self, N_steps, epsilon,flag = False):
        p = np.random.normal(size=self.shape)
        #p= np.zeros(self.shape)
        #p = np.ones(self.shape)
        H = self.action() + np.sum(p**2)/2
        #print(self.action(),np.sum(p**2)/2,H)
        p_new, lattice_new = self.molecular_dynamics(N_steps, epsilon, p.copy(), self.lattice.copy())
        H_new = self.actions(lattice_new) + np.sum(p_new**2)/2
        delta_H = H_new - H
        #print(self.actions(lattice_new), np.sum(p_new**2)/2)

        if delta_H <0 or np.exp(-delta_H) > np.random.random():
        
            self.lattice = lattice_new.copy()
            self.accepted += 1
        if flag:
            self.lattice = lattice_new.copy()


        return delta_H
    def I(self,lattice):
        result = 0
        for i in range(self.dim):
            forward = np.roll(lattice,-1,axis=i)
            backward = np.roll(lattice,1,axis=i)
            result += forward + backward - 2*lattice + (forward-lattice)**2
        return result 
     
    def actions(self,lattice):
        return 1/(2*np.abs(self.lambda_))*np.sum(self.I(lattice)**2)
    
    
    
    def dot_p(self,lattice):
        I = self.I(lattice)
        result = 0
        for i in range(self.dim):
            Fow = np.roll(I,-1,axis=i)
            Back = np.roll(I,1,axis=i)*(1+2*lattice-2*np.roll(lattice,1,i))
            cent = I*(2*lattice-2-2*np.roll(lattice,-1,i))
            result += Fow + Back + cent
        return -1/np.abs(self.lambda_)*result
           
   
    
    
    