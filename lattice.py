import numpy as np
from alive_progress import alive_bar

class Lattice(object):
    def __init__(self, N, lambda_, N_measurements, N_thermalization, width):
        self.width = width
        
        self.lambda_ = lambda_
        self.N = N
        self.lattice = np.zeros((N,N,N,N))
        self.N_thermalization = N_thermalization
        self.N_measurements = N_measurements
        self.accepted = 0
        self.randomize()
        #print(self.action())
        self.thermalize()

    def derivative(self,axis):
        foward = np.roll(self.lattice,-1,axis=axis)
        return (foward-self.lattice)
    
    def second_derivative(self,axis):
        foward = np.roll(self.lattice,-1,axis=axis)
        backward = np.roll(self.lattice,1,axis=axis)
        return (foward-2*self.lattice+backward)
    
    def laplacian(self):
        return self.second_derivative(0)+self.second_derivative(1)+self.second_derivative(2)+self.second_derivative(3)
    def action(self):

        return -1/(2*self.lambda_)*np.sum((self.laplacian()+self.derivative(0)**2+self.derivative(1)**2+self.derivative(2)**2+self.derivative(3)**2)**2)
    
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
                self.sweep()
                bar()
        print("Thermalization Complete----------------")
      
    def generate_measurements(self,observable):
        results = [0 for i in range(self.N_measurements)]
        print("Generating Measurements")
        with alive_bar(self.N_measurements) as bar:
            for i in range(self.N_measurements):
                self.sweep()
                results[i] = observable(self.lattice)
                bar()
        print("Measurements Complete----------------")
        
        return results
    
    
    def randomize(self):
        self.lattice = np.random.normal(size=(self.N,self.N,self.N,self.N))
    def calibration_runs(self,calibration_runs, thermal_runs):
        with alive_bar(thermal_runs) as bar:
            for i in range(thermal_runs):
                 self.sweep()
                 bar()
        self.accepted = 0
        with alive_bar(calibration_runs) as bar:

            for i in range(calibration_runs):
                self.sweep()
                bar()
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
        result = (1+np.roll(lattice,1,axis)+np.roll(np.roll(lattice,1,axis),1,axis))
        return result
    
    def forwards(self,lattice,axis):
        result = (1+np.roll(lattice,-1,axis)+np.roll(np.roll(lattice,-1,axis),-1,axis))
        return result
    def centre(self,lattice,axis):
        return np.roll(lattice,1,axis)+np.roll(lattice,-1,axis) -2*lattice
    def molecular_dynamics(self, N_steps, epsilon,p_0,phi_0):
        
        p = p_0 + epsilon/2*self.dot_p(phi_0)
        phi = phi_0 + epsilon*p
        
        for i in range(N_steps-1):
            p = p + epsilon*self.dot_p(phi)
            phi = phi + epsilon*p
        
        p = p + epsilon/2*self.dot_p(phi)
        return p,phi        
    
    def HMC(self, N_steps, epsilon):
        p = np.random.normal(size=(self.N,self.N,self.N,self.N))
        H = self.action() + np.sum(p**2)/2
        p_new, lattice_new = self.molecular_dynamics(N_steps, epsilon, p.copy(), self.lattice.copy())
        H_new = self.actions(lattice_new) + np.sum(p_new**2)/2
        delta_H = H_new - H

        if np.random.rand() < np.exp(-delta_H):
            self.lattice = lattice_new.copy()
            self.accepted += 1

        return self.lattice
    
     
    def actions(self,lattice):

        return -1/(2*self.lambda_)*np.sum((self.laplacians(lattice)+self.derivatives(0,lattice)**2+self.derivative(1,lattice)**2+self.derivative(2,lattice)**2+self.derivative(3,lattice)**2)**2)
    
    def derivatives(self,axis,lattice):
        foward = np.roll(lattice,-1,axis=axis)
        return (foward-self.lattice)
    
    def second_derivatives(self,axis,lattice):
        foward = np.roll(lattice,-1,axis=axis)
        backward = np.roll(lattice,1,axis=axis)
        return (foward-2*lattice+backward)
    
    def laplacians(self,lattice):
        return self.second_derivative(0,lattice)+self.second_derivative(1,lattice)+self.second_derivative(2,lattice)+self.second_derivative(3,lattice)
    
    def dot_p(self,lattice):
        I = self.I(lattice)
        result = 0
        for i in range(4):
            result += np.roll(I,-1,axis=i)*self.forwards(lattice,i) + np.roll(I,1,axis=i)*self.backwards(lattice,i) +I*self.centre(lattice,i)
        return result
           
   
    def I(self,lattice):
        result = 0
        for i in range(4):
            forward = np.roll(lattice,-1,axis=i)
            backward = np.roll(lattice,1,axis=i)
            result += forward + backward - 2*lattice - (forward-lattice)*(backward-lattice)
        return result 
    
    