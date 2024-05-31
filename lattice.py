import numpy as np
class Lattice(object):
    def __init__(self, N, lambda_, N_measurements, N_thermalization):
        
        self.lambda_ = lambda_
        self.N = N
        self.lattice = np.zeros((N,N,N,N))
        self.N_thermalization = N_thermalization
        self.N_measurements = N_measurements
        self.randomize()
        print(self.action())
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
    
    def metropolis(self):
        x = np.random.randint(0,self.N)
        y = np.random.randint(0,self.N)
        z = np.random.randint(0,self.N)
        t = np.random.randint(0,self.N)
        old = self.lattice[x,y,z,t]
        old_action = self.action()
        self.lattice[x,y,z,t] += np.random.normal()
        delta = self.action()-old_action
        #print(delta)
        if np.random.rand() > np.exp(-delta):
            self.lattice[x,y,z,t] = old
      
   
    def sweep(self):
        for i in range(self.N**4):
            self.metropolis()
    def thermalize(self):
        for i in range(self.N_thermalization):
            print(i)
            self.sweep()
    def measure(self,observable):
        results = np.zeros(self.N_measurements)
        for i in range(self.N_measurements):
            print(i)
            self.sweep()
            results[i] = observable(self.lattice)
        return results
    
    
    def randomize(self):
        self.lattice = np.random.normal(size=(self.N,self.N,self.N,self.N))

def main():
    N = 4
    lambda_ = -1
    N_measurements = 1000
    N_thermalization = 1000
    lattice = Lattice(N,lambda_,N_measurements,N_thermalization)
    print(np.average(lattice.measure(lambda x: np.sum(x))))
main()
        