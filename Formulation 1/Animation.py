from lattice import Lattice
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
from matplotlib import colors
from matplotlib import patches
np.set_printoptions(threshold=sys.maxsize)

class Animation(object):
    def __init__(self, dimension: int, iterations: int,loc_x_vortex_one,loc_x_vortex_two) -> None:
        """
        Consturctor of the Animation class
        """
        self.model = Lattice(dimension,-1, 0,0,0)
        self.model.lattice = self.model.define_potential(loc_x_vortex_one,loc_x_vortex_two)
        self.iterations = iterations
        self.history = np.array([[]])
        self.history = [self.model.return_slice_lattice_x()]
        
    def evolve(self) -> None:
        """
        Method used to evolve the model one step
        """
        for i in range(self.iterations):
            print(i)
            self.model.run_dissipative2(10,0.001,1)
            self.history += [self.model.return_slice_lattice_x()]

    def animate(self) -> None:
        """
        Method used to create an animation of the system, it will first make all the updates
        and then it will create the animation
        """
        self.evolve()
        fig = plt.figure()
        images = []
        count = 0
        for array2D in self.history:
            count+=1
            print(count)
            plt.title("Ising Model: ", fontsize=16)
            renderImage = plt.plot(array2D)
            images.append([renderImage])
        ani = animation.ArtistAnimation(fig, images, interval=0.01, blit=True, repeat=False)
        plt.show()

    def display(self) -> None:
        """
        Method used to update the system a set number of iterations and to display the final 
        result
        """
        for i in range(self.iterations):
            print(i)
            self.model.run_dissipative2(10,0.001,1)
        plt.axis('off')
        plt.title("Ising Model: ", fontsize=16)
        plt.plot(self.model.return_slice_lattice_x())
        plt.show()
        #plt.savefig('../results/Ising Model01.png')

    def updatefig(self,i) -> None:
        """
        Method used to update the system for animation
        """
        print(i)
        self.model.run_dissipative2(10,0.001,1)
        self.ax.plot(self.model.return_slice_lattice_x().copy())
        #return [self.im]

    def init(self) -> None:
        """
        Method used to init the animation
        """
        self.ax.clear()
        self.ax.plot(self.model.return_slice_lattice_x().copy())
        #return [self.im]

    def animation(self) -> None:
        """
        Method used to create an animation of the model
        """
        fig = plt.figure()
        self.ax = plt.subplot(111)
        self.ax.plot(self.model.return_slice_lattice_x().copy())
        ani = animation.FuncAnimation(fig, self.updatefig, interval = 0,blit = True)
        plt.show()

    
        
def main():
    dimension = 8
    iterations = 100
    loc_x_vortex_one = 0
    loc_x_vortex_two = 0
    model = Animation(dimension,iterations,loc_x_vortex_one,loc_x_vortex_two)
    model.animation()
main()