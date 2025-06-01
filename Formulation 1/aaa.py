import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from lattice import Lattice
N = 10
model = Lattice(N,1,0,0,0)
model2 = Lattice(N,1,0,0,0)
x = (2.5,0,0,0)
y = (6.5,0,0,0)
#model.lattice = model.define_potential(x,y,1,1)
plt.plot(model.return_slice_lattice_x())
plt.show()
x = (0,0,0,0)
class myobject():
    def __init__(self):
        self.actions = []
        self.updated = 0
myobject_m = myobject()

#model2.lattice = model.define_potential(x,x,1,0,5)

#plt.plot(model.return_slice_lattice_x())
#plt.show()
x = np.arange(0, N, 1)

fig ,(ax,ax1) = plt.subplots(1,2)

data_skip = 50


def init_func():
    ax.clear()
    
    
    ax.set_xlim((x[0], N))
    #plt.ylim((-1, 1))


def update_plot(i):
    #ax.plot(x[i:i+data_skip], y[i:i+data_skip], color='k')
    #ax.plot(x[i], y[i], color='r')


    #ax.set_ylim(min(model.return_slice_lattice_x().copy()), max(model.return_slice_lattice_x().copy()))
    if myobject_m.updated <= 500:
        myobject_m.actions += [model.run_dissipative2_force_two(10,0.005,1,20,20,5,0,0,0,7,0,0,0)]
    elif myobject_m.updated <= 1000:
        myobject_m.actions += [model.run_dissipative2(10,0.005,1)]
    else:
        myobject_m.actions += [model.run_dissipative2(10,0.001,1)]


    ax1.clear()
    ax1.plot(myobject_m.actions)
    ax.plot(model.return_slice_lattice_x().copy(), color='g',label ='two vortices')
    myobject_m.updated += 1




anim = FuncAnimation(fig,
                     update_plot,
                     frames=np.arange(0, len(x), data_skip),
                     init_func=init_func,
                     interval=20)
plt.show()
#anim.save('sine.mp4', dpi=150, fps = 30, writer='ffmpeg')