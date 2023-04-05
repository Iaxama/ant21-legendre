import matplotlib.pyplot as plt
import nengo
import numpy as np
import scipy.linalg
from scipy.special import legendre

class LDN(nengo.Process):
    def __init__(self, theta, q, size_in=1):
        self.q = q              # number of internal state dimensions (polynomials) per input
        self.theta = theta      # size of time window (in seconds)
        self.size_in = size_in  # number of inputs (channels)

        # Do Aaron's math to generate the matrices A and B so that
        #  dx/dt = Ax + Bu will convert u into a legendre representation over a window theta
        #  https://github.com/arvoelke/nengolib/blob/master/nengolib/synapses/analog.py#L536
        A = np.zeros((q, q))
        B = np.zeros((q, 1))
        for i in range(q):
            B[i] = (-1.)**i * (2*i+1)
            for j in range(q):
                A[i,j] = (2*i+1)*(-1 if i<j else (-1.)**(i-j+1))  # DON'T CHANGE THIS! TRUST IT
        self.A = A / theta
        self.B = B / theta        
        
        super().__init__(default_size_in=size_in, default_size_out=q*size_in)

    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        state = np.zeros((self.q, self.size_in))

        # Handle the fact that we're discretizing the time step
        #  https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
        Ad = scipy.linalg.expm(self.A*dt)
        Bd = np.dot(np.dot(np.linalg.inv(self.A), (Ad-np.eye(self.q))), self.B)

        # this code will be called every timestep
        def step_legendre(t, x, state=state):
            state[:] = np.dot(Ad, state) + np.dot(Bd, x[None, :])
            return state.T.flatten()
        return step_legendre

    def get_weights_for_delays(self, r):
        # compute the weights needed to extract the value at time r
        # from the network (r=0 is right now, r=1 is theta seconds ago)
        r = np.asarray(r)
        m = np.asarray([legendre(i)(2*r - 1) for i in range(self.q)])
        return m.reshape(self.q, -1).T
    
theta = 0.5
q = 20
dt = 0.001
t = np.arange(10000)*dt
stim1 = np.sin(t*2*np.pi).reshape(-1,1)
stim2 = np.sin(t*2*np.pi*2).reshape(-1,1)

plt.figure(figsize=(12,4))
plt.plot(t, stim1)
plt.plot(t, stim2)

x1 = LDN(theta=theta, q=q).apply(stim1) # Encode the data in compressed form
x2 = LDN(theta=theta, q=q).apply(stim2)


plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.plot(x1)
plt.title('1Hz wave')
plt.subplot(1, 2, 2)
plt.plot(x2)
plt.title('2Hz wave')
plt.show()