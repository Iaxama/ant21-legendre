import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import scipy.linalg
from scipy.special import legendre

class LDN():
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
        
    def make_step(self, dt):
        state = np.zeros((self.q, self.size_in))

        # Handle the fact that we're discretizing the time step
        #  https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
        Ad = scipy.linalg.expm(self.A*dt)
        Bd = np.dot(np.dot(np.linalg.inv(self.A), (Ad-np.eye(self.q))), self.B)

        # this code will be called every timestep
        def step_legendre(x, state=state):
            state[:] = np.dot(Ad, state) + np.dot(Bd, x[None, :])
            return state.T.flatten()
        return step_legendre

    def get_weights_for_delays(self, r):
        # compute the weights needed to extract the value at time r
        # from the network (r=0 is right now, r=1 is theta seconds ago)
        r = np.asarray(r)
        m = np.asarray([legendre(i)(2*r - 1) for i in range(self.q)])
        return m.reshape(self.q, -1).T

print('Loading')
stim = np.loadtxt('/home//miacono/data/emg_data/CS/01_03_2023/training_all.txt')[:,0].reshape(-1,1)
print('done')
dt = 0.0005   
q = 20
theta = 0.5

# stim = np.array([np.sin(x) + np.random.rand() for x in np.arange(0, 10*np.pi, dt)]).reshape(-1,1)
ldn = LDN(theta, q)

step = ldn.make_step(dt)
print('compressing')
compressed_stim = [step(x) for x in stim]
print('done')
plt.plot(stim[:10000])
plt.figure()
plt.plot(compressed_stim)
plt.figure()
plt.title('reconstructed')
plt.plot([ldn.get_weights_for_delays(0).dot(x) for x in tqdm(compressed_stim[:10000])])
# plt.figure()
# plt.title('reconstructed delayed')
# plt.plot([ldn.get_weights_for_delays(50).dot(x) for x in tqdm(compressed_stim[:10000])])
plt.show()
# w = np.linspace(0, 1, int(2*np.pi//dt))
# decoder = ldn.get_weights_for_delays(w)
# plt.scatter(w*theta, decoder.dot(compressed_stim[-1]))