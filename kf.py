import numpy as np 
import matplotlib.pyplot as plt 

class train:
    def __init__(self, initial_state, dt=0.01):
        # initial_state = [x_t, \dot{x}_t]
        self.x = initial_state

        # time discretization
        self.dt = dt

        # noise 
        self.pn_mean = np.array([0., 0.])
        self.Q_t = np.array([[0.5, 0.],
                                [0., 0.6]]) # pn_cov
        
        self.mn_mean = np.array([0., 0.])
        self.R_t = np.array([[0.1, 0.],
                                [0., 0.7]]) # mn_cov

        # process matrices
        self.F_t = np.array([[1., self.dt],
                        [0., 1.]])
        self.B_t = np.array([0.5*self.dt**2, self.dt])

        # measurement matrices
        self.H_t = np.array([[1., 0.],
                             [0., 0.]])

    def update_state(self, u): 
        # u is scalar
        self.x = np.matmul(self.F_t, self.x) +  self.B_t*u + np.random.multivariate_normal(self.pn_mean, self.Q_t)
        self.z = np.matmul(self.H_t, self.x) + np.random.multivariate_normal(self.mn_mean, self.R_t)


class kalman_filter:
    def __init__(self, train, xhat_0, P_0):
        self.tr = train
        self.xhat = xhat_0
        self.P  = P_0
        
    def prediction_update(self, u):
        self.xhat = np.matmul(self.tr.F_t, self.xhat) + self.tr.B_t*u
        self.P = np.matmul(self.tr.F_t, np.matmul(self.P, self.tr.F_t.T)) + self.tr.Q_t
    
    def measurement_update(self, z):
        temp1 = np.matmul(self.P, self.tr.H_t.T)
        temp2 = np.linalg.inv(np.matmul(self.tr.H_t, temp1) + self.tr.R_t)
        K = np.matmul(temp1, temp2) # kalman gain
        self.xhat = self.xhat + np.matmul(K, (z - np.matmul(self.tr.H_t, self.xhat)))
        self.P = self.P - np.matmul(K, np.matmul(self.tr.H_t, self.P))


class simulation:
    def __init__(self, train, xhat_0, P_0, simulation_time, dt=0.01):
        self.T = simulation_time
        self.dt = dt
        self.t = np.linspace(0, self.T, int(self.T / self.dt))
        self.tr = train
        self.kf = kalman_filter(self.tr, xhat_0, P_0)

    
    def simulate(self):
        self.x_t = []
        self.xhat_t = []
        for i in range(len(self.t)):
            self.xhat_t.append(self.kf.xhat)
            self.x_t.append(self.tr.x)
            u = 0.2
            self.kf.prediction_update(u) 
            self.tr.update_state(u)
            self.kf.measurement_update(self.tr.z)
    
    def plot(self):
        self.x_t = np.array(self.x_t)
        self.xhat_t = np.array(self.xhat_t)
        x_t = self.x_t[:, 0]
        xhat_t = self.xhat_t[:, 0]

        # uncomment to plot estimation error
        # e = x_t - xhat_t
        # plt.plot(self.t, e, color = 'red') 
        # plt.plot(self.t, np.zeros(len(self.t)), color = 'black', linestyle = 'dashed')
        # plt.ylabel('estimation error')


        plt.plot(self.t, x_t, color='red', label='true')
        plt.plot(self.t, xhat_t, color='black', label='estimate', linestyle = 'dotted')
        plt.xlabel('time (s)')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    train = train(np.array([0., 0.]))
    simulation_time = 5
    xhat_0 = np.array([5., 2.])
    P_0 = np.array([[0.1, 0.],
                    [0., 0.4]])
    sim = simulation(train, xhat_0, P_0, simulation_time)
    sim.simulate()
    sim.plot()



    