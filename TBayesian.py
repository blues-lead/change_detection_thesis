#!/usr/bin/python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from   matplotlib.colors import LogNorm

class TBayesian:
    def __init__(self,mu,kappa,alpha,beta, hazard):
        # mu - after change expected mu
        # alpha - how many points
        # beta - at what rate points should come
        # hazard - 1/beta
        # kappa - will be calculated, initial value might be one
        self.mus = np.array([mu])
        self.kappas = np.array([kappa])
        self.alphas = np.array([alpha])
        self.betas = np.array([beta])
        self.lengths = np.array([1])
        self.hazard = hazard
        self.R = []
        self.R.append(np.array([1]))
        self.maxes = []
        # TEST
        self.counter = 0

    def _ret_ppdf(self, x):
        df = 2 * self.alphas
        loc = self.mus
        scale = np.sqrt(self.betas*(self.kappas + 1) / (self.alphas * self.kappas))
        return stats.t.pdf(x=x, df = df, loc = self.mus, scale = scale)

    def process_point(self, i, x):
        # TEST
        self.counter += 1
        # TEST
        pis = self._ret_ppdf(x)
        growth_probs = pis * (1-self.hazard) * self.lengths # RIGHT
        #growth_probs = pis * (1 - self.kappas) * self.lengths # TEST
        cp_probs = np.sum(pis * self.hazard * self.lengths) # RIGHT
        #cp_probs = np.sum(pis * self.kappas * self.lengths) # TEST
        #
        joint = np.append(cp_probs, growth_probs)
        # if np.sum(joint) > 1e10**8:
        #     print("Helo")
        self.R.append(joint/np.sum(joint))
        self.lengths = np.append(cp_probs, growth_probs)
        self._update_statistics(x)
        self.maxes.append(np.argmax(joint))
        # TEST
        cpoint = np.sum([True for i in range(len(self.maxes)-1) if (self.maxes[i+1] - self.maxes[i] < -3)])
        if cpoint > 0:
            return True, self.counter + 1
        else:
            return False, 0
        # TEST

    def _update_statistics(self, x):
        new_mu = (self.kappas*self.mus + x)/(self.kappas + 1)
        new_kappa = self.kappas + 1
        new_alpha = self.alphas + (1/2)
        new_beta = self.betas + (self.kappas * (x - self.mus)**2) / (2. * (self.kappas + 1.))
        self.mus = np.concatenate(([self.mus[0]],new_mu))
        self.kappas = np.concatenate(([self.kappas[0]],new_kappa))
        self.alphas = np.concatenate(([self.alphas[0]],new_alpha))
        self.betas = np.concatenate(([self.betas[0]],new_beta))

    def get_results(self):
        change_point = 0
        retR = np.zeros((len(self.R),len(self.R)))
        for i in range(len(self.R)):
            l = len(self.R[i])
            retR[i,:l] = self.R[i]
        for i in range(len(self.maxes)-1): # look for falling in the max values. Possibly look for argmax=1, starting from maxes[2:]
            if self.maxes[i+1] - self.maxes[i] < -5:
                change_point = i+1
                break
        return retR, self.maxes, change_point




if __name__ == "__main__":
    directory = "/media/synology/Documents/TUT/MSThesis/Scripts/Data/Baseline/"
    file1 = 'Lemon peelFlaskBaseline1.mat.csv'
    data = pd.read_csv(directory + file1)
    ims_abs = data['IMS_abs14']
    ts = [ims_abs[i+1] - ims_abs[i] for i in range(len(ims_abs)-1)]
    bs = TBayesian(0,1,1,1/50,1/50)
    for i in range(1,len(ts)):
        x = ts[i-1]
        bs.process_point(i,x)
    res, maxes, cp = bs.get_results()
    print(maxes)
    fig, ax = plt.subplots(1,2)
    norm = LogNorm(vmin=0.0001, vmax=1)
    plt.rcParams['figure.figsize'] = [10, 30]
    ax[0].plot(ims_abs)
    ax[0].axvline(cp, color='red')
    ax[1].imshow(np.rot90(res), cmap='gray_r', norm=norm)
    plt.show()


    

    