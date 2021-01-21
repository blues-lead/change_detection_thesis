import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as st

class Shewhart:
    def __init__(self, first_points):
        self.mu0 = np.mean(first_points)
        self.std0 = np.std(first_points)
        self.found = False
        self.counter = 1

    def process_points(self, sample):        
        new_mu = np.mean(sample)
        self.std0 = np.std(sample)
        if np.isclose(self.std0,0):
            self.std0=0.001
        threshold = np.abs(new_mu - self.mu0) + 3*(self.std0/np.sqrt(len(sample)))
        v = 0 - new_mu
        b = v/self.std0
        si = (b/self.std0)*(np.sum(sample) - self.mu0*len(sample) - len(sample)*(v/2))
        self.counter = self.counter + 1
        if si > threshold:
            return True, self.counter + int(len(sample)/2)
        else:
            return False, 0


if __name__ == "__main__":
    directory = "/media/synology/Documents/TUT/MSThesis/Scripts/Data/NoBaseline/"
    file1 = 'JasmineFlaskNoBaseline1.mat.csv'
    data = pd.read_csv(directory + file1)
    ims_abs = data['IMS_abs1']

    ts = [ims_abs[i+1] - ims_abs[i] for i in range(len(ims_abs)-1)]
    cs = Shewhart(ts[:10])
    for i in range(len(ts)):
        sample = ts[i:i+10]
        f,cp = cs.process_points(sample)
        if f==True:
            break
    print(cp)

        
