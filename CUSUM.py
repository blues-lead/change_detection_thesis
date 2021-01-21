import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as st

class CUSUM:
    def __init__(self, first_points):
        self.mu0 = np.mean(first_points)
        self.std0 = np.std(first_points)
        if np.isclose(self.std0,0):
            self.std0=0.001
        self.change_point = 0
        self.min_csi = 0
        self.csi = 0
        self.counter = 0
        self.found = False

    def process_points(self, data):
        self.__llr(data)
        if self.csi - self.min_csi > 0:
            self.change_point = self.counter
            return True, self.change_point
        self.counter += 1
        return False, 0


    def __llr(self, sample):
        v = 0 - self.mu0
        b = v/self.std0
        self.csi = self.csi + (b/self.std0)*(np.sum(sample) - self.mu0*len(sample) - len(sample)*(v/2))
        if self.min_csi > self.csi:
            self.min_csi = self.csi


if __name__ == "__main__":
    directory = "/media/synology/Documents/TUT/MSThesis/Scripts/Data/Baseline/"
    file1 = 'JasmineTableBaseline5.mat.csv'
    data = pd.read_csv(directory + file1)
    ims_abs = data['IMS_abs1']

    ts = [ims_abs[i+1] - ims_abs[i] for i in range(len(ims_abs)-1)]
    cs = CUSUM(ts[:3])
    for i in range(len(ts)):
        sample = ts[i:i+3]
        f,cp = cs.process_points(sample)
        if f==True:
            break
    print(cp)