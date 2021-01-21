#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:47:44 2021
Class requires numpy for functioning
@author: kondrate
"""


class MAXCUSUM:
    def __init__(self, isample):
        # data chunk must be of size 14xW
        # W will be interpreted as moving window size
        self._counter = 1
        self._Si = 0
        self._mu = np.mean(isample, axis=1).reshape(14,1)
        self._cov = np.cov(isample).reshape(14,14)
        # Add small value to the main diagonal to avoid singularity
        np.fill_diagonal(self._cov, self._cov.diagonal() + 1e-10)
        self._w = isample.shape[1]
        # debug
        self._decs = [self._Si]
        
        # a
        up = np.dot((-1*self._mu).T, np.linalg.inv(self._cov))
        down = np.sqrt(np.dot(np.dot((-1*self._mu).T, np.linalg.inv(self._cov)), -1*self._mu))
        self._a = up/down # (shape 1,14)
        
    def iterate(self,data_chunk):
        # data chunk must be of size 14xW
        self._counter += 1
        nmu = np.mean(data_chunk, axis=1).reshape(14,1)
        D = np.sqrt(np.dot(np.dot((-1*self._mu).T, np.linalg.inv(self._cov)), -1*self._mu))
        Z = np.dot(self._a, nmu - self._mu)
        self._Si = np.max([self._Si + Z - 0.5*D, 0])
        # debug
        self._decs.append(self._Si)
        #
        if self._Si > 0:
            return self._counter + self._w, self._decs
        else:
            return 0,0
        
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import pickle
    import os
    import matplotlib.pyplot as plt
    folder='/Users/kondrate/nfs/Documents/TUT/MSThesis/Scripts/Data/Baseline/'
    file = 'JasmineTableBaseline1.mat.csv'
    table = pd.read_csv(folder + file)
    needed_columns = 'IMS_abs1 IMS_abs2 IMS_abs3 IMS_abs4 IMS_abs5 IMS_abs6 IMS_abs7 IMS_abs9 IMS_abs10 IMS_abs11 IMS_abs12 IMS_abs13 IMS_abs14 IMS_abs15'.split(" ")
    table = table[needed_columns]
    tsims = table.to_numpy()
    ims = tsims
    tsims = np.array([tsims[i + 1,:] - tsims[i,:] for i in range(tsims.shape[0]-1)]).T
    gt = pickle.load(open(os.path.join(folder, 'ground truth', file + '.pickle'), 'rb'))
    vpoints = [gt[key]['val'] for key in gt if key != 'state']
    #plt.plot(tsims[0,:])
    mc = MAXCUSUM(tsims[:,:10])
    for i in range(tsims.shape[1]):
        pt, decs = mc.iterate(tsims[:,i:i+10])
        if pt > 0:
            break
    print('Change point found at:',pt)