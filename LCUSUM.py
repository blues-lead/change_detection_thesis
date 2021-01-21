#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 11:08:06 2020

@author: anton

Matrix Form CUSUM
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

#%%
class LCUSUM_static_window:
    '''
    The data must be of size 14x__moving_window and normalized.
    At each iteration method LLR returns bool value whether point found or not. If plotting is not
    needed then iterations must be stopped if LLR returned true-value. If plotting is needed and set True
    then iteration should not be stopped. This class uses predetermined moving window size.
    '''
    def __init__(self, idata, mw, plot = False):
        self.__plot = plot
        self.__moving_window = mw
        self.__data = np.zeros((14,self.__moving_window)).reshape(14,self.__moving_window)
        self.__data = idata
        self.__mu0 = np.zeros((self.__moving_window,1)).reshape(self.__moving_window,1)
        self.__std0 = np.zeros((self.__moving_window,1)).reshape(self.__moving_window,1)
        self.__mins = np.zeros((14,1)).reshape(14,1)
        self.__cll = np.zeros((14,1)).reshape(14,1)
        self.__point = 0
        # Create matrix for storing decision values if plotting is needed
        if plot == True:
            self.__decision = np.zeros((14,1)).reshape(14,1)
        #
        self.__detected = False
        self.__counter = 1
        self.__params()
    
    def __params(self):
        '''
        Function ran only once whne called from the constructor.
        Defines parameters mu0, std0, b, v and calculates the first time CLLR
        '''
        self.__mu0 = np.dot(self.__data, np.ones((self.__moving_window, 1))/self.__moving_window)
        M = np.tile(self.__mu0, (1, self.__moving_window))
        self.__std0 = np.sqrt(np.dot((self.__data - M)**2, np.ones((self.__moving_window, 1)))/self.__moving_window)
        self.__v = -1*self.__mu0
        self.__b = self.__v/self.__std0
        self.LLR(self.__data)
        
    def LLR(self, sample):
        self.__data = sample
        self.__counter += 1
        bsigma = self.__b/self.__std0
        sumfactor = np.dot(self.__data, np.ones((self.__moving_window, 1)))
        mnot = self.__moving_window * self.__mu0
        vo2 = self.__moving_window * (self.__v/2)
        self.__cll += bsigma*(sumfactor - mnot - vo2)
        self.__update_mins()
        # If plotting is needed
        if self.__plot == True:
            self.__decision = np.hstack((self.__decision, self.__cll - self.__mins))
        #
        if np.mean(self.__cll - self.__mins) > 0 and self.__detected == False:
            self.__point = self.__counter + self.__moving_window
            self.__detected = True
        return self.__detected
    
    def __update_mins(self):
        barr = self.__mins > self.__cll
        self.__mins[barr] = self.__cll[barr]
        
    def get_results(self):
        if self.__plot == False:
            return int(self.__point)
        else:
            return int(self.__point), self.__decision
#%%        
class LCUSUM_dynamic_window:
    '''
    The data must be of size 14x__moving_window and normalized.
    At each iteration method LLR returns bool value whether point found or not. If plotting is not
    needed then iterations must be stopped if LLR returned true-value. If plotting is needed and set True
    then iteration should not be stopped. This class uses dynamic moving window size, which can be changed
    at runtime.
    '''
    def __init__(self, idata, threshold_add = 0, plot = False):
        self.__threshold_add = threshold_add
        self.__plot = plot
        self.__moving_window = idata.shape[1]
        self.__data = idata
        self.__mu0 = np.zeros((14,1)).reshape(14,1)
        self.__std0 = np.zeros((14,1)).reshape(14,1)
        self.__mins = np.zeros((14,1)).reshape(14,1)
        self.__cll = np.zeros((14,1)).reshape(14,1)
        self.__point = 0
        self.__detected = False
        self.__counter = 1
        # Create matrix for storing decision values if plotting is needed
        if plot == True:
            self.__decision = np.zeros((14,1)).reshape(14,1)
        #
        self.__avg_mwsize = idata.shape[1]
        self.__params()
    
    def __params(self):
        '''
        Function used only once whne called from the constructor.
        Defines parameters mu0, std0, b, v and calculates the first time CLLR
        '''
        self.__mu0 = np.dot(self.__data, np.ones((self.__moving_window, 1))/self.__moving_window)
        M = np.tile(self.__mu0, (1, self.__moving_window))
        self.__std0 = np.sqrt(np.dot((self.__data - M)**2, np.ones((self.__moving_window, 1)))/self.__moving_window)
        # check if any of std values are very close to zero
        barr = self.__std0 < 1e-100
        self.__std0[barr] = 1e-100
        
        self.__v = -1*self.__mu0
        self.__b = self.__v/self.__std0
        self.LLR(self.__data)
        
    def LLR(self, sample):
        self.__data = sample
        self.__counter += 1
        bsigma = self.__b/self.__std0
        sumfactor = np.dot(self.__data, np.ones((self.__data.shape[1], 1)))
        mnot = self.__data.shape[1] * self.__mu0
        vo2 = self.__data.shape[1] * (self.__v/2)
        self.__cll += bsigma*(sumfactor - mnot - vo2)
        self.__update_mins()
        self.__avg_mwsize += self.__data.shape[1]/self.__counter
        # If plotting is needed
        if self.__plot == True:
            self.__decision = np.hstack((self.__decision, self.__cll - self.__mins))
        #
        if np.mean(self.__cll - self.__mins) > 0 and self.__detected == False:
            self.__point = self.__counter +  self.__threshold_add #self.__avg_mwsize
            self.__detected = True
        return self.__detected
    
    def __update_mins(self):
        barr = self.__mins > self.__cll
        self.__mins[barr] = self.__cll[barr]
        
    def get_results(self):
        if self.__plot == False:
            return int(self.__point)
        else:
            return int(self.__point), self.__decision
#%%
def main():
    folder = '/media/synology/Documents/TUT/MSThesis/Scripts/Data/Baseline/'
    file = 'JasmineFlaskBaseline1.mat.csv'
    data = pd.read_csv(folder + file).to_numpy().T
    ts = np.array([data[:,i+1] - data[:,i] for i in range(data.shape[1]-1)]).T
    lc = LCUSUM_dynamic_window(ts[:,:10])
    for i in range(ts.shape[1]):
        sample = ts[:,i:i+10]
        b = lc.LLR(sample)
        if b == True:
            break
    print(lc.get_results())

        
if __name__ == "__main__":
    main()
