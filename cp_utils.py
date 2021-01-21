#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 12:41:12 2020

@author: kondrate
"""
import matplotlib.pyplot as plt
import numpy as np


def draw_plots(readings, point, fig_size = None, caption = 'Plot', decision = np.empty((3,1))):
    # Readings are pandas frame
    r = 0
    if len(decision) == 3:
        fig, ax = plt.subplots(14,1, figsize=fig_size)
        fig.tight_layout(h_pad=2, rect=[0, 0.03, 1, 0.95])
        fig.suptitle(caption)
        for key in readings:
            ax[r].plot(readings[key])
            ax[r].grid(True)
            ax[r].set_title(key)
            ax[r].axvline(point, color='red', ls='dashed')
            r+=1
        plt.draw()
        for i in range(14):
            locs = list(np.arange(-50,350,50))
            locs += [point]
            labels = [str(w) for w in locs]
            ax[i].set_xticks(locs[1:])
            ax[i].set_xticklabels(labels[1:])
    else:
        fig, ax = plt.subplots(14,2, figsize=fig_size)
        fig.tight_layout(h_pad=2, rect=[0, 0.03, 1, 0.95])
        fig.suptitle(Caption)
        for key in readings:
            ax[r,0].plot(readings[key])
            ax[r,0].grid(True)
            ax[r,0].set_title(key)
            ax[r,0].axvline(point, color='red', ls='dashed')
            #
            ax[r,1].plot(decision[r,:])
            ax[r,1].grid(True)
            ax[r,1].set_title(key)
            ax[r,1].axvline(point, color='red', ls='dashed')
            r+=1
        plt.draw()
        for i in range(14):
            locs = list(np.arange(-50,350,50))
            locs += [point]
            labels = [str(w) for w in locs]
            ax[i,0].set_xticks(locs[1:])
            ax[i,0].set_xticklabels(labels[1:])
            ax[i,1].set_xticks(locs[1:])
            ax[i,1].set_xticklabels(labels[1:])