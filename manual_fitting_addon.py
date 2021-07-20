#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: misiak

"""
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from scipy.optimize import minimize
from tqdm import tqdm

class Manual_fitting():
    
    # induce a bug with the text_boxes list not restarting at each call of 
    # a new instance of Manual_fitting
    # to be fix later i guess :/
    text_boxes = list()
    lines = list()
    func = lambda x: list()
    paramnow = list()
    paraminit = list()
    paramprevious = list()
    fig = None
    callback = lambda *x: None
    lines_previous = list()

    def __init__(self, lines, func, paraminit,
                 callback=None, chi2_fun=None, autoscale=True):
    
        self.lines = lines
        self.func = func
        self.fig = lines[0].get_figure()
        self.canvas = self.fig.canvas
        self.paraminit = list(paraminit)
        self.nparams = len(paraminit)
        if callable(callback):
            self.callback = callback
        self._autoscale_flag = autoscale
        
        # reserving some place for the widgets
        rightl = 0.7
        topl = 0.75
        self.fig.subplots_adjust(right=rightl)
        
        resetax = plt.axes([rightl+0.1, topl, 0.1, 0.045])
        self.reset_button = Button(resetax, 'Reset', hovercolor='0.975')
        self.reset_button.on_clicked(self._reset)   
        
        updatax = plt.axes([rightl+0.1, topl-0.05, 0.1, 0.045])
        self.update_button = Button(updatax, 'Update', hovercolor='0.975')
        self.update_button.on_clicked(self._update)        
        
        for i,p in enumerate(self.paraminit):
            bot_level = topl-0.1-0.05*i
            axbox = self.fig.add_axes([rightl+0.1, bot_level, 0.15, 0.045])
            text_box = TextBox(axbox, 'param{}'.format(i), initial='{:.3e}'.format(p))    
            (self.text_boxes).append(text_box)

        previax = plt.axes([rightl+0.1, bot_level-0.05, 0.1, 0.045])
        self.previous_button = Button(previax, 'Previous', hovercolor='0.975')
        self.previous_button.on_clicked(self._previous)

        miniax = plt.axes([rightl+0.1, bot_level-0.2, 0.1, 0.045])
        self.mini_button = Button(miniax, 'Minimize!', hovercolor='0.975')
        self.mini_button.on_clicked(self._minimize)    

        self.rightl = rightl
        
        self.chi2_fun = chi2_fun

        # finish with initial update
        self._update(0)


    def _minimize(self, event):
        if self.chi2_fun is None:
            print('No chi2 function given.')
        else:
            param = [float(tbox.text) for tbox in self.text_boxes]
            progress_bar = tqdm(total=1000)
            result = minimize(self.chi2_fun, param, method='Nelder-Mead',
                     callback=lambda p: progress_bar.update())

            print(result.message)
            self.set_param(result.x)

    def _update(self, val):
        # getting the values in the text boxes
        param = [float(tbox.text) for tbox in self.text_boxes]
        
        # no update if the same parameters as before
        if param == self.paramnow:
            print('No update. Same param={}'.format(param))
            return None
        
        # saving the previous set of parameters
        self.paramprevious = self.paramnow.copy()
        self.paramnow = param
        
        # evaluating the func for the entered parameters
        new_data = self.func(self.paramnow)
        
        # replacing the ydata of the lines
        for l, nd in zip(self.lines, new_data):            
            l.set_xdata(nd[:,0])
            l.set_ydata(nd[:,1])
        
        # autoscale of the axes
        if self._autoscale_flag:
            for l in self.lines:
                l.axes.relim()
                l.axes.autoscale(True)
            
        # refresh the figure    
        if self.fig:
            self.fig.canvas.draw_idle()
            
        # callback function
        self.callback(param)
        
        # explicit sanity check
        print('Updated with param={}'.format(param))
        
        self.canvas.draw_idle()
        
    def _reset(self, event):
        for p0, text_box in zip(self.paraminit, self.text_boxes):
            text_box.set_val('{:.3e}'.format(p0))
    
    def _previous(self, event):
        # uses an auxiliary list to swap list contents
        param_aux = self.paramprevious.copy()
        self.paramprevious = self.paramnow.copy()
        self.set_param(param_aux)
#        for ppre, tbox in zip(param_aux, self.text_boxes):
#            tbox.set_val(str(ppre))
#        # to finish, update, some memoization might be possible here
#        self._update(0)
    
    def set_param(self, param):
        # set parameters from command prompt
        for p, tbox in zip(param, self.text_boxes):
            tbox.set_val('{:.3e}'.format(p)) 
        self._update(0)
        
    def get_param(self):
        return [float(tbox.text) for tbox in self.text_boxes]
    
    def bonus_button(self, label, on_clicked=None):
        
        if on_clicked is None:
            on_clicked = lambda event: print('Bonus Button ;D')
        
        bonusax = plt.axes([self.rightl+0.1, 0.1, 0.1, 0.045])
        self.bonus_button = Button(bonusax, label, hovercolor='0.975')
        self.bonus_button.on_clicked(on_clicked)    