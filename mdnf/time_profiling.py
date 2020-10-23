# -*- coding: utf-8 -*-
""" Auxiliary functions for measuring time. """

import time
import pandas as pd
import numpy as np


def _count_max(l, tolerance=0.2):
    l = np.array(l)
    t = (np.max(l)-np.min(l))*(1.-tolerance) + np.min(l)
    return np.sum(l>t)


class Timer():
    """ Measuring and recording time of operations. """
    
    shared_dict = {}
    enabled = True;
        
    def __init__(self, timer_name="Timer", dst_dict=None):
        if not self.enabled: return
        if dst_dict is None: dst_dict = Timer.shared_dict
        self.dst_dict = dst_dict
        self.timer_name = timer_name
                        
    def __enter__(self):
        if not self.enabled: return
        self.start = time.time()                
        
    def __exit__(self, type, value, traceback):
        if not self.enabled: return
        elapsed = (time.time()-self.start)
        self.dst_dict[self.timer_name] = self.dst_dict.get(self.timer_name, []) + [elapsed]
      
        
    @classmethod            
    def get_report(Timer):        
        report = pd.DataFrame([(k, len(l), np.sum(l,), np.median(l), np.mean(l),
                        np.min(l), np.max(l), np.percentile(l, 80), _count_max(l)) \
                        for k, l in sorted(Timer.shared_dict.items())] )
        COLS = ["func", "count", "total", "median",  "mean", "min", "max", "q=.8", "#max"]
        report.rename(columns=dict(enumerate(COLS)), inplace=True)
        return report
                                                      
    @classmethod            
    def enable(Timer):
        Timer.enabled = True
        
    @classmethod            
    def disable(Timer):
        Timer.enabled = False

    @classmethod            
    def reset(Timer):
        Timer.shared_dict.clear()


def reset():
    Timer.reset()


def enable():
    Timer.enable()


def disable():
    Timer.disable()


def get_report():
    return Timer.get_report()


def timing(f):
    def measured_function(*args, **kwargs): 
        with Timer(f.__qualname__):
            results = f(*args, **kwargs)   
            return results
        
    measured_function.__name__ = f.__name__
    return measured_function    



