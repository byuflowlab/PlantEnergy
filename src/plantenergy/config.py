"""
Description:    Defines a global variable to track function calls (including calls for finite difference)
Date:           3/26/2016
Author:         Jared J. Thomas
"""

import numpy as np

obj_func_calls = 0
sens_func_calls = 0

nMaxProcs = 100
obj_func_calls_array = np.zeros(nMaxProcs)
sens_func_calls_array = np.zeros(nMaxProcs)


floris_single_component = False
BV = True
