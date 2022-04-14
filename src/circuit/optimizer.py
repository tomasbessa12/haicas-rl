'''
Created on Feb 15, 2019

@author: aida
'''
from circuit.vcamp import VCAmplifierCircuitOptProblem
from circuit import ngspice as ng
import time
import math
import random
import numpy as np

class Optimizer():
  '''
  classdocs
  '''


  def __init__(self):
    '''
    Constructor
    '''
    pass
    
    
    
class SA(Optimizer):  
  '''
  Based on Matthew T. Perry code from https://github.com/perrygeo/simanneal/blob/master/simanneal/anneal.py from 
  
  '''
  def __init__(self):
    '''
    Constructor
    '''
    pass
  
  def minimize(self, problem, steps = 10000, Tmax = 1500.0, Tmin = 2.5, initial_state=None):
    """Minimizes the energy of a system by simulated annealing.
     Parameters
      state : an initial arrangement of the system
        Returns
        (state, energy): the best state and energy found.
    """

    stats = {}
    start = time.time()
  
    # Precompute factor for exponential cooling from Tmax to Tmin
    if Tmin <= 0.0:
      raise Exception('Exponential cooling requires a minimum temperature greater than zero.')
    Tfactor = -math.log(Tmax / Tmin)

    # Note initial state
    T = Tmax
    if initial_state is None :
      self.best_state = prev_state = state = problem.random_values()
    else:
      self.best_state = prev_state = state = initial_state
    self.best_value = prev_value = value = problem.evaluate(state)
    
    step, trials, accepts, improves = 0, 0, 0, 0
    # Attempt moves to new states
    while step < steps:
      stats[step] =(T, value, self.best_value, trials, accepts, improves)
      if step % 20 == 0: 
        yield step, stats[step]
      step += 1
      T = Tmax * math.exp(Tfactor * step / steps)
      
      state = problem.move(state)
      value = problem.evaluate(state)
      
      dV = 100*(value - prev_value)
      trials += 1
      if dV > 0.0 and math.exp(-dV / T) < random.random():
        # Restore previous state
        state, value = prev_state, prev_value
      else:
        # Accept new state and compare to best state
        accepts += 1
        prev_state,prev_value  = state, value
        
        if dV < 0.0: improves += 1
        if value < self.best_value:
          self.best_state, self.best_value, = state, value
    # Return best state and energy
    return self.best_state, self.best_value

  
if __name__ == '__main__':
  seed = 17
  np.random.seed(seed)
  random.seed(seed)
  
  sat_conditions = {}
  sat_conditions["vov_mpm0"] = 0.05
  sat_conditions["vov_mpm1"] = 0.05
  sat_conditions["vov_mpm2"] = 0.05
  sat_conditions["vov_mpm3"] = 0.05
  sat_conditions["vov_mnm4"] = 0.05
  sat_conditions["vov_mnm5"] = 0.05
  sat_conditions["vov_mnm6"] = 0.05
  sat_conditions["vov_mnm7"] = 0.05
  sat_conditions["vov_mnm8"] = 0.05
  sat_conditions["vov_mnm9"] = 0.05
  sat_conditions["vov_mnm10"] = 0.05
  sat_conditions["vov_mnm11"] = 0.05

  sat_conditions["delta_mpm0"] = 0.1
  sat_conditions["delta_mpm1"] = 0.1
  sat_conditions["delta_mpm2"] = 0.1
  sat_conditions["delta_mpm3"] = 0.1
  sat_conditions["delta_mnm4"] = 0.1
  sat_conditions["delta_mnm5"] = 0.1
  sat_conditions["delta_mnm6"] = 0.1
  sat_conditions["delta_mnm7"] = 0.1
  sat_conditions["delta_mnm8"] = 0.1
  sat_conditions["delta_mnm9"] = 0.1
  sat_conditions["delta_mnm10"] = 0.1
  sat_conditions["delta_mnm11"] = 0.1
  
  gt={'gdc': 50,'gbw': 35e6,'pm' : 45.0, 'fom': 900}
  gt.update(sat_conditions)
  
  circuit = VCAmplifierCircuitOptProblem(
    ng.Specifications(objective=[('idd', 1)], lt={'idd': 35e-5,'pm' : 90.0},gt=gt), discrete_actions = False)
  sa = SA()

  print(circuit)

  for iter, stats in sa.minimize(circuit): 
    print("\r iter {}: {}".format(iter, stats))

  print(sa.best_state)
  print(circuit.simulate(sa.best_state))
  print(circuit.target.verify(circuit.simulate(sa.best_state)))

