'''
Created on Oct 2, 2022
@author: aida
'''
import random
import math
import numpy as np
from circuit import ngspice as ng


class FoldedCascodeCircuit:
  '''
  Folded Cascode amp specifications for simulation using ngspice.
  '''
  def __init__(self):
    '''
    Constructor all needeed parameters to simualte circuit 
    '''
    #parameters names wihc are the desing variables
    self.parameters = (
        "_wp5", "_wp3", "_wp1", "_wp0", 
        "_wn7", "_wn5", "_wn3", "_wn2", "_wn1", 
        "_lp1", "_lp0", "_ln7", "_ln5", "_ln3", "_ln1",
        "_nfp5", "_nfp3", "_nfp1", "_nfp0", 
        "_nfn7", "_nfn5", "_nfn3", "_nfn2", "_nfn1")

    #parameters ranges (mion, max, grid)
    self.ranges = np.array([[1e-6, 100e-6, 0.1e-6],[1e-6,100e-6, 0.1e-6],[1e-6,100e-6, 0.1e-6],[1e-6,100e-6, 0.1e-6],
                            [1e-6, 100e-6, 0.1e-6],[1e-6,100e-6, 0.1e-6],[1e-6,100e-6, 0.1e-6],[1e-6,100e-6, 0.1e-6],[1e-6,100e-6, 0.1e-6],
                            [0.34e-6,1e-6, 0.1e-6],[0.34e-6,1e-6, 0.1e-6],[0.34e-6,1e-6, 0.1e-6],[0.34e-6,1e-6, 0.1e-6],[0.34e-6,1e-6, 0.1e-6],[0.34e-6,1e-6, 0.1e-6],
                            [1,8,1],[1,8,1],[1,8,1],[1,8,1],
                            [1,8,1],[1,8,1],[1,8,1],[1,8,1],[1,8,1]])


    assert len(self.parameters) == len (self.ranges)

    self.folder = "./examples/fcas-ngspice/cir/"
  
  def __str__(self):
    return "Running Folder: {}\nParameters: {}\nRanges: {}".format(self.folder, self.parameters, self.ranges)

  def parameters(self):
    return self.parameters
  
  def range_min(self):
    ''' 
    Return range.min as a column vector 
    '''
    return self.ranges[:,0]
  
  def range_max(self):
    ''' 
    Return range.max as a column vector 
    '''
    return self.ranges[:,1]
  
  
  def random_values(self, n = 1):
    if(n == 1): values = np.random.rand(len(self.parameters))
    else : values = np.random.rand(n, len(self.parameters))
    
    values = self.ranges[:, 0] + values*(self.ranges[:,1] - self.ranges[:, 0]) 
    values = np.round(values / self.ranges[:,2])*self.ranges[:,2] 
    return values
  

  def simulate(self, parameter_values):
    '''
    Simulate this circuit using the values as paramters
    inputs:
    values a numpy array with the solution to simulate
    
    Returns: the measures and expended measures as a {name:value} map
    
    '''
    
    assert len(parameter_values) == len(self.parameters)
    #move to grid
    parameter_values = np.round(parameter_values / self.ranges[:,2])*self.ranges[:,2]
    parameter_values = np.fmin(np.fmax(parameter_values,self.ranges[:, 0]),self.ranges[:,1])
    
    return self._extended_meas(
      ng.simulate(
        cwd = self.folder, 
        netlist="tb_ac.cir", 
        param = self.parameters, 
        val = parameter_values))
    

  def _extended_meas(self, measures):
    """
    inputs raw measures from simulator and outputs only relevant measures
    """
    meas_out = measures
    meas_out['gdc'] = float(measures['gdc'])
    meas_out['gbw'] = float(measures['gbw']) if 'gbw' in measures else None
    meas_out['pm'] = float(measures['pm']) if 'pm' in measures else None
    #meas_out['voff'] = float(measures['voff'])
    meas_out['inoise_total'] = float(measures['inoise_total']) if 'inoise_total' in measures else None
    meas_out['onoise_total'] = float(measures['onoise_total']) if 'onoise_total' in measures else None

    meas_out['idd'] = float(measures['idd'])

    meas_out["fom"] =  (((meas_out['gbw']/1000000)*6)/(meas_out['idd']*1000)) if meas_out['gbw'] != None else None



    return meas_out
  
  

class FoldedCascodeRLEnv(FoldedCascodeCircuit):
  '''
  VC amp as an open AI gym.
  Use random initial sizes and target specifications
  OPTION TO TRY: Use predefined sets of initial sizes and target specifications 
  '''

  def __init__(self):
    '''
    Constructor
    defines param
    '''
    FoldedCascodeCircuit.__init__(self)
    
    
    self.values = np.array([
        1.8500e-05, 1.1300e-05, 2.6690e-04, 5.4900e-05, 1.1760e-04, 1.2510e-04,
        8.5000e-05, 9.5000e-06, 5.5000e-06, 8.8000e-07, 1.4000e-06, 1.0500e-06,
        1.4800e-06, 7.4000e-07, 2.3100e-06, 
        1.1000e+01, 5.0000e+00, 5.0000e+00, 1.1000e+01, 1.3000e+01, 1.3000e+01,
        5.0000e+00, 1.5000e+01, 3.0000e+00])

    self.values_init = np.array([
        1.8500e-05, 1.1300e-05, 2.6690e-04, 5.4900e-05, 1.1760e-04, 1.2510e-04,
        8.5000e-05, 9.5000e-06, 5.5000e-06, 8.8000e-07, 1.4000e-06, 1.0500e-06,
        1.4800e-06, 7.4000e-07, 2.3100e-06, 
        1.1000e+01, 5.0000e+00, 5.0000e+00, 1.1000e+01, 1.3000e+01, 1.3000e+01,
        5.0000e+00, 1.5000e+01, 3.0000e+00])
    
    self.target = ng.Specifications(
      lt={'idd': 200e-6,'pm' : 90.0}, 
      gt={'gdc': 70,'gbw': 60e6,'pm' : 45.0, 'fom' : 1000 })


    self.state_scale = self._run_simulation()
    self.state_size = len(self.state_scale)
    self.action_size = len(self.parameters) 

    self.current_performance, _, self.current_performance_log = self.target.verify(self.measures)


  def __str__(self):
    return "Values:\n {}\nPerformance: {}\nEval Log: {}".format(self.values, self.current_performance, self.current_performance_log)  

  def step(self, action):
    """
    Inputs:
    - action - changes to devices sizes as an array [-1 to 1]
         self.values update is self.value += self.value + action*(self.ranges[:,1] - self.ranges[:, 0])    
		-
		Outouts: observation, reward, done, {}
		- observations array of concat [ values, measures]
		- reward +1 improved, -1 worsen, -1000 no sim, 1000 meet specs
    """
    self.iter += 1
    #Converts the action space to simulator input.

    self.values = self.values + 0.05*action*(self.ranges[:,1] - self.ranges[:, 0])     
    self.values = np.round(self.values / self.ranges[:,2])*self.ranges[:,2]
 
    obs = self._run_simulation()
    next_performance, done, log = self.target.verify(self.measures)

    log['performance'] = next_performance
    reward = 0
    # reward 100 if feasible


    if next_performance > -1:
      if next_performance > self.current_performance: reward = next_performance + 1  
    else:
      reward = 0

    if next_performance == 0: reward = 10

    self.current_performance = next_performance
    done = done or (self.iter >= 200)

    return obs, reward, done, log

  def reset(self, values=None):
    """
		Sets new initial values and target specs.
		Can be made to ensure initial sizing simulates all measures
		"""
    if values is None :
      self.values = np.random.rand(len(self.parameters))
      self.values = self.ranges[:, 0] + self.values*(self.ranges[:,1] - self.ranges[:, 0]) 
      self.values = np.round(self.values / self.ranges[:,2])*self.ranges[:,2]
    else:
      self.values = values

    assert len(self.values) == len(self.parameters) 

    self.iter = 0

    obs = self._run_simulation()
    self.current_performance, done, self.current_performance_log = self.target.verify(self.measures)

    return obs

  def sample_action(self, n = 1):
    if(n == 1): return  np.random.rand(18)-0.5
    return np.random.rand(n, 18)-0.5

  def render(self, mode='human', close=False):
     print(self)
  
  def _run_simulation(self):
    self.measures = self.simulate(self.values)
    meas = np.array(list(self.measures.values()), dtype=float)
    obs = np.concatenate((np.array(self.target.asarray()), self.values, meas))

    return obs


class FoldedCascodeRLEnvDiscrete(FoldedCascodeRLEnv):
  '''
  VC amp as an open AI gym with discrete actions.
  use as singleton because is does not copy files to temporary folder
  
  Use random initial sizes and target specifications
  OPTION TO TRY: Use predefined sets of initial sizes and target specifications
  Note refractor to inheritance 
  
  '''
  
  ACTIONS = []
  STEPS = [1, 5, 10, 50, 100]
  
  def __init__(self):
    '''
    Constructor
    '''
    FoldedCascodeRLEnv.__init__(self)

    for var_index in range(16):
      for step in self.STEPS:
        #add only actions that make sense with the ranges
        if(step * self.ranges[var_index,2] + self.ranges[var_index,0] < self.ranges[var_index,1]):
          self.ACTIONS.append((var_index, step))
          self.ACTIONS.append((var_index, -step))

    self.action_size = len(self.ACTIONS)

  def step(self, a):
    """
    Inputs:
    - action - index to ACTION tuple (var_to_change, step) 
         self.values update is self.value[var_to_change] = self.value[var_to_change] + step*(self.ranges[var_to_change,2])  
    -
    Outouts: observation, reward, done, {}
    - observations array of concat [ values, measures]
    - reward +1 improved, -1 worsen, -1000 no sim, 1000 meet specs
    """
    previous_values = self.values 
    previous_obs = self.obs
    action = self.ACTIONS[a]
    self.iter += 1
    #Converts the action space to simulator input.

    self.values[action[0]] = self.values[action[0]] + [action[1]]*self.ranges[[action[0]],2]     
    self.values[action[0]] = np.round(self.values[action[0]] / self.ranges[[action[0]],2])*self.ranges[[action[0]],2]
    
    self.obs = self._run_simulation()
    next_performance, done, log = self.target.verify(self.measures)

    log['performance'] = next_performance
    reward = 0
    # reward 100 if feasible


    reward = math.tanh(0.001*next_performance)
    if next_performance == 0: reward = 1
    
    
    if(self.current_performance > next_performance) and (random.random() > 0.5):
      #todo add some anneling schedlre here
      self.values = previous_values
      self.obs = previous_obs
    
    done = done or (self.iter >= 1000) 

    return self.obs, reward, done, log

  def reset(self, values=None):
    """
    Sets new initial values and target specs.
    Can be made to ensure initial sizing simulates all measures
    """
    if values is None :
      self.values[:15] = np.random.rand(len(self.parameters)-9)
      self.values[:15] = self.ranges[:15, 0] + self.values[:15]*(self.ranges[:15,1] - self.ranges[:15, 0]) 
      self.values[:15] = np.round(self.values[:15] / self.ranges[:15,2])*self.ranges[:15,2]
    else:
      self.values[:15] = values[:15]

    assert len(self.values) == len(self.parameters) 

    self.iter = 0

    self.obs = self._run_simulation()
    self.current_performance, done, self.current_performance_log = self.target.verify(self.measures)

    return self.obs

  def sample_action(self):
    return random.randint(0, len(self.ACTIONS) -1)


if __name__ == '__main__':
  env = FoldedCascodeRLEnvDiscrete()

  observation = env.reset()
  action = env.sample_action()

  # add row 
  # numpy.vstack([A, newrow])

  for i_episode in range(10):
    observation = env.reset()

    for t in range(3):
      
      action = env.sample_action()
      env.render()
      observation, reward, done, info = env.step(action)
     
      if done:
        print("Episode finished after {} timesteps".format(t+1))
        break

