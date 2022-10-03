'''
Created on Feb 1, 2019

@author: aida
'''
import random
import math
import numpy as np
from circuit import ngspice as ng
from circuit.func1 import Func1


class VCAmplifierCircuit:
  '''
  VC amp specifications for simulation using ngspice.
  '''
  def __init__(self):
    '''
    Constructor all needeed parameters to simualte circuit 
    '''
    #parameters names wihc are the desing variables
    self.parameters = ('_w8','_w6','_w4','_w10','_w1','_w0', '_l8',
                       '_l6','_l4','_l10','_l1','_l0',
                       "_nf8","_nf6", "_nf4", "_nf10", "_nf1", "_nf0" )

    #parameters ranges (mion, max, grid)
    self.ranges = np.array([[1e-6, 100e-6, 1e-6],[1e-6,100e-6, 1e-6],
                            [1e-6,100e-6, 1e-6],[1e-6,100e-6, 1e-6],
                            [1e-6,100e-6, 1e-6],[1e-6,100e-6, 1e-6],
                            [0.34e-6,10e-6, 0.1e-6],[0.34e-6,10e-6, 0.1e-6],
                            [0.34e-6,10e-6, 0.1e-6],[0.34e-6,10e-6, 0.1e-6],
                            [0.34e-6,10e-6, 0.1e-6],[0.34e-6,10e-6, 0.1e-6],
                            [1,8, 1],[1,8,1],[1,8,1],[1,8,1],[1,8,1],[1,8,1]])

    self.folder = "./examples/ssvcamp-ngspice/"
  
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
        netlist="3a_VCOTA_OLtb_AC_OP.cir", 
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

    meas_out['idd'] = ((-float(measures['vdc_i'])) - 0.0001) 

    meas_out["fom"] =  (((meas_out['gbw']/1000000)*6)/(meas_out['idd']*1000)) if meas_out['gbw'] != None else None

    meas_out["vov_mpm0"] = float(measures["m_xinova_mpm0_vgs"]) - float(measures["m_xinova_mpm0_vth"])
    meas_out["vov_mpm1"] = float(measures["m_xinova_mpm1_vgs"]) - float(measures["m_xinova_mpm1_vth"])
    meas_out["vov_mpm2"] = float(measures["m_xinova_mpm2_vgs"]) - float(measures["m_xinova_mpm2_vth"])
    meas_out["vov_mpm3"] = float(measures["m_xinova_mpm3_vgs"]) - float(measures["m_xinova_mpm3_vth"])
    meas_out["vov_mnm4"] = float(measures["m_xinova_mnm4_vgs"]) - float(measures["m_xinova_mnm4_vth"])
    meas_out["vov_mnm5"] = float(measures["m_xinova_mnm5_vgs"]) - float(measures["m_xinova_mnm5_vth"])
    meas_out["vov_mnm6"] = float(measures["m_xinova_mnm6_vgs"]) - float(measures["m_xinova_mnm6_vth"])
    meas_out["vov_mnm7"] = float(measures["m_xinova_mnm7_vgs"]) - float(measures["m_xinova_mnm7_vth"])
    meas_out["vov_mnm8"] = float(measures["m_xinova_mnm8_vgs"]) - float(measures["m_xinova_mnm8_vth"])
    meas_out["vov_mnm9"] = float(measures["m_xinova_mnm9_vgs"]) - float(measures["m_xinova_mnm9_vth"])
    meas_out["vov_mnm10"] = float(measures["m_xinova_mnm10_vgs"]) - float(measures["m_xinova_mnm10_vth"])
    meas_out["vov_mnm11"] = float(measures["m_xinova_mnm11_vgs"]) - float(measures["m_xinova_mnm11_vth"])

    meas_out["delta_mpm0"] = float(measures["m_xinova_mpm0_vds"]) - float(measures["m_xinova_mpm0_vdsat"])
    meas_out["delta_mpm1"] = float(measures["m_xinova_mpm1_vds"]) - float(measures["m_xinova_mpm1_vdsat"])
    meas_out["delta_mpm2"] = float(measures["m_xinova_mpm2_vds"]) - float(measures["m_xinova_mpm2_vdsat"])
    meas_out["delta_mpm3"] = float(measures["m_xinova_mpm3_vds"]) - float(measures["m_xinova_mpm3_vdsat"])
    meas_out["delta_mnm4"] = float(measures["m_xinova_mnm4_vds"]) - float(measures["m_xinova_mnm4_vdsat"])
    meas_out["delta_mnm5"] = float(measures["m_xinova_mnm5_vds"]) - float(measures["m_xinova_mnm5_vdsat"])
    meas_out["delta_mnm6"] = float(measures["m_xinova_mnm6_vds"]) - float(measures["m_xinova_mnm6_vdsat"])
    meas_out["delta_mnm7"] = float(measures["m_xinova_mnm7_vds"]) - float(measures["m_xinova_mnm7_vdsat"])
    meas_out["delta_mnm8"] = float(measures["m_xinova_mnm8_vds"]) - float(measures["m_xinova_mnm8_vdsat"])
    meas_out["delta_mnm9"] = float(measures["m_xinova_mnm9_vds"]) - float(measures["m_xinova_mnm9_vdsat"])
    meas_out["delta_mnm10"] = float(measures["m_xinova_mnm10_vds"]) - float(measures["m_xinova_mnm10_vdsat"])
    meas_out["delta_mnm11"] = float(measures["m_xinova_mnm11_vds"]) - float(measures["m_xinova_mnm11_vdsat"])

    return meas_out
  
  

class VcAmpRLEnv(VCAmplifierCircuit):
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
    VCAmplifierCircuit.__init__(self)
    
    
    self.values = np.array([1.0000e-06, 7.1800e-05, 1.5700e-05, 2.2000e-06, 1.6000e-06, 9.0000e-06,
                            9.4000e-07, 8.8000e-07, 6.7000e-07, 8.9000e-07, 8.9000e-07, 8.4000e-07,
                            5.0000e+00, 1.0000e+00, 7.0000e+00, 1.0000e+00, 3.0000e+00, 3.0000e+00])
    
    self.values_init = np.array([1.0000e-06, 7.1800e-05, 1.5700e-05, 2.2000e-06, 1.6000e-06, 9.0000e-06,
                            9.4000e-07, 8.8000e-07, 6.7000e-07, 8.9000e-07, 8.9000e-07, 8.4000e-07,
                            5.0000e+00, 1.0000e+00, 7.0000e+00, 1.0000e+00, 3.0000e+00, 3.0000e+00])
    

    self.target = ng.Specifications(
      lt={'idd': 35e-5,'pm' : 90.0}, 
      gt={'gdc': 50,'gbw': 35e6,'pm' : 45.0})

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
      # self.values = np.random.rand(len(self.parameters))
      self.values = self.values_init
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
    obs = np.concatenate((np.array(self.target.asarray()), self.values))
    # obs = np.concatenate((np.array(self.target.asarray()), self.values, meas))


    return obs


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class VCAmpRLEnvDiscrete(VcAmpRLEnv):
  '''
  VC amp as an open AI gym with discrete actions.
  use as singleton because is does not copy files to temporary folder
  
  Use random initial sizes and target specifications
  OPTION TO TRY: Use predefined sets of initial sizes and target specifications
  Note refractor to inheritance 
  
  '''
  
  ACTIONS = []
  STEPS = [1, 10]
  
  def __init__(self):
    '''
    Constructor
    '''
    VcAmpRLEnv.__init__(self)

    for var_index in range(len(self.ranges)):
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

    done = done or (self.iter >= 200) or (next_performance < -3000)

    reward = (next_performance) if not done else -100
    if next_performance == 0: reward = 200
    # if next_performance<-500: 
    #   reward = -500
    #   done = True


    # UNSTEP FUNCTION
    if(self.current_performance > next_performance): 
      #todo add some anneling schedlre here
     self.values = previous_values
     self.obs = previous_obs
    
    else: 
      self.current_performance = next_performance
   

    return self.obs, reward, done, log


# RESET
  def reset(self, values=None):
    """
    Sets new initial values and target specs.
    Can be made to ensure initial sizing simulates all measures
    """
    if values is None :
      self.values[:12] = np.random.rand(len(self.parameters)-6)
      self.values[:12] = self.ranges[:12, 0] + self.values[:12]*(self.ranges[:12,1] - self.ranges[:12, 0]) 
      self.values[:12] = np.round(self.values[:12] / self.ranges[:12,2])*self.ranges[:12,2]
    else:
      self.values[:12] = values[:12]

    assert len(self.values) == len(self.parameters) 

    self.iter = 0

    self.obs = self._run_simulation()
    self.current_performance, done, self.current_performance_log = self.target.verify(self.measures)

    # Resultado da simulação
    return self.obs

  def sample_action(self):
    return random.randint(0, len(self.ACTIONS) -1)


class VCAmplifierCircuitOptProblem(VCAmplifierCircuit):
  
  def __init__(self, target, discrete_actions = False):
    VCAmplifierCircuit.__init__(self)

    self.target = target
    self.actions = None
    if discrete_actions :
      self.actions = []
      for step in [1,2, 5, 10, 20, 50, 100, 200]:
        for var_index in range(18):
          #add only actions that make sense with the ranges
          if(step * self.ranges[var_index,2] + self.ranges[var_index,0] < self.ranges[var_index,1]):
            self.actions.append((var_index, step))
            self.actions.append((var_index, -step))

  def __str__(self):
    return "Target: {}".format(self.target) 

  def evaluate(self, parameter_values):
    """
    Inputs:
    - action - changes to devices sizes as an array [-1 to 1]
         self.values update is self.value += self.value + action*(self.ranges[:,1] - self.ranges[:, 0])    
    -
    Outouts: observation, reward, done, {}
    - observations array of concat [ values, measures]
    - reward +1 improved, -1 worsen, -1000 no sim, 1000 meet specs
    """

    measures = self.simulate(parameter_values)
    
    next_performance, _, __ = self.target.verify(measures)

    return -next_performance
    

  def move(self, parameter_values):
    """
    Inputs:
    - value - new device sizes
         self.values update is self.value += self.value + action*(self.ranges[:,1] - self.ranges[:, 0])    
    -
    Outouts: observation, reward, done, {}
    - observations array of concat [ values, measures]
    - reward +1 improved, -1 worsen, -1000 no sim, 1000 meet specs
    """

    #Converts the action space to simulator input.
    if self.actions is None:
      action = np.random.normal(scale=0.1, size=18)
      parameter_values = parameter_values + action*(self.ranges[:,1] - self.ranges[:, 0])
      parameter_values = np.round(parameter_values / self.ranges[:,2])*self.ranges[:,2]
      parameter_values = np.fmin(np.fmax(parameter_values,self.ranges[:, 0]),self.ranges[:,1])
    else:
      var, steps = random.choice(self.actions)
      new_val = parameter_values[var] + steps*self.ranges[var,2] 
      while new_val > self.ranges[var,1] or new_val < self.ranges[var,0] :
        var, steps = random.choice(self.actions)
        new_val = parameter_values[var] + steps*self.ranges[var,2]

      parameter_values[var] = new_val 
      parameter_values[var] = np.round(parameter_values[var] / self.ranges[var,2])*self.ranges[var,2]
      parameter_values[var] = np.fmin(np.fmax(parameter_values[var],self.ranges[var, 0]),self.ranges[var,1])
      
    return parameter_values



class VCAmpMOOProb(VCAmplifierCircuit, Func1):
  
  def __init__(self, target):
    VCAmplifierCircuit.__init__(self)
    
    self.target = target
    self.d = len(self.parameters)
    self.M = len(target.objective)
    self.upper = self.range_max()
    self.lower = self.range_min()
    
    

  def __str__(self):
    return "Target: {}".format(self.target) 

  def evaluate(self, parameter_values):
    """
    Inputs:
    - action - changes to devices sizes as an array [-1 to 1]
         self.values update is self.value += self.value + action*(self.ranges[:,1] - self.ranges[:, 0])    
    -
    Outouts: observation, reward, done, {}
    - observations array of concat [ values, measures]
    - reward +1 improved, -1 worsen, -1000 no sim, 1000 meet specs
    """

    measures = self.simulate(parameter_values)
    
    obj, gsum, log = self.target.verifyMOO(measures)

    return obj, gsum
    
  def cost_fun(self, x):
      """
      calculate the objective vectors
      :param x: the decision vectors
      :return: the objective vectors
      """
      n = x.shape[0]
      
      obj = np.zeros((n, self.M))
      cstr = np.zeros((n, 1))
      for i in range(n):
          obj[i,:],cstr[i] = self.evaluate(x[i,:])
      return obj, cstr




 
def transition_as_array(state, gsum_state, next_state,gsum_next_state, action):
  return np.concatenate([np.array([gsum_state, gsum_next_state]), action, state, next_state])

def transition_header(state, action):
  header = ["gsum_state","gsum_next_state"]
  for i in range(action.shape[0]):
    header.append ("action_" + str(i))

  for i in range(len(state)):
    header.append ("state_" + str(i))

  for i in range(len(state)):
    header.append ("next_state_" + str(i))
  return header


def update_tran_table(tran_table, transition):
  if tran_table is None:
    return transition
  return np.vstack((tran_table, transition))

if __name__ == '__main__':
  env = VcAmpRLEnv()

  observation = env.reset()
  action = env.sample_action()

  # add row 
  # numpy.vstack([A, newrow])

  for i_episode in range(10):
    observation = env.reset()

    for t in range(3):
      env.render()
      action = env.sample_action()
      observation, reward, done, info = env.step(action)
     
      if done:
        print("Episode finished after {} timesteps".format(t+1))
        break

