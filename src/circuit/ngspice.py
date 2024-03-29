import sys
import subprocess
import os 


from asyncio.subprocess import DEVNULL

#Output file name
ACEI_OUT = 'ACEI_OUT.dat'
AC_Measures = 'AC_Measures.txt'
OP_Measures = 'OP_Measures.txt'
# print("LOAD NGSPICE\n")


def call(args, cwd = None, timeout = 15):
  try:
    p = subprocess.Popen(args,cwd=cwd, stdout=DEVNULL,stderr=DEVNULL)
    p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
    sys.exit("Simulation ran too long!")
    



def runSimulator(netlist, cwd):
  #runs ngspice and gives a timeout of 15 seconds
  call(["ngspice_con", "-b", netlist, "-o", AC_Measures, "-r", OP_Measures], cwd=cwd, timeout=15)
  # print("ERROOOOOOOOOOOOOOO")

def parseMeasures(cwd):
  measures = {}
  try:
    with open(cwd + AC_Measures, 'r') as file:
      for line in file:
        line = line.replace('nan(ind)', 'nan')
        content = line.split()
        if len(content) == 3 and content[1] == '=':
          measures[content[0]] = content[2]
  except:
    print("File with AC measures not found!")

  listOfMeas = []
  index = 0
  valuesFound = False
  variablesFound = False
  numberVariables = 0
  try:
    with open(cwd + OP_Measures, 'r') as file:
      for line in file:
        if variablesFound and '@' in line:
          listOfMeas.append(line.split('\t')[2].split('@')[1].split(')')[0].replace('[', '_').replace(']', '').replace('.', '_'))
        elif valuesFound and index <= numberVariables:
          if index == 0:
            measures[listOfMeas[index]] = line.split()[1]
          else:
            measures[listOfMeas[index]] = line.split()[0]
          index += 1
        elif 'Values:' in line:
          valuesFound = True
          variablesFound = False
          numberVariables = len(listOfMeas) - 1
        elif 'Variables:' == line.strip():
          variablesFound = True
        elif variablesFound:
          listOfMeas.append(line[line.find("(")+1:line.find(")")])
          
  except IOError:
    # print("File with OP measures not found!")
    pass
  return measures


def removePreviousFiles(cwd):
  #Remove old output files if exist (NGSpice and parser outputs)
  
  try:
    os.remove(AC_Measures)
    os.remove(OP_Measures)
    os.remove(ACEI_OUT)

  except FileNotFoundError:
    pass


def writeDesignVar(cwd, param, val):
  try:
    with open(cwd + "design_var.inc", 'w') as outFile:
      outFile.write('*  Design Var\n')
      outFile.write('.param\n')
      for i in range(len(param)):
        outFile.write('+')
        outFile.write(param[i])
        outFile.write('=')
        outFile.write(str(val[i]))
        outFile.write('\n')
        
  except IOError:
    print("Error opening output file!")




def simulate(cwd, netlist, param, val):
  """
  Executes ngspice simulation in cwd 
  return a dictionary with measures
  """
  removePreviousFiles(cwd)
  writeDesignVar(cwd,param, val)
  runSimulator(netlist, cwd)
  return parseMeasures(cwd)


class Specifications:
  '''
  Specifications for circuit sizing.
  '''
  def __init__(self, lt={}, gt={}, objective=None):
    '''
    Constructor
    Args:
    - lt dictionary with less than measures and their target value
    - gt dictionary with large then measures and their target value
    '''
    self.lt = lt
    # {'idd': 35e-6,'pm' : 90.0}
    self.gt = gt
    # {'gdc': 50,'gbw': 35e6}
    self.objective = objective

  def __str__(self):
    return "obj: {}, lt: {} gt: {}".format(self.objective, self.lt, self.gt)

  def update(self, lt, gt):
    """
    """
    self.lt.update(lt)
    self.gt.update(gt)

  def verifyMOO(self, measures):
    log = {}
    gsum = 0
    obj = []
    for meas, limit in self.lt.items():
      if meas in measures and measures[meas] != None:
        if(measures[meas] > limit):
          gsum += (limit - measures[meas])/abs(limit) 
          log[meas+"_lt"] = (limit, measures[meas])
      else:
        gsum += -1000
        log[meas+"_lt"] = (limit, measures[meas])
        
    for meas, limit in self.gt.items():
      if meas in measures and measures[meas] != None:
        if(measures[meas] < limit):
          gsum += (-limit + measures[meas])/abs(limit) 
          log[meas+"_gt"] = (limit, measures[meas])
      else:
        gsum += -1000
        log[meas+"_gt"] = (limit, measures[meas])
    
    if self.objective is not None:
      obj = [ (measures[obj[0]] if measures[obj[0]] is not None else -1000)*obj[1] for obj in self.objective]
      
    return obj, gsum, log
    
    
    
  def verify(self, measures):
    obj, gsum, log = self.verifyMOO(measures)
    
    return gsum, gsum==0, log
  

  
  def asarray(self):
    return list(self.lt.values()) + list(self.gt.values()) 

if __name__ == '__main__':
  
  parameters = ('_w8','_w6','_w4','_w10','_w1','_w0',
          '_l8','_l6','_l4','_l10','_l1','_l0',
          "_nf8","_nf6", "_nf4", "_nf10", "_nf1", "_nf0" )
          
  values     = ( 1.0000e-06, 7.1800e-05, 1.5700e-05, 2.2000e-06, 1.6000e-06, 9.0000e-06,
           9.4000e-07, 8.8000e-07, 6.7000e-07, 8.9000e-07, 8.9000e-07, 8.4000e-07,
           5.0000e+00, 1.0000e+00, 7.0000e+00, 1.0000e+00, 3.0000e+00, 3.0000e+00)
  
  folder = "./examples/ssvcamp-ngspice/"
  
  measures = simulate(cwd = folder, netlist="3a_VCOTA_OLtb_AC_OP.cir", param = parameters, val = values)
  
  print(measures)	


  fcas = {
    "_wp5": 1.8500e-05,
    "_wp3": 1.1300e-05,
    "_wp1": 2.6690e-04,
    "_wp0": 5.4900e-05,
    "_wn7":1.1760e-04,
    "_wn5":1.2510e-04,
    "_wn3":8.5000e-05,
    "_wn2":9.5000e-06,
    "_wn1":5.5000e-06,
    "_nfp5":1.1000e+01,
    "_nfp3":5.0000e+00,
    "_nfp1":5.0000e+00,
    "_nfp0":1.1000e+01,
    "_nfn7":1.3000e+01,
    "_nfn5":1.3000e+01,
    "_nfn3":5.0000e+00,
    "_nfn2":1.5000e+01,
    "_nfn1":3.0000e+00,
    "_lp1":8.8000e-07,
    "_lp0":1.4000e-06,
    "_ln7":1.0500e-06,
    "_ln5":1.4800e-06,
    "_ln3":7.4000e-07,
    "_ln1":2.3100e-06
  }

  parameters_fcas = [k for k in fcas.keys()]
  values_fcas = [fcas[k] for k in fcas.keys()]

  folder = "./examples/fcas-ngspice/cir/"

  measures = simulate(cwd = folder, netlist="tb_ac.cir", param = parameters_fcas, val = values_fcas)

  for m in ('idd', 'pm', 'gdc', 'gbw', 'fom'):
    print(m, measures.get(m, None))	




    

