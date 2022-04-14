#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from circuit.vcamp import VCAmpMOOProb
from circuit import ngspice as ng
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import random
from itertools import repeat
from collections import Sequence
#
def tournament(K, N, fit):
    '''
    tournament selection
    :param K: number of solutions to be compared
    :param N: number of solutions to be selected
    :param fit: fitness vectors
    :return: index of selected solutions
    '''
    n = len(fit)
    print(n, N)
    mate = []
    for i in range(N):
        a = np.random.randint(n)
        for j in range(K):
            b = np.random.randint(n)
            for r in range(fit[0, :].size):
                if fit[(b, r)] < fit[(a, r)]:
                    a = b
        mate.append(a)
    
    return np.array(mate)

def pareto_dominance(objs, M, a, b):
     
    a_dominates_b = False;
    b_dominates_a = False;
 
    for i in range(M):
        if objs[a,i] < objs[b,i] :
            a_dominates_b = True;
        elif objs[a,i] > objs[b,i] :
            b_dominates_a = True;
 
    if a_dominates_b and (not b_dominates_a):
        return -1
    if (not a_dominates_b) and b_dominates_a: 
        return 1
    return 0


def nd_sort(pop_obj, pop_cstr):
    """
    :rtype:
    :param n_sort:
    :param pop_obj: objective vectors
    :return: [FrontNo, MaxFNo]
    """
    
    n, m_obj = np.shape(pop_obj)
    a, loc = np.unique(pop_obj[:, 0], return_inverse=True)
    index = pop_obj[:, 0].argsort()
    new_obj = pop_obj[index, :]
    front_no = np.inf * np.ones(n)
    max_front = 0
    while np.sum(front_no < np.inf) < min(n/2, len(loc)):
        max_front += 1
        for i in range(n):
            if front_no[i] == np.inf:
                dominated = False
                for j in range(i, 0, -1):
                    if front_no[j - 1] == max_front:
                        m = 2
                        while (m <= m_obj) and (new_obj[i, m - 1] >= new_obj[j - 1, m - 1]):
                            m += 1
                        dominated = m > m_obj
                        if dominated or (m_obj == 2):
                            break
                if not dominated:
                    front_no[i] = max_front
    
    return front_no[loc], max_front


def fnd_sort(pop_obj, pop_cstr):
    """
    Computes and sets the ranks of the population elements  using the fast non-dominated sorting method.
      
    :returns: ranks: an array with the ranks
              max_rank: max rank
    """
    
    N,M = pop_obj.shape
    
    ranks = np.inf * np.ones(N)
    
    # structures for holding the domination info required for fast nd sorting 
    dominate = [[] for x in range(N)]   
    dominatedByCounter = np.zeros(N, dtype=int)   
    
    #list holding the indexes of current pop
    current_front = []
   
    for i in range(N):
        for j in range(i+1,N):    
            #constrained pareto dominance
            if pop_cstr[i] == pop_cstr[j]:
                i_dominates_j = False
                j_dominates_i = False
                                
                if pop_obj[i,0] < pop_obj[j,0] :
                  i_dominates_j = True;
                elif pop_obj[i,0] > pop_obj[j,0] :
                  j_dominates_i = True;
                
                if M >= 2:
                  if pop_obj[i,1] < pop_obj[j,1] :
                    i_dominates_j = True;
                  elif pop_obj[i,1] > pop_obj[j,1] :
                    j_dominates_i = True;
                
                if M >= 3:
                  if pop_obj[i,2] < pop_obj[j,2] :
                    i_dominates_j = True;
                  elif pop_obj[i,2] > pop_obj[j,2] :
                    j_dominates_i = True;

                if M >= 4:  
                  for a in range(3,M):
                    if pop_obj[i,a] < pop_obj[j,a] :
                        i_dominates_j = True;
                    elif pop_obj[i,a] > pop_obj[j,a] :
                        j_dominates_i = True;
             
                if i_dominates_j and (not j_dominates_i):
                    dominate[i].append(j)
                    dominatedByCounter[j]+=1
                if (not i_dominates_j) and j_dominates_i: 
                    # b dominates a*/
                    dominate[j].append(i)
                    dominatedByCounter[i]+=1

            elif pop_cstr[i] < pop_cstr[j]:
                # b dominates a*/
                dominate[j].append(i)
                dominatedByCounter[i]+=1
            else:
                # a dominates b » /*updates the set of dominated solutions*/
                dominate[i].append(j) 
                dominatedByCounter[j]+=1
                                       

        #    if non dominated  is part of front 1*/
        if dominatedByCounter[i] == 0:
            current_front.append(i)
    
        
        
    #    assign rank */
    current_rank = 1
        
    while np.sum(ranks < np.inf) < N/2:
        ranks[current_front] = current_rank
        next_front = []

        for indexA in current_front:
            #  reduce the numbers of domination to the ones in its set of dominance         
            for indexB in dominate[indexA]:
                dominatedByCounter[indexB]-=1
                #  if( they become non dominated - then they are part of next front)
                if dominatedByCounter[indexB] == 0:
                    next_front.append(indexB)

        current_front = next_front
        current_rank+=1
               
    return ranks, current_rank-1


def crowding_distance(pop_obj, front_no):
    """
    The crowding distance of each Pareto front
    :param pop_obj: objective vectors
    :param front_no: front numbers
    :return: crowding distance
    """
    n, M = np.shape(pop_obj)
    crowd_dis = np.zeros(n)
    front = np.unique(front_no)
    Fronts = front[front != np.inf]
    for f in range(len(Fronts)):
        Front = np.array([k for k in range(len(front_no)) if front_no[k] == Fronts[f]])
        Fmax = pop_obj[Front, :].max(0)
        Fmin = pop_obj[Front, :].min(0)
        for i in range(M):
            rank = np.argsort(pop_obj[Front, i])
            crowd_dis[Front[rank[0]]] = np.inf
            crowd_dis[Front[rank[-1]]] = np.inf
            for j in range(1, len(Front) - 1):
                crowd_dis[Front[rank[j]]] = crowd_dis[Front[rank[j]]] + (pop_obj[(Front[rank[j + 1]], i)] - pop_obj[
                    (Front[rank[j - 1]], i)]) / ((Fmax[i] - Fmin[i]) if Fmax[i] != Fmin[i] else 1.0)
    return crowd_dis



def environment_selection(pop_dec, pop_obj, pop_cstr, N):
    '''
    environmental selection in NSGA-II
    :param population: current population
    :param N: number of selected individuals
    :return: next generation population
    '''
    front_no, max_front = fnd_sort(pop_obj, pop_cstr)
    #front_no, max_front = nd_sort(pop_obj, pop_cstr)
    
    
    #for i in range(len(front_no)):
    #    if front_no[i] != front_no2[i]:
    #        print("Rank%d = %f:%f"%(i,front_no[i], front_no2[i])) 
    
    next_label = [False for i in range(front_no.size)]
    for i in range(front_no.size):
        if front_no[i] < max_front:
            next_label[i] = True
    crowd_dis = crowding_distance(pop_obj, front_no)
    last = [i for i in range(len(front_no)) if front_no[i]==max_front]
    rank = np.argsort(-crowd_dis[last])
    delta_n = rank[: (N - int(np.sum(next_label)))]
    rest = [last[i] for i in delta_n]
    for i in rest:
        next_label[i] = True
    index = np.array([i for i in range(len(next_label)) if next_label[i]])
    return pop_dec[index,:], pop_obj[index,:], pop_cstr[index,:], front_no[index], crowd_dis[index],index


def mutPolynomialBounded(offspring_dec, low, up, eta=20, indpb=0.3):
    """Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb.
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param low: A value or a :term:`python:sequence` of values that
                is the lower bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that
               is the upper bound of the search space.
    :returns: A tuple of one individual.
    """
    
    (n, d) = np.shape(offspring_dec)

    mut_pow = 1.0 / (eta + 1.)
   
    
    rand = np.random.random((n, d)) 
    site = np.random.random((n, d)) <= indpb
    
    delta_1 = (offspring_dec - low) / (up - low)
    val1 = 2.0 * rand + (1.0 - 2.0 * rand) * np.power(1 - delta_1, eta + 1)
    
    delta_2 = (up - offspring_dec) / (up - low)
    val2 = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * np.power(1 - delta_2, eta + 1)
    
    delta_q = (rand < 0.5)*(np.power(val1,mut_pow) - 1) + (rand >= 0.5)*(1-np.power(val2,mut_pow))

            
    offspring_dec = np.clip(offspring_dec + site*delta_q * (up - low), low,up)
    return offspring_dec
  
def cxSimulatedBinaryBounded(ind1, ind2, eta, low, up):
    """Executes a simulated binary crossover that modify in-place the input
    individuals. The simulated binary crossover expects :term:`sequence`
    individuals of floating point numbers.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param eta: Crowding degree of the crossover. A high eta will produce
                children resembling to their parents, while a small eta will
                produce solutions much more different.
    :param low: A value or a :term:`python:sequence` of values that is the lower
                bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that is the upper
               bound of the search space.
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    .. note::
       This implementation is similar to the one implemented in the
       original NSGA-II C code presented by Deb.
    """
    size = min(len(ind1), len(ind2))
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of the shorter individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of the shorter individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() <= 0.5:
            # This epsilon should probably be changed for 0 since
            # floating point arithmetic in Python is safer
            if abs(ind1[i] - ind2[i]) > 1e-14:
                x1 = min(ind1[i], ind2[i])
                x2 = max(ind1[i], ind2[i])
                rand = random.random()

                beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha = 2.0 - beta**-(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha)**(1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta + 1))

                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                alpha = 2.0 - beta**-(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha)**(1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha))**(1.0 / (eta + 1))
                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                c1 = min(max(c1, xl), xu)
                c2 = min(max(c2, xl), xu)

                if random.random() <= 0.5:
                    ind1[i] = c2
                    ind2[i] = c1
                else:
                    ind1[i] = c1
                    ind2[i] = c2

    return ind1, ind2



def generateChildren(pop_dec, rank, cdist, lower, upper):

    popSize = pop_dec.shape[0]

    idx1 = np.arange(popSize)
    idx2 = np.arange(popSize)
    
    np.random.shuffle(idx1)
    np.random.shuffle(idx2)
    
    bh, th = np.vstack((idx1[:popSize/2], idx2[:popSize/2])), np.vstack((idx1[popSize/2:], idx2[:popSize/2:]))
    
    rank_bh, rank_th = rank[bh], rank[th]
    cdist_bh, cdist_th = cdist[bh], cdist[th]
    
    same_rank = rank_bh == rank_th
    bh = bh * (rank_bh < rank_th) + same_rank*(cdist_bh > cdist_th)
    th = th * (rank_bh > rank_th) + same_rank*(cdist_bh < cdist_th) 
    
    tournament = th + bh
    
    
    
    
    
#     for (int i=0; i<popSize; i+=4)
#     {
#       parents.clear();
#       parents.add(Comparators.binaryTournament(population.get(idx1[i]), population.get(idx1[i+1]), Comparators.ConstrianedParetoDominanceWithCdist));
#       parents.add(Comparators.binaryTournament(population.get(idx1[i+2]), population.get(idx1[i+3]), Comparators.ConstrianedParetoDominanceWithCdist));
# 
#       List<S> childs = children.subList(i, i + 2);
# 
#       c.execute(parents, childs);
# 
#   
#       parents.clear();
#       parents.add(Comparators.binaryTournament(population.get(idx2[i]), population.get(idx2[i+1]), Comparators.ConstrianedParetoDominanceWithCdist));
#       parents.add(Comparators.binaryTournament(population.get(idx2[i+2]), population.get(idx2[i+3]), Comparators.ConstrianedParetoDominanceWithCdist));
# 
#       childs = children.subList(i+2, i + 4);  
#       c.execute(parents, childs);
#     }
# 
    
#    return mutPolynomialBounded(offspring_dec, lower, upper)




class Nsga2(object):
    """
    NSGA-II algorithm
    """
    def __init__(self, prob, decs=None, pop_size=100,  eva=100 * 500):
        self.prob = prob
        self.decs = decs
        self.eva = eva
        #pop-size
        self.N = pop_size
        

    def run(self):
        start = time.clock()
        if self.decs is None:
            pop_dec, pop_obj, pop_cstr = self.prob.initialize(self.N)
        else:
            pop_dec, pop_obj, pop_cstr = self.prob.individual(self.decs)

        front_no, max_front = fnd_sort(pop_obj, pop_cstr)
        crowd_dis = crowding_distance(pop_obj, front_no)
        evaluation = self.eva
        
        iter = int(evaluation / self.N) + 1  
        
        self.parents_dec = np.zeros((iter,self.N,self.prob.d))
        self.parents_obj = np.zeros((iter,self.N,self.prob.M))
        self.parents_cstr = np.zeros((iter,self.N,1) )
        
        self.children_dec = np.zeros((iter,self.N,self.prob.d))
        self.children_obj = np.zeros((iter,self.N,self.prob.M))
        self.children_cstr = np.zeros((iter,self.N,1 ))
        
        iter = 0
        while self.eva >= 0:
            fit = np.vstack((front_no, crowd_dis)).T
            mating_pool = tournament(2, self.N, fit)
            
            self.parents_dec[iter, ...] = pop_dec[mating_pool, :]
            self.parents_obj[iter, ...] = pop_obj[mating_pool, :]
            self.parents_cstr[iter, ...] = pop_cstr[mating_pool, :]
             
            offspring_dec, offspring_obj, offspring_cstr  = self.prob.individual(self.prob.variation(pop_dec[mating_pool, :]))
            
            self.children_dec[iter, ...] = offspring_dec
            self.children_obj[iter, ...] = offspring_obj
            self.children_cstr[iter, ...] = offspring_cstr
           
            iter = iter + 1
            pop_dec = np.vstack((pop_dec, offspring_dec))
            pop_obj = np.vstack((pop_obj, offspring_obj))
            pop_cstr = np.vstack((pop_cstr, offspring_cstr))
            
            pop_dec, pop_obj,pop_cstr, front_no, crowd_dis,_ = environment_selection(pop_dec, pop_obj, pop_cstr, self.N)
            self.eva = self.eva - self.N
            
            print('.',end='')
            if self.eva%(0.1*evaluation) == 0:
                end = time.clock()
                self.draw(pop_dec, pop_obj, pop_cstr)
                plt.show()
                print('Best gsum %10.2f'%(np.max(pop_cstr)))
                print('Running time %10.2f, percentage %s'%(end-start,100*(evaluation-self.eva)/evaluation))
        return pop_dec, pop_obj,  pop_cstr

    def draw(self, pop_dec, pop_obj, pop_cstr):
        front_no, max_front = fnd_sort(pop_obj, pop_cstr)
        non_dominated = pop_obj[front_no == 1, :]
        
           
        if self.prob.M == 2:
            plt.xlabel("%g"%(non_dominated[0,0]))
            plt.ylabel("%g"%(non_dominated[0,1]))
            
            
            plt.scatter(non_dominated[:,0]/non_dominated[0,0], non_dominated[:,1]/non_dominated[0,1])
        elif self.prob.M == 3:
            x, y, z = non_dominated[:, 0], non_dominated[:, 1], non_dominated[:, 2]
            ax = plt.subplot(111, projection='3d')
            ax.scatter(x, y, z, c='b')
        else:
            for i in range(len(non_dominated)):
                plt.plot(range(1, self.prob.M + 1), non_dominated[i, :])
        plt.show()

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
  
  circuit = VCAmpMOOProb(
    ng.Specifications(objective=[('idd', 1), ('gbw', -1) ], lt={'idd': 35e-5,'pm' : 90.0},gt=gt))
  a = Nsga2(circuit, eva=100*2 )

  pop_dec, pop_obj, pop_cstr = a.run()

  a.draw(pop_dec, pop_obj, pop_cstr )
  plt.show()

  print(circuit)

#   for iter, stats in sa.minimize(circuit): 
#     print("\r iter {}: {}".format(iter, stats))
# 
#   print(sa.best_state)
#   print(circuit.simulate(sa.best_state))
#   print(circuit.target.verify(circuit.simulate(sa.best_state)))



  
