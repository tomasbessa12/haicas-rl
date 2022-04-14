#!/usr/bin/env python
# encoding: utf-8
import numpy as np

class Func1(object):
    """
    The problem related parameters and genetic operations
    """
    def __init__(self, d=10, M=2, lower=-np.ones((1, 10)), upper=np.ones((1, 10))):
        self.d = d
        self.M = M
        self.upper = upper
        self.lower = lower

    def cost_fun(self, x):
        """
        calculate the objective vectors
        :param x: the decision vectors
        :return: the objective vectors
        """
        n = x.shape[0]
        a = np.zeros((self.M, self.d))
        
        for i in range(self.d):
            for j in range(self.M):
                a[j,i] = ((i+0.5)**(j-0.5))/(i+j+1.)
        obj = np.zeros((n, self.M))
        cstr = np.zeros((n, 1))
        for i in range(n):
            for j in range(self.M):
                obj[i, j] = np.dot(x[i, :] ** (j + 1), a[j, :].T)
        return obj, cstr

    def individual(self, pop_vars):
        """
        turn decision vectors into individuals
        :param pop_vars: decision vectors
        :return: (pop_vars, pop_obj, pop_cstr)
        """
        pop_obj, pop_cstr = self.cost_fun(pop_vars)
        return (pop_vars, pop_obj, pop_cstr)

    def initialize(self, N):
        """
        initialize the population
        :param N: number of elements in the population
        :return: the initial population
        """
        pop_dec = np.random.random((N, self.d)) * (self.upper - self.lower) + self.lower
        return self.individual(pop_dec)

    def variation(self, pop_dec):
        """
        Generate offspring individuals
        :param boundary: lower and upper boundary of pop_dec once d != self.d
        :param pop_dec: decision vectors
        :return: 
        """
        boundary=(self.lower,self.upper)
        pro_c = 0.9
        dis_c = 10
        pro_m = 0.3
        dis_m = 20
        pop_dec = pop_dec[:(len(pop_dec) // 2) * 2][:]
        (n, d) = np.shape(pop_dec)
        parent_1_dec = pop_dec[:n // 2, :]
        parent_2_dec = pop_dec[n // 2:, :]
        beta = np.zeros((n // 2, d))
        mu = np.random.random((n // 2, d))
        beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1 / (dis_c + 1))
        beta[mu > 0.5] = np.power(2 * mu[mu > 0.5], -1 / (dis_c + 1))
        beta = beta * ((-1)** np.random.randint(2, size=(n // 2, d)))
        beta[np.random.random((n // 2, d)) < 0.5] = 1
        beta[np.tile(np.random.random((n // 2, 1)) > pro_c, (1, d))] = 1
        offspring_dec = np.vstack(((parent_1_dec + parent_2_dec) / 2 + beta * (parent_1_dec - parent_2_dec) / 2,
                                   (parent_1_dec + parent_2_dec) / 2 - beta * (parent_1_dec - parent_2_dec) / 2))
        site = np.random.random((n, d)) < pro_m
        mu = np.random.random((n, d))
        temp = site & (mu <= 0.5)
        if boundary is None:
            lower, upper = np.tile(self.lower, (n, 1)), np.tile(self.upper, (n, 1))
        else:
            lower, upper = np.tile(boundary[0], (n, 1)), np.tile(boundary[1], (n, 1))

        norm = (offspring_dec[temp] - lower[temp]) / (upper[temp] - lower[temp])
        offspring_dec[temp] += (upper[temp] - lower[temp]) * \
                               (np.power(2. * mu[temp] + (1. - 2. * mu[temp]) * np.power(1. - norm, dis_m + 1.),
                                         1. / (dis_m + 1)) - 1.)
        temp = site & (mu > 0.5)
        norm = (upper[temp] - offspring_dec[temp]) / (upper[temp] - lower[temp])
        offspring_dec[temp] += (upper[temp] - lower[temp]) * \
                               (1. - np.power(
                                   2. * (1. - mu[temp]) + 2. * (mu[temp] - 0.5) * np.power(1. - norm, dis_m + 1.),
                                   1. / (dis_m + 1.)))
        offspring_dec = np.maximum(np.minimum(offspring_dec, upper), lower)
        return offspring_dec
