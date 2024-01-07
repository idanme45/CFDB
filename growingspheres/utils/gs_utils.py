#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import pairwise_distances

import cf_generator


def get_distances(x1, x2, metrics=None):
    x1, x2 = x1.reshape(1, -1), x2.reshape(1, -1)
    euclidean = pairwise_distances(x1, x2)[0][0]
    same_coordinates = sum((x1 == x2)[0])
    
    #pearson = pearsonr(x1, x2)[0]
    kendall = kendalltau(x1, x2)
    out_dict = {'euclidean': euclidean,
                'sparsity': x1.shape[1] - same_coordinates,
                'kendall': kendall
               }
    return out_dict        


def generate_ball(enc, expression, outcome_name, center, r, n):
    def norm(v):
        return np.linalg.norm(v, ord=2, axis=1)
    d = center.shape[1]
    u = np.random.normal(0, 4, (n, d+2))  # an array of (d+2) normally distributed random variables
    norm_ = norm(u)
    u = 1/norm_[:,None]* u
    x = u[:, 0:d] * r #take the first d coordinates
    x = x + center
    x1 = np.apply_along_axis(lambda a: fix_cf(a, enc, False), 1, x)
    x2 = np.apply_along_axis(lambda a: fix_cf(a, enc, True), 1, x)
    x = np.concatenate((x1, x2), axis=0)
    # remain only cf with norm less then r
    if x is not None:
        mask = np.apply_along_axis(lambda a: norm(a - center), 1, x) < r
        #norms = np.apply_along_axis(lambda a: norm(a - center), 1, x)
        x = x[mask.reshape(-1)]
    return x

def fix_cf(cf, enc, is_bigger):
    for i in range(len(enc.features)):
        if i < len(enc.columns):
            cf[i] = enc.closest(i, cf[i], bigger=is_bigger)
        elif enc.is_integer(i):
            if int(cf[i]) == cf[i]:
                if is_bigger:
                    cf[i] = cf[i]+1
                else:
                    cf[i] = cf[i]+1
            else:
                if is_bigger:
                    cf[i] = int(cf[i]+1)
                else:
                    cf[i] = int(cf[i]- 1)
    return cf

def generate_ring(center, segment, n):
    def norm(v):
        return np.linalg.norm(v, ord=2, axis=1)
    d = center.shape[1]
    z = np.random.normal(0, 1, (n, d))
    try:
        u = np.random.uniform(segment[0]**d, segment[1]**d, n)
    except OverflowError:
        raise OverflowError("Dimension too big for hyperball sampling. Please use layer_shape='sphere' instead.")
    r = u**(1/float(d))
    z = np.array([a * b / c for a, b, c in zip(z, r,  norm(z))])
    z = z + center
    return z

def generate_sphere(center, r, n):    
    def norm(v):
            return np.linalg.norm(v, ord=2, axis=1)
    d = center.shape[1]
    z = np.random.normal(0, 1, (n, d))
    z = z/(norm(z)[:, None]) * r + center
    return z