

import numpy as np
import matplotlib.pyplot as plt

import stan_utils as stan


x, y, xerr, yerr = np.loadtxt("data.txt")

data_dict = dict(x=x, y=y, xerr=xerr, yerr=yerr, N=x.size)


model = stan.load_stan_model("line.stan")


p_opt = model.optimizing(data=data_dict)

samples = model.sampling(data=data_dict, chains=2, init=p_opt)

