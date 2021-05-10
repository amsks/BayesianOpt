from math import sin
from math import pi
from numpy import arange, vstack, argmax, asarray
from numpy.random import normal, random
from scipy.stats import norm
from scipy.optimize import minimize
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from pathlib import Path

class BayesianOpt:
    def __init__(self,
                 eps=1,
                 div=2,
                 eps_min=1e-3,
                 GP=GaussianProcessRegressor(),
                 plot_dir="./output/plots/"
                 ):

        self.eps = eps
        self.div = div
        self.eps_min = eps_min
        self.model = GP
        self.plot_dir = plot_dir

    def decay_eps(self):

        # decay_value = self.eps_init/np.abs(self.max_iter)

        if self.eps >= self.eps_min:
            self.eps = self.eps / self.div

        else:
            self.eps = self.eps_min

        print(f"Epsilon ---> {self.eps}")

    # Surrogate or approximation for the objective function
    # using a particular Gaussian Process
    def surrogate(self, X):

        with catch_warnings():
            simplefilter("ignore")
            return self.model.predict(X, return_std=True)

    # Expected Improvement acquisition function

    def acquisition(self, X, Xsamples):

        # calculate the best surrogate score found so far
        yhat, y_std = self.surrogate(X)
        tau = max(yhat)

        # calculate mean and stdev via surrogate function
        mu, std = self.surrogate(Xsamples)
        mu = mu[:, 0]

        # calculate the improvement
        with np.errstate(divide='warn'):
            z = (mu - tau - self.eps) / std
            EI = (mu - tau - self.eps) * norm.cdf(z) + std * norm.pdf(z)
            EI[std == 0.0] = 0

        return EI

    # optimize the acquisition function
    def optimize_acq(self, X, y):

        # Get random samples
        Xsamples = asarray(arange(0, 1, 0.001))
        Xsamples = Xsamples.reshape(len(Xsamples), 1)

        # Calculate Acquisition value for each sample
        scores = self.acquisition(X, Xsamples)

        # Get the largest Score
        ix = argmax(scores)

        return Xsamples[ix, 0]

    # plot observations, posterior mean, uncertainty, surrogate and next sampling pint

    def plot(self, X, y, X_next, iteration):

        plt.figure(figsize=(10, 5))

        # scatter plot of inputs and real objective function
        plt.scatter(X, y)

        clrs = sns.color_palette("husl", 5)
        # line plot of surrogate function across domain
        Xsamples = asarray(arange(0, 1, 0.001))
        Xsamples = Xsamples.reshape(len(Xsamples), 1)
        ysamples, std = self.surrogate(Xsamples)
        plt.plot(Xsamples, ysamples, label='Surrogate Posterior')
        plt.axvline(x=X_next, ls='--', c='k', lw=1,
                    label='Next sampling location')
        plt.fill_between(Xsamples.ravel(),
                         ysamples.ravel() + 1.96 * std,
                         ysamples.ravel() - 1.96 * std,
                         alpha=0.1,
                         facecolor=clrs[3])

        scores = self.acquisition(X,  Xsamples)

        plt.plot(Xsamples, scores*2, label='Acquisition function')

        # Set plot metrics
        plt.xlim([0, 1])
        plt.ylim([-0.2, 1.5])
        plt.title(f"Plot for Iteration -- {iteration}")
        plt.legend()
        plt.savefig(self.plot_dir + f"Plot_iter_{iteration}.png")
        # plt.show()
