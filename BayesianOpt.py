from math import sin
from math import pi
from numpy import arange, vstack, argmax, asarray
from numpy.random import normal, random
from scipy.stats import norm
import scipy
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
                    GP=GaussianProcessRegressor(),
                    plot_dir="./output/plots/",
                    bounds = [0,1]
                ):
        '''
            Class to implement bayesian Optimization

            Parameters
            -----------
            eps: float 
                Parameter to control the exploration in Expected improvement calculation 
            
            GP: sklearn.gaussian_process 
                Gaussian process to be used as a surrogate
            
            plot_dir: str
                Directory to store the outputs

            bounds: list
                Bounds for optimization

        '''

        self.eps = eps
        self.div = 2
        self.eps_min = 1e-3
        self.model = GP
        self.plot_dir = plot_dir
        self.bounds = bounds

    def decay_eps(self):
        '''
            Function to decay the epsilon value based on the specified initial value and division 
            factor in the class definition
        '''

        if self.eps >= self.eps_min:
            self.eps = self.eps / self.div

        else:
            self.eps = self.eps_min

        print(f"Epsilon ---> {self.eps}")

    def surrogate(self, X):
        '''
            Surrogate approximation of the objective function using a Gaussian Process initialized 
            in the class definition

            Parameters
            -----------
            X : N x 1 
                Array of sampled for which surrogate function outputs need to be calculated

            Returns
            -------
            Tuple
                Mean and standard deviation of the gaussian estimate of the samples
            

        '''

        with catch_warnings():
            simplefilter("ignore")
            return self.model.predict(X, return_std=True)

    # Expected Improvement acquisition function

    def _acquisition(self, X, X_samples):
        '''
            Acquisition function using hte Expected Improvement method

            Parameters
            -----------
            X : N x 1 
                Array of parameter points

            X_samples : N x 1
                Array of Sampled points between the bounds

            Returns
            --------
            float
                Expected improvement

        '''

        # calculate the best surrogate score found so far
        y_hat, y_std = self.surrogate(X)
        y_max = max(y_hat)

        # calculate mean and sigma via surrogate function
        mu, std = self.surrogate(X_samples)
        mu = mu[:, 0]

        # calculate the improvement
        with np.errstate(divide='warn'):
            z = (mu - y_max - self.eps) / std
            EI = (mu - y_max - self.eps) * norm.cdf(z) + std * norm.pdf(z)
            EI[std == 0.0] = 0

        return EI

    def optimize_acq(self, X, y):
        '''
            Optimization of the Acquisition function using a maximization check of the outputs

            Parameters
            -----------
            X : N x 1 
                Array of parameter points

            y : N x 1
                Array of Observation points corresponding to the parameter points

            Returns
            --------
            float
                Next location of the sampling point based on the Maximization

        '''


        # Get random samples
        X_samples = asarray(arange(self.bounds[0], self.bounds[1], 0.001))
        X_samples = X_samples.reshape(len(X_samples), 1)

        # Calculate Acquisition value for each sample
        scores = self._acquisition(X, X_samples)

        # Get the index of the largest Score
        max_index = argmax(scores)

        return X_samples[max_index, 0]

    def _min_obj(self, X, X_samples):
        
        return -self._acquisition(X, X_samples)
    
    
    def suggest_next(self, X, y, n_starts=25):
        
        start_points_dict = [self.surrogate(X) for i in range(n_restarts)]
        start_points_arr = np.random.uniform(self.bounds[0], self.bounds[1], size=(n_starts,1))
        
        x_best = np.empty((n_starts,))
        f_best = np.empty((n_starts,))

        for index, start_point in enumerate(start_points_arr):

            print(start_point)
            
            res = scipy.optimize.minimize(
                min_obj, 
                # x0=np.array([start_point]), 
                # method='L-BFGS-B',
                bounds=self.bounds
                )
            x_best[index], f_best[index] = res.x, np.atleast_1d(res.fun)[0]
        
        print(x_best[np.argmin(f_best)])
        return x_best[np.argmin(f_best)]

    
    def plot(self, X, y, X_next, iteration):
        '''
            Function to plot observations, posterior mean, uncertainty, surrogate 
            and next sampling point

            Parameters
            -----------
            X : N x 1 
                Array of parameter points

            y : N x 1
                Array of Observation points corresponding to the parameter points

            X_next: float
                next sampling point 

            iteration: int
                Current iteration


        '''

        plt.figure(figsize=(10, 5))
        clrs = sns.color_palette("husl", 5)

        # Plot the observation
        plt.scatter(X, y)

        # line plot of surrogate function across domain
        X_samples = asarray(arange(self.bounds[0], self.bounds[1], 0.001))
        X_samples = X_samples.reshape(len(X_samples), 1)
        y_samples, std = self.surrogate(X_samples)
        plt.plot(X_samples, y_samples, label='Surrogate Posterior')
        plt.axvline(x=X_next, ls='--', c='k', lw=1,
                    label='Next sampling location')
        plt.fill_between(   X_samples.ravel(),
                            y_samples.ravel() + 1.96 * std,
                            y_samples.ravel() - 1.96 * std,
                            alpha=0.1,
                            facecolor=clrs[3])

        scores = self._acquisition(X,  X_samples)

        plt.plot(X_samples, scores*2, label='Acquisition function')

        # Set plot metrics
        plt.xlim([0, 1])
        plt.ylim([-0.2, 1.5])
        plt.title(f"Plot for Iteration -- {iteration}")
        plt.xlabel("Learning Rate")
        plt.ylabel("Balanced Accuracy")
        plt.legend()
        plt.savefig(self.plot_dir + f"Plot_iter_{iteration}.png")
        # plt.show()
