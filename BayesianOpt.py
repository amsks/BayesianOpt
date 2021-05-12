import scipy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import seaborn as sns


from warnings import catch_warnings
from warnings import simplefilter
from pathlib import Path
import pdb

class BayesianOpt:
    def __init__(self,
                    eps=1,
                    plot_dir="./output/plots/",
                    bounds = (0,1),
                    model=GaussianProcessRegressor(),
                ):
        '''
            Class to implement Bayesian Optimization

            Parameters
            -----------
            eps: float 
                Parameter to control the exploration in Expected improvement calculation 
            
            GP: sklearn.gaussian_process 
                Gaussian process to be used as a surrogate
            
            plot_dir: str
                Directory to store the outputs

            bounds: Tuple
                Bounds for optimization

        '''

        self.eps = eps
        self.model = model
        self.plot_dir = plot_dir
        self.bounds = bounds

        self.X_samples_ = np.asarray(np.arange(self.bounds[0], self.bounds[1], 0.001))
        self.X_samples_ = self.X_samples_.reshape(len(self.X_samples_), 1)


    def decay_eps(self):
        '''
            Function to decay the epsilon value based on the specified initial value and division 
            factor in the class definition
        '''

        div_ = 2 
        eps_min_ = 1e-3 

        if self.eps >= eps_min_:
            self.eps = self.eps / div_

        else:
            self.eps = eps_min_

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

    def _acquisition(self, X, samples):
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

        # calculate the max of surrogate values from history
        mu_x_, _ = self.surrogate(X)
        max_x_ = max(mu_x_)

        # Get the mean and deviation of the samples 
        mu_sample_, std_sample_ = self.surrogate(samples)
        mu_sample_ = mu_sample_[:, 0]

        # Get the improvement
        with np.errstate(divide='warn'):
            z = (mu_sample_ - max_x_ - self.eps) / std_sample_
            EI_ = (mu_sample_ - max_x_ - self.eps) * \
                scipy.stats.norm.cdf(z) + std_sample_ * scipy.stats.norm.pdf(z)
            EI_[std_sample_ == 0.0] = 0

        return EI_

    # TODO: Add n_starts optimization
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
        

        # Calculate Acquisition value for each sample
        scores_ = self._acquisition(X, self.X_samples_)

        # Get the index of the largest Score
        max_index_ = np.argmax(scores_)

        return self.X_samples_[max_index_, 0]
    
    def plot(self, X, y, X_next, iteration, show=False):
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
        colors_ = sns.color_palette("husl", 5)

        # Plot the observation
        plt.scatter(X, y)

        # line plot of surrogate function across domain
        # X_samples_ = np.asarray(np.arange(self.bounds[0], self.bounds[1], 0.001))
        # X_samples_ = X_samples_.reshape(len(X_samples_), 1)
        y_samples_, std_samples_ = self.surrogate(self.X_samples_)
        plt.plot(self.X_samples_, y_samples_, label='Surrogate Posterior')
        plt.axvline(x=X_next, ls='--', c='k', lw=1,
                    label='Next sampling location')
        plt.fill_between(   self.X_samples_.ravel(),
                            y_samples_.ravel() + 1.96 * std_samples_,
                            y_samples_.ravel() - 1.96 * std_samples_,
                            alpha=0.1,
                            facecolor=colors_[3])

        scores = self._acquisition(X,  self.X_samples_)

        plt.plot(self.X_samples_, scores*2, label='Acquisition function')

        # Set plot metrics
        plt.xlim([0, 1])
        plt.ylim([-0.2, 1.5])
        plt.title(f"Plot for Iteration -- {iteration}")
        plt.xlabel("Learning Rate")
        plt.ylabel("Balanced Accuracy")
        plt.legend()
        plt.savefig(self.plot_dir + f"Plot_iter_{iteration}.png")
        
        if show:
            plt.show()
    

