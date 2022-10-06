#%%
from scipy.special import factorial
import numpy as np



# Class for Angular-Central-Gaussian spherical ditribution
#Based on: "Tyler 1987 - Statistical analysis for the angular central Gaussian distribution on the sphere"
class AngularCentralGaussian:
    def __init__(self,p):
        self.p = p
        assert self.p % 2 == 0, 'P must be an even positive integer'
        self.half_p = int(p/2)

    def surfaceArea(self) -> float:
        SurfA = (2*np.pi**self.half_p)/factorial(self.half_p-1)
        return SurfA

    # Propability Density function
    def pdf(self,x,A) -> float:

        ACG_pdf = np.linalg.det(A)**-0.5 * (x.T @ np.linalg.inv(A) @ x)

        return ACG_pdf
    # With surfarea division
    def pdf_a(self, x, A):
        ACG_pdf_A = ((np.linalg.det(A) ** -0.5) * (x.T @ np.linalg.inv(A) @ x))/self.surfaceArea(self.p)
        return ACG_pdf_A

    def likelihood(self,sample_X, A):
        n = len(sample_X[0])



        L = np.linalg.det(A)**(-0.5*n) *
        return L

    def log_likelihood(self, sample_X, A):


    # Log Likelihood
    def log_likelihood(self,mu,k,X):
        n = len(X[0])
        y = 0
        likelihood = n * (k * mu.T @ self.Scatter_matrix(X) @ mu - np.log(self.M(1/2,self.p/2,k)) + y)
        return likelihood
    
    # Estimation of Scatter Matrix
    def Scatter_matrix(self,X):
        S = np.zeros((self.p,self.p))
        for x in X.T:
            S += np.outer(x,x)
        return S





