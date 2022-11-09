import torch
import torch.nn as nn
import numpy as np


class Watson(nn.Module):
    """
    Logarithmic Multivariate Watson distribution class
    """

    def __init__(self, p):
        super().__init__()

        self.p = p
        self.mu = nn.Parameter(torch.rand(self.p))
        self.kappa = nn.Parameter(torch.tensor([1.]))
        self.SoftPlus = nn.Softplus(beta=20, threshold=1)  # Log?
        self.const_a = torch.tensor(0.5)  # a = 1/2,  !constant

    def set_param(self, set_mu, set_kappa):
        assert len(set_mu) == self.p, f'Diagonal tensor most be {self.p} long'
        self.mu = nn.Parameter(set_mu)
        self.kappa = nn.Parameter(set_kappa)

    def log_kummer(self, a, b, kappa):

        # konvergens criterie
        n = torch.arange(1000)  # precicion order

        inner = torch.lgamma(a + n) + torch.lgamma(b) - torch.lgamma(a) - torch.lgamma(b + n) \
                + n * torch.log(kappa) - torch.lgamma(n + torch.tensor(1))

        logkum = torch.logsumexp(inner, dim=0)
        return logkum

    def log_sphere_surface(self):
        logSA = torch.log(torch.tensor(2 * np.pi ** (self.p / 2))) / torch.lgamma(torch.tensor(self.p / 2))
        return logSA

    def log_norm_constant(self):
        # logC = torch.lgamma(torch.tensor(self.p / 2)) - torch.log(torch.tensor(2 * np.pi ** (self.p / 2))) \
        #        - self.log_kummer(self.const_a, torch.tensor(self.p / 2), self.kappa)  # addiction kummer last part?

        logC = 1 / self.log_sphere_surface() - self.log_kummer(self.const_a, torch.tensor(self.p / 2), self.kappa)

        return logC

    def log_pdf(self, X):
        # Constraints
        kappa_positive = self.SoftPlus(self.kappa)  # Log softplus?
        mu_unit = nn.functional.normalize(self.mu, dim=0)  ##### Sufficent for backprop?
        # print(f'Norm of mu:{mu_unit.norm()}')
        # assert torch.abs(mu_unit.norm() - 1.) > 1.e-3, "mu is not properly normalized"

        # log PDF
        if self.p == 1:
            logpdf = self.log_norm_constant() + kappa_positive * (mu_unit * X) ** 2
        else:
            logpdf = self.log_norm_constant() + kappa_positive * (mu_unit @ X.T) ** 2
        return logpdf

    def forward(self, X):
        return self.log_pdf(X)


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.use('Qt5Agg')

    W = Watson(p=2)
    # phi = linspace(0, 2 * pi, 320);
    # x = [cos(phi);sin(phi)];
    #
    # _, inner = W.log_kummer(torch.tensor(0.5), torch.tensor(3/2), torch.tensor())

    phi = torch.arange(0, 2 * np.pi, 0.001)
    phi_arr = np.array(phi)
    x = torch.column_stack((torch.cos(phi), torch.sin(phi)))

    points = torch.exp(W(x))
    props = np.array(points.squeeze().detach())

    # props = props/np.max(props)

    ax = plt.axes(projection='3d')
    ax.plot(np.cos(phi_arr), np.sin(phi_arr), 0, 'gray')
    ax.scatter(np.cos(phi_arr), np.sin(phi_arr), props, c=props, cmap='viridis', linewidth=0.5)

    ax.view_init(30, 135)
    plt.show()
    # plt.show()
    # plt.scatter(phi,props, s=3)
    # plt.show()
