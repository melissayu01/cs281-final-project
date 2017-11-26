import torch
from torch import optim, nn
from torch.autograd import Variable

class SS_AAE():
    def __init__(self, X_dim, y_dim, z_dim, h_dim, lr=1e-3):
        self.X_dim = X_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.lr    = lr

        # Encoder
        self._Q = nn.Sequential(
            nn.Linear(X_dim, h_dim),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim)
            # nn.Linear(h_dim, h_dim),
            # nn.ReLU(),
        )
        self.Qz = nn.Sequential(
            self._Q,
            nn.Linear(h_dim, z_dim)
        )
        self.Qy = nn.Sequential(
            self._Q,
            nn.Linear(h_dim, y_dim),
            nn.Softmax()
        )
        self.Q_solver = optim.Adam(
            set(self.Qz.parameters()).union(set(self.Qy.parameters())),
            lr=lr
        )

        # Decoder
        self.P = nn.Sequential(
            nn.Linear(z_dim + y_dim, h_dim),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim),
            nn.Linear(h_dim, X_dim),
            nn.Sigmoid()
        )
        self.P_solver = optim.Adam(self.P.parameters(), lr=lr)

        # Discriminator
        self.Dz = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim),
            # nn.Linear(h_dim, h_dim),
            # nn.ReLU(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )
        self.Dy = nn.Sequential(
            nn.Linear(y_dim, h_dim),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim),
            # nn.Linear(h_dim, h_dim),
            # nn.ReLU(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )
        self.D_solver = optim.Adam(
            set(self.Dz.parameters()).union(set(self.Dy.parameters())),
            lr=lr
        )

        self.models = (self.Qz, self.Qy, self.P, self.Dz, self.Dy)

    def zero_grad(self):
        for model in self.models:
            model.zero_grad()

    def cuda(self):
        for model in self.models:
            model.cuda()

    def train(self):
        for model in self.models:
            model.train()

    def eval(self):
        for model in self.models:
            model.eval()

    def parameters(self):
        return (
            ('X dim', self.X_dim),
            ('y dim', self.y_dim),
            ('z dim', self.z_dim),
            ('h dim', self.h_dim),
            ('lr'   , self.lr)
        )
