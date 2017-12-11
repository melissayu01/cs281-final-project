import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

class SS_AAE():
    def __init__(self, X_dim, y_dim, z_dim, h_dim, Q_lr, P_lr, D_lr):
        self.X_dim = X_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.Q_lr  = Q_lr
        self.P_lr  = P_lr
        self.D_lr  = D_lr

        # Encoder
        self._Q = nn.Sequential(
            nn.Linear(X_dim, h_dim),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim),
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

        # Decoder
        self.P = nn.Sequential(
            nn.Linear(z_dim + y_dim, h_dim),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim),
            nn.Linear(h_dim, X_dim),
            nn.Sigmoid(),
        )

        # Discriminator
        self.Dz = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim),
            nn.Linear(h_dim, 1),
            nn.Sigmoid(),
        )
        self.Dy = nn.Sequential(
            nn.Linear(y_dim, h_dim),
            # nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim),
            nn.Linear(h_dim, 1),
            nn.Sigmoid(),
        )

        # optimizers
        self.Q_solver = optim.Adam(
            set(self.Qz.parameters()).union(set(self.Qy.parameters())),
            lr=Q_lr
        )
        self.P_solver = optim.Adam(
            self.P.parameters(),
            lr=P_lr
        )
        self.D_solver = optim.Adam(
            set(self.Dz.parameters()).union(set(self.Dy.parameters())),
            lr=D_lr
        )

        # schedulers
        self.Q_scheduler = ReduceLROnPlateau(
            self.Q_solver, 'min', patience=1, threshold=0.01, verbose=True
        )
        self.P_scheduler = ReduceLROnPlateau(
            self.P_solver, 'min', patience=1, threshold=0.01, verbose=True
        )
        self.D_scheduler = ReduceLROnPlateau(
            self.D_solver, 'min', patience=2, threshold=0.01, verbose=True
        )

        self.models = (self.Qz, self.Qy, self.P, self.Dz, self.Dy)
        self.solvers = (self.Q_solver, self.P_solver, self.D_solver)
        self.schedulers = (
            self.Q_scheduler, self.P_scheduler, self.D_scheduler)

    def step_schedulers(self, loss):
        for scheduler in self.schedulers:
            scheduler.step(loss)

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
            ('Q_lr' , self.Q_lr),
            ('P_lr' , self.P_lr),
            ('D_lr' , self.D_lr),
        )
