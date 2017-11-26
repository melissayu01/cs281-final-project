import torch
from torch import optim, nn
from torch.autograd import Variable

class SS_AAE():
    def __init__(self, X_dim, y_dim, z_dim, h_dim,
                 recon_lr, D_lr, G_lr, ss_lr):
        self.X_dim = X_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.recon_lr = recon_lr
        self.D_lr  = D_lr
        self.G_lr  = G_lr
        self.ss_lr = ss_lr

        """ Models """
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

        # Decoder
        self.P = nn.Sequential(
            nn.Linear(z_dim + y_dim, h_dim),
            nn.ReLU(),
            nn.BatchNorm1d(h_dim),
            nn.Linear(h_dim, X_dim),
            nn.Sigmoid()
        )

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

        self.models = (self.Qz, self.Qy, self.P, self.Dz, self.Dy)

        """ Optimizers """
        self.recon_solver = optim.Adam(
            set(self.P.parameters())
            .union(set(self.Qz.parameters()))
            .union(set(self.Qy.parameters())),
            lr=recon_lr
        )
        self.D_solver = optim.Adam(
            set(self.Dz.parameters())
            .union(set(self.Dy.parameters())),
            lr=D_lr
        )
        self.G_solver = optim.Adam(
            set(self.Qz.parameters())
            .union(set(self.Qy.parameters())),
            lr=G_lr
        )
        self.ss_solver = optim.Adam(
            set(self.Qz.parameters())
            .union(set(self.Qy.parameters())),
            lr=ss_lr
        )
        self.optimizers = (
            self.recon_solver, self.D_solver, self.G_solver, self.ss_solver)

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
            ('recon_lr', self.recon_lr),
            ('D_lr' , self.D_lr),
            ('G_lr' , self.G_lr),
            ('ss_lr', self.ss_lr),
        )
