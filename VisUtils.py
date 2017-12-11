import os, errno
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from pandas.tools.plotting import parallel_coordinates

import seaborn
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class Vis2D:
    def __init__(self, epoch, n_classes, tag, out_dir='../results'):
        self.dim_reducer = None

        self.epoch = epoch
        self.n_classes = n_classes

        # make output directory if it does not exist
        self.out_dir = os.path.join(out_dir, tag)
        try:
            os.makedirs(self.out_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.cmap = plt.cm.get_cmap('jet', n_classes)
        self.norm = colors.BoundaryNorm(range(n_classes + 1), self.cmap.N)
        self.sc = None

    def update(self, X, y):
        if not self.dim_reducer:
            self.dim_reducer = PCA(n_components=2)
            self.dim_reducer.fit(X, y)
        X2d = self.dim_reducer.transform(X)
        self.sc = self.ax.scatter(
            X2d[:, 0], X2d[:, 1],
            c=y, cmap=self.cmap, norm=self.norm,
            alpha=0.8, s=10
        )

    def save(self, fname):
        if self.sc:
            cb = self.fig.colorbar(
                self.sc, ticks=np.arange(self.n_classes) + 0.5)
            cb.set_ticklabels(np.arange(self.n_classes))

            self.ax.set_title('Epoch {} - {}'.format(self.epoch, fname))
            self.fig.savefig(
                os.path.join(self.out_dir, '{}-{}.png'.format(
                    fname.replace(' ', ''), self.epoch))
            )
            plt.close(self.fig)
