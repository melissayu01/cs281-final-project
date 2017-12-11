import sys
import argparse
import numpy as np

import torch
from torch.nn import functional as F
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, roc_auc_score

import DataUtils, VisUtils
from SS_AAE import SS_AAE

np.set_printoptions(precision=4, suppress=True)
torch.set_printoptions(precision=4)

TINY = 1e-10

def onehot(values, n_classes):
    enc = np.eye(n_classes)[values].astype('float32')
    return torch.from_numpy(enc)

def sample_categorical(batch_size, n_classes):
    '''
     Sample from a categorical distribution
     of size batch_size and # of classes n_classes
     return: torch.Tensor with the sample
    '''
    values = np.random.randint(0, high=n_classes, size=batch_size)
    return onehot(values, n_classes=n_classes)

def train(epoch, model, data_loader, args):
    model.train()

    n_batches = len(data_loader)
    n_unlabeled_used = 0
    n_labeled_used = 0
    labeled_interval = n_batches * args.labeled_size // args.n_labeled

    for batch_idx, (X, y) in enumerate(data_loader):
        X, y = Variable(X), Variable(y)
        if args.cuda:
            X, y = X.cuda(), y.cuda()
        model.zero_grad()

        batch_sz = min(args.batch_size, len(X))
        unlabeled_sz = batch_sz - args.labeled_size
        if unlabeled_sz < 0 or batch_idx % labeled_interval != 0:
            unlabeled_sz = batch_sz

        ### unlabeled examples
        k = 1
        n_unlabeled_used += unlabeled_sz // k
        Xu = X[:unlabeled_sz // k]

        """ Reconstruction phase """
        latent_sample = torch.cat((model.Qz(Xu), model.Qy(Xu)), dim=1)
        X_sample = model.P(latent_sample)

        recon_loss = F.binary_cross_entropy(X_sample, Xu)

        recon_loss.backward()
        model.Q_solver.step()
        model.P_solver.step()
        model.zero_grad()

        """ Regularization phase """
        # Discriminator
        model.eval()

        z_real = Variable(torch.randn(unlabeled_sz // k, model.z_dim))
        y_real = Variable(sample_categorical(unlabeled_sz // k, model.y_dim))
        if args.cuda:
            z_real = z_real.cuda()
            y_real = y_real.cuda()
        z_fake = model.Qz(Xu)
        y_fake = model.Qy(Xu)

        Dz_real = model.Dz(z_real)
        Dy_real = model.Dy(y_real)
        Dz_fake = model.Dz(z_fake)
        Dy_fake = model.Dy(y_fake)

        Dz_loss = -((Dz_real + TINY).log()
                    + (1 - Dz_fake + TINY).log()).mean()
        Dy_loss = -((Dy_real + TINY).log()
                    + (1 - Dy_fake + TINY).log()).mean()
        D_loss  = Dz_loss + Dy_loss

        D_loss.backward()
        model.D_solver.step()
        model.zero_grad()

        # Generator
        model.train()

        z_fake = model.Qz(Xu)
        y_fake = model.Qy(Xu)

        Dz_fake = model.Dz(z_fake)
        Dy_fake = model.Dy(y_fake)

        Gz_loss = -(Dz_fake + TINY).log().mean()
        Gy_loss = -(Dy_fake + TINY).log().mean()
        G_loss  = 0.05 * (Gz_loss + Gy_loss)

        G_loss.backward()
        model.Q_solver.step()
        model.zero_grad()

        ### labeled
        if unlabeled_sz != batch_sz:
            Xl, yl = X[unlabeled_sz:], y[unlabeled_sz:].long()
            n_labeled_used += len(yl)

            class_loss = 20 * F.cross_entropy(
                model.Qy(Xl), yl,
                weight=torch.Tensor([0.6078, 0.3922])
            )

            class_loss.backward()
            model.Q_solver.step()
            model.zero_grad()

        # logging
        if batch_idx % args.log_interval == 0 or batch_idx == n_batches-1:
            endchar = '\n' if batch_idx == n_batches-1 else '\r'
            print((
                'Train Epoch: {:2} [{:6}/{:6} ({:2.0f}%)] <{:3}:{:5}>\t ' +
                'class_loss: {:7.4f} D_loss: {:7.4f} ' +
                'G_loss: {:7.4f} recon_loss: {:7.4f}'
            ).format(
                epoch,
                batch_idx * args.batch_size + len(X),
                len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                n_labeled_used, n_unlabeled_used,
                class_loss.data[0], D_loss.data[0],
                G_loss.data[0], recon_loss.data[0],
            ), end=endchar, flush=True)

def test(epoch, model, data_loader, args):
    model.eval()

    n_batches = len(data_loader)
    vis_real = VisUtils.Vis2D(epoch, n_classes=model.y_dim, tag=args.tag)
    vis_pred = VisUtils.Vis2D(epoch, n_classes=model.y_dim, tag=args.tag)
    test_loss = 0
    n_unknown, n_unknown_correct = 0, 0
    cm = np.zeros((2, 2))
    pred_y, true_y = [], []

    print('Testing...', end='\r')
    for batch_idx, (X, y) in enumerate(data_loader):
        X, y = Variable(X, volatile=True), Variable(y.long(), volatile=True)
        if args.cuda:
            X, y = X.cuda(), y.cuda()

        # get predictions
        probs = model.Qy(X)
        if args.labeled_size > 0:
            _, preds = probs.max(dim=1)
            preds = preds.data.numpy()
        else: # cluster on y
            probs, clusters = probs.max(dim=1)
            cluster_to_pred = np.zeros(model.y_dim)
            for i in range(model.y_dim):
                if i in clusters.data:
                    mask = (clusters == i).data
                    top_probs, sample = probs[mask].topk(1)
                    preds, _ = y.unsqueeze(1).index_select(
                        dim=0, index=sample).mode(dim=1)
                    cluster_to_pred[i] = preds.data[0]
            preds = np.apply_along_axis(
                lambda c: cluster_to_pred[c], 0,
                clusters.data.numpy()
            )

        # convert preds to binary classes
        bin_y = (y >= 1).int().data.numpy()
        bin_preds = (preds > 0).astype('int')

        # update stats
        if np.random.random() < 0.5:
            pred_y.append(bin_preds)
            true_y.append(bin_y)

            # plot
            z = model.Qz(X)
            vis_real.update(z.data.numpy(), y.data.numpy())
            vis_pred.update(z.data.numpy(), preds)

        n_unknown_correct += np.logical_and(
            bin_preds == bin_y, y.data.numpy() == 2).sum()
        n_unknown += (y.data.numpy() == 2).sum()

        test_loss += (bin_preds != bin_y).sum()
        cm += confusion_matrix(bin_y, bin_preds, labels=range(2))

        # logging
        if batch_idx % 10 == 0 or batch_idx == n_batches-1:
            endchar = '\n' if batch_idx == n_batches-1 else '\r'
            print((
                'Test Epoch : {:2} [{:6}/{:6} ({:2.0f}%)]\t'
            ).format(
                epoch,
                batch_idx * args.batch_size + len(X),
                len(data_loader.dataset),
                100. * batch_idx / len(data_loader)
            ), end=endchar, flush=True)

    vis_real.save('True-{}:{}'.format(model.y_dim, model.z_dim))
    vis_pred.save('Pred-{}:{}'.format(model.y_dim, model.z_dim))

    test_loss /= len(data_loader.dataset)
    model.step_schedulers(test_loss)

    pred_y = np.concatenate(pred_y).ravel()
    true_y = np.concatenate(true_y).ravel()
    auc = roc_auc_score(true_y, pred_y)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)

    print(('====> Test set loss: {:.4f}\t AUC: {:.4f}\t ' +
           'Recall: {:.4f}\t FPR: {:.4f}\t ' +
           'Unknown attacks detected: {:3}/{:3} ({:2.0f}%)')
          .format(
              test_loss, auc, recall, fpr,
              n_unknown_correct, n_unknown,
              100 * n_unknown_correct / n_unknown
          ))
    print(cm)

def main(**kwargs):
    # Initialize arguments and torch
    parser = argparse.ArgumentParser(description=
        'Runs semi-supervised Adversarial Autoencoder experiment for NIDS')
    parser.add_argument('--tag', metavar='ID', help='ID for experiment')
    parser.add_argument('--batch-size', type=int, default=128, metavar='B',
                        help='mini batch size for training (default: 128)')
    parser.add_argument('--labeled-size', type=int, default=16, metavar='L',
                        help='labeled size for each batch (default: 16)')
    parser.add_argument('--n-labeled', type=int, default=100, metavar='N',
                        help='total number of labeled examples to use \
                        in training (default: 100)')
    parser.add_argument('--epochs', type=int, default=10, metavar='E',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--test-interval', type=int, default=1, metavar='T',
                        help='number of epochs between tests of \
                        model on test set (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='I',
                        help='number of batches between logs of \
                        training status (default: 100)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training (default: False)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    n_features, train_loader, test_loader = DataUtils.load_data(
        args.batch_size, args.cuda, shuffle=True
    )

    # Initialize model and optimizers
    model = SS_AAE(X_dim=n_features, **kwargs)
    if args.cuda:
        model.cuda()

    # Log model parameters
    print('Model parameters:')
    for name, value in model.parameters():
        print('{} = {}'.format(name, value))

    # Train/test
    for epoch in range(1, args.epochs + 1):
        train(epoch, model, train_loader, args)
        if epoch % args.test_interval == 0:
            test(epoch, model, test_loader, args)

if __name__ == '__main__':
    for i in range(10):
        for y_dim in [2]:
            for z_dim in [20]:
                for h_dim in [400]:
                    for lr in [1e-4]:
                        kwargs = {
                            'y_dim': y_dim,
                            'z_dim': z_dim,
                            'h_dim': h_dim,
                            'Q_lr' : lr,
                            'P_lr' : lr,
                            'D_lr' : lr,
                        }
                        main(**kwargs)
