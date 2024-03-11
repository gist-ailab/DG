import numpy as np
from scipy.special import logsumexp
import torch


def make_in_class_cov(features, labels):
    num_classes = torch.unique(labels).size(0)
    class_means = []
    for i in range(num_classes):
        class_mask = (labels == i)
        class_features = features[class_mask]
        class_mean = class_features.mean(dim=0)
        class_means.append(class_mean)

    # S_W = torch.zeros(features.size(1), features.size(1))
    cov_sum = 0
    for i in range(num_classes):
        class_mask = (labels == i)
        class_features = features[class_mask]
        class_mean = class_features.mean(dim=0)
        
        for feature in class_features:
            diff = (feature - class_mean).unsqueeze(-1)  # 열 벡터
            cov_sum += torch.triu(torch.matmul(diff, diff.t()), diagonal=1)
    return cov_sum

    centered_features = cls_data - cls_data.mean(dim=0, keepdim=True)
    # 공분산 행렬을 계산합니다
    cov_matrix = torch.matmul(centered_features.T, centered_features) / (cls_data.shape[0] - 1)
    upper_cov = torch.triu(cov_matrix, diagonal=1)
    sum_cov = upper_cov.sum()
    return sum_cov


def one_hot(ids, depth):
    z = np.zeros([len(ids), depth])
    z[np.arange(len(ids)), ids] = 1
    return z

def multivariate_gaussian(X, pi, variances, means):
  """
  X: examples with shape (n, d)
  pi:  priors with shape (K, )
  variances and means shape: (K, d)
  log_r_matrix: log of r matrix with shape (K, n)
  """
  reversed_var = 1 / variances
  log_r_matrix = (X ** 2) @ reversed_var.T
  log_r_matrix -= 2 * X @ (means * reversed_var).T
  log_r_matrix[:] += np.sum((means ** 2) * reversed_var, axis=1)
  log_r_matrix *= -0.5
  log_r_matrix[:] += np.log(pi) - 0.5 * np.sum(np.log(variances), axis=1)
  log_r_matrix = log_r_matrix.T

  sum_log_r = logsumexp(log_r_matrix, axis=0)
  return sum_log_r, log_r_matrix
