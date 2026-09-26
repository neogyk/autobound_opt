"""Compares SafeRate against AdamW on MNIST digit classification.

Both optimizers train an identical MLP from the same initial weights, on the
same batches, so the only difference between runs is the optimizer. Usage:

    python mnist_saferate_vs_adamw.py

MNIST is downloaded once (IDX files, no torchvision/HF dependency needed)
and cached under ./mnist_data/.
"""

import copy
import gzip
import os
import struct
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from saferate import SafeAdamW
from saferate import SafeRate

MNIST_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist_data')

# Kept small so the demo (including SafeRate's extra forward/backward passes
# per step) runs in well under a minute on a CPU.
N_TRAIN = 4000
N_TEST = 1000
BATCH_SIZE = 256
N_EPOCHS = 100
SEED = 0


def _download(fname: str) -> str:
  path = os.path.join(DATA_DIR, fname)
  if not os.path.exists(path):
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f'Downloading {fname}...')
    urllib.request.urlretrieve(MNIST_URL + fname, path)
  return path


def _read_idx_images(path: str) -> np.ndarray:
  with gzip.open(path, 'rb') as f:
    magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
    data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols)


def _read_idx_labels(path: str) -> np.ndarray:
  with gzip.open(path, 'rb') as f:
    magic, n = struct.unpack('>II', f.read(8))
    return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist():
  train_images = _read_idx_images(_download('train-images-idx3-ubyte.gz'))
  train_labels = _read_idx_labels(_download('train-labels-idx1-ubyte.gz'))
  test_images = _read_idx_images(_download('t10k-images-idx3-ubyte.gz'))
  test_labels = _read_idx_labels(_download('t10k-labels-idx1-ubyte.gz'))

  def to_tensors(images, labels, n):
    x = torch.from_numpy(images[:n].astype(np.float32) / 255.0)
    y = torch.from_numpy(labels[:n].astype(np.int64))
    return x, y

  x_train, y_train = to_tensors(train_images, train_labels, N_TRAIN)
  x_test, y_test = to_tensors(test_images, test_labels, N_TEST)
  return x_train, y_train, x_test, y_test


def make_model() -> nn.Module:
  # No BatchNorm/Dropout: SafeRate's line search assumes a deterministic
  # loss surface across the several closure() calls it makes per step.
  return nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))


@torch.no_grad()
def evaluate(model, x, y) -> float:
  logits = model(x)
  preds = logits.argmax(dim=-1)
  return (preds == y).float().mean().item()


def train_adamw(model, x_train, y_train, x_test, y_test):
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
  history = []
  n = x_train.shape[0]
  for epoch in range(N_EPOCHS):
    perm = torch.randperm(n)
    total_loss = 0.0
    for start in range(0, n, BATCH_SIZE):
      idx = perm[start:start + BATCH_SIZE]
      optimizer.zero_grad()
      loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
      loss.backward()
      optimizer.step()
      total_loss += loss.item() * len(idx)
    history.append((total_loss / n, evaluate(model, x_test, y_test)))
  return history


def train_saferate(model, x_train, y_train, x_test, y_test):
  optimizer = SafeRate(model.parameters(), initial_max_eta=1.0)
  return _train_with_closure_optimizer(optimizer, model, x_train, y_train,
                                        x_test, y_test)


def train_safeadamw(model, x_train, y_train, x_test, y_test):
  optimizer = SafeAdamW(model.parameters(), initial_max_eta=1.0,
                         weight_decay=0.01)
  return _train_with_closure_optimizer(optimizer, model, x_train, y_train,
                                        x_test, y_test)


def _train_with_closure_optimizer(optimizer, model, x_train, y_train, x_test,
                                   y_test):
  history = []
  n = x_train.shape[0]
  for epoch in range(N_EPOCHS):
    perm = torch.randperm(n)
    total_loss = 0.0
    for start in range(0, n, BATCH_SIZE):
      idx = perm[start:start + BATCH_SIZE]
      xb, yb = x_train[idx], y_train[idx]

      def closure():
        # Must NOT call .backward(): SafeRate/SafeAdamW run autograd
        # themselves (they need a fresh graph for the curvature estimate).
        return F.cross_entropy(model(xb), yb)

      loss = optimizer.step(closure)
      total_loss += loss.item() * len(idx)
    history.append((total_loss / n, evaluate(model, x_test, y_test)))
  return history


def main():
  torch.manual_seed(SEED)
  x_train, y_train, x_test, y_test = load_mnist()

  torch.manual_seed(SEED)
  base_model = make_model()
  init_state = copy.deepcopy(base_model.state_dict())

  model_adamw = make_model()
  model_adamw.load_state_dict(init_state)
  t0 = time.time()
  history_adamw = train_adamw(model_adamw, x_train, y_train, x_test, y_test)
  time_adamw = time.time() - t0

  model_saferate = make_model()
  model_saferate.load_state_dict(init_state)
  t0 = time.time()
  history_saferate = train_saferate(model_saferate, x_train, y_train, x_test,
                                     y_test)
  time_saferate = time.time() - t0

  model_safeadamw = make_model()
  model_safeadamw.load_state_dict(init_state)
  t0 = time.time()
  history_safeadamw = train_safeadamw(model_safeadamw, x_train, y_train,
                                       x_test, y_test)
  time_safeadamw = time.time() - t0

  print(f'{"epoch":>5}  {"AdamW loss":>10}  {"AdamW acc":>9}  '
        f'{"SafeRate loss":>13}  {"SafeRate acc":>12}  '
        f'{"SafeAdamW loss":>14}  {"SafeAdamW acc":>13}')
  for epoch in range(N_EPOCHS):
    al, aa = history_adamw[epoch]
    sl, sa = history_saferate[epoch]
    wl, wa = history_safeadamw[epoch]
    print(f'{epoch:>5}  {al:>10.4f}  {aa:>9.3f}  {sl:>13.4f}  {sa:>12.3f}  '
          f'{wl:>14.4f}  {wa:>13.3f}')
  print(f'\nWall-clock time: AdamW={time_adamw:.1f}s  '
        f'SafeRate={time_saferate:.1f}s  SafeAdamW={time_safeadamw:.1f}s '
        '(SafeRate/SafeAdamW do extra forward/backward passes per step for '
        'their curvature estimate, so they are expected to be slower per '
        'step than plain AdamW)')

  try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot
    epochs = list(range(N_EPOCHS))
    pyplot.plot(epochs, [h[1] for h in history_adamw], 'o-', label='AdamW')
    pyplot.plot(epochs, [h[1] for h in history_saferate], 's-',
                label='SafeRate')
    pyplot.plot(epochs, [h[1] for h in history_safeadamw], '^-',
                label='SafeAdamW')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Test accuracy')
    pyplot.legend()
    pyplot.title('SafeRate / SafeAdamW vs AdamW on MNIST')
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'mnist_comparison.png')
    pyplot.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'Saved plot to {out_path}')
  except ImportError:
    pass


if __name__ == '__main__':
  main()
