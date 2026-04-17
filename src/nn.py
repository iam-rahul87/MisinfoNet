"""
Lightweight numpy neural network — Linear layers, activations, dropout,
cross-entropy loss, Adam optimiser. Used for the RAG head and fusion MLP.
"""

import numpy as np


def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def gelu(x):
    return x * sigmoid(1.702 * x)


def cross_entropy(logits, labels):
    """logits: (N, C), labels: (N,) int"""
    probs = softmax(logits)
    n = len(labels)
    log_p = -np.log(probs[np.arange(n), labels] + 1e-9)
    return float(log_p.mean())


def cross_entropy_grad(logits, labels):
    """Returns dL/d_logits  shape (N, C)"""
    probs = softmax(logits)
    n = len(labels)
    probs[np.arange(n), labels] -= 1.0
    return probs / n


class Linear:
    def __init__(self, in_dim, out_dim, rng, gain=1.0):
        scale = gain * np.sqrt(2.0 / in_dim)
        self.W = rng.standard_normal((in_dim, out_dim)).astype(np.float32) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)
        # Adam state
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, dy):
        self.dW = self._x.T @ dy
        self.db = dy.sum(axis=0)
        return dy @ self.W.T

    def step(self, lr, t, beta1=0.9, beta2=0.999, eps=1e-8, wd=1e-4):
        for p, m, v, g in [
            (self.W, self.mW, self.vW, self.dW),
            (self.b, self.mb, self.vb, self.db),
        ]:
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * g**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            p -= lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * p)


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta  = np.zeros(dim, dtype=np.float32)
        self.eps   = eps
        self.mgamma = np.zeros(dim, dtype=np.float32)
        self.vgamma = np.zeros(dim, dtype=np.float32)
        self.mbeta  = np.zeros(dim, dtype=np.float32)
        self.vbeta  = np.zeros(dim, dtype=np.float32)

    def forward(self, x):
        self._x = x
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self._xhat = (x - mu) / np.sqrt(var + self.eps)
        self._var  = var
        return self.gamma * self._xhat + self.beta

    def backward(self, dy):
        N, D = dy.shape
        self.dgamma = (dy * self._xhat).sum(axis=0)
        self.dbeta  = dy.sum(axis=0)
        dxhat = dy * self.gamma
        dvar  = (-0.5 * dxhat * self._xhat / (self._var + self.eps)).sum(-1, keepdims=True)
        dmu   = (-dxhat / np.sqrt(self._var + self.eps)).sum(-1, keepdims=True)
        dx    = (dxhat / np.sqrt(self._var + self.eps)
                 + 2 * dvar * self._xhat / D
                 + dmu / D)
        return dx

    def step(self, lr, t, beta1=0.9, beta2=0.999, eps=1e-8, wd=0):
        for p, m, v, g in [
            (self.gamma, self.mgamma, self.vgamma, self.dgamma),
            (self.beta,  self.mbeta,  self.vbeta,  self.dbeta),
        ]:
            m[:] = beta1 * m + (1 - beta1) * g
            v[:] = beta2 * v + (1 - beta2) * g**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)


class MLP:
    """
    Multi-layer perceptron: Linear → LayerNorm → GELU → (repeat) → Linear
    Supports forward, backward, and Adam step.
    """

    def __init__(self, dims: list[int], rng, dropout: float = 0.1):
        assert len(dims) >= 2
        self.linears = []
        self.norms   = []
        self.dropout = dropout
        self._masks  = []
        self._training = True

        for i in range(len(dims) - 1):
            gain = 1.0 if i < len(dims) - 2 else 1.0
            self.linears.append(Linear(dims[i], dims[i+1], rng, gain))
            if i < len(dims) - 2:
                self.norms.append(LayerNorm(dims[i+1]))

    def train(self):  self._training = True
    def eval(self):   self._training = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._masks = []
        self._acts  = []
        h = x
        for i, lin in enumerate(self.linears):
            h = lin.forward(h)
            if i < len(self.linears) - 1:
                h = self.norms[i].forward(h)
                h = gelu(h)
                if self._training and self.dropout > 0:
                    mask = (np.random.rand(*h.shape) > self.dropout).astype(np.float32)
                    h *= mask / (1 - self.dropout + 1e-9)
                    self._masks.append(mask)
                else:
                    self._masks.append(None)
                self._acts.append(h.copy())
        return h   # logits

    def backward(self, dy: np.ndarray) -> np.ndarray:
        dh = dy
        for i in reversed(range(len(self.linears))):
            dh = self.linears[i].backward(dh)
            if i > 0:
                # undo dropout
                mask = self._masks[i-1]
                if mask is not None:
                    dh *= mask / (1 - self.dropout + 1e-9)
                # undo gelu  (∂gelu/∂x ≈ sigmoid(1.702x) + 1.702x·sig·(1-sig))
                a = self._acts[i-1]
                sig = sigmoid(1.702 * a)
                dgelu = sig + 1.702 * a * sig * (1 - sig)
                dh *= dgelu
                dh = self.norms[i-1].backward(dh)
        return dh

    def step(self, lr: float, t: int):
        for lin in self.linears:
            lin.step(lr, t)
        for norm in self.norms:
            norm.step(lr, t)

    def param_count(self) -> int:
        total = 0
        for lin in self.linears:
            total += lin.W.size + lin.b.size
        for norm in self.norms:
            total += norm.gamma.size + norm.beta.size
        return total
