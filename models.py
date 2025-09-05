import jax
import jax.numpy as jnp
from functools import partial
# -----------------------------
# Activation and normalization
# -----------------------------

@jax.jit
def relu(x):
    return jnp.maximum(0, x)

@jax.jit
def normalize_adjacency(A):
    # A is (N, N); add self loops and apply D^{-1/2} A D^{-1/2}
    N = A.shape[0]
    A_hat = A + jnp.eye(N, dtype=A.dtype)
    degree = jnp.sum(A_hat, axis=1)
    D_inv_sqrt = jnp.where(degree > 0, 1.0 / jnp.sqrt(degree), 0.0)
    D_hat = jnp.diag(D_inv_sqrt)
    return D_hat @ A_hat @ D_hat

# -----------------------------
# Xavier Initialization
# -----------------------------

def xavier_init(key, in_dim, out_dim):
    limit = jnp.sqrt(6.0 / (in_dim + out_dim))
    W = jax.random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit)
    b = jnp.zeros((out_dim,))
    return W, b

# -----------------------------
# GCN + Dense model functions
# -----------------------------

@jax.jit
def gcn_layer(A_norm, X, W, b):
    # A_norm: (N,N), X: (N,F), W: (F,H), b: (H,)
    return relu(A_norm @ X @ W + b)

@jax.jit
def _safe_protein_embedding(params, A, X):
    """
    params: [(W1,b1),(W2,b2)]
    A: (N,N) adjacency (may be zeros), X: (N,F)
    Returns mean-pooled embedding (H,) where H = params[-1][0].shape[1]
    If shapes are incompatible, returns zeros.
    """
    num_nodes, num_features = X.shape
    exp_in = params[0][0].shape[0]
    out_dim = params[-1][0].shape[1]
    # Guard small/invalid graphs or feature mismatch
    def _zero():
        return jnp.zeros((out_dim,), dtype=X.dtype)

    ok = (num_nodes >= 2) & (num_features == exp_in) & (A.shape[0] == num_nodes) & (A.shape[1] == num_nodes)
    def _embed():
        A_norm = normalize_adjacency(A)
        (W1, b1), (W2, b2) = params
        h1 = gcn_layer(A_norm, X, W1, b1)
        h2 = gcn_layer(A_norm, h1, W2, b2)
        return jnp.mean(h2, axis=0)  # (H,)
    return jax.lax.cond(ok, _embed, _zero)

@jax.jit
def dense_concat(combined_params, h1, h2, h_common_1, h_common_2):
    """
    combined_params: [(W,b),...], expects input dim = 4 * output_dim
    h* are (H,)
    """
    x = jnp.concatenate([h1, h2, h_common_1, h_common_2])  # (4H,)
    for W, b in combined_params[:-1]:
        x = relu(x @ W + b)
    W_last, b_last = combined_params[-1]
    logits = x @ W_last + b_last
    return jax.nn.sigmoid(logits)

# -----------------------------
# Core forward (JIT-friendly)
# -----------------------------

@partial(jax.jit, static_argnums=(7,))  
def model_forward_core(params, A1, X1, A2, X2, A_common_1, A_common_2, use_mcs: bool=False):
    """
    params: (protein1_params, protein2_params, combined_params, common1_params, common2_params)
    All adjacency args are arrays (no None); when not using MCS, pass zeros and set use_mcs=False.
    """
    h1 = _safe_protein_embedding(params[0], A1, X1)
    h2 = _safe_protein_embedding(params[1], A2, X2)

    def _with_mcs():
        hc1 = _safe_protein_embedding(params[3], A_common_1, X1)
        hc2 = _safe_protein_embedding(params[4], A_common_2, X2)
        return dense_concat(params[2], h1, h2, hc1, hc2)

    def _without_mcs():
        z1 = jnp.zeros_like(h1)
        z2 = jnp.zeros_like(h2)
        return dense_concat(params[2], h1, h2, z1, z2)

    return jax.lax.cond(use_mcs, _with_mcs, _without_mcs)

# -----------------------------
# User-facing forward
# -----------------------------

def _zeros_adj_like(A):
    # Create a zeros adjacency with same (N,N) and dtype as A
    return jnp.zeros_like(A)

def model_forward(params, A1, X1, A2, X2, A_common_1=None, A_common_2=None):
    """
    Wrapper that accepts None for A_common_* and calls the core JIT with proper flags/zeros.
    """
    use_mcs = (A_common_1 is not None) and (A_common_2 is not None)
    if not use_mcs:
        # pass shape-compatible zeros to keep JIT happy
        A_common_1 = _zeros_adj_like(A1)
        A_common_2 = _zeros_adj_like(A2)
    return model_forward_core(params, A1, X1, A2, X2, A_common_1, A_common_2, use_mcs=use_mcs)

# -----------------------------
# Loss 
# -----------------------------

@partial(jax.jit, static_argnums=(8,))  # use_mcs is the 9th arg here
def binary_cross_entropy_loss_core(params, A1, X1, A2, X2, y_true, A_common_1, A_common_2, use_mcs: bool=False, eps=1e-8):
    y_pred = model_forward_core(params, A1, X1, A2, X2, A_common_1, A_common_2, use_mcs=use_mcs)
    loss = -(y_true * jnp.log(y_pred + eps) + (1 - y_true) * jnp.log(1 - y_pred + eps))
    return jnp.squeeze(loss)

def binary_cross_entropy_loss(params, A1, X1, A2, X2, y_true, A_common_1=None, A_common_2=None, eps=1e-8):
    use_mcs = (A_common_1 is not None) and (A_common_2 is not None)
    if not use_mcs:
        # shape-compatible zeros to avoid retraces
        A_common_1 = jnp.zeros_like(A1)
        A_common_2 = jnp.zeros_like(A2)
    return binary_cross_entropy_loss_core(
        params, A1, X1, A2, X2, y_true, A_common_1, A_common_2, use_mcs=use_mcs, eps=eps
    )

# -----------------------------
# GCNN Class for Param Init
# -----------------------------

class GCNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim   # F
        self.hidden_dim = hidden_dim # H1
        self.output_dim = output_dim # H (embedding size per branch)

    def init_params(self, key):
        keys = jax.random.split(key, 11)

        # GCN parameters for drug/protein 1
        W1a, b1a = xavier_init(keys[0], self.input_dim, self.hidden_dim)
        W2a, b2a = xavier_init(keys[1], self.hidden_dim, self.output_dim)
        protein1_params = [(W1a, b1a), (W2a, b2a)]

        # GCN parameters for drug/protein 2
        W1b, b1b = xavier_init(keys[2], self.input_dim, self.hidden_dim)
        W2b, b2b = xavier_init(keys[3], self.hidden_dim, self.output_dim)
        protein2_params = [(W1b, b1b), (W2b, b2b)]

        # Dense head expects 4 streams (h1, h2, h_common_1, h_common_2)
        interm_1 = 512
        interm_2 = 512
        Wc1, bc1 = xavier_init(keys[4], 4 * self.output_dim, interm_1)
        Wc2, bc2 = xavier_init(keys[5], interm_1, interm_2)
        Wc3, bc3 = xavier_init(keys[6], interm_2, 1)
        combined_params = [(Wc1, bc1), (Wc2, bc2), (Wc3, bc3)]

        # GCN parameters for common subgraph branch 1
        WC1a, bC1a = xavier_init(keys[7], self.input_dim, self.hidden_dim)
        WC2a, bC2a = xavier_init(keys[8], self.hidden_dim, self.output_dim)
        common_protein1_params = [(WC1a, bC1a), (WC2a, bC2a)]

        # GCN parameters for common subgraph branch 2
        WC1b, bC1b = xavier_init(keys[9], self.input_dim, self.hidden_dim)
        WC2b, bC2b = xavier_init(keys[10], self.hidden_dim, self.output_dim)
        common_protein2_params = [(WC1b, bC1b), (WC2b, bC2b)]

        return (
            protein1_params,         # params[0]
            protein2_params,         # params[1]
            combined_params,         # params[2] (dense head)
            common_protein1_params,  # params[3]
            common_protein2_params,  # params[4]
        )
