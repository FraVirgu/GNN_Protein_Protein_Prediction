import os
import math
import random
from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import matplotlib.pyplot as plt
import flax.serialization

from models import (
    GCNN,
    model_forward,              # functional forward (unused directly here but kept for parity)
    binary_cross_entropy_loss,  # per-sample BCE loss
)
import data_creation as dc   # ⬅️ import directly
def cosine_decay(step: int, total_steps: int, lr0: float, lr_min: float = 0.0) -> float:
    """Cosine LR schedule from lr0 → lr_min over total_steps."""
    t = min(step, total_steps)
    cos_factor = 0.5 * (1.0 + math.cos(math.pi * t / total_steps))
    return lr_min + (lr0 - lr_min) * cos_factor


def batch_indices(n: int, bs: int, rng: np.random.Generator):
    """Yield shuffled batch indices."""
    idx = np.arange(n)
    rng.shuffle(idx)
    for s in range(0, n, bs):
        yield idx[s : s + bs]


def tree_has_nans(tree):
    flags = jtu.tree_map(lambda x: jnp.any(~jnp.isfinite(x)) if hasattr(x, "dtype") else False, tree)
    return bool(jnp.any(jnp.array(jtu.tree_leaves(flags))))

def plot_test_predictions(y_trues, y_preds):
    plt.figure(figsize=(7,5))
    plt.hist(y_preds[y_trues==1], bins=30, alpha=0.6, label="Positive class")
    plt.hist(y_preds[y_trues==0], bins=30, alpha=0.6, label="Negative class")
    plt.axvline(0.5, color="red", linestyle="--", label="Threshold 0.5")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Distribution of Predictions on Test Set")
    plt.legend()
    plt.show()

# ---------------------------
# Testing
# ---------------------------
def evaluate_model(params, test_set, num_features):
    y_trues = []
    y_preds = []
    loss_sum = 0.0

    for sample in tqdm(test_set, desc="Evaluating", ncols=100):
        A1, X1 = sample[2], sample[3]
        A2, X2 = sample[4], sample[5]
        y      = float(sample[6])

        # forward
        y_pred = model_forward(params, A1, X1, A2, X2)
        y_pred = float(jnp.squeeze(y_pred))   # make sure it's a scalar
        

        # per-sample loss on the *real* graphs
        loss = float(binary_cross_entropy_loss(params, A1, X1, A2, X2, y))

        y_trues.append(y)
        y_preds.append(y_pred)
        loss_sum += loss

    y_trues = jnp.array(y_trues)
    y_preds = jnp.array(y_preds)

    plot_test_predictions(y_trues, y_preds)

    # accuracy at 0.5 threshold
    y_hat = (y_preds >= 0.5).astype(jnp.float32)
    acc = float(jnp.mean(y_hat == y_trues))

    mean_loss = loss_sum / len(test_set)
    return acc, mean_loss, y_trues, y_preds





def main():
    # ----------------------------------
    # Environment
    # ----------------------------------
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ----------------------------------
    # Data
    # ----------------------------------
    # build datasets here (NOT at import)
    train_set, valid_set, test_set, N = dc.get_datasets(max_train=5000, max_valid=2000, max_test=2000, n_jobs=8)

    # ----------------------------------
    # Hyperparameters
    # ----------------------------------
    num_features = train_set[0][3].shape[1]
    
    layers_size = {"num_features": num_features, "hidden_dim":512, "output_dim": 512}

    num_epochs = 200
    batch_size = 64

    # SGD + momentum + cosine decay schedule
    momentum = 0.8
    lr_init = 1e-1
    lr_min  = 1e-4

    
    # ----------------------------------
    # Model init 
    # ----------------------------------
    gcnn_model = GCNN(
        input_dim=layers_size["num_features"],
        hidden_dim=layers_size["hidden_dim"],
        output_dim=layers_size["output_dim"],
    )
    key = jax.random.PRNGKey(0)
    params = gcnn_model.init_params(key)

    
   
    '''
        # ----------------------------------
        # Model init  with PRETRAINED PARAMS
        # ----------------------------------
        ckpt_path = "PARAMETER/gcnn_params_saved.pkl"
        with open(ckpt_path, "rb") as f:
            payload = flax.serialization.from_bytes(None, f.read())

        cfg = payload["config"]
        state_dict = payload["params"] 

        # sanity: dims must match training
        assert cfg["num_features"] == num_features, \
            f"Dataset features ({num_features}) != checkpoint ({cfg['num_features']})"

        # instantiate model with SAME dims as training
        gcnn_model = GCNN(input_dim=cfg["num_features"], hidden_dim=cfg["hidden_dim"], output_dim=cfg["output_dim"])

        print("✅ Parameters + config loaded:", cfg)

        key = jax.random.PRNGKey(0)
        target_params = gcnn_model.init_params(key)  # dummy init with same structure
        # ---------------------------
        # Load trained parameters
        # ---------------------------

        # 4) Map state_dict → target pytree (preserves tuple/list structure)
        params = flax.serialization.from_state_dict(target_params, state_dict)


    '''

  

    # ----------------------------------
    # JITed loss/grad fns
    # ----------------------------------
    loss_jit = jax.jit(binary_cross_entropy_loss)
    grad_jit = jax.jit(jax.grad(binary_cross_entropy_loss, argnums=0))

    # ----------------------------------
    # Train
    # ----------------------------------
    n_train = len(train_set)
    steps_per_epoch = math.ceil(n_train / batch_size)
    total_steps = steps_per_epoch * num_epochs

    history_train = []

    # momentum buffer same tree-structure as params
    velocity = jax.tree_util.tree_map(jnp.zeros_like, params)
    global_step = 0

    
    print(f"Training on {n_train} samples with {num_features} features per node...")

    rng = np.random.default_rng(0)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        epoch_loss_sum = 0.0

        for ids in batch_indices(n_train, batch_size, rng):
            batch = [train_set[i] for i in ids]

            # accumulate grads and loss over batch
            grads_sum = None
            loss_sum = 0.0

            for sample in batch:
                a_adj, a_feat = sample[2], sample[3]
                b_adj, b_feat = sample[4], sample[5]
                #A_common_1 = sample[7]
                #A_common_2 = sample[8]
                A_common_1 = None
                A_common_2 = None
                y = float(sample[6])

                g = grad_jit(params, a_adj, a_feat, b_adj, b_feat, y, A_common_1, A_common_2)
                grads_sum = g if grads_sum is None else jax.tree_util.tree_map(lambda x, y_: x + y_, grads_sum, g)

                loss_sum += float(loss_jit(params, a_adj, a_feat, b_adj, b_feat, y, A_common_1, A_common_2))

            # mean gradients / loss
            bsz = len(batch)
            grads_avg = jax.tree_util.tree_map(lambda g: g / bsz, grads_sum)
            batch_loss = loss_sum / bsz

            # cosine-decayed LR for this step
            lr_t = cosine_decay(global_step, total_steps, lr_init, lr_min)

            # SGD with momentum (Polyak)
            velocity = jax.tree_util.tree_map(lambda v, g: momentum * v - lr_t * g, velocity, grads_avg)
            params = jax.tree_util.tree_map(lambda p, v: p + v, params, velocity)
            if tree_has_nans(params):
                print("❌ NaNs in params at step", global_step)
                break

            global_step += 1
            epoch_loss_sum += batch_loss * bsz

        epoch_loss = epoch_loss_sum / n_train
        history_train.append(epoch_loss)

    # ----------------------------------
    # Plot
    # ----------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(history_train, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross Entropy Loss")
    plt.title("Training Loss over Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss.png")  

    if tree_has_nans(params):
        print("❌ NaNs in params at step", global_step)
        

    # ----------------------------------
    # Save params
    # ----------------------------------
    save_dir = "PARAMETER"
    os.makedirs(save_dir, exist_ok=True)

    config = {
        "num_features": layers_size["num_features"],
        "hidden_dim":   layers_size["hidden_dim"],
        "output_dim":   layers_size["output_dim"],
    }

    payload = {"config": config, "params": params}
    with open(os.path.join(save_dir, "gcnn_params.pkl"), "wb") as f:
        f.write(flax.serialization.to_bytes(payload))

    print("✅ Saved params + config to RDKIT/PARAMETER/gcnn_params.pkl")

    
    accuracy, loss, y_trues, y_preds = evaluate_model(params, test_set,num_features)
    print(f"Test Accuracy: {accuracy*100:.2f}%, Test Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
