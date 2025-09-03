import numpy as np
try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # fallback no-op
    _tqdm = None  # type: ignore

# Minimal NumPy backend implementing only the used TensorFlow-like ops
class _Backend:
    def einsum(self, subscripts, *operands, **kwargs):
        return np.einsum(subscripts, *operands, **kwargs)
    def sign(self, x):
        x = np.sign(x)
        return np.where(x >= 0, 1, -1)
    def tanh(self, x):
        return np.tanh(x)
    def convert_to_tensor(self, x, dtype=None):
        return np.array(x, dtype=dtype if dtype is not None else None)
    def transpose(self, x, perm=None):
        return np.transpose(x, axes=perm)

tf = _Backend()


class Hopfield_Network:
    def __init__(self):
        self.N = None
        self.K = None
        self.J = None
        self.σ = None

    def prepare(self, η):
        self.N = η.shape[1]
        self.K = η.shape[0]
        self.J = tf.einsum('ai,aj->ij', η, η) / self.N

    def dynamics(self, σ0, β, updates, mode="parallel"):
        assert self.N is not None, "Call prepare first"
        N = self.N
        M = σ0.shape[0]
        J = self.J
        σ = tf.convert_to_tensor(σ0, dtype=np.float32)

        if mode == "parallel":
            for _ in range(updates):
                h = tf.einsum('ij,Aj->Ai', J, σ)
                noise = np.random.uniform(-1, 1, size=(M, N))
                self.σ = tf.sign(tf.tanh(β * h) + noise)
                σ = self.σ
        else:  # serial
            for _ in range(updates):
                idx = np.random.randint(0, N)
                h = tf.einsum('ij,Aj->Ai', J, σ)
                noise = np.random.uniform(-1, 1, size=(M, N))
                signal = tf.tanh(β * h) + noise
                σ[:, idx] = tf.sign(signal[:, idx])
                self.σ = σ


class TAM_Network:
    def __init__(self):
        self.N = None
        self.J = None
        self.L = None
        self.σ = None

    def prepare(self, J, L):
        self.N = J.shape[1]
        self.J = J
        self.L = L

    def compute_fields(self, input_field):
        # σ shape: (s, L, N)
        J = self.J
        σ = self.σ
        N = self.N
        # h1: local fields for each layer
        h1 = tf.einsum('ij,slj->sli', J, σ)
        # Replicate original (possibly unconventional) tensor algebra
        temp0 = tf.einsum('sli,ski->slk', σ, h1)
        temp1 = tf.einsum('slk,sli->ski', temp0, h1)
        temp2 = tf.einsum('skk,ski->ski', temp0, h1)
        h2 = (temp1 - temp2) / N
        h3 = input_field
        return h1, h2, h3

    def dynamics(self, σr, β, λ, h, updates, noise_scale: float = 0.3, anneal: bool = True, min_scale: float = 0.02, schedule: str = "linear", show_progress: bool = False, desc: str = "TAM dyn"):
        """Run TAM dynamics with controlled / annealed noise.

        Parameters
        ----------
        σr : array (s, L, N)
            Initial candidate states.
        β : float
            Inverse temperature parameter (gain for tanh fields).
        λ : float
            Coupling weight for higher-order correction (h2 term).
        h : float
            External input coupling (h3 term).
        updates : int
            Number of parallel update steps.
        noise_scale : float
            Initial scale (std-like) for uniform noise in [-scale, scale]. Much smaller than legacy 1.0.
        anneal : bool
            If True reduce noise over iterations.
        min_scale : float
            Lower bound for noise scale if annealing.
        schedule : {'linear','exp'}
            Annealing schedule shape.
        """
        assert self.N is not None and self.L is not None, "Call prepare first"
        σr = tf.convert_to_tensor(σr, dtype=np.float32)  # expected shape (s, L, N)
        self.σ = np.copy(σr)
        s, L, N = self.σ.shape
        iterator = range(updates)
        if show_progress and _tqdm is not None:
            iterator = _tqdm(iterator, desc=desc, leave=False)
        for t in iterator:
            h1, h2, h3 = self.compute_fields(σr)
            ht = h1 - λ * h2 + h * h3
            if anneal and updates > 1:
                if schedule == "linear":
                    scale = noise_scale - (noise_scale - min_scale) * (t / (updates - 1))
                else:  # exponential
                    # decay so that at final step ~min_scale
                    γ = (min_scale / noise_scale) ** (1 / max(updates - 1, 1))
                    scale = noise_scale * (γ ** t)
            else:
                scale = noise_scale
            # Uniform noise scaled & additionally attenuated by 1/β so large β => effectively lower noise
            eff_scale = scale / max(β, 1e-6)
            noise = np.random.uniform(-eff_scale, eff_scale, size=(s, L, N))
            self.σ = tf.sign(tf.tanh(β * ht) + noise)
            σr = self.σ
