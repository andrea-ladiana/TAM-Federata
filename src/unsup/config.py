# -*- coding: utf-8 -*-
"""
Configurazione e iperparametri (single mode only).

Questo modulo centralizza gli iperparametri usati negli esperimenti in modalità
*single* (nessuna accumulazione dei dati tra i round, a parte la memoria Ebraica
da round t-1 per il blending).

Note:
- Le soglie e la pipeline sono allineate alle specifiche del report (τ, ρ, qthr,
  blending, propagazione pseudo-inversa, ecc.).
- Il modulo espone una dataclass principale `HyperParams` con metodi di utilità.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Literal, Optional


Mode = Literal["single"]  # hard-lock alla single-mode


@dataclass(frozen=True)
class TAMParams:
    """Parametri della dinamica TAM."""
    beta_T: float = 2.5
    lam: float = 0.2
    h_in: float = 0.1
    updates: int = 80
    # noise/annealing: i default devono combaciare con TAM_Network
    noise_scale: float = 0.3
    min_scale: float = 0.02
    anneal: bool = True
    schedule: Literal["linear", "exp"] = "linear"


@dataclass(frozen=True)
class PropagationParams:
    """Parametri per la propagazione pseudo-inversa J -> J_KS."""
    iters: int = 200
    eps: float = 1e-2  # usato internamente da `propagate_J` (se rilevante)


@dataclass(frozen=True)
class SpectralParams:
    """Soglie per inizializzazione/pruning nel disentangling."""
    tau: float = 0.5     # cut sugli autovalori di J_KS
    rho: float = 0.6     # allineamento spettrale ξᵀ J_KS ξ / N
    qthr: float = 0.4    # pruning per overlap mutuo tra candidati


@dataclass
class HyperParams:
    # --- scala problema / dataset federato ---
    L: int = 3           # numero client/layer
    K: int = 3           # numero archetipi
    N: int = 300         # dimensione dei pattern
    n_batch: int = 24    # round
    M_total: int = 2400  # numero totale esempi
    r_ex: float = 0.8    # correlazione media campioni/archetipo
    K_per_client: Optional[int] = None  # se None -> ceil(K / L)

    # --- blending e stima single-round ---
    w: float = 0.4  # peso dell'unsupervised vs memoria ebraica del round precedente

    # --- modalità (hard-lock) ---
    mode: Mode = "single"

    # --- seed/repliche ---
    n_seeds: int = 5
    seed_base: int = 1234
    use_tqdm: bool = True

    # --- sottostrutture ---
    tam: TAMParams = field(default_factory=TAMParams)
    prop: PropagationParams = field(default_factory=PropagationParams)
    spec: SpectralParams = field(default_factory=SpectralParams)

    # --- varie utilità/flags ---
    estimate_keff_method: Literal["shuffle", "mp"] = "mp"
    ema_alpha: float = 0.0  # 0.0 = off (EMA su J_unsup se si vuole smussare il rumore)

    def __post_init__(self):
        # lock single-mode
        if self.mode != "single":
            raise ValueError("Questo pacchetto è bloccato alla modalità 'single'.")
        if not (0.0 <= self.w <= 1.0):
            raise ValueError("w dev'essere in [0, 1].")
        if not (0 < self.r_ex <= 1.0):
            raise ValueError("r_ex dev'essere in (0, 1].")
        if self.K_per_client is None:
            object.__setattr__(self, "K_per_client", max(1, ceil(self.K / self.L)))

    # -------- helpers pratici --------
    @property
    def M_per_client_per_round(self) -> int:
        """Numero di esempi per *client* per *round* in single-mode."""
        return max(1, ceil(self.M_total / (self.L * self.n_batch)))

    @property
    def M_eff_round(self) -> int:
        """Effettivo numero di esempi per stima J in *single* (per client)."""
        # Possibile diversa normalizzazione a seconda della tua implementazione di unsupervised_J;
        # qui esponiamo un valore coerente con "single".
        return self.M_per_client_per_round

    def copy_with(self, **kwargs) -> "HyperParams":
        """Clona con override di alcuni campi (comodo in sweep/ablazioni)."""
        data = {**self.__dict__}
        data.update(kwargs)
        return HyperParams(**data)
