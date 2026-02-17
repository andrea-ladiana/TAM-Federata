# -*- coding: utf-8 -*-
"""
Client-Aware Adaptive Weight Experiment
========================================
Studies the evolution of per-client adaptive weights w_c(t)
in a federated unsupervised learning setting, where one client
provides pure noise (r=0) while the others provide genuine data.

The adaptive weight is derived from the complement of the sign-agreement
entropy between the server reconstruction and the local Hebbian correlator:

    p_c(t)      = sign-agreement fraction between A^{t-1} and J_local_c^{t}
    H_{AB}      = h_2(p_c(t))
    w_raw,c(t)  = clip(1 − H_AB, 0, 1)
    w_c(t)      = α · w_raw,c(t) + (1−α) · w_c(t−1)    (EMA smoothing)

Reference: eq. (convex_comb) and Sec. plasticity_w in the manuscript.
"""
