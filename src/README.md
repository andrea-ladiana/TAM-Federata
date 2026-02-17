# Documentazione della gerarchia `src/`

Questa sezione descrive in dettaglio le tre aree funzionali della codebase:

- `mixing/`: pipeline e strumenti per l'**Exp-06** (dataset sintetici e FMNIST) con controllo adattivo del peso di blending `w`.
- `narch/`: toolkit per l'**Exp-07** dedicato alla rilevazione di novita (novelty emergence) e alla reportistica collegata.
- `unsup/`: nucleo riutilizzabile per l'apprendimento non supervisionato in modalita *single round*, da cui dipendono gli altri due pacchetti.

---

## `unsup/` - nucleo riutilizzabile per la modalita *single* (Exp-01)

### Moduli principali

- `config.py`: definisce le dataclass `HyperParams`, `TAMParams`, `PropagationParams`, `SpectralParams` con controlli di consistenza (`HyperParams.__post_init__` blocca la modalita single ed espone proprieta come `M_per_client_per_round`).
- `data.py`: generazione del dataset federato (`gen_dataset_partial_archetypes`), subset per client, estrazione del batch di round (`new_round_single`), metriche di coverage/esposizione.
- `estimators.py`: costruzione di `J_unsup` per round (`build_unsup_J_single`) e blending Hebb con memoria (`blend_with_memory`).
- `functions.py`: backend NumPy che ingloba generazione pattern, proiezioni pseudo-inverse (`propagate_J`), stimatori unsup/sup e `estimate_K_eff_from_J` (shuffle/MP).
- `dynamics.py`: pipeline TAM (`eigen_cut`, `init_candidates_from_eigs`, `disentangling`, `dis_check`) che usa `src.unsup.networks.TAM_Network`.
- `networks.py`: implementazioni pure NumPy delle reti di Hopfield e TAM, con dinamiche parallele o stocastiche.
- `metrics.py`: funzioni di valutazione (Frobenius relativo, retrieval Ungherese, magnetizzazioni, coverage, robust z-score, `K_eff` da autovalori).
- `mnist_hfl.py`: adattatori per dataset MNIST reali (binarizzazione, prototipi di classe, generazione dataset SINGLE mode).
- `hopfield_eval.py`: valutazione e persistenza della dinamica Hopfield post-hoc (`run_hopfield_test`, funzioni di plotting, serializzazione JSON/NPY).
- `runner_single.py` e `single_round.py`: orchestratori dell'Exp-01. Il runner gestisce l'intera run multi-seed (generazione archetipi, dataset, log round-by-round, salvataggi), mentre `single_round_step` incapsula la sequenza di trasformazioni 1-7 (stima `J`, blend, propagazione, cut, TAM, metriche, aggiornamento memoria).
- `spectrum.py`: wrapper per `eigen_cut` e `estimate_keff` esposto al resto della codebase con simmetrizzazione inclusa.
- `merge.py`: concatena i `.py` locali in `codebase.txt` per tracciamento.


---


## `mixing/` - pipeline Exp-06 con controllo adattivo del mixing

| File | Ruolo |
| --- | --- |
| `pipeline_core.py` | Motore round-by-round per dataset sintetici (`run_seed_synth`): genera ETA seguendo una mixing schedule `pis`, stima `J_unsup`, esegue il blend Hebb (peso `w` fisso o adattivo), propaga `J -> J_KS`, estrae candidati via TAM e calcola metriche/Hopfield salvandole in `outdir/round_{t}`. |
| `pipeline_fmnist.py` | Variante per dati strutturati (FMNIST) che binarizza le immagini (`_binarize_pm1`), costruisce gli archetipi `sign(mean)` per classe e genera round reali `_build_eta_from_images`. Condivide lo stesso blocco di controllo `w` e l'intera diagnostica del core. |
| `control.py` | Calcola i segnali di drift/mismatch (`compute_drift_signals`) e implementa le policy `threshold`, `sigmoid` e `pctrl` per aggiornare dinamicamente `w`. |
| `adaptive_w.py` / `adaptive_w_damped.py` | Calcolo di `w` basato sull'entropia di coerenza di segno tra memoria Hebb e dato corrente; la versione "damped" aggiunge EMA, rate limiting, momentum e modalita `adaptive_ema`. |
| `scheduler.py` | Generatori di mixing schedule (`cyclic`, `piecewise_dirichlet`, `random_walk`) con factory `make_schedule` e misure di distanza TV fra round. |
| `metrics.py` | Wrapper per stimare `pi_hat` dai soli esempi, distanza TV, stima `K_eff` (shuffle o MP) e funzioni specifiche per lag/ampiezza, forgetting ed equity delle magnetizzazioni. |
| `io.py` | Helper I/O (JSON, NPY/NPZ, scansione `round_XXX`) piu `atomic_write`. |
| `hopfield_hooks.py` | Ponte verso `src.unsup.hopfield_eval`: esegue la valutazione Hopfield round-wise, aggrega i risultati o ricostruisce matrici `m_{mu}(t)`. |
| `reporting.py` | Carica `metrics.json` per ogni round (`collect_round_metrics`), calcola lag/ampiezza dal simplesso e `build_run_report` per produrre JSON/CSV e statistiche su `w`. |
| `plotting.py` | Figure richieste dai pannelli (simplesso, magnetizzazioni, lag/fase, scatter esposizione->magnetizzazione). |
| `merge.py` e `codebase_mixing.txt` | Utility per consolidare i sorgenti Python in un'unica snapshot testuale (usata per audit o allegati). |

**Interazioni principali**

- Tutte le pipeline importano gli stimatori/metriche core da `src.unsup` (es. `HyperParams`, `build_unsup_J_single`, `propagate_J`, `dis_check`, `run_or_load_hopfield_eval`).
- Gli script salvano nel run folder gli stessi artefatti (`metrics.json`, `J_rec.npy`, `xi_aligned.npy`) che vengono poi letti da `hopfield_hooks.py` e `reporting.py`.

---

## `narch/` - toolkit Exp-07 (novelty emergence)

| File | Ruolo |
| --- | --- |
| `novelty.py` | API per generare mixing schedule con novita (`novelty_schedule`), calcolare serie temporali dai run folder (`compute_series_over_run`) e stimare indicatori chiave: gap spettrale al confine `K_old`, round di rilevazione (`detect_novelty_round`), magnetizzazioni Hopfield coerenti con il pipeline Exp-06. |
| `adaptive_w_exp07.py` | Trasposizione delle policy di entropia adattiva con damping opzionale pensata per Exp-07 (modalita `ema`, `rate_limit`, `momentum`, `adaptive_ema`). |
| `mnist_utils.py` | Helper per dataset reali (`binarize_pm1`, `build_class_prototypes`, `sample_round_single`, `prepare_mnist_triplet_single`) senza dipendenze esterne: costruisce `xi_true` e round SINGLE-mode da array gia caricati. |
| `plots.py` | Estensioni grafiche dedicate (simplesso colorato nel tempo, scree plot pre/post novita, heatmap delle magnetizzazioni, ablation sulle scheduler). |
| `reporting.py` | Genera l'output richiesto dal panel Exp-07: `series.json` (K_eff, gap, TV, L1, m_old/m_new, eventuali `pi_true/pi_hat`) e figure `fig_timeseries.png`, `fig_pi_error.png`, `fig_simplex.png`. |
| `merge.py` / `codebase.txt` | Stesso scopo di `mixing`: consolidare il pacchetto per audit. |

**Interazioni principali**

- Riusa il nucleo `src.unsup` per dinamiche TAM/Hopfield, layout dei run folder e stimatori (`tv_distance`, `simplex_embed_2d`, ecc.).
- Offre funzioni di alto livello come `report_novelty_summary` per elaborare una run generata dal pipeline base, aggiungendo serie temporali e figure pronte per il panel.

---


### Sottostrutture e componenti speciali

- `__old__/`: contiene versioni archiviate delle dinamiche (`dynamics.py`, `dynamics_single_mode.py`) utili come riferimento storico.
- `spectral/`: mini pacchetto autonomo con:
  - `deformed_mp.py`: helper per stimare il bordo destro MP (classico e "deformed" via Monte-Carlo).
  - `robust.py`: implementazione della Tyler shape matrix e normalizzazioni di traccia/diagonale per scatter robusti.
  - `tests/test_end2end_fixed.py`: test S8 che genera round sintetici (`gen_spiked_rounds`, `gen_no_spike_rounds`) e verifica Keff, gap e stime `kappa` usando la strumentazione dei pannelli.

### Dipendenze trasversali

- I moduli `mixing` e `narch` importano direttamente `HyperParams`, `build_unsup_J_single`, `propagate_J`, `dis_check`, `run_or_load_hopfield_eval` e gli helper spettrali per evitare duplicazioni.
- Tutti i pipeline e i report assumono il layout dei run folder definito qui (`round_XXX/metrics.json`, `round_XXX/xi_aligned.npy`, `round_XXX/J_rec.npy`, ecc.) garantendo compatibilita incrociata.

---

## Script ausiliari condivisi

- Ogni sottocartella include uno script `merge.py` e il relativo `codebase*.txt` per aggregare rapidamente i `.py` locali quando serve allegare la "foto" del codice alle relazioni ufficiali.
- I file `codebase*.txt` vengono rigenerati lanciando `python merge.py` dalla cartella corrispondente; non servono per l'esecuzione ma documentano esattamente la versione dei moduli impiegata in un esperimento.

Questa panoramica ti aiuta a capire dove implementare nuove feature o dove reperire le primitive gia pronte (per esempio riutilizzare `mixing/control.py` per altre policy su `w`, oppure estendere `narch/reporting.py` con figure aggiuntive sfruttando le serie esposte in `novelty.py`).
