# Experiment 07: Novelty Emergence with Entropy-Based Adaptive Damping

## Overview

Questo esperimento simula l'emergenza graduale di nuovi archetipi durante il training federato e confronta diverse strategie di adattamento basate su **entropy-based damping**.

## Strategie Disponibili

### 1. `baseline` - Fixed Weight
Peso fisso `w_baseline` senza adattamento.
- **Pro**: Semplice, stabile
- **Contro**: Non si adatta ai cambiamenti distributivi

### 2. `ema` - Exponential Moving Average
Smoothing EMA del peso adattivo:
```
w_t = α * w_raw + (1-α) * w_{t-1}
```
- **Parametri**: `--alpha-ema` (default: 0.3)
- **Pro**: Transizioni graduali
- **Quando usare**: Adattamento moderato senza salti bruschi

### 3. `rate_limit` - Rate Limiting
Hard clipping della variazione massima di w per round:
```
Δw = clip(w_raw - w_{t-1}, -max_delta_w, +max_delta_w)
```
- **Parametri**: `--max-delta-w` (default: 0.15)
- **Pro**: Previene jumps catastrofici
- **Quando usare**: Evitare catastrophic forgetting

### 4. `momentum` - Momentum-Based
Aggiornamenti con momentum:
```
delta = (1-β) * (w_raw - w_{t-1})
w_t = w_{t-1} + delta
```
- **Parametri**: `--momentum` (default: 0.7)
- **Pro**: Bilanciato tra reattività e stabilità
- **Quando usare**: Risposta rapida ma smooth

### 5. `adaptive_ema` - Adaptive EMA (⭐ Raccomandato)
EMA con alpha adattivo basato sull'incertezza entropica:
```
uncertainty = (H_AB - H_min) / (H_max - H_min)
α = α_max - uncertainty * (α_max - α_min)
w_t = α * w_raw + (1-α) * w_{t-1}
```
- **Parametri**: `--alpha-min` (default: 0.1), `--alpha-max-adapt` (default: 0.5)
- **Pro**: Context-aware, rileva automaticamente distributional shifts
- **Quando usare**: Migliore per rilevamento novelty

## Metriche di Entropy

Tutte le strategie adattive (tranne `baseline`) usano **sign-consistency entropy** per rilevare cambiamenti distributivi:

- **H_AB**: Entropia dell'accordo tra segni di `J_memory` e `J_current`
- **Alta H_AB** → Distribuzione instabile → Peso alto (fiducia nei nuovi dati)
- **Bassa H_AB** → Distribuzione stabile → Peso basso (fiducia nella memoria)

## Esempi di Utilizzo

### 1. Quick Start (Raccomandato)
```powershell
# Confronto baseline vs adaptive_ema
python scripts\exp07_novelty_emergence.py --strategy baseline adaptive_ema
```

### 2. Test Singola Strategia
```powershell
# Solo EMA damping
python scripts\exp07_novelty_emergence.py --strategy ema --alpha-ema 0.2

# Solo rate limiting aggressivo
python scripts\exp07_novelty_emergence.py --strategy rate_limit --max-delta-w 0.1

# Solo momentum
python scripts\exp07_novelty_emergence.py --strategy momentum --momentum 0.5
```

### 3. Confronto Completo
```powershell
# Tutte le strategie
python scripts\exp07_novelty_emergence.py `
    --strategy baseline ema rate_limit momentum adaptive_ema `
    --n-seeds 10
```

### 4. Configurazione Custom
```powershell
# Adaptive EMA con parametri custom
python scripts\exp07_novelty_emergence.py `
    --strategy adaptive_ema `
    --alpha-min 0.15 `
    --alpha-max-adapt 0.6 `
    --K-old 5 `
    --K-new 3 `
    --t-intro 15 `
    --ramp-len 6 `
    --n-seeds 8
```

### 5. Esperimento Veloce
```powershell
# Test rapido con pochi seeds e rounds
python scripts\exp07_novelty_emergence.py `
    --strategy baseline adaptive_ema `
    --n-seeds 3 `
    --n-batch 16 `
    --no-progress
```

## Parametri Principali

### Modello
- `--K-old`: Archetipi iniziali (default: 3)
- `--K-new`: Nuovi archetipi da introdurre (default: 3)
- `--N`: Dimensione pattern (default: 400)
- `--L`: Numero di client (default: 3)
- `--n-batch`: Round totali (default: 24)
- `--r-ex`: Signal-to-noise ratio (default: 0.8)

### Novelty Schedule
- `--t-intro`: Round di introduzione nuovi archetipi (default: 12)
- `--ramp-len`: Lunghezza rampa introduzione (default: 4)
- `--alpha-max`: Allocazione massima ai nuovi archetipi (default: 0.5)

### Baseline
- `--w-baseline`: Peso fisso per baseline (default: 0.8)

### EMA Damping
- `--alpha-ema`: Coefficiente EMA (default: 0.3)
  - Valori alti (0.5-0.7): Adattamento veloce
  - Valori bassi (0.1-0.3): Smoothing aggressivo

### Rate Limiting
- `--max-delta-w`: Variazione massima w per round (default: 0.15)
  - Valori alti (0.2-0.3): Cambiamenti rapidi
  - Valori bassi (0.05-0.1): Cambimenti graduali

### Momentum
- `--momentum`: Coefficiente momentum (default: 0.7)
  - Valori alti (0.8-0.9): Inerzia maggiore
  - Valori bassi (0.3-0.5): Risposta più rapida

### Adaptive EMA
- `--alpha-min`: Alpha minimo (alta incertezza) (default: 0.1)
- `--alpha-max-adapt`: Alpha massimo (bassa incertezza) (default: 0.5)

### Experiment
- `--n-seeds`: Numero di seed (default: 6)
- `--seed-base`: Seed iniziale (default: 200001)
- `--no-progress`: Disabilita progress bar
- `--out-dir`: Directory output custom

## Output

L'esperimento genera:

### File
- `hyperparams.json`: Configurazione esperimento
- `results_detailed.csv`: Risultati completi per round/seed/strategy
- `summary.csv`: Statistiche finali per strategy
- `log.jsonl`: Log esecuzione

### Plot (`fig_novelty_emergence.png`)
1. **Panel A**: K_eff detection - Rilevamento dimensionalità
2. **Panel B**: Retrieval old vs new - Performance per tipo di archetipo
3. **Panel C**: Spectral gap - Gap spettrale al boundary K_old
4. **Panel D**: Mixing error - Errore stima distribuzione
5. **Panel E**: Adaptive weight evolution - Evoluzione peso w_t
6. **Panel F**: Sign-consistency entropy - Entropia H_AB (solo adaptive)

## Risultati Attesi

### Detection Rate
Frazione di seed che rilevano correttamente i nuovi archetipi (K_eff ≥ K_old + 1)

### Retrieval Performance
- **m_old**: Overlap medio con archetipi vecchi (dovrebbe rimanere alto)
- **m_new**: Overlap medio con archetipi nuovi (dovrebbe crescere dopo t_intro)

### Strategie Migliori per Novelty Detection
1. **adaptive_ema**: Miglior bilanciamento, rileva shift automaticamente
2. **ema**: Buono con alpha moderato (0.2-0.3)
3. **rate_limit**: Stabile ma può essere lento
4. **momentum**: Reattivo ma può oscillare
5. **baseline**: Worst case, nessun adattamento

## Best Practices

### Per Rilevamento Rapido
```powershell
python scripts\exp07_novelty_emergence.py `
    --strategy adaptive_ema `
    --alpha-min 0.2 `
    --alpha-max-adapt 0.6
```

### Per Stabilità Massima
```powershell
python scripts\exp07_novelty_emergence.py `
    --strategy ema `
    --alpha-ema 0.15
```

### Per Prevenire Catastrophic Forgetting
```powershell
python scripts\exp07_novelty_emergence.py `
    --strategy rate_limit `
    --max-delta-w 0.08
```

## Troubleshooting

### Problema: Detection Rate basso
**Soluzione**: Aumentare reattività
```powershell
--alpha-ema 0.4  # per ema
--alpha-max-adapt 0.7  # per adaptive_ema
```

### Problema: m_old degrada dopo t_intro
**Soluzione**: Aumentare smoothing
```powershell
--alpha-ema 0.15  # per ema
--max-delta-w 0.08  # per rate_limit
```

### Problema: Oscillazioni in w_t
**Soluzione**: Usare damping più aggressivo
```powershell
--strategy rate_limit --max-delta-w 0.1
```

## Riferimenti

- **Entropy-based adaptation**: `src/narch/adaptive_w_exp07.py`
- **Novelty schedule**: `src/narch/novelty.py`
- **TAM dynamics**: `src/unsup/functions.py`

## Note Tecniche

- Tutte le strategie adattive computano `w_raw` tramite entropy, poi applicano damping
- `baseline` usa peso fisso, NON computa entropy
- Sign-consistency entropy misura agreement tra `sign(J_memory)` e `sign(J_current)`
- Adaptive EMA modula alpha in base a uncertainty per context-awareness ottimale
