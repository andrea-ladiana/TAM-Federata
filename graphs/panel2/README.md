# Panel 2: BBP Theorem Validation

Validazione sperimentale del teorema BBP (Baik-Ben Arous-Péché) per l'apprendimento Hebbiano federato.

---

## Quick Start

```bash
python bbp_theorem_demo.py           # Genera dati (3 archetipi forti + 3 deboli)
python plot_panel2.py                # Crea figura 4-panel (A,B,C,D)
python utils/verify_experiment.py   # Verifica risultati
```

**Output**: `output/panel2_bbp_validation.pdf` ⭐

---

## Struttura Cartella

```
panel2/
├── bbp_theorem_demo.py          ← Script principale generazione dati
├── plot_panel2.py               ← Script principale visualizzazione
├── README.md                    ← Questa guida
├── 04_experiments.tex           ← Framework teorico completo
│
├── utils/                       ← Script diagnostici e verifica
│   ├── verify_experiment.py    ← Verifica rilevazione vs teoria
│   ├── plot_diagnostic_spectrum.py
│   ├── plot_exposure_comparison.py
│   └── print_summary.py
│
├── scripts_extra/               ← Script avanzati (batch, fix, alt. plots)
│   ├── run_panel2_experiment.py ← Wrapper interattivo
│   ├── batch_experiments.py     ← Esperimenti batch
│   ├── fix_empirical_sharp.py   ← Fix vecchi dati (deprecated)
│   └── plot_panel_C_vs_q.py     ← Plot alternativo Panel C
│
├── config_template.json         ← Template configurazione
├── example_config.json          ← Esempio configurazione
│
└── output/                      ← Risultati (generati automaticamente)
    ├── panel2_bbp_validation.pdf
    ├── bbp_demo_data.npz
    └── bbp_demo_log.json
```

---

## Configurazione Esperimento

### Parametri
- **N** = 400 (dimensione)
- **K** = 3 archetipi **forti**, exposure α ≈ [0.50, 0.30, 0.20]
- **K_weak** = 3 archetipi **deboli**, exposure α ≈ [0.0002, 0.00015, 0.0001]
- **r** = 0.95 (qualità canale), **σ²** = 0.0975
- **M** = [400, 1600, 6400] (sample sizes)
- **n_trials** = 50 (per statistiche robuste)

### Risultati Attesi (M=400, q=1.0)

| Archetipo | Tipo | κ | λ_out | Rilevato? |
|-----------|------|---|-------|-----------|
| ξ₁ | Forte | 1850 | 180.6 | ✓ SI (κ >> √q) |
| ξ₂ | Forte | 1110 | 108.5 | ✓ SI |
| ξ₃ | Forte | 740 | 72.4 | ✓ SI |
| ξ₄ | Debole | 0.74 | - | ✗ NO (κ < √q) |
| ξ₅ | Debole | 0.56 | - | ✗ NO |
| ξ₆ | Debole | 0.37 | - | ✗ NO |

**Validazione**: 9/9 archetipi forti rilevati, 3/3 deboli nascosti ✓

---

## File Principali

### Core (nella root)
- **`bbp_theorem_demo.py`** - Generazione dati sperimentali
- **`plot_panel2.py`** - Figura pubblicazione 4-panel

### Utils (cartella `utils/`)
- **`verify_experiment.py`** - Verifica correttezza rilevazione
- `plot_diagnostic_spectrum.py` - Plot spettro completo (top 30 autovalori)
- `plot_exposure_comparison.py` - Visualizza exposure ↔ rilevabilità
- `print_summary.py` - Tabelle dettagliate parametri/predizioni

### Scripts Extra (cartella `scripts_extra/`)
- `run_panel2_experiment.py` - Wrapper interattivo per esperimenti
- `batch_experiments.py` - Esecuzione batch multipli esperimenti
- `plot_panel_C_vs_q.py` - Plot alternativo Panel C (vs q invece di γ)
- `fix_empirical_sharp.py` - Fix vecchi dati (deprecated)

---

## I 4 Panel

### A (Top-Left): Eigenspectrum
Top 20 autovalori. **Cerchi pieni** = rilevati (λ > λ₊). **Cerchi vuoti** = bulk (λ ≤ λ₊).

### B (Top-Right): Validazione λ_out
λ_out empirico vs teorico BBP. **R² > 0.95**, errore < 2%.

### C (Bottom-Left): Allineamento Autovettori
Overlap |⟨v,u⟩|² empirico vs γ(κ,q) teorico. **Gap mediano ≈ 0.016**.

### D (Bottom-Right): Effetto Sharpening
Errore relativo pre/post sharpening. **RMS error ↑ ~14x** dopo sharpening.

---

## Personalizzazione

### Cambiare Exposure
Edita `bbp_theorem_demo.py`:
```python
alpha = np.array([0.5, 0.3, 0.2])                      # Forti
alpha_weak = np.array([0.0002, 0.00015, 0.0001])       # Deboli
```

### Cambiare Numero Archetipi
```python
run_exposure_sweep(K=3, K_weak=3, ...)
```

### Esperimenti Batch
```bash
python scripts_extra/run_panel2_experiment.py --config my_config.json
```

---

## Teoria BBP

### Soglia di Rilevabilità
**κ_μ > √q** ⟺ archetipo μ rilevabile spettralmente

Dove:
- **κ_μ = (r²/σ²) × α_μ × N** (signal-to-noise)
- **q = N/M** (aspect ratio)
- **α_μ** (exposure)

### Formule Chiave

```
λ_out(κ) = σ²(1 + κ)(1 + q/κ)          [per κ > √q]
γ(κ,q) = (1 - q/κ²) / (1 + q/κ)        [overlap autovettore]
λ₊(q) = σ²(1 + √q)²                     [bordo bulk MP]
```

---

## Caricamento Dati

```python
import numpy as np
data = np.load('output/bbp_demo_data.npz', allow_pickle=True)
results = data['results'].item()

# Per M specifico
res = results[400]
theory = res['theory']           # Predizioni teoriche
emp = res['empirical']           # Risultati empirici
all_eigs = emp['all_eigs']       # Spettro completo (N autovalori)
```

---

## Troubleshooting

**Archetipi deboli rilevati?** → Riduci `alpha_weak` (target: κ < √q)  
**Archetipi forti non rilevati?** → Aumenta `alpha` o M  
**File not found?** → Esegui `bbp_theorem_demo.py` prima  

---

**Documentazione completa**: `04_experiments.tex` (framework teorico matematico)  
*Versione 2.0 - Novembre 2025*
