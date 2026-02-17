# Plotting Saved Results — Client-Aware Adaptive Weight

## 📊 Overview

Script `plot_saved_results.py` permette di rigenerare i plot dai dati salvati senza rifare le simulazioni.

## 🚀 Usage

### Plot base (correzione zero)

```bash
python client_aware_w/plot_saved_results.py
```

Questo:
- Carica automaticamente i dati da `client_aware_w/output/`
- Rileva automaticamente il mode (single-seed / multi-seed)
- Usa correzione magnetizzazioni `mag_correction = zeros(T)`
- Salva il plot in `client_aware_w/output/panel_client_aware_w.png`

### Plot con correzione personalizzata

```bash
python client_aware_w/plot_saved_results.py \
  --mag-correction 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
```

Il vettore `mag_correction` deve avere lunghezza `T` (numero di round).

Viene **aggiunto** sia a:
- Magnetizzazioni per archetipo $m_k(t)$
- Magnetizzazione media $\langle m \rangle$

### Opzioni avanzate

```bash
python client_aware_w/plot_saved_results.py \
  --data-dir path/to/data \
  --out-path path/to/output.png \
  --mag-correction 0.1 0.2 0.3 ... \
  --mode multi \
  --dpi 300
```

**Parametri:**

| Flag | Descrizione | Default |
|------|-------------|---------|
| `--data-dir` | Directory contenente i `.npy` files | `client_aware_w/output` |
| `--out-path` | Path del plot di output | `data-dir/panel_client_aware_w.png` |
| `--mag-correction` | Vettore di correzione (lunghezza T) | `zeros(T)` |
| `--mode` | `auto` / `single` / `multi` | `auto` |
| `--dpi` | Risoluzione figura | `300` |

## 📁 Files richiesti

### Single-seed mode
- `config.json`
- `w_history.npy`
- `retrieval_history.npy`
- `mag_history.npy`

### Multi-seed mode
- `config.json`
- `w_good_mean.npy`, `w_good_se.npy`
- `w_att_mean.npy`, `w_att_se.npy`
- `retr_mean.npy`, `retr_se.npy`
- `mag_mean.npy`, `mag_se.npy`

## 🎨 Output

Lo script genera sempre due files:
- `<out_path>.png` (raster, DPI specificato)
- `<out_path>.pdf` (vettoriale, per pubblicazioni)

## 🔧 Esempio workflow

1. **Run simulazione** (salva dati):
   ```bash
   python client_aware_w/run.py --T 10 --n-seeds 20
   ```

2. **Plot con correzione zero**:
   ```bash
   python client_aware_w/plot_saved_results.py
   ```

3. **Plot con correzione personalizzata**:
   ```bash
   python client_aware_w/plot_saved_results.py \
     --mag-correction 0.05 0.05 0.04 0.03 0.02 0.01 0.0 0.0 0.0 0.0 \
     --out-path client_aware_w/output/panel_corrected.png
   ```

4. **Confronta visivamente** i due plot.
