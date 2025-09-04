# Report MNIST Federated

Questa cartella contiene i report generati automaticamente.

## File
* metrics_rounds.csv / .json: metriche per round (retrieval, frobenius_rel, K_eff, coverage)
* metrics_summary.json: valori finali, medie e best.
* hopfield_summary.json: correlazioni e statistiche distribuzioni magnetizzazione.
* hopfield_exposure_vs_magnetization.csv: relazione esposizione vs magnetizzazione media.
* hopfield_magnetizations_long.csv: valori grezzi delle magnetizzazioni (long-form).
* config_and_digests.json: iperparametri e digest SHA delle matrici principali.

## Note
- I digest consentono di verificare se le matrici cambiano tra run.
- Per rigenerare completamente i dati: impostare FORCE_RERUN=True nello script.

Generato da `exp01_mnist.py`.
