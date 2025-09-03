import itertools, json
from pathlib import Path
from dataclasses import replace

from typing import TypedDict, List, Any, Dict

from src.unsup.config import HyperParams
from src.unsup.runner_single import run_exp01_single


class AggregateDict(TypedDict):
    retrieval_mean: List[float]
    fro_mean: List[float]
    keff_mean: List[float]
    coverage_mean: List[float]
    retrieval_se: List[float]
    fro_se: List[float]
    keff_se: List[float]
    coverage_se: List[float]

def main() -> None:
    root = Path("out_01/synth_sweep"); root.mkdir(parents=True, exist_ok=True)

    # Regimi che coprono lo span:
    Ks_per_client = [3, 4, 5, 6]        # da disgiunto → forte overlap
    r_ex_vals     = [1.0, 0.9, 0.8, 0.7]
    w_vals        = [0.2, 0.4, 0.6, 0.8]
    prop_iters    = [50, 200]           # poca vs molta propagazione
    regimes_M     = {                   # data-scarce / medium / rich
      "scarce": dict(M_total=1200),
      "medium": dict(M_total=2400),
      "rich":   dict(M_total=4800),
    }

    combos = list(itertools.product(regimes_M.items(), Ks_per_client, r_ex_vals, w_vals, prop_iters))
    total = len(combos)

    for idx, (reg_pair, Kpc, rx, w, pit) in enumerate(combos, start=1):
        regime, md = reg_pair  # md contiene solo M_total
        out_dir = root / f"r{idx:03d}_{regime}_Kpc{Kpc}_rex{rx}_w{w}_it{pit}"
        try:
            hp = HyperParams(
                mode="single", L=3, K=9, N=300, n_batch=24,
                K_per_client=Kpc, r_ex=rx, w=w, M_total=md["M_total"],
                n_seeds=5, seed_base=1000, use_tqdm=False,
            )
            # override iterazioni di propagazione
            hp = replace(hp, prop=replace(hp.prop, iters=pit))

            res = run_exp01_single(hp, out_dir=str(out_dir), do_plot=True)

            # quick summary per filtrare in seguito
            agg = res["aggregate"]  # type: ignore[assignment]
            # cast per type checker
            agg_t: AggregateDict = agg  # type: ignore[assignment]
            quick = {
                "retrieval_last": agg_t["retrieval_mean"][-1],
                "fro_last": agg_t["fro_mean"][-1],
                "keff_last": agg_t["keff_mean"][-1],
                "coverage_last": agg_t["coverage_mean"][-1],
            }
            with (out_dir / "quick.json").open("w", encoding="utf-8") as f:
                json.dump(quick, f, indent=2)
            print(f"[{idx}/{total}] OK -> {out_dir}")
        except Exception as e:
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "ERROR.txt").open("w", encoding="utf-8") as f:
                f.write(repr(e))
            print(f"[{idx}/{total}] ERRORE -> {out_dir}: {e}")


if __name__ == "__main__":
    main()
