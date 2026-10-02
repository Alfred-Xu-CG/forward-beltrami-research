"""Read completed H-proxy and matched gray scores; never optimize or select maps."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "stain_proxy_all25_t21"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    prediction = read(OUT / "predictions.json")
    assert prediction["prediction_complete"] and len(prediction["rows"]) == 25
    assert all(r["status"] in ("ok", "failed") for r in prediction["rows"])
    hm = read(OUT / "miit_scores.json")
    he = read(OUT / "existing_scores.json")
    gm = read(ROOT / "miit_multiscale_control_t19" / "landmark_scores.json")
    ge = read(ROOT / "existing22_shared_affine_a300_t20" / "landmark_scores.json")
    groups = {"miit": (gm["aggregate"]["analytic"]["all_three"], hm["aggregate"]["analytic"]["all_three"])}
    groups.update({key: (ge[key]["all_directions"], he[key]["all_directions"])
                   for key in ("lung_all20", "histo", "rat_kidney")})
    aggregates = {}
    for name, (gray, stain) in groups.items():
        aggregates[name] = None if gray is None or stain is None else {
            "gray": gray["canvas_pixels"], "hematoxylin_proxy": stain["canvas_pixels"],
            "delta": {key: stain["canvas_pixels"][key] - gray["canvas_pixels"][key]
                      for key in ("mean_pair_mean", "mean_pair_p90")}}
    rows = []
    for cohort, gray, stain in (("miit", gm, hm), ("existing", ge, he)):
        g = {r["name"]: r for r in gray["rows"]}
        for r in stain["rows"]:
            old = g[r["name"]]
            left = old["methods"]["analytic"] if cohort == "miit" else old
            right = r["methods"]["analytic"] if cohort == "miit" else r
            item = dict(name=r["name"], status=right["status"])
            if left["status"] == right["status"] == "ok":
                for metric in ("mean", "p90", "maximum"):
                    a, b = left["metrics"]["canvas_pixels"][metric], right["metrics"]["canvas_pixels"][metric]
                    item[metric] = dict(gray=a, hematoxylin_proxy=b, delta=b-a)
            rows.append(item)
    ok = [r for r in prediction["rows"] if r["status"] == "ok"]
    costs = {key: dict(minimum=min(r[key] for r in ok), maximum=max(r[key] for r in ok),
                      mean=float(np.mean([r[key] for r in ok])))
             for key in ("complete_call_seconds", "peak_allocated_bytes", "actual_minimum_corner_ratio")}
    result = dict(scope="four previously viewed specimens; 25 correlated directions, not 25 patients",
                  units="512 moving-canvas pixels", prediction_manifest=str(OUT / "predictions.json"),
                  complete=len(ok) == 25, cohorts=aggregates, rows=rows, costs=costs,
                  experiment_wall_seconds=prediction["elapsed_seconds"],
                  warning="Different H/gray image functionals; do not compare E totals as convergence.")
    destination = OUT / "comparison.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(cohorts=aggregates, costs=costs), indent=2))


if __name__ == "__main__":
    main()
