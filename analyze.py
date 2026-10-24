"""Equal-quality comparison and decision rule (fixed BEFORE the measurements).

For each model and corpus, interpolate (log-perplexity linear in bits per weight, between the two
trellis points that bracket it) the bits per weight the trellis needs to match the perplexity of each
scalar format (best variant, with or without rotation). Saving = 1 - bpw_trellis / bpw_scalar.

Rule: keep the trellis codec only if, on EVERY model and EVERY corpus, it matches the best 4-bit scalar
format (the one least favourable to it) with at least 10 % fewer bits.

Usage: python analyze.py [--quick]
"""
import glob, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = "--quick" in sys.argv
THRESHOLD = 0.10
BASELINES = ["NVFP4", "NVFP4 + scale search", "MXFP4", "INT4 g128", "Lloyd-Max 4b g32", "Lloyd-Max 3b g32"]


def trellis_bpw(points, target):
    """points: [(bpw, ppl)] of the trellis. Returns (bpw, status)."""
    pts = sorted(points)
    for (b1, p1), (b2, p2) in zip(pts, pts[1:]):
        if p1 >= target >= p2:
            a = (math.log(p1) - math.log(target)) / (math.log(p1) - math.log(p2))
            return b1 + a * (b2 - b1), "interpolated"
    if target < pts[-1][1]:
        return pts[-1][0], "not reached (needs more than)"
    return pts[0][0], "already reached at (at most)"


def main():
    pattern = "*.quick.json" if QUICK else "*.json"
    files = [f for f in glob.glob(os.path.join(HERE, "results", pattern))
             if (QUICK or ".quick" not in f) and not os.path.basename(f).startswith("analysis")]
    summary, keep = {}, True
    for f in sorted(files):
        r = json.load(open(f, encoding="utf-8"))
        model = r["_meta"]["model"]
        tre = {k: v for k, v in r.items() if k.startswith("Trellis")}
        if len(tre) < 2:
            print(model, ": not enough trellis points")
            continue
        summary[model] = {}
        for corpus in ["wikitext2", "c4"]:
            pts = [(v["bpw"], v[corpus]) for v in tre.values()]
            rows, worst = [], None
            for base in BASELINES:
                variants = [(n, r[n]) for n in (base, base + " +RHT") if n in r]
                if not variants:
                    continue
                best_name, best = min(variants, key=lambda nv: nv[1][corpus])
                b, status = trellis_bpw(pts, best[corpus])
                saving = 1 - b / best["bpw"]
                rows.append({"format": base, "best_variant": best_name, "bpw": best["bpw"], "ppl": round(best[corpus], 3),
                             "trellis_bpw_at_equal_ppl": round(b, 3), "status": status, "saving": round(saving, 3)})
                if base != "Lloyd-Max 3b g32" and (worst is None or saving < worst["saving"]):
                    worst = rows[-1]
            summary[model][corpus] = {"bf16": round(r["BF16"][corpus], 3), "comparisons": rows, "worst_case": worst}
            print(f"\n{model} / {corpus} (BF16 {r['BF16'][corpus]:.3f})")
            for row in rows:
                print(f"  {row['format']:20s} {row['bpw']:5.3f} bpw ppl {row['ppl']:8.3f} -> trellis {row['trellis_bpw_at_equal_ppl']:5.3f} bpw "
                      f"({row['status']}) saving {100*row['saving']:+5.1f} %")
            if worst is not None:
                print(f"  => worst case (4-bit formats): {worst['format']}: {100*worst['saving']:+.1f} %")
                if worst["saving"] < THRESHOLD or worst["status"].startswith("not reached"):
                    keep = False
    decision = "KEEP the trellis codec" if (keep and summary) else "DROP the trellis codec"
    print(f"\nRule: >= {100*THRESHOLD:.0f} % saving against the best 4-bit scalar format, everywhere. Decision: {decision}")
    json.dump({"threshold": THRESHOLD, "decision": decision, "summary": summary},
              open(os.path.join(HERE, "results", "analysis" + (".quick" if QUICK else "") + ".json"), "w", encoding="utf-8"),
              indent=1)


if __name__ == "__main__":
    main()
