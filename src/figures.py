"""Figures and tables for the report, read only from results/ (nothing is typed in by hand).

Run:  python -m src.figures            all figures
      python -m src.figures 3 5        only figures 3 and 5
Writes figures/*.png (300 dpi) and figures/main_results.csv / .md.
A run that has not finished is skipped and printed as TODO.
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.data import load_config
from src.evaluate import known_labels_long
from src.experiments import load_sets
from src.schemas import CATEGORIES
from src.stats import paired_difference, set_b_counts, set_b_metrics

OUT = Path("figures")
INK, SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#ffffff"
# Color follows the system, in every figure. LearnedCounsel is always blue.
COLOR = {"B1": "#8a8985", "B2": "#b9b8b3", "B3": "#eb6834", "B4": "#1baf7a", "B5": "#eda100",
         "LearnedCounsel": "#2a78d6"}
RUNS = {"B1": "b1_main", "B2": "b2_A", "B3": "b3_main", "B4": "b4_main", "B5": "b5_B",
        "LearnedCounsel": "learnedcounsel_main"}
ABLATIONS = [("Full system", "learnedcounsel_ablation"), ("(a) no skeptic", "learnedcounsel_no_skeptic_ablation"),
             ("(b) no router", "learnedcounsel_no_router_ablation"),
             ("(c) no consults", "learnedcounsel_no_consults_ablation"),
             ("(d) one general specialist", "learnedcounsel_single_specialist_ablation"),
             ("(e) 0 debate rounds", "learnedcounsel_rounds0_ablation"),
             ("(e) 1 debate round", "learnedcounsel_rounds1_ablation"),
             ("(f) tau 0.5", "learnedcounsel_tau0.5_ablation"), ("(f) tau 0.7", "learnedcounsel_tau0.7_ablation"),
             ("(g) 4B specialists", "learnedcounsel_mixed_4b_ablation"),
             ("(h) few-shot", "learnedcounsel_few_shot_ablation")]
PAIRS = [("divorce", "family"), ("domestic_violence", "family"), ("consumer", "business"),
         ("courts", "crime"), ("torts", "crime"), ("torts", "traffic"), ("torts", "housing"), ("torts", "health")]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": SECONDARY,
                     "xtick.color": SECONDARY, "ytick.color": SECONDARY, "axes.edgecolor": GRID,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def metrics(run):
    path = Path("results") / run / "metrics.json"
    if not path.exists():
        print(f"TODO: {run} has not run")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def predictions(run):
    return pd.read_csv(Path("results") / run / "predictions.csv")


def clean(ax, grid="y"):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("saved", OUT / name)


# ---------------- 1. architecture ----------------
def fig1():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(-1, 111), ax.set_ylim(0, 56), ax.axis("off")

    def box(x, y, w, h, title, text="", color="#e8f1fc", edge="#2a78d6"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=color, ec=edge, lw=1.3))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="center", fontsize=9.5, weight="bold")
        ax.text(x + w / 2, y + (h - 4) / 2, text, ha="center", va="center", fontsize=7.8, color=SECONDARY)

    def arrow(a, b, label="", curve=0.0, dx=0, dy=1.3, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, color=SECONDARY, lw=1.2, ls=ls,
                                     connectionstyle=f"arc3,rad={curve}"))
        if label:
            ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, label, ha="center", fontsize=7.8, color=SECONDARY)

    box(0, 34, 13, 10, "Intake", "facts, goals,\nurgency")
    box(18, 34, 13, 10, "Router", "which specialists\nto call")
    groups = [("Housing", "housing"), ("Family", "family, divorce,\nDV, estates"), ("Work & Money", "employment, consumer,\nbenefits, business"),
              ("Courts & Crime", "courts, crime,\ntraffic"), ("Health & More", "health, education,\nimmigration, torts")]
    for i, (name, cats) in enumerate(groups):
        box(37, 45.5 - i * 8.6, 19, 7, name, cats, color="#e6f6ef", edge="#1baf7a")
    ax.text(46.5, 54.3, "5 specialists (independent)", ha="center", fontsize=8.5, color=SECONDARY)
    box(64, 33, 18, 11, "Skeptic", "challenges weak claims,\nflags possible misses", color="#fdeee7", edge="#eb6834")
    box(64, 14, 18, 10, "Rebuttal", "same specialist,\nnew call", color="#e6f6ef", edge="#1baf7a")
    box(90, 24, 17, 12, "Coordinator", "final 16 labels,\nurgency, referral note,\nsettled or reopen")
    arrow((13.6, 39), (17.4, 39)), arrow((31.6, 39), (36.3, 39))
    arrow((56.8, 38.5), (63.4, 38.5), "claims,\nquotes", dy=0.9)
    arrow((73, 32.4), (73, 24.8), "challenges", dx=6, dy=-0.5)
    arrow((82.6, 20.5), (89.4, 27), "answers", dx=-2.4, dy=1.4)
    arrow((98, 23.4), (82.6, 16), "reopen (within 2 rounds)", curve=-0.25, dx=8, dy=-5)
    arrow((56.8, 24.5), (56.8, 14), "", curve=-0.6, ls="--")
    ax.text(57.5, 9.6, "consult requests", ha="left", fontsize=7.5, color=SECONDARY)
    box(0, 0.8, 107, 6.2, "Shared case file (JSON, validated with pydantic)", "", color="#f3f2ef", edge="#8a8985")
    ax.text(53.5, 2.3, "every agent reads only the parts it needs and writes its result; specialists never see each other's output in round 0",
            ha="center", va="center", fontsize=7.8, color=SECONDARY)
    save(fig, "fig1_architecture.png")


# ---------------- 2. main results table ----------------
def fig2():
    rows = []
    for name, run in RUNS.items():
        m = metrics(run)
        if m is None:
            continue
        ci, cost = m["ci"], m.get("cost")
        cell = lambda d, k: f"{d[k]['value']:.3f} [{d[k]['ci95'][0]:.3f}, {d[k]['ci95'][1]:.3f}]" if d else "-"
        covers_a = m["set_a"]["n_items"] >= 0.5 * m["set_a"]["n_items_in_set"]
        lat = metrics({"B3": "latency_b3", "B4": "latency_b4", "B5": "latency_b5",
                       "LearnedCounsel": "latency_learnedcounsel"}[name]) if name in ["B3", "B4", "B5", "LearnedCounsel"] else None
        rows.append({"System": name, "Set A accuracy": cell(ci["set_a"], "mean_accuracy") if covers_a else "-",
                     "Set B micro-F1": cell(ci["set_b"], "micro_f1"), "Set B macro-F1": cell(ci["set_b"], "macro_f1"),
                     "Set B recall": cell(ci["set_b"], "micro_recall"),
                     "Calls/post": ("1 per item" if name == "B2" else f"{cost['calls_per_post']:.1f}") if cost else "-",
                     "Tokens/post": (f"{(cost['total_tokens_in'] + cost['total_tokens_out']) / m['set_a']['n_items']:,.0f} per item"
                                     if name == "B2" else f"{cost['tokens_in_per_post'] + cost['tokens_out_per_post']:,.0f}") if cost else "-",
                     "Latency s/post": f"{lat['cost']['latency_s_per_post']:.1f}" if lat else "-"})
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "main_results.csv", index=False)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(r) + " |" for r in df.values]
    (OUT / "main_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(13, 0.32 * len(df) + 0.3))
    ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center",
                 colWidths=[0.12, 0.165, 0.165, 0.165, 0.165, 0.085, 0.1, 0.115])
    t.auto_set_font_size(False), t.set_fontsize(8.5), t.scale(1, 1.6)
    for (r, c), cell_ in t.get_celld().items():
        cell_.set_edgecolor(GRID)
        cell_.set_text_props(weight="bold" if r == 0 or (r > 0 and df.iloc[r - 1, 0] == "LearnedCounsel") else "normal")
    save(fig, "fig2_main_results.png")


# ---------------- 3. recall by number of issues ----------------
def fig3():
    systems = ["B1", "B3", "B4", "B5", "LearnedCounsel"]
    data = {s: m["set_b"]["recall_by_known_positives"] for s in systems if (m := metrics(RUNS[s]))}
    labels = known_labels_long(load_sets(load_config())["set_b_sample"], CATEGORIES)
    n = labels[labels.label == 1].n_known_pos.map(lambda k: "3+" if k >= 3 else str(k)).value_counts()
    fig, ax = plt.subplots(figsize=(8.5, 4))
    width, buckets = 0.15, ["1", "2", "3+"]
    for i, (s, rec) in enumerate(data.items()):
        xs = [b + (i - (len(data) - 1) / 2) * width for b in range(3)]
        ys = [rec[b] for b in buckets]
        ax.bar(xs, ys, width * 0.9, color=COLOR[s], label=s)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.012, f"{y:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(3), [f"{b} known issue{'s' if b != '1' else ''}\n({n[b]} labels)" for b in buckets])
    ax.set_ylim(0, 1.08), ax.set_ylabel("Recall on known issues")
    ax.set_title("Recall by number of known issues per post (Set B)", pad=28)
    ax.legend(ncol=5, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.12), fontsize=9, handlelength=1)
    clean(ax)
    save(fig, "fig3_recall_by_issues.png")


# ---------------- 4. ablations ----------------
def fig4():
    sets = load_sets(load_config())
    base = predictions(ABLATIONS[0][1])
    rows = []
    for name, run in ABLATIONS:
        m = metrics(run)
        if m is None:
            continue
        ci = m["ci"]["set_b"]
        d = paired_difference(set_b_counts, set_b_metrics, predictions(run), base, sets["set_b_sample"], CATEGORIES)
        rows.append({"variant": name, "f1": ci["micro_f1"], "recall": ci["micro_recall"], "d": d["micro_f1"],
                     "tokens": m["cost"]["tokens_in_per_post"] + m["cost"]["tokens_out_per_post"]})
    rows = rows[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(rows) + 1.4), sharey=True)
    for ax, key, title in [(axes[0], "f1", "Micro-F1 (95% CI)"), (axes[1], "d", "Change vs full system (paired, 95% CI)")]:
        for y, r in enumerate(rows):
            color = COLOR["LearnedCounsel"] if r["variant"] == "Full system" else SECONDARY
            ax.plot(r[key]["ci95"], [y, y], color=color, lw=2)
            ax.plot(r[key]["value"], y, "o", color=color, ms=8, mec=SURFACE, mew=1.5)
            text = f"{r[key]['value']:+.3f}" if key == "d" else f"{r[key]['value']:.3f}"
            ax.text(r[key]["value"], y + 0.24, text, ha="center", fontsize=7.8)
        ax.set_title(title)
        clean(ax, "x")
    axes[1].axvline(0, color=INK, lw=1)
    axes[0].set_yticks(range(len(rows)), [f"{r['variant']}  ({r['tokens'] / 1000:.1f}k tokens)" for r in rows])
    axes[0].set_ylim(-0.6, len(rows) - 0.3)
    save(fig, "fig4_ablations.png")


# ---------------- 5. F1 vs tokens ----------------
def fig5():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
    left = [(s, RUNS[s], COLOR[s]) for s in ["B3", "B4", "B5", "LearnedCounsel"]]
    right = [(f"tau {t}", run, COLOR["LearnedCounsel"]) for t, run in
             [("0.5", "learnedcounsel_tau0.5_ablation"), ("0.7", "learnedcounsel_tau0.7_ablation"),
              ("0.9", "learnedcounsel_ablation")]]
    for ax, points in [(axes[0], left), (axes[1], right)]:
        for name, run, color in points:
            m = metrics(run)
            if m is None:
                continue
            x = (m["cost"]["tokens_in_per_post"] + m["cost"]["tokens_out_per_post"]) / 1000
            f1 = m["ci"]["set_b"]["micro_f1"]
            ax.plot([x, x], f1["ci95"], color=color, lw=2)
            ax.plot(x, f1["value"], "o", color=color, ms=9, mec=SURFACE, mew=1.5)
            side = -1 if name in ["B3", "LearnedCounsel", "tau 0.7"] else 1
            ax.annotate(f"{name}\n{f1['value']:.3f}", (x, f1["value"]), xytext=(9 * side, 0), textcoords="offset points",
                        fontsize=8.5, va="center", ha="left" if side > 0 else "right")
        ax.set_xlabel("Thousand tokens per post")
        clean(ax, "both")
    axes[0].set_xlim(0, 15.5), axes[1].set_xlim(11.8, 13.4)
    axes[0].set_ylabel("Set B micro-F1 (95% CI)")
    axes[0].set_title("Systems (120 posts)"), axes[1].set_title("Tau sweep (60-post ablation subset)")
    save(fig, "fig5_f1_vs_tokens.png")


# ---------------- 6. per-category heatmap ----------------
def fig6():
    systems = [s for s in ["B1", "B2", "B3", "B4", "LearnedCounsel"] if metrics(RUNS[s])]
    f1 = pd.DataFrame({s: {c: metrics(RUNS[s])["set_a"]["per_category"][c]["f1"] for c in CATEGORIES} for s in systems}).T
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(systems) + 1.8))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#5598e7", "#0d366b"])
    im = ax.imshow(f1.values, cmap=cmap, vmin=0.4, vmax=1, aspect="auto")
    for i in range(len(systems)):
        for j in range(len(CATEGORIES)):
            v = f1.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="#ffffff" if v > 0.75 else INK)
    ax.set_xticks(range(16), [c.replace("_", " ") for c in CATEGORIES], rotation=40, ha="right")
    ax.set_yticks(range(len(systems)), systems)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("F1 per category on Set A (15 items each, so single cells are noisy)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).outline.set_visible(False)
    save(fig, "fig6_category_heatmap.png")


# ---------------- 7. confusable pairs ----------------
def fig7():
    """Of posts where a is a known positive and b a known negative, how often is b predicted?"""
    sets = load_sets(load_config())
    known = known_labels_long(sets["set_b"], CATEGORIES).set_index(["post_id", "category"])["label"].unstack()
    rows = []
    for s in ["B3", "B4", "LearnedCounsel"]:
        if metrics(RUNS[s]) is None:
            continue
        pred = predictions(RUNS[s]).set_index(["post_id", "category"])["pred"].unstack()
        k = known.reindex(pred.index)
        for a, b in [p for x, y in PAIRS for p in [(x, y), (y, x)]]:
            sel = (k[a] == 1) & (k[b] == 0)
            if sel.sum() >= 3:
                rows.append({"system": s, "pair": f"{a} is there, {b} is not\n({sel.sum()} posts)", "rate": pred.loc[sel, b].mean(),
                             "n": int(sel.sum()), "hits": int(pred.loc[sel, b].sum())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fig7_confusable_pairs.csv", index=False)
    pairs = list(df.pair.unique())[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(pairs) + 1.3))
    for i, s in enumerate(["LearnedCounsel", "B4", "B3"]):
        d = df[df.system == s].set_index("pair").reindex(pairs)
        ys = [y + (i - 1) * 0.26 for y in range(len(pairs))]
        ax.barh(ys, d.rate, 0.23, color=COLOR[s], label=s)
        for y, r, h, n in zip(ys, d.rate, d.hits, d.n):
            ax.text(r + 0.01, y, f"{h:.0f}/{n:.0f}", va="center", fontsize=7.5)
    ax.set_yticks(range(len(pairs)), [p.replace("_", " ") for p in pairs], fontsize=8)
    ax.set_xlim(0, 1.08), ax.set_xlabel("How often the absent category was predicted anyway")
    ax.set_title("Confusable pairs (all 342 posts, known labels only)", pad=24)
    ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9, handlelength=1, reverse=True)
    clean(ax, "x")
    save(fig, "fig7_confusable_pairs.png")


# ---------------- 8. worked example ----------------
def fig8(post_id=None):
    """One post's transcript. Default: the shortest post where a challenge or miss changed a label."""
    folder = Path("logs/transcripts/learnedcounsel_main")
    best = None
    for path in sorted(folder.glob("*.jsonl")):
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        case = lines[-1]["case"]
        flips = [d for d in case["debate"] if d["type"] == "rebuttal" and d["before"] != d["after"] and d["kind"] != "reopen"]
        if (post_id == case["post_id"]) or (post_id is None and flips and len(lines) <= 12):
            if best is None or len(case["post"]) < len(best[-1]["case"]["post"]):
                best = lines
    case = best[-1]["case"]
    out = [("Post", case["post"])]
    for m in best[:-1]:
        o, a = m["output"] or {}, m["agent"]
        if a == "intake":
            text = f"urgency {o['urgency']}; facts: " + "; ".join(o["facts"][:4])
        elif a == "router":
            text = "call " + ", ".join(g for g, v in o.items() if v["call"])
        elif a.startswith("specialist"):
            text = "; ".join(f'{c} PRESENT ({v["confidence"]:.1f}): "{v["evidence_quote"]}"' for c, v in o.items()
                             if c != "consult_requests" and v["present"]) or "nothing present"
            if o.get("consult_requests"):
                called = set(case["specialist_outputs"])
                text += "; asks to consult " + ", ".join(
                    q["specialist"] + ("" if q["specialist"] in called else " [not honored: one level only]")
                    for q in o["consult_requests"])
        elif a == "skeptic":
            kept = {i["category"] for d in case["debate"] if d["type"] == "skeptic" for i in d["items"]}
            mark = lambda c: "" if c["category"] in kept else " [not passed on: the claim was not flagged for review]"
            text = "; ".join([f'challenge {c["category"]} ({c["type"]}): {c["argument"]}{mark(c)}' for c in o["challenges"]] +
                             [f'possible miss {c["category"]}: {c["reason"]}{mark(c)}' for c in o["possible_misses"]]) or "no objections"
        elif a.startswith("rebuttal"):
            text = f'{o["decision"]}, present={o["present"]}: {o["response"]}'
        else:
            text = "labels: " + ", ".join(c for c, v in o["labels"].items() if v["present"]) + f'; urgency {o["urgency"]}; settled {o["settled"]}. Note: ' + o["referral_note"]
        out.append((f"{a} (round {m['round']})", text))
    blocks = [(h, textwrap.wrap(t, 145)) for h, t in out]
    height = sum(1.35 + len(w) for _, w in blocks)
    fig, ax = plt.subplots(figsize=(11, 0.19 * height + 0.4))
    ax.axis("off"), ax.set_xlim(0, 1), ax.set_ylim(height, 0)
    y = 0.5
    for head, wrapped in blocks:
        ax.text(0, y, head, fontsize=9, weight="bold", va="center", color="#184f95")
        for line in wrapped:
            y += 1
            ax.text(0.012, y, line.replace("$", r"\$"), fontsize=8.2, va="center")
        y += 1.35
    print("fig8 post:", case["post_id"])
    save(fig, "fig8_worked_example.png")


# ---------------- 9. v2 on fresh posts ----------------
def fig9():
    runs = [("B3", "b3_v2_test", COLOR["B3"]), ("B4", "b4_v2_test", COLOR["B4"]),
            ("LearnedCounsel v1", "learnedcounsel_v1_test", "#9ec5f4"),
            ("LearnedCounsel v2", "learnedcounsel_v2_test", COLOR["LearnedCounsel"])][::-1]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.6), sharey=True)
    for ax, key, title in zip(axes, ["micro_f1", "micro_recall", "micro_precision"],
                              ["Micro-F1", "Recall", "Precision"]):
        for y, (name, run, color) in enumerate(runs):
            m = metrics(run)
            if m is None:
                continue
            v = m["ci"]["set_b"][key]
            ax.plot(v["ci95"], [y, y], color=color, lw=2.5)
            ax.plot(v["value"], y, "o", color=color, ms=9, mec=SURFACE, mew=1.5)
            ax.text(v["value"], y + 0.24, f"{v['value']:.3f}", ha="center", fontsize=8.5)
        ax.set_title(f"{title} (95% CI)")
        ax.set_xlim(0.78, 1.0)
        clean(ax, "x")
    axes[0].set_yticks(range(len(runs)), [r[0] for r in runs])
    axes[0].set_ylim(-0.5, len(runs) - 0.2)
    save(fig, "fig9_v2_test.png")


if __name__ == "__main__":
    for n in sys.argv[1:] or "123456789":
        globals()[f"fig{n}"]()
