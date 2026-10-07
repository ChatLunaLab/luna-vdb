# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Draw the README performance charts from a CI comparison run.

The `Compare with pre-rewrite engine` CI job uploads `compare.txt` as the
`compare-results` artifact: the output of `compare/src/main.rs` followed by
`compare/src/sweep.rs`. This script parses that text and writes SVGs to
`docs/charts/`:

    gh run download <run-id> -n compare-results -D /tmp/compare
    uv run scripts/charts.py /tmp/compare/compare.txt

Charts are drawn from the parsed numbers only, so a new CI run regenerates
them without editing anything here.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "charts"

OLD = "#9aa0a6"
NEW = "#1a73e8"
PQ = "#e8710a"
COLORS = ["#1a73e8", "#e8710a", "#188038", "#d93025", "#9334e6"]

CJK_FONTS = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Songti SC",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Microsoft YaHei",
    "WenQuanYi Zen Hei",
]


def setup_fonts() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    else:
        print("warning: no CJK font found; Chinese labels will not render", file=sys.stderr)
    plt.rcParams["axes.unicode_minus"] = False
    # Glyphs become paths, so the SVG renders the same without the font, and
    # the output is byte-stable for the same input.
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams["svg.hashsalt"] = "luna-vdb"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

NUM = r"([0-9.]+)"


@dataclass
class Case:
    dim: int
    size: int
    search: tuple[float, float, float, float] | None = None  # old ms, new ms, old recall, new recall
    clustered: tuple[float, float, float, float] | None = None
    build: tuple[float, float] | None = None
    serialise: tuple[float, float] | None = None
    restore: tuple[float, float] | None = None


@dataclass
class Sweep:
    data: str
    dim: int
    size: int
    scan_ms: float = 0.0
    has_pq: bool = False  # calibration may drop PQ; the "pq" sweep is then plain IVF
    default: tuple[int, float, float] | None = None  # nprobe, ms, recall
    pq_default: tuple[int, float, float] | None = None
    flat: list[tuple[float, float]] = field(default_factory=list)  # (recall, ms)
    pq: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class Results:
    cases: list[Case]
    sweeps: list[Sweep]
    ingest: tuple[float, float] | None  # old ms/add, new ms/add


def parse(text: str) -> Results:
    cases: list[Case] = []
    sweeps: list[Sweep] = []
    ingest_old = ingest_new = None
    in_sweep = False

    for line in text.splitlines():
        if line.startswith("luna-vdb recall/latency sweep"):
            in_sweep = True
            continue

        if not in_sweep:
            if m := re.match(r"dim\s+(\d+)\s+n\s+(\d+)\s+\(new:", line):
                cases.append(Case(int(m[1]), int(m[2])))
            elif m := re.match(
                rf"\s+search\s+old\s+{NUM} ms\s+new\s+{NUM} ms.*recall old {NUM} new {NUM}", line
            ):
                cases[-1].search = (float(m[1]), float(m[2]), float(m[3]), float(m[4]))
            elif m := re.match(
                rf"\s+search/clustered\s+old\s+{NUM} ms\s+new\s+{NUM} ms.*recall old {NUM} new {NUM}",
                line,
            ):
                cases[-1].clustered = (float(m[1]), float(m[2]), float(m[3]), float(m[4]))
            elif m := re.match(rf"\s+(build|serialise|restore)\s+old\s+{NUM} ms\s+new\s+{NUM} ms", line):
                setattr(cases[-1], m[1], (float(m[2]), float(m[3])))
            elif m := re.match(rf"\s+old\s+{NUM} ms/add", line):
                ingest_old = float(m[1])
            elif m := re.match(rf"\s+new\s+{NUM} ms/add", line):
                ingest_new = float(m[1])
            continue

        if m := re.match(r"== (\w+) dim (\d+) n (\d+)", line):
            sweeps.append(Sweep(m[1], int(m[2]), int(m[3])))
        elif not sweeps:
            continue
        elif m := re.match(rf"\s+exact scan\s+{NUM} ms", line):
            sweeps[-1].scan_ms = float(m[1])
        elif m := re.match(r"\s+ivf-pq build .* pq (true|false)", line):
            sweeps[-1].has_pq = m[1] == "true"
        elif m := re.match(rf"\s+default \(nprobe\s+(\d+)\)\s+{NUM} ms\s+recall {NUM}", line):
            sweeps[-1].default = (int(m[1]), float(m[2]), float(m[3]))
        elif m := re.match(rf"\s+pq default \(nprobe\s+(\d+)\)\s+{NUM} ms\s+recall {NUM}", line):
            sweeps[-1].pq_default = (int(m[1]), float(m[2]), float(m[3]))
        elif m := re.match(rf"\s+ivf-flat cells\s+{NUM}\s+{NUM} ms\s+recall {NUM}", line):
            sweeps[-1].flat.append((float(m[3]), float(m[2])))
        elif m := re.match(rf"\s+ivf-pq\s+cells\s+{NUM}\s+{NUM} ms\s+recall {NUM}", line):
            sweeps[-1].pq.append((float(m[3]), float(m[2])))

    ingest = (ingest_old, ingest_new) if ingest_old and ingest_new else None
    return Results(cases, sweeps, ingest)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def label(dim: int, size: int) -> str:
    count = f"{size // 1000} 千" if size < 10_000 else f"{size // 10_000} 万"
    return f"{dim} 维\n{count}条"


def times(value: float) -> str:
    return f"{value:.0f}×" if value >= 10 else f"{value:.1f}×"


def search_chart(results: Results, path: Path) -> None:
    """Old vs new search latency, uniform and clustered data, recall on top."""
    cases = [c for c in results.cases if c.search and c.clustered]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    width = 0.38

    for ax, attr, title in [
        (axes[0], "search", "均匀随机数据（无簇结构，最坏情况）"),
        (axes[1], "clustered", "簇状数据（高斯混合，接近真实 embedding）"),
    ]:
        xs = range(len(cases))
        for i, case in zip(xs, cases):
            old_ms, new_ms, old_recall, new_recall = getattr(case, attr)
            ax.bar(i - width / 2, old_ms, width, color=OLD, label="旧引擎" if i == 0 else None)
            ax.bar(i + width / 2, new_ms, width, color=NEW, label="新引擎（默认参数）" if i == 0 else None)
            for x, value, recall in [(i - width / 2, old_ms, old_recall), (i + width / 2, new_ms, new_recall)]:
                bad = recall < 0.9
                ax.text(
                    x,
                    value * 1.15,
                    f"{recall:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#d93025" if bad else "#3c4043",
                    fontweight="bold" if bad else "normal",
                )
            speedup = old_ms / new_ms
            ax.text(
                i,
                max(old_ms, new_ms) * 2.4,
                times(speedup),
                ha="center",
                fontsize=10,
                fontweight="bold",
                color=NEW if speedup >= 1 else "#5f6368",
            )
        ax.set_yscale("log")
        ax.set_ylim(0.01, 200)
        ax.set_xticks(list(xs), [label(c.dim, c.size) for c in cases], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", which="both", alpha=0.25)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("单次查询耗时 (ms，对数坐标)")
    axes[1].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle(
        "查询耗时：旧引擎 vs 新引擎（k = 10）　柱顶数字为召回率@10，红色表示低于 0.9；粗体为加速比",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


def operations_chart(results: Results, path: Path) -> None:
    """old ÷ new for build, serialise, restore, plus ingest."""
    cases = [c for c in results.cases if c.build and c.serialise and c.restore]
    ops = [("build", "构建索引"), ("serialise", "序列化"), ("restore", "反序列化")]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    width = 0.26

    for j, (attr, name) in enumerate(ops):
        for i, case in enumerate(cases):
            old_ms, new_ms = getattr(case, attr)
            speedup = old_ms / new_ms
            x = i + (j - 1) * width
            ax.bar(x, speedup, width, color=COLORS[j], label=name if i == 0 else None)
            ax.text(x, speedup * 1.12, times(speedup), ha="center", va="bottom", fontsize=8)

    ax.axhline(1, color="#3c4043", linewidth=1, linestyle="--")
    ax.set_xlim(-0.8, len(cases) - 0.3)
    ax.text(-0.77, 1.08, "持平", ha="left", va="bottom", fontsize=8, color="#3c4043")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 60)
    ax.set_xticks(range(len(cases)), [label(c.dim, c.size) for c in cases], fontsize=9)
    ax.set_ylabel("旧耗时 ÷ 新耗时（对数坐标，越高越好）")
    title = "其他操作的加速比"
    if results.ingest:
        old_add, new_add = results.ingest
        title += f"　｜　逐条写入：{old_add:,.0f} ms → {new_add:.3f} ms / 次（{old_add / new_add:,.0f}×）"
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncols=3)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


def recall_chart(results: Results, path: Path) -> None:
    """Recall vs speedup over the exact scan, across nprobe, per dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

    for ax, data, title, legend_at in [
        (axes[0], "clustered", "簇状数据", "lower left"),
        (axes[1], "uniform", "均匀随机数据", "upper right"),
    ]:
        sweeps = [s for s in results.sweeps if s.data == data and s.scan_ms > 0]
        for colour, sweep in zip(COLORS, sweeps):
            name = f"{sweep.dim} 维 / {sweep.size:,} 条"
            curves = [(sweep.flat, "-", "")]
            if sweep.has_pq:
                curves.append((sweep.pq, ":", " + PQ"))
            for points, style, suffix in curves:
                if not points:
                    continue
                recalls = [recall for recall, _ in points]
                speedups = [sweep.scan_ms / ms for _, ms in points]
                ax.plot(
                    recalls,
                    speedups,
                    style,
                    marker="o",
                    markersize=3,
                    color=colour,
                    linewidth=1.4,
                    label=name + suffix,
                )
            if sweep.default:
                _, ms, recall = sweep.default
                ax.plot(
                    recall,
                    sweep.scan_ms / ms,
                    marker="*",
                    markersize=15,
                    color=colour,
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    zorder=5,
                )

        # Clustered recall stays near 1 at every nprobe; zoom in so the
        # curves are distinguishable rather than one vertical smear.
        recalls = [r for s in sweeps for r, _ in s.flat + (s.pq if s.has_pq else [])]
        low = 0.0 if not recalls or min(recalls) < 0.8 else round(min(recalls) - 0.015, 2)
        ax.set_xlim(low, 1.005)
        if low < 0.9:
            ax.axvline(0.9, color="#d93025", linewidth=1, linestyle="--")
            ax.text(0.893, 0.25, "召回率 0.9", color="#d93025", fontsize=8, rotation=90, va="bottom", ha="right")
        if data == "uniform":
            ax.annotate(
                "★ 校准后探测全部簇，\n即精确搜索",
                xy=(1.0, 1.0),
                xytext=(0.45, 0.3),
                fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#3c4043"},
            )
        ax.axhline(1, color="#3c4043", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.2, 300)
        ax.set_xlabel("召回率@10")
        ax.set_title(title, fontsize=11)
        ax.grid(which="both", alpha=0.25)
        ax.set_axisbelow(True)
        ax.legend(fontsize=7, frameon=False, loc=legend_at, ncols=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("相对精确扫描的加速比（对数坐标）")
    fig.suptitle(
        "新引擎近似搜索的召回率与速度：曲线为逐个调大 nprobe 的结果，实线不带 PQ，点线带 PQ；★ 为自动校准的默认值",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} compare.txt")

    results = parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if not results.cases or not results.sweeps:
        sys.exit("no comparison or sweep output found in the input")

    setup_fonts()
    OUT.mkdir(parents=True, exist_ok=True)
    search_chart(results, OUT / "search.svg")
    operations_chart(results, OUT / "operations.svg")
    recall_chart(results, OUT / "recall.svg")
    for name in ("search.svg", "operations.svg", "recall.svg"):
        print(OUT / name)


if __name__ == "__main__":
    main()
