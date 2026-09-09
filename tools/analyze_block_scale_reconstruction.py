#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from validate_block_scale_reconstruction_results import validate

COLORS = {16: "#1f77b4", 32: "#d95f02", 128: "#2ca02c"}
BASE_COLORS = {"raw_fp32": "#222222", "fp32_to_fp64": "#777777", "raw_fp64": "#aaaaaa"}


def quantiles(values):
    return np.quantile(values, [0.25, 0.5, 0.75]).tolist()


def label(variant):
    if variant == "raw_fp32": return "raw FP32, FP32 accumulation"
    if variant == "fp32_to_fp64": return "FP32 to FP64, FP64 accumulation"
    if variant == "raw_fp64": return "raw FP64, FP64 accumulation"
    if variant.startswith("deferred_local"):
        return f"B=128, S{variant[-2:]}, local deferred FP64"
    reconstruction, tail = variant.split("_b")
    block, scale = tail.split("_s")
    precision = "FP32 reconstruction" if reconstruction == "reconstruct32" else "FP64 reconstruction"
    return f"B={block}, S{scale}, {precision}"


def appearance(variant):
    if variant in BASE_COLORS:
        return BASE_COLORS[variant], {"raw_fp32": "-", "fp32_to_fp64": "--", "raw_fp64": ":"}[variant], "s"
    block = 128 if variant.startswith("deferred_local") else int(variant.split("_b")[1].split("_")[0])
    if variant.startswith("deferred_local"): return COLORS[block], "--", "^"
    if variant.startswith("reconstruct32"): return COLORS[block], ":", "D"
    if variant.endswith("s64"): return COLORS[block], "--", "o"
    return COLORS[block], "-", "o"


def spread(ax, entries, pixels=25):
    ax.figure.canvas.draw()
    inverse = ax.transData.inverted()
    ordered = sorted(entries, key=lambda item: ax.transData.transform((item[1], item[2]))[1])
    positions = []
    last = ax.bbox.y0 + 5
    for item in ordered:
        current = max(ax.transData.transform((item[1], item[2]))[1], last)
        positions.append(current)
        last = current + pixels
    if positions[-1] > ax.bbox.y1 - 5:
        shift = positions[-1] - (ax.bbox.y1 - 5)
        positions = [position - shift for position in positions]
    for index in range(len(positions) - 2, -1, -1):
        positions[index] = min(positions[index], positions[index + 1] - pixels)
    if positions[0] < ax.bbox.y0 + 5:
        shift = ax.bbox.y0 + 5 - positions[0]
        positions = [position + shift for position in positions]
    for item, position in zip(ordered, positions):
        item[3] = inverse.transform((ax.bbox.x1, position))[1]


def embed(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def main():
    parser = argparse.ArgumentParser()
    for name in ("samples", "manifest", "audit", "correctness", "environment", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    rows, manifest = validate(args.samples, args.manifest, "full")
    audit_bytes = Path(args.audit).read_bytes()
    audit = json.loads(audit_bytes)
    expected = {("reconstruct64", b, s) for b in (16, 32, 128) for s in ("fp32", "fp64")}
    expected |= {("reconstruct32", b, "fp32") for b in (16, 32, 128)}
    expected |= {("deferred_local", 128, s) for s in ("fp32", "fp64")}
    expected |= {(x, 0, "none") for x in ("raw_fp32", "fp32_to_fp64", "raw_fp64", "reduce32", "reduce64")}
    observed = {(item["kind"], int(item["B"]), item["scale_type"]) for item in audit}
    if len(audit) != 16 or len({item.get("symbol") for item in audit}) != 16 or observed != expected or any(item["status"] != "pass" for item in audit):
        raise ValueError("assembly audit inventory failed")
    if hashlib.sha256(audit_bytes).hexdigest() != manifest["audit_sha256"]:
        raise ValueError("audit digest does not match manifest")
    checks = Path(args.correctness).read_text()
    check_lines = checks.splitlines()
    for marker in ("all_passed=1", "validated_case_count=238", "rounding_contracts_distinct=1"):
        if marker not in check_lines:
            raise ValueError("correctness inventory failed: " + marker)
    digits = [line for line in check_lines if line.startswith("long_double_digits=")]
    if len(digits) != 1 or int(digits[0].split("=", 1)[1]) <= 53:
        raise ValueError("long double precision evidence failed")
    generated = [0, 1, 15, 16, 17, 31, 32, 33, 127, 128, 129, 257, 4099, 1048576]
    expected_checks = {f"validated_n={n},fixture=generated,cases=14" for n in generated}
    expected_checks |= {"validated_n=4099,fixture=edge,cases=14", "validated_n=4099,fixture=all_positive,cases=14", "validated_n=257,fixture=rounding_sensitive,cases=14"}
    observed_checks = [line for line in check_lines if line.startswith("validated_n=")]
    if len(observed_checks) != 17 or set(observed_checks) != expected_checks:
        raise ValueError("correctness dataset inventory failed")
    environment_bytes = Path(args.environment).read_bytes()
    if hashlib.sha256(environment_bytes).hexdigest() != manifest["environment_sha256"]:
        raise ValueError("environment digest does not match manifest")
    environment = environment_bytes.decode(errors="replace")

    groups = defaultdict(list)
    for row in rows:
        groups[(row["stage"], int(row["N"]), row["variant_id"])].append(float(row["time_ms"]))
    official = {n: "rerun" if any(key[0] == "rerun" and key[1] == n for key in groups) else "initial" for n in manifest["sizes"]}
    variants = sorted({row["variant_id"] for row in rows})
    summary = []
    for n in manifest["sizes"]:
        baseline = quantiles(groups[(official[n], n, "fp32_to_fp64")])[1]
        for variant in variants:
            q1, median, q3 = quantiles(groups[(official[n], n, variant)])
            summary.append(dict(N=n, stage=official[n], variant_id=variant, n=50,
                                q1_ms=q1, median_ms=median, q3_ms=q3,
                                ratio_fp32_to_fp64=median / baseline))
    output = Path(getattr(args, "output_dir"))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "timing_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)

    fig, ax = plt.subplots(figsize=(17, 11))
    ends = []
    for variant in variants:
        points = [item for item in summary if item["variant_id"] == variant]
        xs = [item["N"] for item in points]
        medians = [item["median_ms"] for item in points]
        q1 = [item["q1_ms"] for item in points]
        q3 = [item["q3_ms"] for item in points]
        color, style, marker = appearance(variant)
        ax.plot(xs, medians, color=color, ls=style, marker=marker, lw=2)
        ax.fill_between(xs, q1, q3, color=color, alpha=0.07)
        ends.append([variant, xs[-1], medians[-1], None, color])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(manifest["sizes"], [f"2^{int(np.log2(n))}" for n in manifest["sizes"]])
    ax.set_xlabel("DOT length N, log2 spacing")
    ax.set_ylabel("Two-kernel time, milliseconds, log scale")
    ax.set_title("Reconstruction precision and thread-local deferred scaling on H200")
    ax.grid(True, which="both", alpha=0.2)
    spread(ax, ends)
    for variant, x, y, label_y, color in ends:
        ax.annotate("", (x, y), xytext=(x * 1.52, label_y), arrowprops=dict(arrowstyle="-", linestyle=":", color=color), annotation_clip=False)
        ax.text(x * 1.57, label_y, label(variant), color=color, fontsize=8.7, va="center")
    ax.set_xlim(manifest["sizes"][0] / 1.25, manifest["sizes"][-1] * 4.5)
    fig.tight_layout()
    fig.savefig(output / "dot_block_scale_reconstruction_all.png", dpi=180)
    fig.savefig(output / "dot_block_scale_reconstruction_all.svg")
    plt.close(fig)

    n28 = manifest["sizes"][-1]
    points = sorted([item for item in summary if item["N"] == n28], key=lambda item: item["median_ms"])
    fig, ax = plt.subplots(figsize=(17, 10))
    label_entries = []
    for item in points:
        color, style, marker = appearance(item["variant_id"])
        ax.hlines(item["median_ms"], 0.03, 1.0, color=color, ls=style, lw=2)
        ax.fill_between([0.03, 1.0], item["q1_ms"], item["q3_ms"], color=color, alpha=0.07)
        ax.plot(1.0, item["median_ms"], marker=marker, color=color)
        label_entries.append([item["variant_id"], 1.0, item["median_ms"], None, color])
    ax.set_xlim(0, 1.72)
    ax.set_xticks([])
    ax.set_ylabel("Two-kernel time, milliseconds, linear scale")
    ax.set_title("N=2^28 reconstruction and local-deferred comparison")
    ax.grid(True, axis="y", alpha=0.2)
    spread(ax, label_entries, pixels=25)
    for variant, x, y, label_y, color in label_entries:
        ax.plot([x, 1.10], [y, label_y], color=color, ls=":", lw=1)
        ax.text(1.12, label_y, label(variant), color=color, va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "dot_block_scale_reconstruction_n28.png", dpi=180)
    fig.savefig(output / "dot_block_scale_reconstruction_n28.svg")
    plt.close(fig)

    drift = []
    for n in manifest["sizes"]:
        for stage in ("initial", "rerun"):
            if not any(row["stage"] == stage and int(row["N"]) == n for row in rows): continue
            for variant in ("raw_fp32", "fp32_to_fp64", "raw_fp64"):
                values = [float(row["time_ms"]) for row in sorted(rows, key=lambda item: int(item["round"])) if row["stage"] == stage and int(row["N"]) == n and row["variant_id"] == variant]
                ratio = abs(float(np.median(values[25:])) - float(np.median(values[:25]))) / float(np.median(values[:25]))
                drift.append(f"N={n}, {stage}, {label(variant)}: {ratio:.2%}" + (" (persistent above 5%)" if stage == "rerun" and ratio > 0.05 else ""))
    comparisons = [f"N={item['N']}, {label(item['variant_id'])}: median {item['median_ms']:.6g} ms, IQR {item['q1_ms']:.6g} to {item['q3_ms']:.6g} ms, {item['ratio_fp32_to_fp64']:.3f}x FP32-to-FP64" for item in summary]
    body = f"""<!doctype html><meta charset=utf-8><title>Block-scale reconstruction</title><style>body{{font:16px system-ui;max-width:1550px;margin:32px auto;color:#17202a}}img{{width:100%;height:auto}}pre{{background:#f4f5f6;padding:12px;white-space:pre-wrap}}li{{margin:.3em}}</style><h1>Reconstruction precision and thread-local deferred scaling</h1><p>The full inventory has 70 initial cases and 3,500 samples. Fourteen configurations ran at each N in 50 shuffled rounds. Points show medians and bands show interquartile ranges. Inputs repeat without explicit cache flushing.</p><img src='{embed(output/'dot_block_scale_reconstruction_all.png')}'><img src='{embed(output/'dot_block_scale_reconstruction_n28.png')}'><h2>Arithmetic contracts</h2><p>FP64 reconstruction widens each payload and scale before multiplication. FP32 reconstruction rounds each q*s product in FP32, widens both results, then uses FP64 DOT accumulation. Raw FP32 alone accumulates in FP32. Local deferred B128 forms one dependent four-product FP64 subtotal per lane, applies the scale product to every lane, and has no intermediate warp reduction. These are different rounding contracts.</p><p>Both operands have independent scales. S64 stores the exact widening of S32. The generator did not fit accuracy-optimal block scales; this experiment measures decoder and DOT cost. Scale storage costs 32+scale_bits/B bits per value.</p><h2>GPU environment</h2><pre>{html.escape(environment)}</pre><h2>Baseline drift</h2><ul>{''.join('<li>'+html.escape(item)+'</li>' for item in drift)}</ul><h2>All medians and quartiles</h2><ul>{''.join('<li>'+html.escape(item)+'</li>' for item in comparisons)}</ul><h2>Correctness evidence</h2><pre>{html.escape(checks)}</pre><h2>Run manifest</h2><pre>{html.escape(json.dumps(manifest,indent=2))}</pre><p>These timings compare full two-kernel DOT implementations. Local deferred scaling is DOT-specific and changes summation order. The measurements do not establish an accuracy ranking or a generic accessor speedup.</p>"""
    (output / "report.html").write_text(body)
    (output / "screenshot.html").write_text(f"<!doctype html><meta charset=utf-8><title>Block-scale reconstruction</title><style>body{{font-family:system-ui;margin:24px;background:#fff}}h1,p{{max-width:1600px;margin-left:auto;margin-right:auto}}img{{display:block;width:min(1800px,100%);margin:24px auto}}</style><h1>Block-scale reconstruction on H200</h1><p>N = 2^16, 2^20, 2^24, 2^26, 2^28. All 14 cases.</p><img src='{embed(output/'dot_block_scale_reconstruction_all.png')}'><img src='{embed(output/'dot_block_scale_reconstruction_n28.png')}'>")
    print(f"analysis passed: {len(rows)} rows, {len(summary)} official summaries")


if __name__ == "__main__":
    main()
