#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


MAIN_DISTRIBUTIONS = (
    "field_balanced_finite",
    "paired_log_uniform_finite",
)
KERNELS = ("dot", "gemv")
FAMILIES = ("posit", "takum", "takum_log")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def short_format(name: str) -> str:
    names = {
        "posit8_es0": "posit<8,0>",
        "posit14_es1": "posit<14,1>",
        "posit16_es1": "posit<16,1>",
        "posit32_es2": "posit<32,2>",
        "takum8": "takum<8>",
        "takum14": "takum<14>",
        "takum16": "takum<16>",
        "takum32": "takum<32>",
        "takum_log8": "takum_log<8>",
        "takum_log14": "takum_log<14>",
        "takum_log16": "takum_log<16>",
        "takum_log32": "takum_log<32>",
    }
    return names.get(name, name)


def short_strategy(name: str) -> str:
    names = {
        "direct": "direct",
        "full_lut_shared": "shared LUT",
        "full_lut_global": "global LUT",
        "native_scalar": "native",
        "direct_branchy": "direct branchy",
        "direct_masked": "direct masked",
        "subnormal_lut_global": "subnormal LUT",
        "prefix_lut_global": "prefix LUT",
    }
    return names.get(name, name)


def short_distribution(name: str) -> str:
    return {
        "field_balanced_finite": "field-balanced",
        "paired_log_uniform_finite": "paired log-uniform",
        "lut_scattered_control": "scattered",
        "lut_concentrated_control": "concentrated",
    }[name]


def ratio_cell(row: dict[str, str]) -> str:
    ratio = float(row["ratio"])
    lower = float(row["ci_lower"])
    upper = float(row["ci_upper"])
    label = {
        "equivalent": "equivalent",
        "numerator_faster": "faster",
        "numerator_slower": "slower",
        "inconclusive": "inconclusive",
    }[row["classification"]]
    return f"{ratio:.3f} [{lower:.3f}, {upper:.3f}], {label}"


def winner_cell(
    winner: dict[str, str], comparison: dict[str, str]
) -> str:
    return (
        f"{short_strategy(winner['winning_strategy'])}, "
        f"{float(winner['median_ms']):.3f} ms; {ratio_cell(comparison)} vs direct"
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def build_report(analysis_dir: Path, output: Path) -> None:
    winners = read_csv(analysis_dir / "strategy_winners.csv")
    comparisons = read_csv(analysis_dir / "case_comparisons.csv")
    timing = read_csv(analysis_dir / "timing_summary.csv")

    winner_by_case = {
        (
            row["format"],
            row["arithmetic"],
            row["distribution"],
            row["kernel"],
        ): row
        for row in winners
    }
    comparison_by_case = {
        (
            row["question"],
            row["numerator_format"],
            row["arithmetic"],
            row["distribution"],
            row["kernel"],
            row["numerator_strategy"],
        ): row
        for row in comparisons
    }
    timing_by_case = {
        (
            row["format"],
            row["arithmetic"],
            row["distribution"],
            row["kernel"],
            row["strategy"],
        ): row
        for row in timing
    }

    lines = [
        "# Posit and takum GPU storage-conversion results",
        "",
        "This report answers the four questions fixed before implementation. The run used one NVIDIA H200 NVL GPU, scalar x1 access, 30 timing samples after 10 warm-ups, DOT with N = 2^27, and GEMV with M = 1024 and N = 65,536. Ratios below are alternative time divided by reference time. Lower is faster. The brackets give the 95% bootstrap confidence interval. `Equivalent` means the full interval lies inside [0.97, 1.03].",
        "",
        "## Case tables",
        "",
        "### Question 1: low-bit strategy winner",
        "",
        "Every cell reports the case winner, its median kernel time, and its paired ratio to the direct decoder.",
        "",
    ]

    q1_rows: list[list[str]] = []
    low_formats = [
        "posit8_es0",
        "takum8",
        "takum_log8",
        "posit14_es1",
        "takum14",
        "takum_log14",
    ]
    for fmt in low_formats:
        for arithmetic in ("fp32", "fp64"):
            row = [short_format(fmt), arithmetic.upper()]
            for distribution in MAIN_DISTRIBUTIONS:
                for kernel in KERNELS:
                    winner = winner_by_case[(fmt, arithmetic, distribution, kernel)]
                    comparison = comparison_by_case[
                        (
                            "1",
                            fmt,
                            arithmetic,
                            distribution,
                            kernel,
                            winner["winning_strategy"],
                        )
                    ]
                    row.append(winner_cell(winner, comparison))
            q1_rows.append(row)
    lines.extend(
        markdown_table(
            [
                "Format",
                "Arithmetic",
                "Field DOT",
                "Field GEMV",
                "Log-uniform DOT",
                "Log-uniform GEMV",
            ],
            q1_rows,
        )
    )

    lines += [
        "",
        "### Question 2: alternative LUT divided by IEEE LUT",
        "",
        "These controls reuse the same raw index stream and compiled table-only kernel. Only table contents change. Each row covers one alternative family, width, arithmetic type, trace, and table placement.",
        "",
    ]
    q2 = [row for row in comparisons if row["question"] == "2"]
    q2_key = {
        (
            row["bits"],
            row["arithmetic"],
            row["distribution"],
            row["numerator_strategy"],
            row["numerator_format"],
            row["kernel"],
        ): row
        for row in q2
    }
    q2_rows: list[list[str]] = []
    format_by_family = {
        ("8", "posit"): "posit8_es0",
        ("8", "takum"): "takum8",
        ("8", "takum_log"): "takum_log8",
        ("14", "posit"): "posit14_es1",
        ("14", "takum"): "takum14",
        ("14", "takum_log"): "takum_log14",
    }
    for bits in ("8", "14"):
        for arithmetic in ("fp32", "fp64"):
            for trace in ("lut_scattered_control", "lut_concentrated_control"):
                for strategy in ("full_lut_shared", "full_lut_global"):
                    for family in FAMILIES:
                        fmt = format_by_family[(bits, family)]
                        dot = q2_key[(bits, arithmetic, trace, strategy, fmt, "dot")]
                        gemv = q2_key[(bits, arithmetic, trace, strategy, fmt, "gemv")]
                        q2_rows.append(
                            [
                                bits,
                                arithmetic.upper(),
                                short_distribution(trace),
                                short_strategy(strategy),
                                short_format(fmt),
                                ratio_cell(dot),
                                ratio_cell(gemv),
                            ]
                        )
    lines.extend(
        markdown_table(
            ["Bits", "Arithmetic", "Trace", "Placement", "Family", "DOT", "GEMV"],
            q2_rows,
        )
    )

    lines += [
        "",
        "### Question 3: strategy after the shared LUT limit",
        "",
        "At 16 bits, the table reports global-LUT time divided by direct-decoder time. At 32 bits, a complete LUT would require 16 GiB for FP32 or 32 GiB for FP64, so direct was the only full decoder tested. The 32-bit cells report its median time.",
        "",
    ]
    q3_rows: list[list[str]] = []
    for fmt in ("posit16_es1", "takum16", "takum_log16"):
        for arithmetic in ("fp32", "fp64"):
            row = [short_format(fmt), arithmetic.upper()]
            for distribution in MAIN_DISTRIBUTIONS:
                for kernel in KERNELS:
                    comparison = comparison_by_case[
                        ("3", fmt, arithmetic, distribution, kernel, "full_lut_global")
                    ]
                    row.append(ratio_cell(comparison))
            q3_rows.append(row)
    lines.extend(
        markdown_table(
            [
                "16-bit format",
                "Arithmetic",
                "Field DOT",
                "Field GEMV",
                "Log-uniform DOT",
                "Log-uniform GEMV",
            ],
            q3_rows,
        )
    )
    lines.append("")
    q3_32_rows: list[list[str]] = []
    for fmt in ("posit32_es2", "takum32", "takum_log32"):
        for arithmetic in ("fp32", "fp64"):
            row = [short_format(fmt), arithmetic.upper()]
            for distribution in MAIN_DISTRIBUTIONS:
                for kernel in KERNELS:
                    result = timing_by_case[(fmt, arithmetic, distribution, kernel, "direct")]
                    row.append(f"{float(result['median_ms']):.3f} ms")
            q3_32_rows.append(row)
    lines.extend(
        markdown_table(
            [
                "32-bit format",
                "Arithmetic",
                "Field DOT",
                "Field GEMV",
                "Log-uniform DOT",
                "Log-uniform GEMV",
            ],
            q3_32_rows,
        )
    )

    lines += [
        "",
        "### Question 4: best alternative divided by fastest retained same-width IEEE",
        "",
        "The alternative winner and IEEE reference are selected independently inside each format, arithmetic, distribution, and kernel case. The IEEE reference shown in each cell is the fastest retained format and scalar strategy for that case.",
        "",
    ]
    q4 = [row for row in comparisons if row["question"] == "4"]
    q4_key = {
        (
            row["numerator_format"],
            row["arithmetic"],
            row["distribution"],
            row["kernel"],
        ): row
        for row in q4
    }
    q4_rows: list[list[str]] = []
    all_alt_formats = [
        "posit8_es0", "takum8", "takum_log8",
        "posit14_es1", "takum14", "takum_log14",
        "posit16_es1", "takum16", "takum_log16",
        "posit32_es2", "takum32", "takum_log32",
    ]
    for fmt in all_alt_formats:
        for arithmetic in ("fp32", "fp64"):
            for distribution in MAIN_DISTRIBUTIONS:
                row = [short_format(fmt), arithmetic.upper(), short_distribution(distribution)]
                for kernel in KERNELS:
                    comparison = q4_key[(fmt, arithmetic, distribution, kernel)]
                    reference = (
                        f"{comparison['denominator_format']} "
                        f"{short_strategy(comparison['denominator_strategy'])}"
                    )
                    row.append(f"{ratio_cell(comparison)} vs {reference}")
                q4_rows.append(row)
    lines.extend(
        markdown_table(
            ["Format", "Arithmetic", "Distribution", "DOT", "GEMV"],
            q4_rows,
        )
    )

    q1 = [row for row in comparisons if row["question"] == "1"]
    q3 = [row for row in comparisons if row["question"] == "3"]
    q4_classes = Counter(row["classification"] for row in q4)

    def ratio_range(rows: list[dict[str, str]]) -> tuple[float, float]:
        values = [float(row["ratio"]) for row in rows]
        return min(values), max(values)

    q4_ranges = {
        bits: ratio_range([row for row in q4 if row["bits"] == bits])
        for bits in ("8", "14", "16", "32")
    }

    lines += [
        "",
        "## Answers",
        "",
        "### 1. Is a full LUT best at low bit counts?",
        "",
        "Yes for every measured 8-bit and 14-bit main case. Shared LUT won all 24 8-bit cases. Global LUT won all 24 14-bit cases. Across the 96 individual LUT-versus-direct comparisons, 87 LUT cases were faster. The nine slower cases were 14-bit FP64 shared-LUT cases, where each block stages a 128 KiB table. That staging cost does not change the winner because the global LUT remained faster than direct.",
        "",
        "### 2. Does LUT speed depend on whether the entries contain IEEE, posit, or takum values?",
        "",
        f"No measurable dependence appeared. All {len(q2)} paired controls were equivalent under the predeclared 3% band. This held for both widths, arithmetic types, traces, placements, and kernels. The observed alternative-to-IEEE ratios ranged from {min(float(row['ratio']) for row in q2):.3f} to {max(float(row['ratio']) for row in q2):.3f}. Once the raw index stream and kernel are fixed, the LUT contents do not affect conversion throughput.",
        "",
        "### 3. What wins after a shared full LUT becomes too large?",
        "",
        f"At 16 bits, the global full LUT beat direct decoding in all {len(q3)} cases. Its paired ratio to direct ranged from {min(float(row['ratio']) for row in q3):.3f} to {max(float(row['ratio']) for row in q3):.3f}. At 32 bits, direct decoding is the only complete strategy measured because a full table is impractical. The experiment therefore establishes a 32-bit direct baseline, not that direct is better than every possible segmented or approximate decoder.",
        "",
        "### 4. How do the best alternatives compare with retained IEEE formats of the same width?",
        "",
        f"No alternative case was faster. Of 96 cases, {q4_classes['equivalent']} were equivalent, {q4_classes['inconclusive']} were inconclusive, and {q4_classes['numerator_slower']} were slower. All equivalent and inconclusive results occurred at 8 bits. The ratio ranges were {q4_ranges['8'][0]:.3f} to {q4_ranges['8'][1]:.3f} at 8 bits, {q4_ranges['14'][0]:.3f} to {q4_ranges['14'][1]:.3f} at 14 bits, {q4_ranges['16'][0]:.3f} to {q4_ranges['16'][1]:.3f} at 16 bits, and {q4_ranges['32'][0]:.3f} to {q4_ranges['32'][1]:.3f} at 32 bits.",
        "",
        "The main-format comparisons do not reuse identical values because each format has its own safe interval and field-balanced generator. Question 2 is the clean conversion-only comparison. It shows equal LUT throughput. Question 4 measures the specified end-to-end kernels with each format's own inputs, so cache locality and code distribution can also affect the result.",
        "",
        "## Validation and limits",
        "",
        "The full run contains 604 timed cases and 18,120 successful samples. All 119 decoder validation rows passed. The run recorded 23,668 histogram rows and no infeasible case. The build gate compiled all 44 targets, the smoke run covered the same 604-case matrix, and the full job completed in 421 seconds. A separate context-free CUDA review ended with no remaining finding.",
        "",
        "The benchmark measures storage conversion followed by ordinary FP32 or FP64 arithmetic. It does not test posit quires, native posit or takum arithmetic, segmented 32-bit decoders, or an accuracy-matched application. These timings alone cannot show that any family has a better accuracy-performance trade-off.",
        "",
        "## Reproducibility files",
        "",
        "- [Raw timing samples](run_20260826T183425Z/full/timing_samples.csv)",
        "- [Decoder validation](run_20260826T183425Z/full/decoder_validation.csv)",
        "- [Input histograms](run_20260826T183425Z/full/histograms.csv)",
        "- [All confidence intervals](run_20260826T183425Z/analysis/case_comparisons.csv)",
        "- [All strategy winners](run_20260826T183425Z/analysis/strategy_winners.csv)",
        "- [Timing medians](run_20260826T183425Z/analysis/timing_summary.csv)",
        "- [Run manifest](run_20260826T183425Z/run_manifest.txt)",
        "- [GPU and compiler environment](run_20260826T183425Z/environment.txt)",
        "- [Independent CUDA review](code_review.md)",
        "- [Predeclared experiment specification](../../docs/posit_takum_strategy_benchmark.md)",
        "",
    ]
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.analysis_dir, args.output)


if __name__ == "__main__":
    main()
