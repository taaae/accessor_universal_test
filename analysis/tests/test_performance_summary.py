import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tools.build_storage_performance_report as performance_report
import tools.summarize_storage_performance as performance_summary

IDENTITY = performance_summary.IDENTITY
MODEL_FIELDS = performance_summary.MODEL_FIELDS
bottleneck = performance_summary.bottleneck


def test_timing_summary_and_packed_speedup(tmp_path):
    performance_summary.EXPECTED_FORMATS = {"fp32_e8m23"}
    performance_summary.EXPECTED_DISTRIBUTIONS = {"normal_0_1"}
    performance_summary.EXPECTED_COMPONENTS = {"dot"}
    samples = tmp_path / "samples.csv"
    fields = IDENTITY + MODEL_FIELDS + [
        "round",
        "sample",
        "order_position",
        "time_ms",
    ]
    rows = []
    for lanes, time_ms in ((1, 4.0), (2, 2.0), (4, 1.0)):
        for round_index in range(2):
            for sample_index in range(2):
                identity = {
                    "gpu": "test_gpu",
                    "compute_capability": "sm_90",
                    "distribution": "normal_0_1",
                    "format": "fp32_e8m23",
                    "storage_bits": "32",
                    "component": "dot",
                    "lanes": str(lanes),
                    "n": "1024",
                    "m": "1",
                    "leading_dimension": "1024",
                    "blocks": str(4 // lanes + 1),
                    "threads": "256",
                    "decode_repeats": "0",
                }
                model = {field: "1" for field in MODEL_FIELDS}
                rows.append(
                    {
                        **identity,
                        **model,
                        "round": round_index,
                        "sample": sample_index,
                        "order_position": (lanes + round_index + sample_index)
                        % 3,
                        "time_ms": time_ms,
                    }
                )
    with samples.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    performance_summary.summarize_timings(samples, tmp_path)

    with (tmp_path / "packed_speedups.csv").open(newline="") as stream:
        speedups = {int(row["lanes"]): row for row in csv.DictReader(stream)}
    assert float(speedups[1]["speedup_vs_x1"]) == 1.0
    assert float(speedups[2]["speedup_vs_x1"]) == 2.0
    assert float(speedups[4]["speedup_vs_x1"]) == 4.0


def test_bottleneck_classification():
    assert bottleneck(80.0, 30.0, 70.0, 60.0) == "memory_bandwidth"
    assert bottleneck(20.0, 80.0, 70.0, 60.0) == "compute_or_conversion"
    assert bottleneck(60.0, 60.0, 70.0, 60.0) == "mixed_memory_compute"
    assert bottleneck(10.0, 10.0, 12.0, 10.0) == "parallelism_or_occupancy"


def test_same_bit_groups_cover_every_non_fp64_format_once():
    grouped = [
        format_name
        for formats in performance_report.BIT_FORMATS.values()
        for format_name in formats
    ]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(performance_report.FORMAT_ORDER) - {"fp64_e11m52"}

    all_grouped = [
        format_name
        for formats in performance_report.ALL_BIT_FORMATS.values()
        for format_name in formats
    ]
    assert len(all_grouped) == len(set(all_grouped))
    assert set(all_grouped) == set(performance_report.FORMAT_ORDER)


def test_primary_accuracy_metrics_and_x4_selection():
    assert performance_report.ACCURACY_METRICS["dot"][0] == "rms_normalized_error"
    assert performance_report.ACCURACY_METRICS["gemv"][0] == "relative_l2"

    rows = [
        {
            "kernel": "dot",
            "distribution": "normal_0_1",
            "comparison": comparison,
            "format": "fp32_e8m23",
            "n": n,
        }
        for comparison, n in (
            ("total_x4", "4096"),
            ("total_x1", "1024"),
            ("total_x4", "1024"),
        )
    ]
    selected = performance_report._accuracy_rows(
        rows,
        component="dot",
        distribution="normal_0_1",
        format_name="fp32_e8m23",
    )
    assert [row["n"] for row in selected] == ["1024", "4096"]
    assert {row["comparison"] for row in selected} == {"total_x4"}


def test_interactive_performance_chart_filters(tmp_path):
    svg = tmp_path / "chart.svg"
    groups = "".join(
        f'<g id="{performance_report.series_gid(component, "e1m6", lanes)}"></g>'
        for component in ("dot", "gemv")
        for lanes in (1, 2, 4)
    )
    svg.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg">{groups}</svg>',
        encoding="utf-8",
    )

    document = performance_report.interactive_chart_document(
        svg,
        title="Test chart",
        description="Test description",
        formats=("e1m6",),
        colors={"e1m6": "#0072B2"},
        include_bit_filters=True,
    )

    assert document.count('data-series="true"') == 6
    assert document.count('data-filter="lanes"') == 3
    assert document.count('data-filter="format"') == 1
    assert document.count('data-filter="bits"') == 4
    assert 'data-format="e1m6"' in document
    assert 'data-bits="8"' in document
    assert "filterEnabled('lanes', group.dataset.lanes)" in document
    assert "filterEnabled('format', group.dataset.format)" in document
    assert "filterEnabled('bits', group.dataset.bits)" in document


def test_strategy_chart_filters_and_fp64_baseline(tmp_path):
    svg = tmp_path / "strategies.svg"
    strategies = ("branchless_fp32_x4", "raw_pointer_x1")
    groups = "".join(
        f'<g id="{performance_report.strategy_series_gid("e2m5", distribution, component, strategy)}"></g>'
        for distribution in ("uniform_0_1", "normal_0_1")
        for component in ("dot", "gemv")
        for strategy in strategies
    )
    svg.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg">{groups}</svg>',
        encoding="utf-8",
    )

    document = performance_report.strategy_interactive_document(
        svg,
        format_name="e2m5",
        strategies=strategies,
    )

    assert document.count('data-series="true"') == 8
    assert 'data-strategy="raw_pointer_x1"' in document
    assert '>F64<' in document
    assert 'data-filter="lanes" value="8"' in document
    assert 'data-filter="family" value="branchless"' in document
    assert 'data-filter="strategy" value="branchless_fp32_x4"' in document
    assert "group.dataset.lanes === 'baseline'" in document


def test_strategy_abbreviations_are_compositional():
    assert performance_report.strategy_abbreviation("raw_pointer_x1") == "F64"
    assert (
        performance_report.strategy_abbreviation("lut_prefix_shared_x4")
        == "LP-S ×4"
    )
    assert (
        performance_report.strategy_abbreviation(
            "lut_fp32_global_pipelined_x4"
        )
        == "L32-G-P ×4"
    )
    assert (
        performance_report.strategy_abbreviation(
            "lut_high_word_swizzled_shared_x8"
        )
        == "LHW-SW ×8"
    )
    assert (
        performance_report.strategy_abbreviation(
            "direct_fp64_words_masked_x4"
        )
        == "DW-M ×4"
    )


def test_strategy_top_picks_rank_by_speedup_not_strategy_name():
    rows = []
    for format_name in ("e2m5", "e3m4"):
        for component in ("dot", "gemv"):
            for distribution in ("uniform_0_1", "normal_0_1"):
                for n in ("1024", "4096"):
                    rows.extend(
                        (
                            {
                                "format": format_name,
                                "component": component,
                                "distribution": distribution,
                                "strategy": "branchless_fp32_x4",
                                "n": n,
                                "speedup_vs_fp64": "3.0",
                            },
                            {
                                "format": format_name,
                                "component": component,
                                "distribution": distribution,
                                "strategy": "lut_subnormal_shared_x8",
                                "n": n,
                                "speedup_vs_fp64": "1.5",
                            },
                        )
                    )

    table = performance_report.strategy_top_picks(rows)
    assert table.count("BR32 ×4") == 8
    assert "3.00×" in table
    assert "SN-S ×8" not in table


def test_register_packing_uses_value_throughput(tmp_path):
    performance_summary.EXPECTED_FORMATS = {"fp32_e8m23"}
    performance_summary.EXPECTED_DISTRIBUTIONS = {"normal_0_1"}
    performance_summary.EXPECTED_COMPONENTS = {"register_decode"}
    samples = tmp_path / "samples.csv"
    fields = IDENTITY + MODEL_FIELDS + [
        "round",
        "sample",
        "order_position",
        "time_ms",
    ]
    rows = []
    for lanes in (1, 2, 4):
        for round_index in range(2):
            for sample_index in range(2):
                rows.append(
                    {
                        "gpu": "test_gpu",
                        "compute_capability": "sm_90",
                        "distribution": "normal_0_1",
                        "format": "fp32_e8m23",
                        "storage_bits": "32",
                        "component": "register_decode",
                        "lanes": str(lanes),
                        "n": "1024",
                        "m": "1",
                        "leading_dimension": "1024",
                        "blocks": "4",
                        "threads": "256",
                        "decode_repeats": "64",
                        "decoded_values": str(1024 * lanes),
                        "unique_storage_bytes": "4096",
                        "requested_storage_bytes": "4096",
                        "useful_flops": str(1024 * lanes),
                        "modeled_flops": str(1024 * lanes),
                        "arithmetic_intensity_unique": "1",
                        "arithmetic_intensity_requested": "1",
                        "theoretical_hbm_gb_per_s": "1",
                        "modeled_fp64_gflop_per_s": "1",
                        "round": round_index,
                        "sample": sample_index,
                        "order_position": (lanes + round_index + sample_index)
                        % 3,
                        "time_ms": str(float(lanes)),
                    }
                )
    with samples.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    performance_summary.summarize_timings(samples, tmp_path)
    with (tmp_path / "packed_speedups.csv").open(newline="") as stream:
        speedups = {int(row["lanes"]): row for row in csv.DictReader(stream)}
    assert float(speedups[4]["time_speedup_vs_x1"]) == 0.25
    assert float(speedups[4]["speedup_vs_x1"]) == 1.0
    assert speedups[4]["comparison_metric"] == "decoded_values_per_second"


def test_stream_load_packing_uses_storage_throughput(tmp_path):
    performance_summary.EXPECTED_FORMATS = {"fp32_e8m23"}
    performance_summary.EXPECTED_DISTRIBUTIONS = {"normal_0_1"}
    performance_summary.EXPECTED_COMPONENTS = {"stream_load"}
    samples = tmp_path / "samples.csv"
    fields = IDENTITY + MODEL_FIELDS + [
        "round",
        "sample",
        "order_position",
        "time_ms",
    ]
    rows = []
    for lanes in (1, 2, 4):
        for round_index in range(2):
            for sample_index in range(2):
                rows.append(
                    {
                        "gpu": "test_gpu",
                        "compute_capability": "sm_90",
                        "distribution": "normal_0_1",
                        "format": "fp32_e8m23",
                        "storage_bits": "32",
                        "component": "stream_load",
                        "lanes": str(lanes),
                        "n": "1024",
                        "m": "1",
                        "leading_dimension": "1024",
                        "blocks": "4",
                        "threads": "256",
                        "decode_repeats": "0",
                        "decoded_values": "1024",
                        "unique_storage_bytes": str(4096 * lanes),
                        "requested_storage_bytes": str(4096 * lanes),
                        "useful_flops": "1",
                        "modeled_flops": "1",
                        "arithmetic_intensity_unique": "1",
                        "arithmetic_intensity_requested": "1",
                        "theoretical_hbm_gb_per_s": "1",
                        "modeled_fp64_gflop_per_s": "1",
                        "round": round_index,
                        "sample": sample_index,
                        "order_position": (lanes + round_index + sample_index)
                        % 3,
                        "time_ms": str(float(lanes)),
                    }
                )
    with samples.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    performance_summary.summarize_timings(samples, tmp_path)
    with (tmp_path / "packed_speedups.csv").open(newline="") as stream:
        speedups = {int(row["lanes"]): row for row in csv.DictReader(stream)}
    assert float(speedups[4]["time_speedup_vs_x1"]) == 0.25
    assert float(speedups[4]["speedup_vs_x1"]) == 1.0
    assert speedups[4]["comparison_metric"] == "storage_bytes_per_second"
