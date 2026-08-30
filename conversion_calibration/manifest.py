"""Authoritative case inventory and fixed experimental constants.

The manifest is deliberately executable data: code generation, validation,
model fitting, reports, and cluster scripts all consume the same objects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


LEFT_SEED = 0x6BD87C012A53F9E1
KEY_XOR = 0x9E3779B97F4A7C15
RIGHT_SEED = LEFT_SEED ^ KEY_XOR
FULL_N = 1 << 27
BLOCKS = 512
THREADS = 256
ITERATIONS_PER_THREAD = FULL_N // (BLOCKS * THREADS)
CUDA_MODULE = "cuda/13.1.1"
CUDA_ARCH = 90
WARMUPS = 10
ROUNDS = 3
SAMPLES_PER_ROUND = 10
TARGET_INTERVAL_MS = 20.0
SHUFFLE_SEED = 0x5C41B3A7


@dataclass(frozen=True)
class Case:
    case_id: str
    split: str
    group: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    branch_probability: float | None = None
    lut_memory: str | None = None
    lut_index_bits: int = 0
    lut_loads: int = 0
    assumptions: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()

    def row(self) -> dict[str, Any]:
        result = asdict(self)
        result["assumptions"] = list(self.assumptions)
        return result


def c(case_id: str, split: str, group: str, kind: str, **kwargs: Any) -> Case:
    return Case(case_id, split, group, kind, **kwargs)


def training_cases() -> list[Case]:
    cases: list[Case] = [
        c("anchor_bitcast_f64", "train", "anchors", "anchor_bitcast"),
        c("anchor_u32_to_f64", "train", "anchors", "numeric", params={"source": "u32", "count": 1}),
        c("anchor_s32_to_f64", "train", "anchors", "numeric", params={"source": "s32", "count": 1}),
    ]

    for op in ("iadd3", "lop3", "shf", "imad"):
        for count in (1, 4, 16):
            for dependency in ("chain", "ilp4"):
                cases.append(c(f"{op}_{count}_{dependency}", "train", "integer_bit", "integer", params={"op": op, "count": count, "dependency": dependency}))

    for count in (1, 2, 4):
        cases.append(c(f"clz_{count}", "train", "expensive_integer", "clz", params={"count": count}))
    for count in (1, 4, 8):
        cases.append(c(f"int64_shift_alu_{count}", "train", "expensive_integer", "int64", params={"count": count}))

    for source in ("u32", "s32", "f32"):
        for count in (1, 2, 4):
            cases.append(c(f"cvt_{source}_f64_{count}", "train", "numeric_conversion", "numeric", params={"source": source, "count": count}))
    cases.extend([
        c("cvt_u32_f32_f64", "train", "numeric_conversion", "numeric_chain", params={"source": "u32"}),
        c("cvt_s32_f32_f64", "train", "numeric_conversion", "numeric_chain", params={"source": "s32"}),
    ])

    for count in (1, 4, 16):
        for dependency in ("chain", "parallel"):
            cases.append(c(f"fp64_fma_{count}_{dependency}", "train", "floating_point", "fp", params={"precision": "fp64", "op": "fma", "count": count, "dependency": dependency}))
    for op in ("add", "mul"):
        for count in (4, 16):
            cases.append(c(f"fp64_{op}_{count}", "train", "floating_point", "fp", params={"precision": "fp64", "op": op, "count": count, "dependency": "chain"}))
    for count in (4, 16):
        for dependency in ("chain", "parallel"):
            cases.append(c(f"fp32_fma_{count}_{dependency}", "train", "floating_point", "fp", params={"precision": "fp32", "op": "fma", "count": count, "dependency": dependency}))

    for bitops, conversions, fmas in ((4, 1, 1), (8, 1, 4), (16, 1, 8), (8, 2, 4), (16, 4, 8)):
        for dependency in ("chain", "parallel"):
            cases.append(c(f"mixed_b{bitops}_c{conversions}_f{fmas}_{dependency}", "train", "mixed_pipeline", "mixed", params={"bitops": bitops, "conversions": conversions, "fmas": fmas, "dependency": dependency}))

    for denominator in (2, 16, 64, 256):
        for body, count in (("integer", 4), ("integer", 16), ("fp64_fma", 4), ("fp64_fma", 16)):
            cases.append(c(f"branch_p1_{denominator}_{body}_{count}", "train", "branches", "branch", params={"denominator": denominator, "body": body, "count": count}, branch_probability=1.0 / denominator, assumptions=("uniform independent raw u32 codes",)))

    for bits in (4, 8, 12, 16, 20, 24):
        cases.append(c(f"lut_global_1x_{bits}", "train", "luts", "lut", params={"memory": "global", "bits": bits, "loads": 1}, lut_memory="global", lut_index_bits=bits, lut_loads=1, assumptions=("high-bit uniform LUT index",)))
    for memory in ("constant", "shared"):
        for bits in (4, 8, 12):
            cases.append(c(f"lut_{memory}_1x_{bits}", "train", "luts", "lut", params={"memory": memory, "bits": bits, "loads": 1}, lut_memory=memory, lut_index_bits=bits, lut_loads=1, assumptions=("high-bit uniform LUT index",)))
    for bits in (8, 16):
        cases.append(c(f"lut_global_2x_{bits}", "train", "luts", "lut", params={"memory": "global", "bits": bits, "loads": 2}, lut_memory="global", lut_index_bits=bits, lut_loads=2, assumptions=("high-bit uniform LUT index",)))
    for bits in (8, 16, 20):
        cases.append(c(f"lut_global_3x_{bits}", "train", "luts", "lut", params={"memory": "global", "bits": bits, "loads": 3}, lut_memory="global", lut_index_bits=bits, lut_loads=3, assumptions=("high-bit uniform LUT index",)))

    for op in ("exp2", "log2", "sqrt", "div_f64", "div_u32"):
        cases.append(c(f"special_{op}", "train", "special_operations", "special", params={"op": op}))

    for chains in (1, 2, 4, 8, 16, 32):
        cases.append(c(f"latency_32fma_{chains}chains", "train", "latency_occupancy", "latency", params={"fmas": 32, "chains": chains}))
    return cases


def synthetic_validation_cases() -> list[Case]:
    result = [
        c("val_iadd3_2_chain", "synthetic_validation", "synthetic_validation", "integer", params={"op": "iadd3", "count": 2, "dependency": "chain"}),
        c("val_lop3_8_parallel", "synthetic_validation", "synthetic_validation", "integer", params={"op": "lop3", "count": 8, "dependency": "ilp4"}),
        c("val_shf_8_chain", "synthetic_validation", "synthetic_validation", "integer", params={"op": "shf", "count": 8, "dependency": "chain"}),
        c("val_clz2_shift4", "synthetic_validation", "synthetic_validation", "clz_shift", params={"clz": 2, "shifts": 4}),
        c("val_fp64_fma_8_chain", "synthetic_validation", "synthetic_validation", "fp", params={"precision": "fp64", "op": "fma", "count": 8, "dependency": "chain"}),
        c("val_fp64_fma_24_8chains", "synthetic_validation", "synthetic_validation", "latency", params={"fmas": 24, "chains": 8}),
        c("val_mixed_b6_c1_f3", "synthetic_validation", "synthetic_validation", "mixed", params={"bitops": 6, "conversions": 1, "fmas": 3, "dependency": "chain"}),
        c("val_mixed_b12_c2_f6", "synthetic_validation", "synthetic_validation", "mixed", params={"bitops": 12, "conversions": 2, "fmas": 6, "dependency": "parallel"}),
        c("val_branch_p1_4_integer_8", "synthetic_validation", "synthetic_validation", "branch", params={"denominator": 4, "body": "integer", "count": 8}, branch_probability=0.25, assumptions=("uniform independent raw u32 codes",)),
        c("val_branch_p1_128_fp64_fma_8", "synthetic_validation", "synthetic_validation", "branch", params={"denominator": 128, "body": "fp64_fma", "count": 8}, branch_probability=1.0 / 128, assumptions=("uniform independent raw u32 codes",)),
        c("val_lut_global_2x_14", "synthetic_validation", "synthetic_validation", "lut", params={"memory": "global", "bits": 14, "loads": 2}, lut_memory="global", lut_index_bits=14, lut_loads=2, assumptions=("high-bit uniform LUT index",)),
        c("val_lut_global_3x_18", "synthetic_validation", "synthetic_validation", "lut", params={"memory": "global", "bits": 18, "loads": 3}, lut_memory="global", lut_index_bits=18, lut_loads=3, assumptions=("high-bit uniform LUT index",)),
    ]
    return result


def real_cases() -> list[Case]:
    return [
        c("dev_raw_fp32_to_fp64", "development", "real_development", "real", params={"format": "raw_fp32"}, provenance=("include/format_decoder_strategies.cuh:native_direct", "main@3cbce41")),
        c("dev_e11m20_shifter", "development", "real_development", "real", params={"format": "e11m20_shifter"}, provenance=("include/decoder_strategy_core.hpp:decode_prefix_words<20>", "main@3cbce41")),
        c("dev_e9m22_branchy", "development", "real_development", "real", params={"format": "e9m22_branchy"}, provenance=("include/bitwidth_benchmark_core.hpp:decode_direct_fp64_branchy<storage::e9m22>", "main@3cbce41")),
        c("dev_e9m22_prefix_global", "development", "real_development", "real", params={"format": "e9m22_prefix_global"}, lut_memory="global", lut_index_bits=10, lut_loads=1, assumptions=("prefix index is sign plus exponent",), provenance=("include/format_decoder_strategies.cuh:prefix_high_lut", "main@3cbce41")),
        c("dev_quadnormal32", "development", "real_development", "real", params={"format": "quadnormal32"}, provenance=("src/normal32_dot_bench.cu:qn32_view", "codex/t16-dot-benchmark@51a416b")),
        c("final_pwlnormal32_16_16", "final", "real_final", "real", params={"format": "pwlnormal32_16_16"}, lut_memory="global", lut_index_bits=16, lut_loads=1, provenance=("src/normal32_dot_bench.cu:pwl_view", "codex/t16-dot-benchmark@51a416b")),
        c("final_pwqnormal32_8_24", "final", "real_final", "real", params={"format": "pwqnormal32_8_24"}, lut_memory="shared", lut_index_bits=8, lut_loads=1, assumptions=("256 coefficients are staged from global to shared once per block",), provenance=("src/normal32_dot_bench.cu:decode_pwq_shared", "codex/t16-dot-benchmark@51a416b")),
        c("final_posit32_es2", "final", "real_final", "real", params={"format": "posit32_es2"}, provenance=("include/posit_takum_core.hpp:decode_posit<32,2,double>", "main@c069094")),
        c("final_takum32_linear", "final", "real_final", "real", params={"format": "takum32_linear"}, provenance=("include/posit_takum_core.hpp:decode_linear_takum<32,double>", "main@c069094")),
        c("final_takum32_log", "final", "real_final", "real", params={"format": "takum32_log"}, provenance=("include/posit_takum_core.hpp:decode_log_takum<32,double>", "main@c069094")),
        c("final_lns32_r23_reference_exp2", "final", "real_final", "real", params={"format": "lns32_r23_reference_exp2"}, provenance=("include/lns_decoder_strategies.cuh:reference_exp2", "main@3cbce41")),
    ]


CASES = tuple(training_cases() + synthetic_validation_cases() + real_cases())


def manifest_document() -> dict[str, Any]:
    return {
        "experiment": "conversion_cost_calibration",
        "version": 1,
        "fixed": {
            "gpu": "NVIDIA H200 NVL",
            "preferred_node": "gpu-nvidia-h200-2",
            "cuda_module": CUDA_MODULE,
            "cuda_arch": CUDA_ARCH,
            "compile_flags": ["-O3", "-lineinfo", "-Xptxas=-v"],
            "fast_math": False,
            "storage": "two uint32 arrays",
            "arithmetic": "fp64",
            "kernel": "scalar x1 DOT",
            "n": FULL_N,
            "blocks": BLOCKS,
            "threads": THREADS,
            "iterations_per_thread": ITERATIONS_PER_THREAD,
            "left_seed_hex": f"0x{LEFT_SEED:016x}",
            "right_seed_hex": f"0x{RIGHT_SEED:016x}",
            "generator": "Philox4x32-10",
            "warmups": WARMUPS,
            "rounds": ROUNDS,
            "samples_per_round": SAMPLES_PER_ROUND,
            "target_event_interval_ms": TARGET_INTERVAL_MS,
            "shuffle_seed_hex": f"0x{SHUFFLE_SEED:08x}",
            "timing_scope": "first reduction plus final reduction",
        },
        "acceptance": {
            "synthetic_validation_median_ape_max_percent": 5.0,
            "synthetic_validation_p90_ape_max_percent": 10.0,
            "final_real_median_ape_max_percent": 7.5,
            "final_real_max_ape_max_percent": 15.0,
            "final_real_spearman_min": 0.9,
        },
        "cases": [case.row() for case in CASES],
    }


def validate_manifest() -> None:
    ids = [case.case_id for case in CASES]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case id")
    counts = {split: sum(case.split == split for case in CASES) for split in {case.split for case in CASES}}
    if counts != {"train": 112, "synthetic_validation": 12, "development": 5, "final": 6}:
        raise ValueError(f"wrong split counts: {counts}")
    expected_groups = {
        "anchors": 3, "integer_bit": 24, "expensive_integer": 6,
        "numeric_conversion": 11, "floating_point": 14,
        "mixed_pipeline": 10, "branches": 16, "luts": 17,
        "special_operations": 5, "latency_occupancy": 6,
    }
    actual_groups = {group: sum(case.split == "train" and case.group == group for case in CASES) for group in expected_groups}
    if actual_groups != expected_groups:
        raise ValueError(f"wrong training group counts: {actual_groups}")
    if ITERATIONS_PER_THREAD != 1024:
        raise ValueError("fixed launch geometry no longer gives 1024 iterations/thread")
    for case in CASES:
        if case.kind == "branch" and case.branch_probability is None:
            raise ValueError(f"missing branch probability: {case.case_id}")
        if case.kind == "lut" and (not case.lut_memory or not case.lut_index_bits or not case.lut_loads):
            raise ValueError(f"missing LUT metadata: {case.case_id}")
        if case.kind == "real" and len(case.provenance) < 2:
            raise ValueError(f"missing real-converter provenance: {case.case_id}")


validate_manifest()
