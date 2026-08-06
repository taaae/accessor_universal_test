# 004: IEEE-like storage-format validation

This experiment validates the storage codecs that will later feed the DOT and
GEMV kernels. All decoded values become FP64 before any arithmetic.

The shortlist contains:

- custom finite E1M6, E1M14, and E1M30 formats;
- custom IEEE-like E2M5, E3M4, E2M13, E3M12, E2M29, and E3M28 formats;
- CUDA FP8 E4M3/E5M2, FP16, BF16, FP32, and FP64 controls;
- RNE-encoded FP64-prefix E11M4 and E11M20 formats.

Every format is decoded through scalar, packed-2, and packed-4 CUDA paths over
deterministic U(-1,1) and N(0,1) inputs. The test requires each GPU result to
agree bit-for-bit with the host decoder. The CSV also records encoding NMSE,
maximum error, decoded zeros, and infinities as validation diagnostics; these
are not performance measurements.

The host test exhaustively validates all 8- and 16-bit custom/prefix encodings
and samples 100,000 encodings for each 32-bit custom/prefix format.

## H200 command

Submit from the repository root:

```bash
sbatch --wait --nodelist=gpu-nvidia-h200-2 \
  scripts/run_storage_formats_h200.sbatch
```

Each run writes a timestamped directory containing the environment, manifest,
CTest log, validation CSV, validator output, CUDA resource usage, and SASS.
