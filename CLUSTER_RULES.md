# GPU cluster working rules

These rules apply whenever an AI-assisted session uses the COMA GPU cluster for
this repository. The cluster provider does not allow unsupervised AI-agent use.
The user must remain in control of every resource-consuming job submission.

## Supervision and resource use

1. Never submit or resubmit a Slurm job without the user's explicit approval.
   Before asking, state:
   - what the job will do;
   - the expected runtime range;
   - the hard Slurm time limit;
   - requested GPUs, CPUs, and memory;
   - the proposed node or node-selection policy.
2. Approval applies to the described submission only. A retry after a failure
   needs new approval unless the user explicitly authorized automatic retries.
3. Every job must have a finite `--time` limit. Prefer a realistic limit with
   safety margin rather than the partition maximum.
4. Request only the resources the job needs. Compilation-only validation should
   not request a GPU. Do not reserve excess host memory or CPUs.
5. Immediately before GPU submission, inspect current availability with
   `sinfo` and active jobs with `squeue`. Prefer a completely idle suitable GPU
   node. If none is idle, determine which suitable GPU becomes free soonest and
   tell the user before submitting.
6. Do not start an unattended background monitor or leave an interactive SSH
   session open. Use short, non-interactive SSH commands initiated from the
   supervised conversation.

## Permitted cluster operations

Cluster access is limited to:

- fast-forward pulling commits made and pushed from the local workstation;
- submitting and running approved jobs;
- checking scheduler state and Slurm `.out` files;
- deleting explicitly identified artifacts from failed jobs;
- committing and pushing generated result files after a successful run.

Do not develop or edit source files directly on the cluster. Make changes in the
local repository, test what can be tested locally, commit and push them, then use
`git pull --ff-only` on the cluster. Do not inspect arbitrary remote files when
the same information is available in the job's `.out` log.

The configured SSH destination is:

```text
Host: 10.152.225.230
User: timofeirusanov
Repository: /storage/home/timofeirusanov/accessor_universal_test
```

Use the SSH configuration already installed on the workstation. Never copy,
print, or modify private key material.

## Before submitting a job

1. Make local changes with the repository's normal editing workflow.
2. Run local syntax and static checks, including `bash -n`, Python compilation,
   and `git diff --check` where applicable.
3. Commit the change in a focused commit and push it.
4. On the cluster, verify that the worktree can be fast-forwarded, then run
   `git pull --ff-only`. Never overwrite cluster data to force a pull.
5. For new or substantially changed CUDA code, first run the CPU-only
   `scripts/check_precision_packing_build.sbatch` job. Compile every
   format-specialized target before requesting an expensive GPU run.
6. Check `sinfo`/`squeue` immediately before the approved submission. Keep node
   selection on the `sbatch` command line; do not hard-code a node in reusable
   batch scripts.
7. Submit with `sbatch --parsable`, record the returned job ID, and close the SSH
   connection. Avoid long quoted `--wrap` payloads; use version-controlled batch
   scripts.
8. About one minute after submission, reconnect briefly and check both the job's
   state with `squeue` and the relevant tail of its Slurm `.out` file. This early
   check is required to catch module, path, build, allocation, and startup errors
   before the job wastes its time allocation. If the job is still pending, record
   the scheduler reason; an empty output file is then expected. Check the log
   again shortly after the job actually starts.

## Cluster environment details learned here

- Non-interactive SSH does not put CUDA tools on `PATH` automatically.
- Source `/etc/profile`, then load `cuda/13.1.1` explicitly. This module provides
  both `nvcc` and `ncu`.
- Source `/etc/profile` **before** enabling `set -u`. A system profile script
  reads `DEBUGINFOD_URLS` without first guaranteeing it is set. The safe order
  in a batch script is:

  ```bash
  set -eo pipefail
  source /etc/profile
  set -u
  module load cuda/13.1.1
  ```

- Slurm copies a submitted batch script to its spool directory. Resolve the
  repository through `SLURM_SUBMIT_DIR`, validate that it is the repository
  root, and require submission from that directory. Do not derive the repository
  from `BASH_SOURCE` inside an `.sbatch` file.
- `sacct` may fail because the accounting database is unavailable. Use `squeue`
  for live state and the job's `.out` file for completion/failure evidence.
- `TMPDIR=/scratch/timofeirusanov/tmp` does not exist on these nodes. Set
  `TMPDIR=/tmp` explicitly in jobs that need it.
- The target GPU is Hopper `sm_90`; configure CMake with
  `CMAKE_CUDA_ARCHITECTURES=90`.

## Failure handling and cleanup

1. A job that leaves `squeue` unexpectedly should be treated as finished or
   failed. Read only the relevant tail of its Slurm `.out` file first.
2. Diagnose and fix code locally. Do not patch the cluster checkout.
3. Before retrying expensive work, use the smallest adequate validation job.
   Environment and compiler failures should be caught without reserving a GPU.
4. Ask for approval again before resubmitting under the supervision rule.
5. Delete failed results only when the exact targets are known. Safe targets are
   the failed job's explicit `slurm-<job-id>.out` (or `slurm-build-<job-id>.out`)
   and its uniquely identified incomplete `run_<timestamp>` directory.
6. Validate that deletion targets are children of the intended experiment
   directory. If multiple run directories might match, stop rather than guess.
7. Use an exact path with `rm --` for a file and `rm -rf --` only for a validated
   run directory. Report what was deleted and that normal command-line deletion
   is not recoverable. Never delete successful results.

## Results and profiling

- Put every experiment in its numbered `results/` directory and every run in a
  timestamped `run_<UTC timestamp>` child directory. Slurm `.out` files belong
  in the experiment directory.
- Full timing must be profiler-free. Nsight Compute replay contaminates CUDA
  event timing, so profiler-reported runs must never be used as normal kernel
  performance samples.
- Keep raw timing samples, summaries, profiler CSV/text exports, environment
  metadata, and a manifest connecting profiler launches to logical variants.
- Before committing generated results, check every file size. GitHub rejects
  individual files above 100 MB. The profiling script moves `.ncu-rep` files
  above 95 MB out of the tracked results tree; do not bypass that guard.
- After a successful job, inspect its `.out` first. Then commit only the intended
  generated result directory on the cluster and push it. Pull that result commit
  locally before analysis.

## Mistakes already encountered

- A full GPU job was initially submitted before confirming that all new CUDA
  targets compiled. It failed on template deduction. Future changes must pass
  the CPU-only all-target build job first.
- The first non-interactive submission inherited no CUDA path and failed at
  `nvcc`. Batch scripts must load the CUDA module explicitly.
- Enabling `set -u` before sourcing `/etc/profile` caused an immediate failure in
  the system `debuginfod` profile. Use the environment initialization order
  documented above.
- Polling only `squeue` can produce an empty row after a fast failure. Follow it
  with the exact `.out` path instead of assuming success.
- Do not use an instrumented Nsight Compute duration as the performance result;
  kernel replay can inflate it by orders of magnitude.
