# Changelog

## 0.2.1rc1 - 2026-10-20

- Preserve requested automatic fan mode in every old-API fallback, instead of
  silently requesting a manual zero-percent fan target. Tested with mocked APIs.
- Refuse to treat an unavailable PCI bus read as real bus zero.
- Add installed module launching/version output, restore redirected streams and
  close each command's diagnostic log, tolerate failed optional log writes, and
  report missing dependencies cleanly.
- Reject non-finite monitoring intervals before NVML and contain memory-workload
  failures without a traceback. Include complete synthetic tests in source builds.
- Preserve original source refs and document each module's reviewed/tested limits.
- No live hardware or stress tests were run; this is a prerelease.

## 0.2.0 source history (previously unreleased)

### Fixed in the current local batch

- Kept machine-readable JSON and CSV stdout free of human log banners and
  shutdown text.
- Made every hardware-changing CLI path plan-only by default and required a
  literal `--execute`; direct public Python writers also require
  `execute=True`.
- Added exact versioned profile identity, driver-range validation, read-back,
  rollback, and restoration of the actual pre-sweep memory offset.
- Correlated NVAPI and NVML devices by normalized PCI identity instead of
  assuming that both APIs use the same ordinal order.
- Required the CUDA test device's complete PCI BDF to match the selected
  NVAPI/NVML device before a memory sweep can write an offset.
- Rejected `oc --status` combined with any setting, `--reset`, or `--load`
  before constructing the control backend.
- Contained NVML startup, invalid-index, and snapshot errors in `monitor` and
  `info` as stable stderr diagnostics and nonzero exit codes.
- Refused system-RAM-only evidence for GPU memory sweeps and made pattern
  verification run for the requested duration at each step unless an error
  ends it early.
- Added fake backends, authorization regressions, deterministic tests, lint,
  package builds, and offline Linux and Windows GitHub Actions jobs.

### Known limitations

- The proposed workflow validates only offline fake-backend paths; it does not
  validate physical NVAPI writes, read-back behavior, or a real GPU.

### Current local validation

- 73 synthetic pytest cases pass; Ruff, Python compilation,
  `git diff --check`, and isolated source/wheel builds pass.
- At the 2026-09-20 audit, the workflow had no GitHub run because the batch was
  uncommitted.
- No GPU write or stress test was run for this audit. Earlier scanner results
  marked unstable are not treated as recommended or validated settings.
