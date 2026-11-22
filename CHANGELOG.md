# Changelog

## 2026-11-22

### Added
- The actual code. Earlier uploads had only the README and docs, which described
  directories (`simulation/`, `navigation/`, `tests/`, `runs/`) that weren't in the repo.
- Module 1: navigation env, LiDAR/GPS/IMU + EKF, A\*/RRT/pure pursuit, PPO from
  scratch, optional BC warm start, curriculum training, paired benchmark with Wilson
  CIs and McNemar tests, sensor-noise stress test.
- Module 2: simulated Hadoop cluster telemetry with five fault types, five online
  detectors, val-calibrated alarm thresholds, fault-type classifier.
- 45 tests, GitHub Actions CI, and scripts that regenerate every number and figure.
- Checkpoints, training histories and result JSON in `runs/`.

### Changed
- README and docs rewritten around results that can be reproduced from the repo. The
  old headline table couldn't be reproduced from anything in the repo. The new
  numbers are different, so please don't compare the two.
- Moved `architecture.svg` and the results doc into `docs/`, where the old README
  already linked to them, and fixed the `<repo-name>` placeholders in the instructions.

### Removed
- `module1_results.md` and `training_curves.png` at the repo root, replaced by
  `docs/results.md` and `docs/figures/`.
