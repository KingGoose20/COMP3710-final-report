# Proposed title

Add reproducible ConvNeXt Alzheimer's interfaces and report slice-level assessment results

# Proposed description

The Alzheimer's prototype's selected ConvNeXt path previously depended on private preparation pointers and historical report trees. This change exposes explicit-input preparation, A1 training and strict checkpoint prediction through the required `dataset.py`, `train.py` and `predict.py` entry points, while preserving the existing baseline and experimental history.

The retained model is scratch-trained ConvNeXt-Tiny A1 (`scratch_augmented`, seed 42, selected epoch 13). The user-supplied Ed clarification establishes individual 2D test slices as the assessed accuracy unit: its existing recorded result is 3,157/4,640 (68.04%) against 80%, versus CNN 3,193/4,640 (68.81%). Patient-test accuracy 76/102 (74.51%) is secondary. The retained checkpoint was originally selected by patient validation metrics; that historical rule is unchanged. The three-seed schedule-only follow-up, fixed G012 seed-42 scan-supervision study and G3 gated-attention study are completed negative results under their original patient-based screens. The bounded retrospective slice review found no substantial quick improvement from retained checkpoints; missing per-epoch validation metrics/weights are explicit. A1 remains retained, without new inference or model promotion. The report discloses the cohort's earlier test exposure, repeated validation use, all relevant negative experiments, confidence/referral limitations, resource scopes, the original feasibility review, and ChatGPT/Codex assistance. Personal review and earlier executions are explicitly author-reported.

## Implementation

- Include the reviewed metadata/split helpers and reproduce only the exact approved assignments into a new output directory, with fixed metadata/manifest hashes and grouping/class counts. Original manifests and private approval/pointer files are not required inputs.
- Route `train.py a1` to the existing checked V2 recipe and create preparation state from explicit data/manifests. Keep the original accuracy, macro-F1 and log-loss selection rule; epoch 13 is a recorded result rather than a forced selection.
- Add `predict.py infer` with explicit checkpoint digest, data/manifests and optional frozen calibration/policy files. Enforce exact source, architecture, state-key/shape/dtype and manifest checks and retain hierarchical slice-to-scan-to-patient aggregation.
- Preserve the pre-change checkpoint runtime byte for byte, document historical inspection/replay, and protect its bytes from Windows line-ending conversion. No historical expected hashes or integrity guards are weakened.

## Submission-interface verification completed previously

- Fifteen focused synthetic interface tests passed, including strict-load failure cases, preparation fidelity, recipe preservation and legacy test guards.
- A clean Git export without author pointers or historical reports reproduced all three manifest files byte for byte. The fixed validation manifest contains 4,660 slices, 233 scans and 102 patients.
- Two bounded training-only smoke executions used 32 training slices and one optimizer update each across a packaging correction; checkpoint round trips matched exactly. These were not cohort epochs or candidate experiments.
- In the corrected clean export, retained A1 validation replay matched every slice/scan/patient probability exactly, within the existing 2e-5 tolerance, with unchanged class decisions. Explicit frozen calibration and all three policy outputs matched the recorded validation results without fitting.
- The initial clean-export provenance refusal exposed a line-ending packaging defect; the exact original bytes and a regression check corrected it. The failed result remains preserved.
- Verification used the existing pinned Python 3.11.9 / PyTorch 2.7.1+cu126 environment. No separate dependency environment or full retraining was tested. No test inference, threshold/calibration fitting, additional model experiment or promotion was performed during packaging.

## Review and assessment limits

The branch contains code, documentation, hashes and aggregate plots. It contains no datasets, model weights, patient manifests, private predictions or medical-image panels. Authorized checkpoint/data access or the documented training-to-checkpoint route is necessary. Exact cross-machine training accuracy is not promised.

The revised failure-case records and existing private panels have been reconciled; the author confirms reviewing the revised cases. Medical-image redistribution permission is not established, so the report retains textual analyses without panels or participant identifiers. Consultation check-off details and the untested fresh dependency environment remain explicit limitations. The accuracy requirement remains unmet; full-mark readiness is not claimed.

Method references: [ConvNeXt](https://arxiv.org/abs/2201.03545), [temperature scaling](https://arxiv.org/abs/1706.04599), and the supplied COMP3710 Version 2.5 specification. The README details source provenance and AI assistance.

The current closeout adds aggregate slice counts/per-class/probability metrics, the selected-versus-retrospective checkpoint inventory, corrected patient-only uncertainty/failure-case framing, an updated AI disclosure and local integration/submission checklist. All sixteen saved selected validation score sets were recomputed on CPU and checkpoint epoch/availability checked; no model was executed. Test reporting reads only the existing published aggregate summary. No independent-slice statistical claims or new calibrated slice policy are made.

G3 uses the gated-attention method of Ilse et al., [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712); G012 uses the declared uniform objectives. All scientific/source provenance and original selection records remain intact.

This file is a description draft only. No pull request, push, Turnitin submission or review response has been performed by this task. At submission, target upstream `topic-recognition` and retain the progressive commit history.
