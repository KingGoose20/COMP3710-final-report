# Alzheimer's classification with ConvNeXt: accuracy, confidence and referral trade-offs

**COMP3710 — Semester 2, 2026**  
**Author:** Angus Walker  
**Student identifier supplied:** s4963173  
**Status:** A1 retained; schedule-only experiment completed as a negative result. Explicit preparation/training/prediction interfaces have passed clean-code verification in the existing pinned environment. The 0.80 test requirement remains unmet. Author-reported review and the remaining submission actions are listed below; full-mark readiness is not claimed.

## 1. Problem, intended user and principal finding

This project investigates a research-only assistant that classifies structural brain MRI images from the supplied ADNI data as Alzheimer's disease (AD) or cognitively normal (CN), and refers uncertain predictions for human review. The supplied folders use `NC`; the code maps this to the control class described as CN in the assessment. Neither class predictions nor confidence scores constitute clinical diagnoses.

The main question is whether a scratch-trained ConvNeXt improves patient classification and confidence-based referral enough to justify its computational cost compared with a small CNN. The selected model is ConvNeXt A1 (`scratch_augmented`, seed 42, selected epoch 13). The baseline is the regularized CNN selected at epoch 14.

A1 achieved 83.33% selected validation-patient accuracy but **74.51% (76/102)** in the disclosed revised evaluation, compared with **77.45% (79/102)** for the CNN. The specified minimum 0.80 test-accuracy target was **not met**. A1 detected one more AD patient but produced four additional false positives. Calibration improved some revised-evaluation probability-quality measurements. Stricter negative acceptance reduced automated false negatives at the cost of additional clinical review; it did not make the classifier more accurate or establish clinical safety.

**Evaluation qualification:** the same patient-independent test cohort had already been evaluated for Version 1. Subsequent development used training and repeatedly consulted validation data; the revised checkpoint, temperatures and policies were fixed before revised inference. This is not a new untouched independent test. Prior test exposure, validation reuse and limited patient numbers constrain the conclusions.

## 2. Feasibility review and changes from the proposal

Retained project records report approval of the feasibility review and the corrected patient-independent split. This is an attributed project record, not independent confirmation of the compulsory consultation hurdle. The exact consultation date, verbatim feedback and final hurdle status require author confirmation. Appendix A preserves the original proposal as historical material, rather than retrospectively describing it as a completed experiment.

The project moved from Rangpur execution to a local NVIDIA RTX 4080 Laptop GPU. Dataset access, patient mapping, model training and resource measurements were subsequently checked. The supplied train/test directory split was replaced after the metadata audit identified 216 shared patient identifiers; the corrected patient grouping and split change were approved as reported by the author. All original files were retained, with manifests defining the new partitions.

The proposed 12 GiB memory limit and 12 GPU-hour initial-experiment budget were planning constraints, not total measured project cost. Training includes an interrupted original attempt; per-run recorded times do not necessarily include discarded partial epochs or downtime. Measured resource scopes are distinguished below.

## 3. Data integrity and experimental partitions

The local dataset contains 30,520 JPEG slices from 1,526 scans and 680 distinct mapped patients. The exact JPEG-prefix-to-metadata-key join links scans to records; the prefix itself is not the patient identifier. Participant tokens were checked for agreement across the seven metadata path fields. Metadata codes 0 and 2 map to model labels NC=0 and AD=1. The audit found no unreadable images or exact duplicate encoded/decoded images in the inspected copy. This does not rule out all forms of dataset bias or near-duplication.

The corrected split groups all scans and slices of a patient together. Within each class, a deterministic SHA-256 ordering using seed 42 and largest-remainder allocation implements the declared approximate 70/15/15 patient proportions. The proportions of scans and slices need not match these patient proportions.

| Partition | Patients (CN / AD) | Scans | Slices |
|---|---:|---:|---:|
| Training | 476 (321 / 155) | 1,061 | 21,220 |
| Validation | 102 (69 / 33) | 233 | 4,660 |
| Test | 102 (69 / 33) | 232 | 4,640 |

All three patient sets are disjoint. Manifest hashes and image hashes were recorded. Later image-integrity checks compare against the local audit, not against an independently hashed Rangpur source. Multiple slices from one patient are correlated and must not be counted as independent patients.

### Input preparation

Each 256 x 240 greyscale JPEG is resized to 224 x 210 using bilinear interpolation. Seven black pixels are added above and below to produce a one-channel 224 x 224 tensor without stretching the original aspect ratio. Values are converted to float32 and scaled by `(pixel / 255 - 0.5) / 0.5`; this does not recover physical MRI intensities from JPEG data and does not fit dataset-wide statistics.

A1 uses training-only rotations sampled between -5 and +5 degrees. Validation and test inputs have no random augmentation. Rotation can clip peripheral content; a private preview was produced for inspection. No patient, scan or slice was discarded on the basis of the diagnostic results. Final partitions are read from manifests, not inferred from the old directory names.

## 4. Models and working principles

### ConvNeXt-Tiny

ConvNeXt-Tiny follows the architectural family described by Liu et al. [2], adapted to one input channel and two output classes. It was implemented using PyTorch and trained from random initial weights; external pretrained weights were not used in the reported experiments.

| Stage | Operation / blocks | Channels | Spatial size |
|---|---|---:|---:|
| Input | Preprocessed greyscale MRI slice | 1 | 224 x 224 |
| Stem | 4 x 4 convolution, stride 4, channel normalization | 96 | 56 x 56 |
| Stage 1 | 3 ConvNeXt blocks | 96 | 56 x 56 |
| Stage 2 | Downsample, then 3 blocks | 192 | 28 x 28 |
| Stage 3 | Downsample, then 9 blocks | 384 | 14 x 14 |
| Stage 4 | Downsample, then 3 blocks | 768 | 7 x 7 |
| Output | Spatial mean, LayerNorm, two-logit linear head | 2 | One output pair |

Each block applies a 7 x 7 depthwise convolution, channel LayerNorm, a fourfold-expanded linear bottleneck, GELU, a projection, LayerScale and a residual connection with stochastic depth during training. The two logits correspond to NC and AD; a softmax supplies the slice AD score. The model has 27,818,594 trainable parameters. Large capacity alone does not guarantee generalisation from a limited patient cohort.

### CNN baseline

The baseline contains four stride-2 Conv/BatchNorm/ReLU blocks with channels 16, 32, 64 and 128. A fixed spatial pooling layer, a 128-unit dense layer, ReLU and dropout precede the two-class head. Its 359,922 trainable parameters provide a substantially smaller comparison. Input preparation, partitions and evaluation definitions match those used for the main model, while architecture and intrinsic regularization differ.

### Aggregation and evaluation units

Slice probabilities are averaged within each scan; scan probabilities are then averaged within each patient. This gives each scan equal weight within its patient. Classification is AD for probability >=0.5, otherwise CN. Patient-level performance is primary; scan and slice results are reported separately. Single-slice cross-entropy training and hierarchical probability aggregation are not the same objective. In A1, patients with more scans contribute more training examples.

## 5. Development experiments and selection

Training used AdamW, float32, gradient clipping and the recorded deterministic/recovery settings. A1 used batch size 8 with four-batch accumulation, peak learning rate 0.0003, three warmup epochs, cosine decay to 0.000001, weight decay 0.05 for matrix/kernel parameters and no decay for biases/one-dimensional parameters. The limit was 30 epochs, with patience 8 after at least 12 epochs. Workers were zero in the checked Windows recovery path.

Version 1 selected checkpoints by validation patient macro-F1, then lower log loss. V2 explicitly changed to patient accuracy, then macro-F1, then lower log loss, retaining the earliest exact tie. This is a documented change, not retrospective relabelling of the first experiments.

| Development experiment | Selected epoch | Patient accuracy | Macro-F1 | AUROC | AD recall |
|---|---:|---:|---:|---:|---:|
| Initial CNN | 7 | 78.43% | 0.7292 | 0.8327 | 51.52% |
| Regularized CNN | 14 | 78.43% | 0.7496 | 0.8037 | 63.64% |
| Original ConvNeXt | 11 | 76.47% | 0.7167 | 0.8006 | 54.55% |
| **A1: rotations, seed 42** | **13** | **83.33%** | **0.8012** | **0.8331** | **66.67%** |
| A2: additional patient weighting | 12 | 80.39% | 0.7683 | 0.7874 | 63.64% |
| A1 replication, seed 43 | 8 | 78.43% | 0.7496 | 0.7980 | 63.64% |
| A1 replication, seed 44 | 7 | 78.43% | 0.7574 | 0.8169 | 69.70% |
| S1: label smoothing | 9 | 78.43% | 0.7403 | 0.7848 | 57.58% |
| S2: smoothing and revised schedule | 9 | 78.43% | 0.7496 | 0.8292 | 63.64% |

All rows use the same validation patients and are selected development scores. They are not independent test estimates. A1 exceeded 80% in one selected epoch, but its two replications did not reproduce that accuracy. The three-run mean does not represent an ensemble or three independent patient cohorts. Eleven validation patients were misclassified by all three A1 seeds, including nine AD patients. This finding does not establish the cause or justify removing these patients.

The training/validation slice-accuracy gap and increasing validation log loss indicated persistent overfitting. A2's equal-patient weighting did not improve the comparison. S1 and S2 reduced some highly confident errors, and S2 improved patient log loss, but neither matched A1's classification result. These negative findings are retained. Changes from Version 1 to V2 include more than augmentation, so the improvement is not attributed causally to augmentation alone.

A1 remained selected based on development evidence. No new model, ensemble or classification cutoff was chosen using the revised test outcomes.

### Later controlled schedule experiment

After a scoped source/evidence review, one predeclared experiment changed only the A1 cosine endpoint from epoch 30 to 12. It retained hard labels, unweighted loss, the 3e-4 peak, 1e-6 floor, three warmup epochs, rotations, batch/accumulation, optimizer grouping, clipping, precision and selection rule. The peak remained at epoch 4; the floor was reached at epoch 12 and then held. The experiment plan and code were committed before training.

| Seed | Selected / completed epoch | Correct / 102 | AD detected / 33 | Patient accuracy | Patient NLL | Paired correct change |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 7 / 15 | 81 | 19 | 79.41% | 0.505059 | -4 |
| 43 | 5 / 13 | 76 | 17 | 74.51% | 0.495243 | -4 |
| 44 | 9 / 17 | 81 | 24 | 79.41% | 0.521898 | +1 |

All three completed by the original eight-epoch patience rule. Their first four epochs matched each seed's control on all declared scalar measurements exactly (loss tolerance 1e-6, scores/recalls 1e-12, updates exact). Full validation replay of every selected checkpoint also matched exactly under the 2e-5 probability tolerance with unchanged classes. Seed 43 recovered the same attempt after an atomic-file error and two disk-guard stops; all execution, including discarded partial work, remained charged. No replacement attempt was created.

The predeclared screen failed: summed correct was 238 rather than at least 251; only one seed improved; two lost four patients. Mean AD recall fell from 66.67% to 60.61%, and seed 42 failed the 85-correct/22-AD guard. Mean ordinary patient NLL improved from the exact control mean 0.542208826357733 to 0.5074003661741057. The integrity/replay gate passed. Lower final-five-epoch variability did not provide repeatable recognition improvement. These are repeated development measurements on the same 102 patients, not 306 independent subjects.

**Decision:** record this as a negative experiment and retain A1, scratch_augmented, seed 42, epoch 13. No model promotion, calibration refit, threshold search or new test query followed. Detailed private histories, paired differences and fixed last-five summaries are retained under the separate schedule-only namespace; the public plan remains `SCHEDULE_ONLY_PLAN.md`.


![A1 training and validation slice loss](figures/a1_training_loss.png)

![A1 patient validation trajectory](figures/a1_validation_trajectory.png)

These are the unchanged curves saved during A1 development. Their original “final test not evaluated” title describes the state when they were generated; the later disclosed test exposure is reported in Section 7.

## 6. Confidence, calibration and referral

Temperature scaling [3] was fitted to aggregated patient log odds using validation data. Its scalar temperature preserves the class decision at 0.5 while changing confidence. The fitted temperatures are approximately 1.61343 for the CNN and 1.85536 for A1. Five-fold cross-fitting was used as a development diagnostic: it held each fold out of calibration/rule fitting, but did not undo earlier checkpoint selection on the complete validation cohort.

The reference rule accepts a prediction only at sufficient calibrated confidence, otherwise referring it. The thresholds are 0.85 for the CNN and 0.75 for A1. They were chosen on a finite development grid subject to the declared empirical coverage/error constraints; those constraints are not clinical standards. The reference procedure was unstable across development folds.

Two additional policies were declared before their comparison: `strict_cn` adds calibrated P(AD)<=0.05 to any automatic CN proposal, and `no_auto_cn` sends every CN proposal to review. AD flags are unchanged. All three are reported; there is no test-based selection of a winning policy. An AD flag is not a confirmed diagnosis, and a referral is not a correct diagnosis.

### Development confidence evidence

| Model | Raw NLL | Cross-fit NLL | Raw Brier | Cross-fit Brier | Raw errors at confidence >=0.90 | Full-fit cutoff | Cross-fit accepted / errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.5205 | 0.5012 | 0.1715 | 0.1692 | 4 | 0.85 | 27 / 4 |
| A1 | 0.5141 | 0.5042 | 0.1421 | 0.1519 | 4 | 0.75 | 20 / 4 |

A1's full-validation fitted rule accepts 48/102 with 4/48 errors, but its cross-fit diagnostic accepts 20/102 with 4/20 errors. The CNN cross-fit accepts 27/102 with 4/27 errors. Thus the apparent full-fit advantage does not establish robust operational superiority. Cross-fitting separates calibration/referral fitting from held-out folds only; earlier model selection already consulted all validation patients. Temperature scaling cannot improve the fixed 0.5 classification decisions or create discriminative information.

The strict-CN and no-auto-CN policies are reported with their correct denominators. Automatically clearing no CN predictions removes automated negative errors by construction; it is not 100% diagnostic sensitivity or proof of safety. Specialist review outcomes and real operational workload have not been measured.


## 7. Disclosed revised evaluation

The original ConvNeXt achieved 72.55% on the Version 1 patient test. A1's later evaluation reused that patient-independent cohort after development using training/validation evidence. The checkpoint, preprocessing, temperatures and all policies were recorded before revised inference. This process limits additional test-based selection but cannot erase prior exposure or adaptive-development bias.

### All-patient classification (primary)

| Model | Correct / N | Accuracy | Macro-F1 | AUROC | AD recall | CN recall |
|---|---:|---:|---:|---:|---:|---:|
| Regularized CNN (epoch 14) | 79/102 | 77.45% | 0.7258 | 0.8177 | 54.55% | 88.41% |
| ConvNeXt A1 (seed 42, epoch 13) | 76/102 | 74.51% | 0.7040 | 0.8142 | 57.58% | 82.61% |

### Per-class precision and confusion counts

| Model | CN precision | AD precision | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|
| cnn | 0.8026 | 0.6923 | 61 | 8 | 15 | 18 |
| a1 | 0.8028 | 0.6129 | 57 | 12 | 14 | 19 |

### Other units and target reporting

| Model | Patient accuracy | Scan accuracy | Slice accuracy |
|---|---:|---:|---:|
| cnn | 77.45% | 72.41% | 68.81% |
| a1 | 74.51% | 68.10% | 68.04% |



cnn: descriptive Wilson 95% patient-accuracy interval 68.43%–84.48%.

a1: descriptive Wilson 95% patient-accuracy interval 65.27%–81.97%.

**Observed A1 accuracy >=0.80:** patient=False, scan=False, slice=False. The primary unit was declared as patient. The brief does not explicitly settle the threshold unit. These observed flags do not guarantee marks, a fresh independent estimate or clinical suitability.

### All fixed policies: safety and workload

| Model | Policy | Auto CN | AD flagged | Refer | AD incorrectly auto CN / all AD | Accepted error | Clinical reviews* |
|---|---|---:|---:|---:|---:|---:|---:|
| cnn | reference | 28 | 4 | 70 | 0/33 | 3.12% | 74 |
| cnn | strict_cn | 15 | 4 | 83 | 0/33 | 5.26% | 87 |
| cnn | no_auto_cn | 0 | 4 | 98 | 0/33 | 25.00% | 102 |
| a1 | reference | 50 | 6 | 46 | 5/33 | 14.29% | 52 |
| a1 | strict_cn | 26 | 6 | 70 | 1/33 | 12.50% | 76 |
| a1 | no_auto_cn | 0 | 6 | 96 | 0/33 | 50.00% | 102 |

*Clinical reviews assume all AD flags also need confirmation; this workload is calculated, not observed. Referrals are not successful diagnoses. No-auto-CN removes automated negatives by construction. Accepted-subset accuracy does not replace all-patient accuracy. The 0.05 strict-CN cap is a fixed research choice, not a clinical safety bound. Descriptive policy confidence intervals and denominators are in the JSON.

### Probability quality

| Model | Raw NLL | Fixed-T NLL | Raw Brier | Fixed-T Brier |
|---|---:|---:|---:|---:|
| cnn | 0.5138 | 0.4839 | 0.1655 | 0.1607 |
| a1 | 0.6040 | 0.4979 | 0.1780 | 0.1661 |

Paired A1 minus CNN accuracy: -2.94%; exploratory patient-bootstrap interval -10.78% to 4.90%. This interval excludes training-seed/site uncertainty and the effects of previous test exposure.

![Revised patient ROC](figures/a1_revised_test_roc.png)

![Revised fixed-temperature reliability](figures/a1_revised_test_reliability.png)

## 8. Resource cost

| Model | Parameters | Inference peak allocated GiB | Prior batch-1 forward median ms | Completed revised test-batch seconds |
|---|---:|---:|---:|---:|
| cnn | 359,922 | 0.0461 | 1.062 | 14.943 |
| a1 | 27,818,594 | 0.2364 | 13.963 | 17.226 |

A1 has approximately 77 times as many parameters as the baseline. Batch-size-one forward latency is model-only timing with inputs already on the device; it excludes loading and patient aggregation. The inference allocator peaks above are not training-memory peaks or total Windows VRAM. The revised per-batch timings include integrity checks, decoding, preprocessing, transfer and forward conversion, but exclude setup, cache writes and recovery verification. These scopes must not be interchanged.

| Selected run | Recorded training time, seconds | Training peak allocated GiB | Timing scope |
|---|---:|---:|---|
| CNN | 1609.03 | 0.1170 | Wall time around the original training/validation loop, including its in-loop checkpoint/history I/O; excludes subsequent selected-checkpoint replay/profiling. |
| A1 | 6437.32 | 1.3108 | Excludes repeated work in interrupted epochs, setup/checkpoint I/O and downtime. |

## 9. Qualitative failure analyses

### Evidence scope and provenance

These analyses retain the reviewed private draft's case discussions, now checked against the revised A1 local records. The old and revised selections use different patients for F01, F04 and F05; F02 and F03 retain the patient alias correspondence but the run/model outputs still differ. The older worksheet is therefore not the source for revised outcomes.

For documentation verification, Codex matched the revised selection records to the already saved A1/CNN patient predictions, reconstructed the panel's documented scan/slice selection from cached records, and inspected the existing revised derived panels. Displayed slice indices and rounded probabilities agree in all five cases:

| Case | Revised displayed slice indices | Revised displayed slice P(AD) |
|---|---|---|
| F01 | 100, 105, 109 | 0.000, 0.000, 0.000 |
| F02 | 83, 88, 92 | 1.000, 0.999, 1.000 |
| F03 | 99, 104, 108 | 0.984, 0.999, 0.993 |
| F04 | 100, 105, 109 | 0.990, 0.001, 0.004 |
| F05 | 93, 98, 102 | 0.002, 0.000, 0.176 |

This resolves record/display correspondence; it is not an independent reconstruction of panel pixels from raw MRI data or a clinical interpretation. No raw test images were reopened and no test inference was run for this check. Current panel hashes and detailed correspondence evidence are retained privately. The author explicitly confirmed review of the revised cases during submission preparation; this is an author-reported review, distinct from Codex's record/display check. Public redistribution permission for the medical panels is not established, so no panels, source paths or participant identifiers are included in the repository. These cases inform documentation of the frozen result, not model or policy changes.

The panels show three representative inputs with rounded slice-level AD scores. Patient probabilities in the tables are aggregates of all recorded slices and scans and need not equal the mean of the three displayed scores. The images are not saliency maps, clinical diagnoses or causal explanations. Values shown as 0.000 or 1.000 are rounded displays, not demonstrations of certainty.

Actions: `auto_CN` means an automatic CN proposal; `AD_flag` means an AD alert needing confirmation; `refer` means unresolved human review. The dataset uses NC for the class described here as CN. All numerical table entries below are preserved from the supplied revised table. Medical images and original participant identifiers are excluded from this report-input document.


### F01

| Model | Dataset class | Prediction | Raw P(AD) | Calibrated P(AD) | Reference | Strict CN | No auto CN |
|---|---|---|---:|---:|---|---|---|
| cnn | AD | CN | 0.1231 | 0.2284 | refer | refer | refer |
| a1 | AD | CN | 0.0019 | 0.0331 | auto_CN | auto_CN | refer |

**Observed image/data properties:** The supplied panel shows three side-on brain views with visible tissue boundaries against a black background. No missing image or grossly blank input is apparent at this display resolution. All three displayed slice AD scores round to 0.000; these are rounded slice outputs, not proof of exactly zero disease probability. The revised case table records an AD-labelled patient with A1 raw P(AD)=0.0019 and calibrated P(AD)=0.0331, establishing a strongly confident false negative against the dataset label.

**Possible explanation and uncertainty:** Agreement among these three displayed slices does not establish correctness. The case is consistent with a learned representation that fails to distinguish this patient's labelled class, but the panel cannot identify the features responsible. Insufficient diagnostic information in the sampled slices, unmodelled variation and a mismatch between slice supervision and patient diagnosis are hypotheses, not demonstrated causes. Neither early disease nor a specific anatomical abnormality can be inferred from these images alone.

**CNN comparison and policy consequences:** Both models classify the patient as CN. However, the CNN's calibrated P(AD)=0.2284 leads to referral under all three policies. A1's calibrated score corresponds to CN confidence of 0.9669: its incorrect CN proposal survives both the reference policy and the stricter P(AD)<=0.05 rule. Only the no-automatic-CN policy refers this A1 result. Thus, the additional negative-confidence requirement fails to protect this particular patient.

**Recommended limitation or safeguard:** Do not interpret a high-confidence CN prediction as clinical clearance. This case supports mandatory clinical review rather than autonomous reassurance. Investigating the full available scan context and input characteristics could be useful in future independent work, but the cause must remain unresolved here and this case must not be used to tune the already evaluated model.

### F02

| Model | Dataset class | Prediction | Raw P(AD) | Calibrated P(AD) | Reference | Strict CN | No auto CN |
|---|---|---|---:|---:|---|---|---|
| cnn | CN | AD | 0.9975 | 0.9760 | AD_flag | AD_flag | AD_flag |
| a1 | CN | AD | 0.9962 | 0.9525 | AD_flag | AD_flag | AD_flag |

**Observed image/data properties:** The three supplied views retain visible brain contours and internal tissue boundaries, with the central dark region changing shape across slice positions. The displayed AD scores are 1.000, 0.999 and 1.000, rounded to three decimal places. In the revised case table the dataset class is CN, while A1 has raw P(AD)=0.9962 and calibrated P(AD)=0.9525. The displayed evidence therefore accompanies a highly confident false positive; confidence does not override the recorded class.

**Possible explanation and uncertainty:** Both architectures may respond to shared image characteristics that resemble patterns learned from AD-labelled training examples, but this is only a possible explanation. Variation in anatomy, appearance or acquisition could contribute, yet none is established as the trigger. A visually prominent dark region cannot be called pathological enlargement or used to diagnose AD from these illustrative slices. Agreement between two models trained on the same data is not independent clinical confirmation.

**CNN comparison and policy consequences:** The CNN also produces a false positive, with raw P(AD)=0.9975 and calibrated P(AD)=0.9760. Every policy returns an AD flag for both models. This is expected because strict-CN and no-automatic-CN alter negative proposals only; they do not modify the positive side. Calibration softens the scores but does not change the classifications. The consequence is an unnecessary AD alert and confirmation workload, not an established clinical diagnosis.

**Recommended limitation or safeguard:** Present an AD flag as a request for specialist assessment, not a definitive diagnosis or a treatment decision. Keep this example in the false-positive analysis to show that false-negative safeguards do not address all harms. Future investigation could compare acquisition and appearance characteristics across independent cases, without assuming the dataset label is wrong or changing the current result.

### F03

| Model | Dataset class | Prediction | Raw P(AD) | Calibrated P(AD) | Reference | Strict CN | No auto CN |
|---|---|---|---:|---:|---|---|---|
| cnn | CN | CN | 0.3308 | 0.3925 | refer | refer | refer |
| a1 | CN | AD | 0.9532 | 0.8353 | AD_flag | AD_flag | AD_flag |

**Observed image/data properties:** The supplied panel shows variation in the outlines and internal appearance across three slice positions, but all displayed AD scores are high: 0.984, 0.999 and 0.993. The revised case table identifies a CN-labelled patient. A1's patient-level raw P(AD)=0.9532 and calibrated P(AD)=0.8353 produce a false positive. These three selected slices illustrate consistent high-scoring outputs; they are not the complete set used to calculate the patient probability.

**Possible explanation and uncertainty:** The different model outcomes show that the larger architecture's learned decision function is not necessarily better for this individual. A1 may rely on image characteristics that the smaller CNN treats differently, but the panels do not reveal feature attribution or establish why. The CNN's correct classification should not be interpreted as proof that it recognised a particular clinical biomarker. No precise anatomical cause can be established from the available evidence.

**CNN comparison and policy consequences:** The CNN correctly predicts CN, with raw P(AD)=0.3308 and calibrated P(AD)=0.3925, but refers the patient because its confidence is insufficient for automatic acceptance. A1's calibrated AD score exceeds its 0.75 reference threshold and results in an AD flag under every policy. This contrasts a correct but cautious baseline prediction with an incorrect, more confident advanced-model prediction.

**Recommended limitation or safeguard:** Do not prefer a model merely because it is larger or more confident. The recorded disagreement is a useful failure-analysis signal and supports specialist confirmation of A1's flag. Using model disagreement as an additional referral mechanism would be a new policy requiring separate development and evaluation; it was not part of the frozen system assessed here.

### F04

| Model | Dataset class | Prediction | Raw P(AD) | Calibrated P(AD) | Reference | Strict CN | No auto CN |
|---|---|---|---:|---:|---|---|---|
| cnn | CN | CN | 0.3887 | 0.4303 | refer | refer | refer |
| a1 | CN | AD | 0.5295 | 0.5159 | refer | refer | refer |

**Observed image/data properties:** The supplied panel shows substantial score disagreement: the displayed slice AD scores are 0.990, 0.001 and 0.004. The middle image appears to have less distinct light-dark tissue contrast than the flanking views, but this display-level observation does not establish an acquisition defect. In the revised table this patient is CN-labelled, while A1's raw patient P(AD)=0.5295 and calibrated P(AD)=0.5159 give a borderline false positive.

**Possible explanation and uncertainty:** A near-boundary aggregate can coexist with highly confident but opposing slice predictions. Sensitivity to slice content or contrast is a plausible hypothesis, but the panel does not demonstrate causation. The three displayed probabilities must not be averaged to reconstruct the patient score: the implemented calculation uses all available slices within each scan and then averages the scan scores. The supplied images do not establish disagreement between different scans or identify a particular anatomical trigger.

**CNN comparison and policy consequences:** The CNN correctly predicts CN with raw P(AD)=0.3887 and calibrated P(AD)=0.4303. Both models refer the case under the reference, strict-CN and no-automatic-CN policies. The referral mechanism therefore prevents A1's borderline positive classification from becoming an accepted AD flag. It does not correct the classifier or supply a confirmed diagnosis; it transfers the unresolved decision to review.

**Recommended limitation or safeguard:** Retain explicit referral for low-confidence aggregates and document disagreement among available slice outputs where measured. No new scan-level model or disagreement policy was developed from this case. Do not drop the high-scoring slice or adjust the class threshold after inspecting this test case. The present evidence supports referral, not an anatomical diagnosis.

### F05

| Model | Dataset class | Prediction | Raw P(AD) | Calibrated P(AD) | Reference | Strict CN | No auto CN |
|---|---|---|---:|---:|---|---|---|
| cnn | AD | CN | 0.1964 | 0.2945 | refer | refer | refer |
| a1 | AD | CN | 0.0193 | 0.1073 | auto_CN | refer | refer |

**Observed image/data properties:** The three supplied views show visible brain tissue against a black background. Their displayed AD scores are 0.002, 0.000 and 0.176: they vary, but all remain below the 0.5 slice-classification boundary. The revised case table labels the patient AD. A1's raw patient P(AD)=0.0193 and calibrated P(AD)=0.1073 yield a false-negative CN prediction. The displayed zero is rounded, and the panel is not the complete patient input.

**Possible explanation and uncertainty:** Variation between the displayed slice outputs indicates sensitivity to slice content, but does not identify the responsible image feature. The patient-level aggregation remains on the CN side despite the higher score in the third view. Limited diagnostic information in individual slices or learned associations that fail for this case are possible explanations, not established findings. No disease stage, label error or causal anatomical abnormality is demonstrated by the panel.

**CNN comparison and policy consequences:** The CNN also predicts CN, with raw P(AD)=0.1964 and calibrated P(AD)=0.2945, but refers under every policy. A1's calibrated CN confidence is 0.8927, sufficient for its reference policy to accept the wrong CN proposal. Because its AD score 0.1073 exceeds the strict-CN maximum of 0.05, the stricter rule instead refers it; no-automatic-CN also refers it. Unlike F01, this is an example where stricter negative acceptance changes an unsafe automated proposal into review.

**Recommended limitation or safeguard:** Describe the improvement as prevention of one automated negative decision, not as successful AD diagnosis. The clinician's eventual result is unmeasured. This case illustrates a potential benefit of stricter referral, while F01 shows that the same safeguard can leave more confident errors untouched. The aggregate evaluation, rather than this selected success alone, must determine the overall safety-workload interpretation.

### Cross-case conclusion

These selected cases illustrate complementary failure modes. F01 and F05 are false negatives shared by both classifiers, but the CNN refers both whereas A1's reference policy accepts their CN proposals. The stricter CN rule redirects F05 but not the more confident F01, showing why increasing a confidence requirement does not guarantee removal of all dangerous errors. F02 is a shared high-confidence false positive; F03 is an A1 false positive that the CNN classifies correctly, although cautiously. F04 demonstrates that a borderline patient-level output can conceal strongly opposing slice scores, and that referral can prevent acceptance of an uncertain prediction without making the diagnosis correct.

Calibration changes the reported probabilities but does not change these class decisions at 0.5. Negative-side policies do not correct AD flags, and referrals are not counted as successful diagnoses. The panels contain no feature-attribution information and cannot establish the anatomical or acquisition cause of an error. These five deliberately selected examples describe mechanisms and operational consequences; they do not estimate population error rates. Their individual outcomes should be interpreted alongside the complete confusion matrices, calibration results and policy-workload measurements. The evidence supports human assessment rather than autonomous diagnosis or reassurance, with causal explanations left uncertain where they are not demonstrated.

## 10. Engineering recommendation and limitations

Under the tested conditions, A1 did not justify its additional model complexity for the primary accuracy and macro-F1 objectives, and the required 0.80 test accuracy was not attained. It detected one more AD patient than the CNN but produced four more false positives. This small paired comparison does not establish that the CNN architecture is universally superior.

Fixed calibration improved A1's probability quality in the revised cohort without changing classifications. Stricter CN acceptance redirected four AD false negatives to review while also referring twenty additional CN patients; one AD patient still received automatic CN. The evidence does not support using this research prototype for autonomous diagnosis or negative clearance; its outputs should remain subject to clinical review. Referring everyone removes automatic negative decisions by definition, but provides no workload reduction when AD flags also require confirmation.

Limitations include limited independent patient counts, correlated slices and scans, slice-level supervision versus patient-level evaluation, repeatedly reused validation evidence, prior test exposure, seed sensitivity, JPEG rather than original physical MRI intensities, uncertainty about causal error triggers, and no prospective or cross-site clinical validation. Intervals do not account for every source of selection or training uncertainty. The later schedule-only experiment used training/validation evidence and did not meet its declared screen. This report does not select another model or policy from the exposed test outcomes.

## 11. Reproduction and tested interfaces

### Recorded environment and verification levels

The experiments and submission checks used Windows, Python 3.11.9, an NVIDIA GeForce RTX 4080 Laptop GPU and CUDA-build 12.6. Exact observed top-level packages are pinned in `requirements-pinned.txt`:

```text
torch==2.7.1+cu126
torchvision==0.22.1+cu126
numpy==2.2.6
Pillow==11.2.1
scikit-learn==1.6.1
matplotlib==3.10.3
```

This is an observed top-level specification, not a complete transitive dependency lock. The existing training environment was not changed. The official [PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/#v271) provide the 2.7.1/0.22.1 CUDA 12.6 wheel combination. Installing a separate pinned environment is a later user action, not a successful test claimed here; no unpinned/latest packages were installed.

| Verification level | Status and limits |
|---|---|
| (a) Clean code using the existing pinned environment | **Tested.** A Git archive export of `989d1096c967af33e6fcef3ce56dee695c24a1f6`, without .git, private pointers or historical run/report tree, reproduced manifests, executed the bounded training smoke, and predicted the fixed full validation manifest using explicit authorized data/weights/rules. Later report-only edits do not change that tested runtime. |
| (b) Separately created dependency environment | **Not created or tested.** No fresh-install success is asserted. |
| (c) Full retraining of A1 | **Not performed in submission verification.** The full route is implemented/documented; its recorded epoch-13 selection and accuracy are not guaranteed on another machine/run. |

Fifteen focused synthetic interface tests passed. The first clean export correctly refused the retained checkpoint because Git/Windows line-ending conversion changed the copied historical bytes. The fix restores the exact pre-change snapshot, protects frozen bytes using `.gitattributes`, and tests them against the recorded A1 selection hashes, rather than weakening comparisons. The failed check and its output are retained privately. Two bounded 32-slice, one-update smoke executions were performed across that correction; neither is a cohort epoch or a candidate experiment.

### Explicit data preparation

The reviewed preparation helpers are included unchanged in `preparation_helpers/`. `dataset.py prepare` reuses their metadata-to-patient rule, deterministic seed-42 allocation and CSV encoding. `APPROVED_PARTITIONS.json` binds the recorded metadata hash, approved manifest hashes/counts and accurate reported-approval provenance. This reproduces previously reported approval for exact existing assignments; it grants no new approval and requires no old author pointer files. Different metadata or assignments are rejected. Reproduction reads filenames/metadata, not image pixels; the existing integrity audit must not be confused with a newly repeated pixel audit.

The following PowerShell commands show the tested invocation syntax with explicit local resources. Replace paths for another authorized machine. The named verification outputs already exist; choose a **new** private output root to execute again. Existing destinations and outputs inside Git/data/manifests are refused.

```powershell
Set-Location '<checkout>/recognition/ConvNeXt_Alzheimers'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$data = 'C:/Users/Angus/COMP3710-private/data/ADNI'
$out = 'C:/Users/Angus/COMP3710-private/submission-readiness/clean-verification-v2'
& $py -B dataset.py prepare --data-root $data --output-dir "$out/manifests"
& $py -B test_submission_interfaces.py
& $py -B train.py a1-smoke --data-root $data --splits-dir "$out/manifests" --output-dir "$out/training-smoke" --device cuda
```

Manifest reproduction matched all original bytes, hashes and counts: train `90e42891babac142ba0693afc4d23e142c018c3a8dc274032552268f3e580be9`; validation `7b0755f4a7a561e501f8142a92f64356e4ebcd20c874404e66cd4e74b80eee99`; test `55f5dd35078a67af0c33d36c587047856aad5f0ba1720579d3f8b049a0337db7`. `--reference-splits-dir` is an optional byte-comparison oracle for verification, not a required author-only input. Original manifests are never overwritten or reshuffled.

The smoke uses the original scratch ConvNeXt-Tiny/configuration, 32 fixed training slices, four physical batches of eight and one optimizer update through the existing V2 epoch function. It checks finite loss, changed weights and exact checkpoint round-trip outputs using training input only. It opens zero validation/test images and is labelled `TRAINING_SMOKE_NOT_A_CANDIDATE`.

### Full selected-recipe training route

`train.py a1` calls the existing V2 implementation with the exact A1 recipe. It creates its own new training/validation data-check state from explicit inputs instead of requiring previous run reports or pointers. The old baseline CLI and `--step8` dispatch remain available; original function bodies are unchanged. Historical executions requiring exact old source hashes use the frozen revision described in `HISTORICAL_REPLAY.md`.

The following full-training command is **documented, not executed in this task**. Use it only for an authorized reproduction run, after actually reviewing preprocessing:

```powershell
& $py -B train.py a1 --full-training --confirm-preprocessing-review --data-root $data --splits-dir "$out/manifests" --output-dir '<NEW_PRIVATE_A1_RUN>' --device cuda
# Recovery uses the same inputs and existing output directory, adding --resume.
```

Maximum epochs 30, minimum 12 and patience 8 are retained. The optimizer is AdamW with the recorded grouping/clipping and original 30-epoch cosine endpoint. Selection is validation patient accuracy, then macro-F1, then lower ordinary patient log loss, earliest exact tie. Epoch 13 is the retained historical result, not a hard-coded target or guaranteed best epoch. The selected export is `<NEW_PRIVATE_A1_RUN>/run/PRIVATE_best_model.pt`; hashes/configuration and predictions remain beside it for verification.

### Independent selected-checkpoint prediction

The retained A1 weights are available in the author's authorized local workspace at the explicit path below. No public redistribution permission or download link is established. Readers without authorized access must use authorized data and the full training-to-checkpoint route above. Datasets, weights, manifests and private predictions must not be committed.

```powershell
$checkpoint = 'C:/Users/Angus/COMP3710-private/v2-scratch_augmented-20261024-161044-951159/PRIVATE_best_model.pt'
& $py -B predict.py infer --checkpoint $checkpoint --checkpoint-sha256 efde4d21993085119d3ef1bc0efb88f837954c187c95b036373cd00e59d4d5ef --data-root $data --splits-dir "$out/manifests" --split validation --output-dir "$out/prediction" --device cuda
```

Ordinary inference requires no old baseline/comparison/test report tree. It checks the explicit checkpoint digest and exact source provenance against the current or frozen runtime, then exact architecture/state keys/shapes/dtypes/finite values, label map and manifest identities. It executes the matching dataset/model/aggregation source bytes in a dedicated process. A fixed input list may be supplied with `--input-list`: it must select complete patients from the declared train/validation manifest, preserving all scans/slices for hierarchical probabilities. Ordinary inference exposes only train/validation; historical test commands remain separate and explicitly guarded.

For frozen calibration/reference routing, add explicit `--rules <a1_operating_rules.json> --rules-sha256 <trusted-file-hash>`. For all already frozen policies, also add `--policy-rules <negative_policy_rules.json> --policy-rules-sha256 <trusted-file-hash>`. These files must bind to the exact checkpoint, epoch, manifests/configuration and fixed definitions. They are applied, never fitted. The clean verification exercised both optional inputs and reproduced the recorded apparent full-validation policy counts (auto-CN / AD flag / refer): reference 41/7/54; strict-CN 21/7/74; no-auto-CN 0/7/95. This does not turn in-sample policy results into independent validation.

The retained checkpoint's clean-interface replay matched **all 4,660 slice, 233 scan and 102 patient probabilities exactly**, below the existing 2e-5 tolerance, with unchanged class decisions. A1's 83.33% validation accuracy was reproduced as a loading/interface check. No new test inference, threshold fitting, calibration refitting or model selection occurred.

### Source preservation and remaining submission actions

Before interface edits, all 70 original assignment files and README were copied privately with SHA-256 hashes and Git revision `b67bb1327c744fdd9a30fea439efc4a5005fca27`. The original checkout and pointers remain intact. `historical_runtime/` preserves exact checkpoint-relevant source bytes; `HISTORICAL_REPLAY.md` documents inspection and validation replay. No historical checkpoint's expected hashes or checks were rewritten.

The author explicitly confirmed the listed code reviews, earlier local executions and revised-case review during this task. Consultation/check-off status, exact dates/feedback, image-publication permission, PR submission, Turnitin PDF submission and later code-review response are not invented. Review and place the local commits on the appropriate submission branch, inspect the PR draft, then perform the authorized push/PR and Turnitin actions yourself. No push, PR creation or submission was performed by this packaging task. The measured 0.80 accuracy requirement remains unmet.


## Artificial Intelligence Usage Disclosure

ChatGPT assisted the earlier project with assessment interpretation, experiment planning, data/model interfaces, Python implementations and synthetic tests, PowerShell instructions, debugging, confidence/policy analysis and report/failure-case drafting. Scientific methods are separately attributed in the references. The work is not presented as wholly unaided authorship.

Codex subsequently performed the scoped source/evidence audit, implemented and executed the predeclared negative schedule-only experiment, and implemented this submission packaging: explicit-input preparation/training/prediction routes, source preservation, exact-byte provenance correction, focused tests, bounded training-only smoke checks, validation-only replay, case-record/panel correspondence verification, README consolidation and PR-description drafting. Codex's checks ran through the existing pinned environment on the author's Windows host. No separate dependency environment was created, no full retraining occurred during packaging, no test inference or calibration/threshold fitting was repeated, and no push/PR/submission occurred.

The author explicitly confirmed review of labels/grouping, preprocessing, model/loss/accumulation and confidence logic; execution of the earlier local checks; and review of the revised F01–F05 cases. The retained author account also describes reporting the PyTorch 2.7.1 scalar-shape loading issue. These are author-reported contributions, distinguished from the checks directly executed by Codex; they are not independent certification of personal actions.

The submitting author remains responsible for reviewing and understanding the code, calculations, case discussions and statements. Automated tests and replay establish the specific checked behaviours, not complete correctness or clinical reliability. Negative experiments, prior test exposure, the unmet accuracy target, unresolved causal error explanations and the untested fresh dependency environment remain disclosed.


## References

1. University of Queensland, *COMP3710 / COMP7710 Report: Pattern Recognition*, Version 2.5, Semester 2, 2026. Sections 2.1, 3-6.
2. Liu et al., *A ConvNet for the 2020s*, CVPR 2022. https://arxiv.org/abs/2201.03545 . ConvNeXt architectural reference; no pretrained weights used.
3. Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017. https://arxiv.org/abs/1706.04599 . Temperature-scaling method reference.
4. PyTorch, [previous versions](https://pytorch.org/get-started/previous-versions/#v271), exact 2.7.1 / torchvision 0.22.1 CUDA 12.6 installation combination. This packaging task did not perform that installation.

The result tables are taken from retained project experiment summaries and histories; submission checks reproduce the recorded validation outputs. No external benchmark has been substituted for the project's measurements. See the private evidence snapshot and the publicly appropriate plan/hash records for provenance.

## Appendix A. Original feasibility proposal

The following preserves the original supplied proposal content with heading levels adjusted for this README; the exact original file is [FEASIBILITY_ORIGINAL.md](FEASIBILITY_ORIGINAL.md). It is a historical proposal, not a retrospective claim that its checks had already been completed. Approval was reported by the author; an exact date or undocumented feedback is not fabricated.

### One-page feasibility review
#### Alzheimer’s classification with ConvNeXt
**Status:** Initial planning draft based on supplied directory listings; experiments and resource measurements pending.

#### User need and research question
Develop a research-only assistant for medical-image reviewers: classify ADNI brain MRI images as Alzheimer’s disease (AD) versus cognitively normal (CN), and flag uncertain predictions. Does ConvNeXt improve classification and confidence-based referral over a simple CNN enough to justify its computational cost? This is not a clinically validated diagnostic system.

#### Model choice and course concepts
Proposed main model: ConvNeXt-Tiny in PyTorch, within the approved Hard Difficulty ConvNeXt family; baseline: a small CNN using the same preprocessing and evaluation split [1, 2]. Relevant concepts are convolutional feature learning, cross-entropy optimisation, regularisation, generalisation and confidence calibration. Input dimensions and channel handling will follow image inspection.

#### Preliminary feasibility evidence
Read-only listings confirm access to /home/groups/comp3710/ADNI, AD_NC/{train,test}/{AD,NC}, sampled .jpeg files under test/NC, and meta_data_with_label.json. This is directory-level evidence only: image decoding, patient counts/mapping, split independence and compute performance remain unverified. The disk label NC must be reconciled with the brief’s CN terminology.

#### Five testable acceptance criteria
| Criterion | Target / evidence |
|---|---|
| 1. Data integrity | Verify patient identifiers and zero patient overlap across training, validation and test sets; keep all visits and slices of a patient together. |
| 2. Classification | Target ≥0.80 held-out test accuracy for ConvNeXt [1]. Report per-class precision/recall, macro-F1 and AUROC; predeclare the evaluation unit and any slice aggregation. |
| 3. Fair comparison | Train a reproducible simple CNN baseline and ConvNeXt, record learning curves, and compare them on identical evaluation data without test-set tuning. |
| 4. Confidence and errors | Select a referral threshold on validation data only. Report held-out coverage, error among accepted predictions and AD recall; analyse 3–5 representative failures and compare both models. |
| 5. Resources | Proposed cap: ≤12 GiB peak allocated GPU memory at the chosen training settings. Record both models’ parameter counts, inference latency and training time; confirm the cap after a feasibility smoke test, before full training. |

#### Data splitting and next experiment
Use metadata to establish patient/scan/visit relationships; do not treat a filename prefix as a verified patient ID. Retain the supplied test partition only if patient-independent, and derive validation from training patients. If overlap or missing identifiers prevent this, pause and agree a corrected split with the tutor. Next: inspect metadata, audit counts/overlap, decode training samples, then run a small baseline smoke test in an allocated GPU session.

#### Budget, risks and fallback
Provisional planning budget: one GPU and 12 GPU-hours for initial experiments; availability, quota and runtime are not yet confirmed. Risks include leakage, label ambiguity, imbalance, overfitting, overconfidence and memory limits. Reduce batch size/resolution or use supported mixed precision if necessary, while retaining ConvNeXt as the Hard main model. A baseline-only fallback does not preserve full-mark eligibility. Revise the budget transparently using measured evidence.

**Sources and status:** [1] Supplied COMP3710 assessment brief, §§2.1, 3–4. [2] Liu et al., “A ConvNet for the 2020s,” CVPR 2022 (brief reference [7]). All budgets/configurations above are proposals, not measured results.
