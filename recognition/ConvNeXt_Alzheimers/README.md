# Pattern Analysis
Pattern Analysis of various datasets by COMP3710 students in 2025 at the University of Queensland.

We create pattern recognition and image processing library for Tensorflow (TF), PyTorch or JAX.

This library is created and maintained by The University of Queensland [COMP3710](https://my.uq.edu.au/programs-courses/course.html?course_code=comp3710) students.

The library includes the following implemented in Tensorflow:
* fractals 
* recognition problems

In the recognition folder, you will find many recognition problems solved including:
* segmentation
* classification
* graph neural networks
* StyleGAN
* Stable diffusion
* transformers
etc.



<!-- COMP3710-STEP6-DATA-PIPELINE -->
## Data pipeline

The finalized patient-independent manifests, rather than the original directory names, determine the training, validation and test partitions. The CSV hashes and grouping consistency were verified before loading images. Model labels are NC = 0 and AD = 1; the original metadata codes 0 and 2 are retained as provenance.

| Partition | Patients | Scans | Images |
|---|---:|---:|---:|
| train | 476 | 1061 | 21220 |
| validation | 102 | 233 | 4660 |
| test | 102 | 232 | 4640 |

### Preprocessing and checks

Each supplied 256-by-240 greyscale JPEG is resized to 224-by-210 using bilinear interpolation and padded with seven black pixels above and below. No anatomy is cropped and the original aspect ratio is preserved. The result is one channel of 224-by-224 float32 values, scaled using (pixel / 255 - 0.5) / 0.5. This fixed scaling does not fit statistics to validation or test images. It does not recover physical MRI intensities from the supplied JPEGs.

The initial configuration uses no random augmentation. A small training-only rotation is implemented but disabled; enabling it later is a separate, documented experiment. Both the CNN baseline and ConvNeXt will use the same selected preprocessing.

A CPU-only pass successfully checked every final training and validation sample for shape, dtype, finite/range-correct values and exactly-once coverage, including the last partial batch. It did not train a network or decode the held-out test images. The test manifest was read only for split-integrity verification.

Subject and scan identifiers are retained privately in each batch for later hierarchical aggregation of probabilities; slice counts must not be confused with independent patient counts. Patient-level evaluation and confidence-based referral will be implemented with the training/evaluation stages.

A private before/after preview was generated for separate visual inspection. A local-byte versus Rangpur-source hash comparison was not performed. Model training, GPU memory and runtime measurements remain outstanding.

### Artificial Intelligence Usage Disclosure — update

ChatGPT assisted with drafting the data loader, verification script and synthetic tests. The code was checked by automated synthetic tests and a local CPU loader pass. I will document my own code review, understanding, visual inspection and any changes; these are not implied by the automated pass.

Manual visual check (2026-10-22): I inspected the private before/after training preview and found no unintended orientation, cropping or stretching changes.


<!-- COMP3710-STEP7-BASELINE -->
## Initial CNN baseline experiment

A four-block convolutional classifier was trained from scratch as the simpler comparison for the forthcoming ConvNeXt model. It uses the unchanged one-channel, 224-by-224 data pipeline. The fixed model labels are NC = 0 and AD = 1.

### Predeclared training and model selection

Configuration: `{'batch_size': 32, 'epochs': 20, 'patience': 5, 'learning_rate': 0.001, 'weight_decay': 0.0001, 'seed': 42, 'num_workers': 0, 'gradient_clip_norm': 5.0, 'device': 'cuda'}`. Training uses ordinary unweighted slice-level cross-entropy and uniformly shuffled images; repeated scans therefore contribute more training examples. No class weighting, data augmentation, pretrained weights, learning-rate search, calibration or threshold tuning was used in this initial experiment.

Validation aggregation averages slice probabilities within each scan, then averages the scan probabilities within each patient. The probability threshold is 0.5. Checkpoint selection uses patient macro-F1, with lower patient log loss breaking exact ties; the earliest exact tie is retained. Early stopping uses the same rule. These validation results are selected development results, not an unbiased final-test estimate.

Completed epochs: **12**; selected epoch: **7**.

| Validation unit | N | Accuracy | Macro-F1 | AUROC |
|---|---:|---:|---:|---:|
| slice | 4660 | 0.6910 | 0.6747 | 0.7790 |
| scan | 233 | 0.7082 | 0.6867 | 0.8236 |
| patient | 102 | 0.7843 | 0.7292 | 0.8327 |

Patient AD recall: **0.5152**. Patient NC recall: **0.9130**.
Training-majority-class reference validation accuracy: **0.6765**.

### Resources and reproducibility

Learnable parameters: **359,922**. Hardware: `{'torch_version': '2.7.1+cu126', 'device': 'cuda', 'precision': 'float32; mixed precision and TF32 disabled', 'device_name': 'NVIDIA GeForce RTX 4080 Laptop GPU', 'total_device_memory_gib': 11.99365234375}`.
Training-loop wall time (all completed epochs, including validation/checkpoints/history): **555.47 seconds**.
GPU allocator peaks: `{'peak_allocated_gib': 0.1967020034790039, 'peak_reserved_gib': 0.29296875, 'note': 'PyTorch process allocator peaks, not total VRAM used by Windows/other processes.'}`.
Model-only batch-size-1 latency: `{'batch_size': 1, 'repeats': 50, 'warmup': 10, 'median_ms_per_slice': 0.8040000102482736, 'p95_ms_per_slice': 0.8248999947682023, 'scope': 'Forward pass only, tensor already on device; not patient/end-to-end latency.'}`. This excludes image decoding/preprocessing and is not a full-patient latency.

CSV hashes, source-code hashes, the exact configuration, per-epoch history and the selected checkpoint were retained. Checkpoints and patient-linked prediction files are private and are not committed to GitHub. The final test images have not been evaluated by this step.

### Limitations and next experiment

This is one initial seed/configuration. The small training-subset fitting check is a debugging test, not evidence of generalisation. Validation performance may be optimistic because it selected the checkpoint. The test comparison, confidence-based referral, failure analysis and comparison with ConvNeXt remain outstanding.

### AI assistance for this stage

ChatGPT assisted with the baseline architecture, training/evaluation code, synthetic tests and this report draft. The student must document the checks they personally performed and must not treat generated code or test results as a substitute for understanding the model.

![Baseline training and validation losses](figures/baseline_loss.png)

![Baseline validation patient metrics](figures/baseline_validation.png)


<!-- COMP3710-STEP8-CONVNEXT -->
## Step 8: baseline regularized experiment

This model was trained from random initial weights on the unchanged patient-independent training partition. Preprocessing and validation aggregation match the initial CNN. No pretrained weights, image augmentation, calibration, decision-threshold search or final-test evaluation were performed. The operating threshold remains 0.5.

Configuration: `{'model': 'baseline', 'batch_size': 8, 'accumulation_steps': 4, 'epochs': 30, 'min_epochs': 12, 'patience': 8, 'learning_rate': 0.0003, 'min_learning_rate': 1e-06, 'warmup_epochs': 3, 'weight_decay': 0.05, 'gradient_clip_norm': 5.0, 'seed': 42, 'num_workers': 0, 'device': 'cuda', 'budget_seconds': 10800.0}`.

AdamW uses stronger weight decay on kernels/matrices, excluding biases and one-dimensional normalization/LayerScale parameters. Learning rate has linear warmup followed by cosine decay. Both Step 8 architectures use the same settings, including physical batch size and gradient accumulation. Accumulation does not make BatchNorm behave like a larger physical batch.

Checkpoint selection: validation patient macro-F1, then lower patient log loss, then earliest. These are selected development results, not independent final-test estimates.

Completed epochs: 22; selected epoch: 14; stop reason: patience_after_minimum_epochs.

| Unit | N | Accuracy | Macro-F1 | AUROC | AD recall |
|---|---:|---:|---:|---:|---:|
| slice | 4660 | 0.6833 | 0.6724 | 0.7440 | 0.5670 |
| scan | 233 | 0.7382 | 0.7276 | 0.8069 | 0.6117 |
| patient | 102 | 0.7843 | 0.7496 | 0.8037 | 0.6364 |

Parameters: 359,922. Hardware: `{'torch_version': '2.7.1+cu126', 'device': 'cuda', 'precision': 'float32; mixed precision and TF32 disabled', 'device_name': 'NVIDIA GeForce RTX 4080 Laptop GPU', 'total_device_memory_gib': 11.99365234375}`.
Training loop time: 1609.03 seconds. GPU allocator peaks: `{'peak_allocated_gib': 0.11696767807006836, 'peak_reserved_gib': 0.166015625, 'note': 'PyTorch process allocator peaks, not total VRAM used by Windows/other processes.'}`.
Batch-1 model-only latency: `{'batch_size': 1, 'repeats': 50, 'warmup': 10, 'median_ms_per_slice': 0.8052500197663903, 'p95_ms_per_slice': 0.8132000220939517, 'scope': 'Forward pass only, tensor already on device; not patient/end-to-end latency.'}`. This excludes JPEG loading, preprocessing and patient aggregation.

Limitations: one seed per configuration, repeated use of the development validation set, slice-weighted training (patients with more scans contribute more examples), and no clinical validation. The two Step 8 runs share a recipe, but it is not claimed to be optimal for both architectures. The Step 7-to-Step 8 change is a multi-setting intervention, not an isolated causal test of weight decay. Data remain restricted; private predictions/checkpoints are not committed. Confidence referral and final held-out comparison remain outstanding.

AI disclosure: ChatGPT assisted with implementation, tests and this draft. Record the actual source review and local checks performed; do not claim unperformed checks.

## Step 8: convnext regularized experiment

This model was trained from random initial weights on the unchanged patient-independent training partition. Preprocessing and validation aggregation match the initial CNN. No pretrained weights, image augmentation, calibration, decision-threshold search or final-test evaluation were performed. The operating threshold remains 0.5.

Configuration: `{'model': 'convnext', 'batch_size': 8, 'accumulation_steps': 4, 'epochs': 30, 'min_epochs': 12, 'patience': 8, 'learning_rate': 0.0003, 'min_learning_rate': 1e-06, 'warmup_epochs': 3, 'weight_decay': 0.05, 'gradient_clip_norm': 5.0, 'seed': 42, 'num_workers': 0, 'device': 'cuda', 'budget_seconds': 10800.0}`.

AdamW uses stronger weight decay on kernels/matrices, excluding biases and one-dimensional normalization/LayerScale parameters. Learning rate has linear warmup followed by cosine decay. Both Step 8 architectures use the same settings, including physical batch size and gradient accumulation. Accumulation does not make BatchNorm behave like a larger physical batch.

Checkpoint selection: validation patient macro-F1, then lower patient log loss, then earliest. These are selected development results, not independent final-test estimates.

Completed epochs: 19; selected epoch: 11; stop reason: patience_after_minimum_epochs.

| Unit | N | Accuracy | Macro-F1 | AUROC | AD recall |
|---|---:|---:|---:|---:|---:|
| slice | 4660 | 0.6601 | 0.6508 | 0.7205 | 0.5621 |
| scan | 233 | 0.6953 | 0.6829 | 0.7879 | 0.5631 |
| patient | 102 | 0.7647 | 0.7167 | 0.8006 | 0.5455 |

Parameters: 27,818,594. Hardware: `{'torch_version': '2.7.1+cu126', 'device': 'cuda', 'precision': 'float32; mixed precision and TF32 disabled', 'device_name': 'NVIDIA GeForce RTX 4080 Laptop GPU', 'total_device_memory_gib': 11.99365234375}`.
Training loop time: 5262.38 seconds. GPU allocator peaks: `{'peak_allocated_gib': 1.3108057975769043, 'peak_reserved_gib': 1.392578125, 'note': 'PyTorch process allocator peaks, not total VRAM used by Windows/other processes.'}`.
Batch-1 model-only latency: `{'batch_size': 1, 'repeats': 50, 'warmup': 10, 'median_ms_per_slice': 9.246149999853515, 'p95_ms_per_slice': 14.084000000366359, 'scope': 'Forward pass only, tensor already on device; not patient/end-to-end latency.'}`. This excludes JPEG loading, preprocessing and patient aggregation.

Limitations: one seed per configuration, repeated use of the development validation set, slice-weighted training (patients with more scans contribute more examples), and no clinical validation. The two Step 8 runs share a recipe, but it is not claimed to be optimal for both architectures. The Step 7-to-Step 8 change is a multi-setting intervention, not an isolated causal test of weight decay. Data remain restricted; private predictions/checkpoints are not committed. Confidence referral and final held-out comparison remain outstanding.

AI disclosure: ChatGPT assisted with implementation, tests and this draft. Record the actual source review and local checks performed; do not claim unperformed checks.

## Step 8: validation comparison

The original CNN experiment was retained. One more strongly regularized CNN and one ConvNeXt-Tiny were trained from scratch using a shared, predeclared Step 8 recipe. The data splits, image preprocessing, validation aggregation, selection criterion and 0.5 decision threshold were unchanged. Final-test images were not evaluated.

| Model | Patient accuracy | Macro-F1 | AUROC | AD recall | NC recall | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|
| Initial CNN | 78.43% | 0.7292 | 0.8327 | 51.52% | 91.30% | 7 |
| Regularized CNN | 78.43% | 0.7496 | 0.8037 | 63.64% | 85.51% | 14 |
| ConvNeXt-Tiny | 76.47% | 0.7167 | 0.8006 | 54.55% | 86.96% | 11 |

Preferred CNN under the predefined validation selection rule: **Regularized CNN**. Retain both CNN runs and report this selection; do not select a weaker baseline to favour ConvNeXt.

The initial-to-regularized CNN comparison changes multiple training settings at once; it does not isolate weight decay. The Step 8 CNN and ConvNeXt have matched optimizer settings and batch configuration, but their architectures and intrinsic regularizers differ. One seed, repeated validation use and a small validation cohort limit generalisation claims. Check stop reasons for any time-limited runs. No result here demonstrates attainment of the final-test accuracy target. Confidence calibration, referral analysis and the final test remain outstanding.

### Implementation and verification

ConvNeXt-Tiny uses stage depths (3,3,9,3), widths (96,192,384,768), a one-channel stem, depthwise 7x7 convolutions, channel LayerNorm, 4x inverted bottlenecks, GELU, LayerScale and stochastic depth. It was implemented in PyTorch without downloading pretrained weights. Reference: Liu et al., *A ConvNet for the 2020s*, CVPR 2022, https://arxiv.org/abs/2201.03545 (assessment reference [7]).

ChatGPT assisted with code, tests and this draft. Record the checks and code review actually performed in the report's Artificial Intelligence Usage Disclosure.

![baseline loss](figures/step8_baseline_loss.png)

![baseline validation](figures/step8_baseline_validation.png)

![convnext loss](figures/step8_convnext_loss.png)

![convnext validation](figures/step8_convnext_validation.png)
