# Local closeout and remaining submission actions

29 October 2026. Documentation preparation only; no push, PR, message or submission was performed.

## Verified locally

- Required five files modules.py, dataset.py, train.py, predict.py and README.md exist under recognition/ConvNeXt_Alzheimers. Explicit preparation/A1 training/independent prediction corrections from submission commit 5851524 are included by ancestry; original source and scientific selection remain unchanged. Historical inspection/replay route is [HISTORICAL_REPLAY.md](HISTORICAL_REPLAY.md).
- Retained A1 is scratch_augmented seed 42 epoch 13. Primary recorded test-slice accuracy is 3157/4640 (68.04%), not 74.51% patient-test or 83.33% patient-validation accuracy. The 80% target remains unmet.
- Original feasibility proposal, model principles/diagram table, exact top-level dependencies, baseline comparison, preserved aggregate learning curves, scoped resources and five author-reviewed patient-case autopsies remain in README. F04's opposing slice scores are explicitly distinguished from its wrong patient aggregate. No MRI/identifiers are included. Reuse prior panel correspondence verification; no new panel was opened.
- Clean-code verification in the existing pinned environment was completed previously: byte-identical manifests, two bounded training-only smoke checks across a packaging correction and exact full-validation A1 replay (all4660/233/102 scores), within 2e-5 and unchanged decisions. This task did **not** repeat those executions. Current runtime-source drift is checked statically. No separately created dependency environment or full retraining was tested.
- Progressive history includes submission-readiness, G012, G3 and export-fix commits. The new closeout branch descends from daf136c, including 5851524 and 500bfe8. Original topic-recognition was clean at b67bb13 at inspection. Nothing was reset, squashed, force-updated or integrated into the original checkout automatically.
- Static preservation check: 126 tracked files outside the two edited report/draft files match the completed G3 worktree byte for byte, including all 11 frozen historical runtime files. The seven preparation/training/prediction interface files have identical Git content to the completed submission branch; three submission helper working files inherit CRLF/LF differences from G3 compared with the older submission worktree. This task changes no runtime source and does not rewrite any checkpoint's expected source hashes or integrity guard. Static continuity is not a new real-data replay.

## User actions remaining, in order

1. Read the corrected README and tables, confirm the author/student details and code understanding, and confirm actual practical consultation/check-off completion. Approval was reported previously; exact date, feedback and official hurdle record are not independently verified. Do not invent them.
2. Review the local closeout diff and progressive commits. Bring the completed closeout branch onto your fork's topic-recognition through the history-preserving commands below, only after confirming its working tree is clean and checking whether remotes have changed. These commands use current local ancestry; no fetch/network check was performed here. If the branch has diverged, stop and inspect rather than reset/force/squash.
3. Inspect README rendering and permitted aggregate figures locally/on GitHub, then export that corrected README to PDF and visually check tables, headings, figures and links. PDF export was not performed here. Medical-panel permission remains unestablished; do not add MRI or participant identifiers without it. Five textual patient autopsies are present; whether teaching staff require permitted image panels or more primary-image uncertainty analysis remains a marking question, not a claimed completed requirement.
4. Review the professional [PR draft](PR_DESCRIPTION_DRAFT.md). When ready, push your fork's topic-recognition and open a PR from KingGoose20/COMP3710-final-report:topic-recognition to shakes76/PatternAnalysis-2026:topic-recognition. **No PR is claimed to exist.** Confirm actual course deadline and required repository destination. Do not include data, weights, manifests or private predictions.
5. Submit the corrected README PDF to Turnitin, retain receipts and verify the PR/Turnitin status yourself. These actions remain undone by Codex. Later read the teaching-team review, commit responses and reply within the announced review window.

## Local integration commands — review only, not executed

Run from your original repository after reviewing the closeout branch. The first two commands are read-only; merge refuses non-fast-forward divergence. Preserve any new uncommitted work before proceeding.

```powershell
Set-Location 'C:/Users/Angus/Documents/GitHub/COMP3710-final-report'
git status --short
git log --oneline topic-recognition..codex/slice-alignment-closeout
# Continue only if status is clean and the reviewed local ancestry is still appropriate.
git switch topic-recognition
git merge --ff-only codex/slice-alignment-closeout
git status --short
git log -8 --oneline
```

This is a single fast-forward of progressive history, not a series of cherry-picks of overlapping submission/G012/G3 branches. Publication/submission is a separate manual action. Old G012 scientific replay still uses its preserved original worktree/revision; the G3 branch intentionally includes its compatible supervisor extension and export adapter. Frozen source/manifest checks are not weakened.

## Unresolved assessment question — unsent Ed draft

> Thank you for clarifying that the 0.80 requirement is slice-level test accuracy. Could you clarify how falling below it affects the marking: is it an absolute hurdle or a condition for particular implementation/difficulty marks, or is performance assessed with partial credit alongside the other criteria? Our retained model records 68.04% slice-level test accuracy, and we will report this explicitly. We have retained the baseline, patient-independent splits, uncertainty investigation and negative experiments.

No message was sent. This does not request or assume a waiver, and the report proceeds with the stated shortfall. Primary-unit calibrated uncertainty/referral evidence and an untouched external cohort remain unavailable; fitting or evaluating them is not authorised by this closeout.
