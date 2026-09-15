# Methods (draft)

Edit freely — this is a starting point sized for a medical imaging journal, not
finished copy. Bracketed items are placeholders to fill once results are in.

---

## Dataset

Sinus CT studies from [N] patients were retrospectively collected at
[institution] under [IRB/ethics approval number]. Each study carried an expert
segmentation of 15 ethmoid and frontal structures, annotated in 3D Slicer with a
per-case terminology table mapping label values to anatomical names.

Of 35 studies, one was excluded because its image and label volumes had
mismatched dimensions, leaving **34 studies** for analysis.

Voxel spacing varied across the cohort: 0.298–0.469 mm in-plane and 0.50–1.25 mm
through-plane, across 15 distinct spacings. All volumes were 512 × 512 in-plane
with 70–433 slices.

## Label schemes

Three segmentation tasks of increasing granularity were derived from the same
expert annotations by merging source structures:

| Task | Classes | Composition |
|---|---|---|
| Experiment 1 | 2 | Anterior (RANC, RSAC, RSAFC, LANC, LSAC, LSAFC); Posterior (RBE, RSBC, RSBFC, RSOEC, LBE, LSBC, LSBFC, LSOEC) |
| Experiment 2 | 4 | Left/Right × Anterior/Posterior, from the same source structures |
| Experiment 3 | 5 | The four above plus the frontal sinus cell (FSC) |

Merging was performed by structure **name**, read from each case's terminology
table, rather than by label value. Label values were not consistent across
patients, so a value-based merge would have silently assigned structures to the
wrong class. Structures not listed above (RFSDP, LFSDP) were mapped to background.

Class prevalence across the 34 cases: [Anterior 34/34, Posterior 34/34,
LeftAnterior 34/34, RightAnterior 31/34, LeftPosterior 34/34, RightPosterior
34/34, FSC 10/34]. Three patients had no right anterior cells and 24 had no
frontal sinus cell segmented; these absences are anatomical or annotation-level
and are handled explicitly in the evaluation (below).

## Preprocessing

All volumes were resampled once to **0.35 × 0.35 × 1.25 mm**, the cohort median
spacing, using cubic interpolation for images and nearest-neighbour for labels.
This grid was chosen to preserve sub-millimetre ethmoid septa; coarser isotropic
resampling (e.g. 1 mm) would have downsampled in-plane resolution roughly
threefold.

Two intensity variants were produced from the same resampling pass:

- **Raw Hounsfield units**, used by nnU-Net, which derives its own foreground
  intensity statistics and normalisation from the training set.
- **Windowed to [-1000, 2000] HU and scaled to [0, 1]**, used by the two
  foundation models, matching the input convention of their public checkpoints.

Geometry is therefore identical across models; only intensity scaling differs.

## Data splitting

Five cases were held out as a fixed test set. The remaining 29 were used for
5-fold cross-validation. The split was drawn once from a fixed seed and reused
without modification for all three models and all three experiments, so
between-model differences cannot arise from different data partitions.

Splitting was **stratified on rare classes**. With FSC present in only 10 of 34
cases, unstratified splitting produced at least one fold containing no positive
validation case in approximately half of random seeds, which would make that
fold's Dice for the class undefined. Stratification eliminated this. For
nnU-Net, the same partition was enforced by overwriting its
`splits_final.json` prior to training.

## Models

Three models were compared. They are **not architecturally equivalent**, and the
differences below are material to interpreting the results.

### nnU-Net v2 (task-specific baseline)

Self-configuring 3D U-Net trained from random initialisation. Planning selected
the `3d_fullres` configuration with patch size 56 × 192 × 192, batch size 2, six
resolution stages, and CT normalisation, retaining the study's 0.35 × 0.35 ×
1.25 mm grid. Trained with `[nnUNetTrainer_250epochs]` for [250] epochs per
fold. Inference is fully automatic and requires no prompt.

### SAM-Med3D (3D foundation model)

Initialised from the public `sam_med3d_turbo` checkpoint and fine-tuned on this
dataset. The model accepts binary masks only, so each N-class task was
decomposed into N binary sub-tasks trained independently and merged post hoc,
with overlaps resolved in order of decreasing structure volume.

Two properties must be stated when interpreting its results. First, inference
requires a **point prompt derived from the ground-truth mask**, so reported
performance reflects prompted rather than automatic segmentation. Second, the
model resamples internally to **1.5 mm isotropic**, approximately fourfold
coarser in-plane than the grid nnU-Net used.

### MedSAM3 (2D foundation model)

A LoRA adapter over SAM3, fine-tuned on this dataset. Because the model operates
on 2D images with text prompts, volumes were sliced along the axial axis and
exported in COCO format; slices containing no foreground were excluded. Class
names were expressed as anatomical text prompts (e.g. "left anterior ethmoid air
cells"). Per-slice predictions were re-stacked into 3D volumes, and the largest
connected components per class were retained. Zero-shot performance was recorded
prior to fine-tuning as a reference point.

## Evaluation

All predictions were evaluated **volumetrically** on the shared grid, so the
three models are scored on the same quantity despite differing internally.

Metrics: Dice similarity coefficient, intersection over union, 95th-percentile
Hausdorff distance, and normalised surface Dice at a 2 mm tolerance.

Structures absent from a patient yield undefined Dice and were excluded from
means rather than scored as zero; a false positive on an absent structure was
scored as zero. This convention is stated explicitly because it materially
affects FSC results in Experiment 3.

Two protocols are reported. **Cross-validation** covers all 29 training cases,
each scored by the fold that did not train on it, and carries the statistical
comparison. The **held-out test set** (5 cases) is reported to confirm the
cross-validation estimate is not inflated; it is too small to separate models on
its own.

## Statistical analysis

Models were compared pairwise on cross-validation results using the Wilcoxon
signed-rank test, paired per (case, class). A non-parametric test was chosen
because Dice is bounded and skewed at this sample size. [Three pairwise tests
were performed; report whether a Bonferroni correction (p < 0.0167) was applied
or the comparisons are framed as exploratory.] Fold-to-fold standard deviation
is reported alongside means, and between-model differences smaller than that
spread are not interpreted as meaningful.

## Reproducibility

[Record: nnU-Net version and trainer, SAM-Med3D commit hash, MedSAM3 commit and
LoRA revision, checkpoint checksums, split seed (42), target spacing, CT window,
and NSD tolerance.]

---

## Limitations to state in the Discussion

1. **The three models do not solve identical tasks.** SAM-Med3D receives a
   ground-truth-derived point prompt at inference; nnU-Net receives nothing.
   Reporting a single comparison table without this caveat would overstate
   prompted methods.
2. **Effective input resolution differs.** SAM-Med3D resamples internally to
   1.5 mm isotropic and MedSAM3 operates on 2D slices, so the models do not see
   equivalent data.
3. **MedSAM3 has no inter-slice consistency mechanism.** Volumetric metrics are
   computed on stacked 2D predictions.
4. **Sample size.** With 29 training and 5 test cases, confidence intervals are
   wide and the study is best framed as a feasibility comparison.
5. **FSC prevalence.** Present in 10 of 34 cases, limiting what can be concluded
   about the fifth class in Experiment 3.
6. **Single-centre, single-annotator data.** No inter-observer variability
   estimate is available, so the annotation is treated as ground truth.
