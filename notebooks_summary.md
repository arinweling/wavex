
# wavelet_explanation_pipeline.ipynb

This notebook distills the entire training pipeline into a single, step-by-step notebook. It trains a U-Net to produce sparse binary masks in the Haar wavelet domain that explain a frozen classifier's decisions.

**Pipeline:**
1. Load a YAML config 
2. Build the Haar DWT, U-Net mask generator, and frozen classifier
3. Train the U-Net with the full loss suite (L_act, L_CE, L_KL, L_rob, L_area, L_bin)
4. Visualize masks and explanations
5. Compute evaluation metrics (sparsity, label preservation, confidence delta)

**Key idea:** Instead of masking pixels directly, we mask *wavelet coefficients* — learning which frequency content (shape, edges, texture) the classifier actually uses.
## Cell 1 — Imports
## Cell 2 — Load Config

Point `CONFIG_PATH` at any YAML config from `wavelet_explanation/configs/`.
Set `DATA_PATH` to the folder that contains your images.

Input options:
- Set `SINGLE_IMAGE_PATH` for one-image mode
- Set `IMAGE_NAME_LIST` to a Python array of image filenames (e.g. `["a.jpg", "b.jpg"]`)
- Leave both unset to use dataset mode from `DATA_PATH`

Set `RETRAIN_FROM_SCRATCH_PER_IMAGE = True` to reset the explainer weights when moving to each new image in `IMAGE_NAME_LIST`.
Set `SAVE_INTERMEDIATE_EXPLANATIONS = False` to save only final explanation + graphs.
## Cell 3 — Discrete Wavelet Transform (via pytorch_wavelets)

Uses `pytorch_wavelets` for GPU-accelerated, autograd-compatible DWT. Splits an image into subbands:
- **LL** — coarse/low-frequency structure (shape)
- **LH** — horizontal edges
- **HL** — vertical edges
- **HH** — fine texture / diagonal edges

Wavelet family and decomposition levels are configurable. The wrapper provides a simple `(x_LL, x_LH, x_HL, x_HH)` interface used by the rest of the pipeline.
## Cell 4 — U-Net Mask Generator

Lightweight encoder-decoder that takes an image and outputs 4 binary masks (one per wavelet subband).

Architecture: `3 → 32 → 64 → 128 → 256 (bottleneck) → 128 → 64 → 32 → 4`

Critical details:
- **STE (Straight-Through Estimator):** Forward pass binarizes at 0.5; backward pass passes gradients through unchanged. This enables gradient-based optimization of discrete masks.
- **Mask downsampling:** Output is full-resolution `(B, 4, H, W)`, then `avg_pool2d` to `(B, 4, H//2, W//2)` after sigmoid but before STE, matching subband spatial dimensions.
- Uses InstanceNorm (not BatchNorm) for stability.
## Cell 5 — Frozen Classifier + Activation Hooks

The classifier is frozen (`requires_grad=False`, **not** `torch.no_grad()`) so gradients still flow *through* it from the loss back to the explanation and then to the U-Net.

Forward hooks on Conv2d, Linear, and ReLU layers capture intermediate activations for the activation matching loss (L_act).
## Cell 6 — Loss Functions

All six loss terms in one place:

| Term | What it does |
|------|-------------|
| **L_act** | MSE (conv/relu) or cosine distance (linear) between activations of x and e |
| **L_CE** | Cross-entropy: preserve the classifier's top-1 prediction |
| **L_KL** | KL divergence: match the full softmax distribution |
| **L_rob** | Robustness: explanation should be stable under wavelet-domain noise |
| **L_area** | Per-subband L1 sparsity (LL < LH=HL < HH penalty weighting) |
| **L_bin** | Push mask values toward {0, 1}: minimizes `m - m^2` |
## Cell 7 — Load Data

Supports four modes:
- **Single image** — set `SINGLE_IMAGE_PATH` above to overfit on one image
- **Image-name list** — set `IMAGE_NAME_LIST` (array of filenames; files are loaded from `DATA_PATH`)
- **Dataset from config** — ImageNet, MNIST, STL-10, or any ImageFolder layout
- **Single image from dataset** — set `single_image_mode: true` in the YAML config
## Cell 8 — Training Step

One gradient update that:
1. U-Net → 4 binary masks
2. Construct explanation `e = IDWT(m * DWT(x))`
3. Forward x and e through frozen classifier, capture activations
4. Compute all loss terms, assemble `L_EXP`, backward, clip, step
## Cell 9 — Training Loop

Runs for the configured number of epochs. In single-image mode each "epoch" is one gradient step on the fixed image. In dataset mode it iterates over the full dataloader.
## Cell 10 — Loss Curves

Plot the training loss history to check convergence.
## Cell 10b — Pareto Sweep: Confidence vs Sparsity

Sweep `lambda_area_pixel` across a log-spaced range and plot the Pareto frontier of confidence (faithfulness) vs sparsity. Set `PARETO_EPOCHS` to an integer before running to use fewer training steps than the full config value.
## Cell 11 — Visualize Explanation + Grad-CAM

In standard mode, this visualizes one sample explanation and a Grad-CAM baseline.
In image-list retrain mode, it also saves final explanation and Grad-CAM images for each processed image into that image's output folder.
## Cell 12 — Subband Attribution Map

Colour-coded visualization showing which frequency content each pixel's explanation came from:
- **Cyan** = LL (coarse structure)
- **Green** = LH (horizontal edges)
- **Red** = HL (vertical edges)
- **Yellow** = HH (fine texture)
## Cell 13 — Evaluation Metrics (Per-Image + GradCAM + Averages)

For each processed image, this cell computes the same comparison metrics for:
- **Your method** (wavelet/pixel explainer output)
- **GradCAM** baseline

Outputs saved by this cell:
- Per-image JSON metrics file in each image output folder
- Per-image bar chart comparing your method vs GradCAM
- Aggregate summary JSON with averages across all images
- Aggregate average bar chart for easy method comparison
## Cell 14 — Save / Load Checkpoint

Save the trained U-Net weights for later use. Only the U-Net is saved (the classifier is frozen and never changes).
## Cell 15 — ADCC Evaluation (Poppi et al. CVPRW 2021)

Implements Average Drop (AD), Complexity (COMP), Coherency (COH), ADCC, and IIC
for EXP-CAM vs Grad-CAM. All metrics computed from scratch with no external XAI libraries.

| Metric | Formula | Better |
|--------|---------|--------|
| AD | mean(max(0, p(y|x) - p(y|e)) / p(y|x)) | ↓ |
| IIC | fraction where p(y|e) > p(y|x) | ↑ |
| COMP | ‖cam‖₁ / (H·W) | ↓ |
| COH | PearsonCorr(cam(x), cam(e)) | ↑ |
| ADCC | HarmonicMean(COH, 1−COMP, 1−AD) | ↑ |

================================================================================
FILE: /home/arin_weling/wavex/notebooks/frequency_ig_explanation.ipynb
# Frequency-Stratified Integrated Gradients

This notebook implements a frequency-aware explainability method by running Integrated Gradients (IG)
with **blurred baselines at increasing σ levels**. By differencing attributions between consecutive
blur levels, we isolate *which spatial frequency bands* the model relied on to make its prediction.

**Pipeline:**
1. Load a pretrained model + image
2. Generate baselines: blurred versions of the image at σ = [0, 2, 4, 8, 16, 32]
3. For each σ, compute IG attribution (integral of gradients from blurred → original)
4. Differential attributions: `ΔAttr(band_i) = Attr(σᵢ) - Attr(σᵢ₊₁)` → captures contribution of the removed frequency band
5. Visualise all bands + aggregate in a single inferno heatmap
## 1 — Configuration
## 2 — Load Model
## 3 — Image Loading & Preprocessing
## 4 — Determine Target Class
## 5 — Blurred Baselines

For each σ we apply a Gaussian blur **in pixel space** (before normalisation) and then renormalise.
σ=0 means no blur → the original image acts as the "sharpest" baseline.
## 6 — Integrated Gradients

For a given baseline `x'` we compute:

$$\text{IG}_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} \, d\alpha$$

Approximated via the **Riemann sum** over `IG_STEPS` steps.
## 7 — Compute IG for Every Sigma Level
## 8 — Differential Frequency Band Attributions

The attribution map at σᵢ captures everything the model uses that is *above* the σᵢ cutoff frequency.

So the **differential**:

$$\Delta\text{Attr}(\text{band}_i) = \text{Attr}(\sigma_i) - \text{Attr}(\sigma_{i+1})$$

isolates the contribution of the frequency content that was present at σᵢ but absent at σᵢ₊₁.
Lower σ bands = high-frequency details; higher σ bands = coarse/low-frequency structure.
## 9 — Visualise: Frequency Band Attributions + Aggregate

Each panel shows **where in the image** the model used information in that frequency band.

The last panel is the **weighted aggregate** — a single importance map across all frequencies.
## 10 — Frequency Importance Bar Chart

A global scalar importance score per band — answers *'which frequency range mattered most overall?'*
---
## Interpretation Guide

| Band σ range | Frequency content | What a high score means |
|---|---|---|
| Low σ (e.g. 0→2) | Very high frequencies — fine edges, grain | Model is **texture-biased** |
| Mid σ (e.g. 5→10) | Mid frequencies — object parts, contours | Model uses **shape features** |
| High σ (e.g. 20→40) | Low frequencies — colour blobs, silhouette | Model relies on **global structure** |

**Tips:**
- If most attribution is in low-σ bands → model is texture-biased (common in standard ImageNet CNNs)
- If attribution spreads into high-σ bands → model is shape/structure-biased (common after Stylized-ImageNet training)
- Bright hotspots in a high-σ panel but not low-σ = the region matters for its **coarse shape**, not fine detail
- You can swap `SIGMA_LEVELS` for finer or coarser frequency analysis

================================================================================
FILE: /home/arin_weling/wavex/notebooks/minimal_frequency_explanation.ipynb
# Fourier Minimal Explanation Pipeline - Self-Contained Notebook

This notebook refactors the explanation pipeline to learn **minimal soft masks in polar Fourier space** instead of wavelet, pixel, or square-grid Fourier masks.

**Pipeline:**
1. Load a YAML config
2. Build FFT/IFFT utilities and a shared-mask autoencoder
3. Train the explainer with fidelity + robustness + soft polar minimality losses
4. Visualize polar frequency masks and reconstructed explanations
5. Compute evaluation metrics (sparsity, label preservation, confidence delta)

**Key idea:** We transform images into Fourier coefficients, group coefficients by radius and angle, and train a shared soft mask that keeps only the minimal polar frequency regions needed to preserve the classifier decision.

## Cell 1 — Imports
## Cell 2 - Load Config

Point `CONFIG_PATH` at your YAML config and set data inputs as before.

Fourier-specific controls in this notebook:
- `FFT_RADIAL_BINS`, `FFT_ANGULAR_BINS`: number of polar frequency bins
- `mask_temperature`: sigmoid temperature for the soft mask
- `mask_threshold`: threshold used only for reporting hard active-bin ratios
- `lambda_bins`: weight for minimal soft active-bin objective

Input modes are unchanged (single image, image list, or dataset mode).

## Cell 3 - Fourier Transform + Polar Frequency Binning

Uses `torch.fft` for autograd-compatible 2D FFT. Coefficients are grouped into polar bins by radius and orientation. Each polar bin receives one shared **soft** mask value.

This gives a direct minimality objective: minimize average soft mask activity while preserving classifier behavior.

## Cell 4 - Shared Soft Polar Fourier Mask Autoencoder

Encoder-decoder network that outputs a **single shared soft polar mask** for Fourier coefficients.

Architecture idea: image -> latent -> mask logits -> global average pooling + linear projection to `(fft_radial_bins, fft_angular_bins)` -> temperature-scaled sigmoid.

This shared soft mask is then expanded to coefficient resolution by polar radius/orientation lookup and applied to all channels.

## Cell 5 — Frozen Classifier + Activation Hooks

The classifier is frozen (`requires_grad=False`, **not** `torch.no_grad()`) so gradients still flow *through* it from the loss back to the explanation and then to the U-Net.

Forward hooks on Conv2d, Linear, and ReLU layers capture intermediate activations for the activation matching loss (L_act).
## Cell 6 - Losses + Fourier Explanation Utilities

Kept losses:
- **L_act** activation matching
- **L_CE** target class preservation
- **L_KL** output distribution alignment
- **L_rob** robustness under frequency perturbations
- **L_energy** explanation Fourier-energy minimization on softly kept coefficients
- **L_entropy** optional entropy penalty that pushes soft mask values away from gray 0.5 values

Changed losses:
- Removed smoothness losses (mask TV and explanation TV)
- Removed hard-mask prior
- Replaced hard active-bin count with **soft polar activity loss** `L_bins`
- Added optional soft-mask entropy loss controlled by `lambda_entropy`

## Cell 7 — Load Data

Supports four modes:
- **Single image** — set `SINGLE_IMAGE_PATH` above to overfit on one image
- **Image-name list** — set `IMAGE_NAME_LIST` (array of filenames; files are loaded from `DATA_PATH`)
- **Dataset from config** — ImageNet, MNIST, STL-10, or any ImageFolder layout
- **Single image from dataset** — set `single_image_mode: true` in the YAML config
## Cell 8 - Training Step

One gradient update that:
1. Autoencoder predicts a shared soft polar Fourier mask
2. Build explanation `e = IFFT(m * FFT(x))`
3. Forward `x` and `e` through the frozen classifier and capture activations
4. Optimize fidelity + robustness + minimal soft-mask objectives

## Cell 9 — Training Loop

Runs epochs over the selected input mode. Supports optional per-image retraining when `IMAGE_NAME_LIST` and `RETRAIN_FROM_SCRATCH_PER_IMAGE` are enabled.
## Cell 10 — Loss Curves

Plot the training loss history to check convergence.
## Cell 11 - Visualize Soft Polar Fourier Explanation

Visualizes:
- Original image
- Shared soft polar mask
- Hard thresholded polar mask for reporting
- FFT log-magnitude before and after masking
- Reconstructed explanation image

In per-image retrain mode, saves one final visualization per image folder.

## Cell 12 - Polar Frequency Attribution Map

Shows where the model keeps frequency information:
- Soft polar mask
- Hard thresholded polar mask
- Expanded soft mask in full Fourier resolution
- Masked FFT magnitude

## Cell 13 - Evaluation Metrics (Per-Image + GradCAM + Averages)

For each processed image, this cell compares:
- **Your soft polar Fourier method**
- **GradCAM baseline**

Shared metrics are preserved (label preservation, confidence delta, insertion/deletion AUC).
Fourier-specific metrics include soft mask mean, hard active-bin ratio, and low/mid/high frequency retention fractions.

## Cell 14 — Save / Load Checkpoint

Save the trained Fourier mask autoencoder for later reuse. The classifier is frozen and not saved.

================================================================================
FILE: /home/arin_weling/wavex/notebooks/wavelet_ig_attribution.ipynb
# Wavelet Integrated Gradients — Texture vs Shape Attribution

This notebook explains **where** in an image a ResNet-18 looks, and **what frequency content** (texture vs shape vs edges) it uses — using Integrated Gradients computed on Wavelet coefficients.

**Output:**
- Individual maps (shape/edge/texture) are shown with their original IG-normalized intensities.
- Combined map uses a single colormap with frequency-aware intensity bands:
  - low frequency (shape) mapped to [0.00, 0.33]
  - mid frequency (edge) mapped to [0.33, 0.66]
  - high frequency (texture) mapped to [0.66, 1.00]
## Cell 1 — Install Dependencies
## Cell 2 — Imports
## Cell 3 — Load Classifier (PCAM or ImageNet)

Two options are provided below. Keep one active and comment out the other.
## Cell 4 — Load and Preprocess Your Image

Set IMAGE_PATH for the selected model. Two example paths are provided below.

IMAGE_SIZE is taken from Cell 3 so the same notebook works for both model options.
## Cell 14 — CAM Baseline (Grad-CAM or Score-CAM)

This cell computes a class activation map on the same image and target class for comparison with IG.

- Set `CAM_METHOD = "gradcam"` for fast gradient-based CAM
- Set `CAM_METHOD = "scorecam"` for forward-pass weighting (slower but gradient-free)

================================================================================
FILE: /home/arin_weling/wavex/notebooks/wavelet_ig_dtcwt.ipynb
# Wavelet Integrated Gradients via Dual-Tree Complex Wavelet Transform

## What this does differently

| Method | Frequency localisation | Spatial localisation | Orientation |
|---|---|---|---|
| Blur IG | Approximate | ✓ | ✗ |
| Spectral IG | Exact (radial rings) | ✗ (global Fourier) | ✗ |
| **Wavelet IG (this)** | **Exact (subbands)** | **✓ per coefficient** | **✓ 6 directions** |

The DT-CWT decomposes an image into `J×6 + 1` spatially-indexed subbands:
- **J levels** of highpass detail (level 1 = finest, level J = coarsest)
- **6 oriented subbands per level** at ±15°, ±45°, ±75° — capturing directional edges
- **1 lowpass residual** — the coarse approximation

For a 224×224 image with J=4: `4×6 + 1 = 25 subbands`.

Each subband's coefficients are **complex** — magnitude gives a shift-invariant response
(the key advantage of DT-CWT over standard DWT).

### Integration path for subband $(j, o)$:

$$x(\alpha) = \mathcal{W}^{-1}\!\Bigl[\underbrace{\mathbf{C} \setminus \mathbf{C}_{j,o}}_{\text{all other subbands, fixed}} + \alpha \cdot \underbrace{\mathbf{C}_{j,o}}_{\text{target subband, scaled}}\Bigr]$$

At α=0: subband $(j,o)$ is absent. At α=1: full image is restored.
Only the target subband's coefficients change along the path — all others are frozen.
Subbands are orthogonal → completeness axiom holds exactly.
## 1 — Configuration
## 2 — Load Model
## 3 — Load Image & Determine Target Class
## 4 — DT-CWT Decomposition

**Shapes after decomposition of a 224×224 image with J=4:**

| Subband | Shape | Spatial res | Frequency |
|---|---|---|---|
| Yh[0] (level 1) | (1,3,6,112,112,2) | 112×112 | Finest detail |
| Yh[1] (level 2) | (1,3,6, 56, 56,2) | 56×56   | Fine |
| Yh[2] (level 3) | (1,3,6, 28, 28,2) | 28×28   | Medium |
| Yh[3] (level 4) | (1,3,6, 14, 14,2) | 14×14   | Coarse |
| Yl (lowpass)    | (1,3,  28, 28)    | 28×28   | DC + very low |

Last dimension of Yh = [real, imaginary] — magnitude = shift-invariant subband energy.
## 5 — Visualise Subband Magnitudes

Each cell shows the **complex magnitude** of a DT-CWT subband, averaged over colour channels.
Bright = high energy at that location, scale, and orientation.
## 6 — Wavelet IG Core Function

For subband $(j, o)$ the integration path is:

$$x(\alpha) = \mathcal{W}^{-1}\bigl[(\mathbf{C} - \mathbf{C}_{j,o}) + \alpha \cdot \mathbf{C}_{j,o}\bigr]$$

All S+1 interpolation steps are batched along the batch dimension for efficiency.
Gradients are computed through the differentiable IDWT (pytorch_wavelets uses fixed-weight convolutions).
## 7 — Run Wavelet IG Over All Subbands

We compute attributions for all `J × 6 + 1 = 25` subbands.
Results are stored in a dict keyed by `(level, orientation)`, where `level=-1` means lowpass.
## 8 — Main Visualisation: (Scale × Orientation) Attribution Grid

Each cell is the spatial attribution heatmap for one subband.
A bright pixel means *that spatial location drove the prediction via that scale and orientation*.

This is the output that no other attribution method can produce.
## 9 — Aggregate Spatial Heatmap
## 10 — Scale Importance & Orientation Selectivity
## 11 — Completeness Verification

Since DT-CWT is a complete basis, the sum of all subband attributions should equal
the standard pixel-space IG attribution (with a black baseline). We verify this here.
Large deviation would indicate a bug in the implementation.
---
## Interpretation Guide

### Reading the Grid (Cell 8)
- **Row** = spatial scale. Top rows = fine detail (hair, texture, grain). Bottom rows = coarse structure (silhouette, colour blobs).
- **Column** = orientation. Each column highlights edges in a specific direction (±15°, ±45°, ±75°).
- **Brightness** = attribution magnitude. A bright pixel at (Level 1, 45°) means *that location's diagonal fine-detail edges drove the prediction*.
- **Empty/dark cells** = the model didn't use that frequency + orientation combination.

### Reading the Bar Charts (Cell 10)

**Scale importance:**
- Level 1–2 dominant → **texture-biased** model (uses fine-grained local patterns)
- Level 3–4 + Lowpass dominant → **shape-biased** model (uses global structure)
- This is a direct, quantitative measure of the texture/shape bias studied in Geirhos et al. 2019

**Orientation selectivity:**
- Uniform → model integrates all edge directions equally (likely shape-based reasoning)
- One dominant orientation → model is keying on a specific structural feature (e.g., horizontal lines in a horizon, vertical edges in a building)
- This is completely novel — no existing attribution method exposes orientation preference

### Why DT-CWT over standard DWT?
- Standard DWT is **shift-sensitive** — a 1px translation can completely change coefficients at fine scales, making attributions jittery and unreliable near edges
- DT-CWT is **approximately shift-invariant** — the complex wavelet magnitude is stable under small translations, giving smooth, reliable spatial attribution maps
- DT-CWT has **6 oriented subbands** vs DWT's 3 — capturing ±15°, ±45°, ±75° rather than just H/V/D, giving finer directional resolution

================================================================================
FILE: /home/arin_weling/wavex/notebooks/frequency_ig_explanation_fourier.ipynb
# Frequency-Stratified Integrated Gradients

This notebook implements a frequency-aware explainability method by running Integrated Gradients (IG)
with **blurred baselines at increasing σ levels**. By differencing attributions between consecutive
blur levels, we isolate *which spatial frequency bands* the model relied on to make its prediction.

**Pipeline:**
1. Load a pretrained model + image
2. Generate baselines: blurred versions of the image at σ = [0, 2, 4, 8, 16, 32]
3. For each σ, compute IG attribution (integral of gradients from blurred → original)
4. Differential attributions: `ΔAttr(band_i) = Attr(σᵢ) - Attr(σᵢ₊₁)` → captures contribution of the removed frequency band
5. Visualise all bands + aggregate in a single inferno heatmap
## 1 — Configuration
## 2 — Load Model
## 3 — Image Loading & Preprocessing
## 4 — Determine Target Class
## 5 — Fourier Low-Pass Baselines

For each cutoff radius `r` we apply a **square low-pass filter in Fourier space** (before normalisation) and then re-normalise.

The filter keeps all FFT coefficients `(u, v)` with `|u| ≤ r` AND `|v| ≤ r` (a square centred on DC) and zeros out the rest.

- `r = 0`   → only the DC term survives → flat mean-colour image (maximally blurred)
- `r = 112` → full spectrum retained → original image unchanged

This is the Fourier equivalent of Gaussian blur: a smaller square = stronger low-pass = blurrier baseline.
## 6 — Integrated Gradients

For a given baseline `x'` we compute:

$$\text{IG}_i(x) = (x_i - x'_i) \times \int_{\alpha=0}^{1} \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} \, d\alpha$$

Approximated via the **Riemann sum** over `IG_STEPS` steps.
## 7 — Compute IG for Every Sigma Level
## 8 — Differential Frequency Band Attributions

The attribution map at cutoff `rᵢ` captures everything the model uses that is *above* the `rᵢ` low-pass cutoff — i.e. spatial frequencies higher than what the `rᵢ`-square baseline retains.

So the **differential**:

$$\Delta\text{Attr}(\text{band}_i) = \text{Attr}(r_i) - \text{Attr}(r_{i+1})$$

isolates the contribution of the frequency content that was present in the baseline at `rᵢ` but absent at `rᵢ₊₁`.

Smaller `r` bands → high-frequency details;  larger `r` bands → coarse / low-frequency structure.

The square low-pass mask (rather than Gaussian blur) gives **hard frequency boundaries** in Fourier space: each band is exactly the annular ring `rᵢ < max(|u|, |v|) ≤ rᵢ₊₁`.
## 9 — Visualise: Frequency Band Attributions + Aggregate

Each panel shows **where in the image** the model used information in that frequency band.

The last panel is the **weighted aggregate** — a single importance map across all frequencies.
## 10 — Frequency Importance Bar Chart

A global scalar importance score per band — answers *'which frequency range mattered most overall?'*
---
## Interpretation Guide
| Band r range | Frequency content | What a high score means |
|---|---|---|
| Low r (e.g. 0→2) | Very high frequencies — fine edges, grain | Model is **texture-biased** |
| Mid r (e.g. 5→10) | Mid frequencies — object parts, contours | Model uses **shape features** |
| High r (e.g. 20→40) | Low frequencies — colour blobs, silhouette | Model relies on **global structure** |
**Tips:**
- If most attribution is in low-r bands → model is texture-biased (common in standard ImageNet CNNs)
- If attribution spreads into high-r bands → model is shape/structure-biased (common after Stylized-ImageNet training)
- Bright hotspots in a high-r panel but not low-r = the region matters for its **coarse shape**, not fine detail
- You can adjust `N_BANDS` for finer or coarser frequency analysis

================================================================================
FILE: /home/arin_weling/wavex/notebooks/minimal_wavelet_explanation.ipynb
# Wavelet Minimal Explanation Pipeline - Self-Contained Notebook

This notebook learns **minimal soft masks in wavelet subband space** instead of Fourier space.
Each subband (LL approximation + 3 detail subbands per level) receives one learned scalar
weight in [0,1]. The explanation is the IDWT reconstruction from the masked subbands.
## Cell 1 — Imports
## Cell 2 - Load Config

Point `CONFIG_PATH` at your YAML config and set data inputs as before.

Wavelet-specific controls:
- `wavelet_levels`: number of DWT decomposition levels
- `wavelet_name`: wavelet family (e.g. `db4`, `haar`, `sym4`)
- `mask_threshold`: sigmoid threshold for binarising the soft mask

## Cell 3 - Wavelet Transform + Subband Masking

Uses `pytorch_wavelets` for autograd-compatible multi-level 2D DWT. The image is
decomposed into `1 + 3*L` subbands: one LL approximation and three detail subbands
(LH, HL, HH) per level. Each subband gets one learned scalar mask weight in [0, 1].
Coarser subbands come first in the mask vector.

## Cell 4 - Shared Soft Wavelet Subband Mask Autoencoder

Multi-scale U-Net producing a *spatial* mask at every wavelet subband resolution. 
It taps into the decoder at every level (112x112, 56x56... 7x7) to generate spatial soft masks that exactly match the downsampled dimensions of each wavelet subband, preserving spatial locality.

## Cell 5 — Frozen Classifier + Activation Hooks

The classifier is frozen (`requires_grad=False`) so gradients still flow *through* it from the loss back to the explanation network.
## Cell 6 - Losses + Wavelet Explanation Utilities

Same loss structure as the Fourier notebook, with wavelet-domain equivalents:
- **L_act** activation matching
- **L_CE** target class preservation
- **L_KL** output distribution alignment
- **L_rob** robustness under masked-out-subband noise
- **L_bins** soft subband activity (sparsity)
- **L_energy** normalized wavelet energy in kept subbands
- **L_entropy** mask entropy (encourages sharp 0/1 decisions)

## Cell 7 — Load Data

Same data-loading logic as the Fourier notebook (unchanged).
## Cell 8 - Training Step

One gradient update:
1. UNet predicts soft wavelet subband mask `(B, 1, n_subbands)`
2. Build explanation `e = IDWT(m * DWT(x))`
3. Forward `x` and `e` through the frozen classifier
4. Compute combined loss and back-propagate into the UNet only

## Cell 9 — Training Loop

Runs epochs over the selected input mode. Supports optional per-image retraining when `IMAGE_NAME_LIST` and `RETRAIN_FROM_SCRATCH_PER_IMAGE` are enabled.
## Cell 10 — Loss Curves

Plot the training loss history to check convergence.
## Cell 11 - Visualize Soft Wavelet Subband Explanation

Shows original, soft/hard masked reconstructions, and per-subband mask bar charts.
## Cell 12 - Wavelet Subband Attribution Map

Shows:
- Soft and hard subband mask values (bar charts, coarsest to finest)
- Soft and hard masked reconstruction images
- Level-1 (finest) wavelet detail subbands: LH, HL, HH

## Cell 13 - Evaluation Metrics (Per-Image + GradCAM + Averages)

Compares the wavelet subband method against GradCAM baseline.
Wavelet-specific metrics: per-level mask weight fractions (approx_fraction, level1..levelL).

## Cell 14 — Save / Load Checkpoint

Save the trained wavelet subband mask autoencoder for later reuse.
