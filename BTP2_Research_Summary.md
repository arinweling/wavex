# WaveX Project: Comprehensive Architecture & Research Analysis

## 1. Executive Summary

The WaveX repository focuses on analyzing and explaining deep learning image classifiers not just spatially (where the model looks), but in the **frequency domain** (what kind of structural information it uses: shape, texture, edges). 

The repository pursues two distinct paradigms for frequency-aware explainability:
1. **Generative Masking / Optimization (The "Minimal Explanation" approach):** Training an explainer (like a U-Net or an Autoencoder) to generate sparse masks in a transformed frequency domain (Haar Wavelet, Polar Fourier, Subband Wavelet) that retain only the minimal necessary coefficients to preserve the classifier's original prediction.
2. **Attribution via Integrated Gradients (The "IG" approach):** Extending standard spatial Integrated Gradients (IG) into frequency spaces (Wavelets, DT-CWT, blurred Fourier rings) to attribute predictions directly to specific frequency bands or orientations.

---

## 2. Generative Masking Explanations

### `wavelet_explanation_pipeline.ipynb`
**Research Objective:** 
Distill the entire WaveX U-Net training pipeline into a reproducible, self-contained notebook. It explains model decisions by finding a minimal subset of Haar wavelet coefficients that preserve the prediction.

**Architecture Details:**
- **Transform:** Discrete Wavelet Transform (DWT) via `pytorch_wavelets` using Haar wavelets. Decomposes image into LL (shape), LH (horizontal edges), HL (vertical edges), and HH (texture).
- **Explainer:** A U-Net encoder-decoder architecture (`3 → 32 → 64 → 128 → 256 → 128 → 64 → 32 → 4`). 
- **Masking Mechanism:** 
  - Generates 4 spatial masks at full resolution.
  - Applies `avg_pool2d` to match the downsampled subband resolution.
  - Uses a **Straight-Through Estimator (STE)**: masks are binarized at a 0.5 threshold during the forward pass, but gradients pass through unaltered during backpropagation.
- **Evaluation / Losses:** 
  - **Frozen Classifier:** Classifier weights are frozen (`requires_grad=False`, not `no_grad()`) so gradients flow through it back to the U-Net.
  - **Losses:** $L_{act}$ (activation matching), $L_{CE}$ (top-1 label preservation), $L_{KL}$ (distribution matching), $L_{rob}$ (robustness under wavelet noise), $L_{area}$ (L1 sparsity heavily penalized on HH texture), and $L_{bin}$ (forces masks to {0,1}).

**Critical Takeaway:** This is the flagship approach. The use of STE is clever for learning discrete masks, and the weighted area loss acts as an interpretability prior, coercing the model to prefer dropping high-frequency noise (HH) over fundamental structure (LL).

---

### `minimal_frequency_explanation.ipynb`
**Research Objective:**
Shift from the discrete, spatial-wavelet domain to **Polar Fourier Space**. It seeks to learn a single shared soft mask for polar frequency bins.

**Architecture Details:**
- **Transform:** 2D Fast Fourier Transform (`torch.fft`). The 2D frequency plane is binned radially and angularly.
- **Explainer:** An Autoencoder that maps the image into a latent space and uses a global average pool + linear projection to output a shared polar mask of size `(fft_radial_bins, fft_angular_bins)`.
- **Masking Mechanism:** 
  - Produces continuous (soft) masks, avoiding the need for STE. 
  - The compact polar mask is expanded to the full resolution of the Fourier coefficients via radial/orientation lookup, and applied uniformly across all color channels.
- **Losses:** Replaces TV smoothness and hard-mask priors with $L_{bins}$ (minimizing soft polar activity), $L_{energy}$ (minimizing preserved Fourier energy), and $L_{entropy}$ (pushing soft values away from 0.5).

**Critical Takeaway:** By operating in polar Fourier space, this approach explicitly tackles rotation and scale invariance in explanations. However, Fourier transforms lack spatial localization (a global change affects the whole image), which makes this approach excellent for identifying "what" frequency matters globally, but poor for identifying "where" in the image it occurs.

---

### `minimal_wavelet_explanation.ipynb`
**Research Objective:**
A compromise between spatial masking and global frequency masking. It learns a minimal soft mask per wavelet subband, not per spatial coefficient.

**Architecture Details:**
- **Transform:** Multi-level DWT (e.g., `db4`, `sym4`). Yields $1 + 3L$ subbands (1 LL approx, plus LH/HL/HH for each level $L$).
- **Explainer:** A multi-scale U-Net that taps into its decoder at every level to output *spatial* masks that perfectly match the downsampled resolution of each wavelet subband (e.g., 112x112, 56x56, down to 7x7).
- **Masking Mechanism:** Soft masking. Unlike single scalar weights, it applies full spatial soft masks to every subband, preserving the spatial locality inherent to the wavelet transform.

**Critical Takeaway:** This approach elegantly combines the spatial mapping capabilities of a U-Net with the multi-resolution decomposition of wavelets. By outputting a 112x112 spatial mask for Level 1 details, and a 7x7 spatial mask for the coarse LL shape, it learns exactly *where* and at *what frequency* the classifier is looking.

---

## 3. Integrated Gradients (IG) Attribution Models

While generative masks optimize for a "sufficient subset" of information, the Integrated Gradients notebooks attempt to mathematically decompose the existing gradients of a frozen classifier into frequency bands.

### `wavelet_ig_attribution.ipynb`
**Research Objective:**
Determine *where* a model looks and *what* it is looking for (Texture vs. Shape) by interpolating IG in wavelet coefficient space.

**Architecture Details:**
- **Methodology:** Instead of fading pixel intensities from black/white to the image, this interpolates wavelet coefficients from a zeroed baseline to the original values.
- **Completeness Axiom:** It accurately reconstructs the prediction change ($f(image) - f(baseline) = \sum IG$) because IDWT is a linear mapping.
- **Outputs:** Merges frequency levels into standard color bands: LL (Shape), LH+HL (Edges), HH (Texture). 

**Critical Takeaway:** Standard IG suffers from noise and lack of semantic meaning. Wavelet IG elegantly solves this by decoupling high-frequency gradient noise from actual structural attributions.

---

### `wavelet_ig_dtcwt.ipynb`
**Research Objective:**
Solve the limitations of standard Wavelet IG (which is shift-sensitive and jittery) by using the **Dual-Tree Complex Wavelet Transform (DT-CWT)**.

**Architecture Details:**
- **Transform:** DT-CWT. Decomposes image into $J \times 6 + 1$ subbands. It provides 6 orientations ($\pm 15^\circ, \pm 45^\circ, \pm 75^\circ$) and uses complex coefficients whose magnitude is approximately shift-invariant.
- **Integration Path:** 
  $x(\alpha) = \mathcal{W}^{-1} [(\mathbf{C} \setminus \mathbf{C}_{j,o}) + \alpha \cdot \mathbf{C}_{j,o}]$
  It computes IG for *one subband at a time* while keeping all other subbands frozen at their actual values. 
- **Outputs:** Generates a granular grid of attributions (Scale $\times$ Orientation). 

**Critical Takeaway:** This is arguably the most mathematically sophisticated notebook in the repository. Standard DWT artifacts (aliasing and shift variance) ruin spatial attribution near edges. DT-CWT provides smooth, shift-invariant maps and reveals the specific edge angles a network prefers—a feature entirely absent in spatial IG.

---

### `frequency_ig_explanation.ipynb` & `frequency_ig_explanation_fourier.ipynb`
**Research Objective:**
Calculate Differential Frequency Attributions using blurred baselines. 

**Architecture Details:**
- **`frequency_ig_explanation.ipynb`:** Uses standard spatial Gaussian blur with increasing $\sigma$ values to create baselines.
- **`frequency_ig_explanation_fourier.ipynb`:** Uses a hard square low-pass filter in Fourier space (annular rings).
- **Mechanism:** Computes IG for each cutoff level. The contribution of a specific frequency band is the difference between two consecutive IG runs: $\Delta Attr = Attr(\sigma_i) - Attr(\sigma_{i+1})$. 

**Critical Takeaway:** These notebooks act as a diagnostic tool for "Texture vs. Shape bias" (e.g., Geirhos et al. 2019). By looking at the bar charts of $\Delta Attr$, you can instantly quantify if a model is texture-biased (heavy reliance on high frequencies) or shape-biased. 

---

## 4. Synthesis & Critical Review

1. **Axiomatic Soundness:** The IG-based approaches in the repo are highly robust because linear transforms (FFT, DWT, DT-CWT) preserve the IG Completeness Axiom. The gradients accumulated in the transformed domains sum up perfectly to the change in the classifier's output. 
2. **Spatial vs. Frequency Tradeoff:**
   - *Fourier methods* provide perfect frequency isolation but obliterate spatial localization.
   - *Spatial methods (Pixel IG, GradCAM)* provide spatial localization but conflate textures and shapes.
   - *Wavelet methods (especially DT-CWT)* offer the optimal middle ground, maintaining both spatial and frequency/orientation resolution.
3. **Generative vs Attribution:** The U-Net/Autoencoder notebooks are practical for creating masked, human-viewable images that "fool" or "satisfy" the classifier. The IG notebooks are analytical, explaining the exact gradients the frozen network *already* relies on. 

Overall, the repository provides an exhaustive, multi-angled assault on frequency-aware deep learning interpretability, transitioning from basic spatial explanations to highly advanced shift-invariant complex wavelets.
