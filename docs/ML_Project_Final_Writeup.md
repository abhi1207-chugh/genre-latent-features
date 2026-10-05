# Latent Feature Extraction for Musical Genres from Raw Audio

## 1. Project Title

**Latent Feature Extraction for Musical Genres from Raw Audio Using a CNN-Based Autoencoder**

---

## 2. Problem Statement

The goal of this project is to learn a compact latent representation of musical audio directly from raw audio signals, without relying on manually engineered audio features such as MFCCs or Mel-spectrograms.

The learned representation should capture information useful for distinguishing musical genres.

The project is based on the research paper **“Latent Feature Extraction for Musical Genres from Raw Audio”**, which proposes a Deep Softmax Autoencoder that jointly learns an audio representation and genre classification.

For this project, the original approach is extended with a lightweight **1D CNN encoder, center loss, improved data splitting, and systematic evaluation**.

---

## 3. Dataset

### GTZAN Genre Collection Dataset

The GTZAN dataset contains approximately 1,000 audio tracks belonging to 10 musical genres, with each track approximately 30 seconds long.

For this project, four genres are selected:

1. **Classical**
2. **Country**
3. **Disco**
4. **Hip-Hop**

The dataset is obtained from Kaggle.

The four selected genres differ from the original paper, which used Classical, Jazz, Metal, and Pop. Therefore, results from this project should not be directly compared numerically with the original paper's results.

---

## 4. Audio Preprocessing

The project intentionally works with **raw audio** rather than manually engineered features.

### Preprocessing Pipeline

```text
Audio Track
     ↓
Track-level dataset split
     ↓
1-second audio clips
     ↓
Raw amplitude waveform
     ↓
Average pooling
     ↓
500-dimensional representation
     ↓
Standardization
     ↓
Model
```

Each track is loaded as a raw amplitude waveform and divided into **1-second clips**.

Average pooling is then used to reduce each clip to a **500-dimensional vector**, following the basic representation used by the original paper.

No MFCC, Mel-spectrogram, chroma features, or other manually engineered audio features are used in the main model.

The resulting inputs are standardized before training.

---

## 5. Data Splitting and Leakage Prevention

The dataset is divided at the **track level before generating clips**.

All clips originating from the same song belong to the same split.

```text
Track A
 ├── Clip 1
 ├── Clip 2
 ├── Clip 3
 └── ...

Track B
 ├── Clip 1
 ├── Clip 2
 └── ...
```

Therefore:

```text
TRAIN TRACKS
      ↓
TRAIN CLIPS

VALIDATION TRACKS
      ↓
VALIDATION CLIPS

TEST TRACKS
      ↓
TEST CLIPS
```

This prevents clips originating from the same song from appearing in both training and testing data.

The split is stratified by genre so that the four genres remain reasonably balanced.

For final evaluation, **5-fold StratifiedGroupKFold** is used with `track_id` as the grouping variable.

---

## 6. Proposed Architecture

The final model consists of four major components:

```text
                     Raw Audio
                         ↓
                     500-D Input
                         ↓
                  1D CNN Encoder
                         ↓
                   64-D Latent
                    Embedding
                         ↓
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
       Decoder       Classifier     Center Loss
          ↓              ↓
       500-D          4 Genres
   Reconstruction
```

### 6.1 1D CNN Encoder

The model uses a lightweight **1D Convolutional Neural Network encoder**.

The encoder receives the 500-dimensional input and progressively transforms it into a compact **64-dimensional latent representation**.

The 64-dimensional output of the encoder is the primary **genre embedding**.

### 6.2 Latent Representation

The encoder produces:

```text
500-D input
      ↓
CNN Encoder
      ↓
64-D latent vector
```

The 64-dimensional vector represents the learned latent features of the audio clip.

The objective is to learn a representation that contains useful genre-related information.

### 6.3 Decoder

The decoder receives the 64-dimensional latent representation and attempts to reconstruct the original 500-dimensional input.

```text
64-D embedding
      ↓
Decoder
      ↓
500-D reconstructed input
```

The reconstruction objective helps preserve useful information in the latent representation.

### 6.4 Genre Classifier

A classification head is connected to the 64-dimensional latent representation.

```text
64-D embedding
      ↓
Fully Connected Layers
      ↓
4-class Softmax
      ↓
Classical
Country
Disco
Hip-Hop
```

The classifier is trained using **cross-entropy loss**.

---

## 7. Center Loss

The model uses **center loss** as an additional objective.

Each genre has a learnable center in the 64-dimensional latent space:

```text
Classical → Center C₁
Country   → Center C₂
Disco     → Center C₃
Hip-Hop   → Center C₄
```

For each training sample, center loss encourages its embedding to remain close to the center of its corresponding genre.

Conceptually:

```text
Before center loss:

Classical:  •  •      •
                •
Country:          • •
              •


After center loss:

Classical:   •••
             •C₁•

Country:     •••
             •C₂•
```

The purpose is to encourage more compact intra-class clusters in the latent space.

---

## 8. Overall Loss Function

The final model uses three objectives:

### Reconstruction Loss

Measures how accurately the decoder reconstructs the original 500-dimensional input.

### Classification Loss

Cross-entropy measures the accuracy of genre prediction.

### Center Loss

Encourages embeddings belonging to the same genre to cluster around their corresponding class center.

The complete objective is:

\[
L =
\gamma L_{reconstruction}
+
(1-\gamma)L_{classification}
+
\lambda L_{center}
\]

where:

- \(L_{reconstruction}\) = reconstruction loss
- \(L_{classification}\) = cross-entropy loss
- \(L_{center}\) = center loss
- \(\gamma\) = reconstruction/classification trade-off
- \(\lambda\) = center-loss weight

---

## 9. Vanilla Autoencoder Baseline

A **plain autoencoder** is trained as a baseline.

```text
500-D input
     ↓
Encoder
     ↓
64-D embedding
     ↓
Decoder
     ↓
500-D reconstruction
```

It does not use genre labels, classification loss, or center loss.

Its purposes are:

1. To provide a reconstruction baseline.
2. To provide an unsupervised latent representation for comparison.
3. To provide a reference reconstruction error for model selection.

---

## 10. Model Selection

The complete model is trained with different combinations of the main hyperparameters.

### Reconstruction/Classificaton Weight

\[
\gamma \in \{0.5, 0.9, 0.99\}
\]

### Center-Loss Weight

\[
\lambda \in \{0.001, 0.01, 0.1\}
\]

### Center Learning Rate

\[
\alpha \in \{0.01, 0.1, 0.5\}
\]

This produces:

\[
3 \times 3 \times 3 = 27
\]

experiments.

### Model Selection Constraint

The validation reconstruction error of the selected model must satisfy:

\[
Validation\ Reconstruction\ Error
\leq
1.5\times
Plain\ AE\ Reconstruction\ Error
\]

Among the models satisfying this constraint, the model with the highest **5-NN validation accuracy on frozen 64-dimensional embeddings** is selected.

If two models are within approximately 1 percentage point of each other, the model with lower reconstruction error is preferred.

---

## 11. Training Configuration

Initial training configuration:

- Framework: **PyTorch**
- Optimizer: **Adam**
- Learning rate: **1e-4**
- Batch size: **512**, subject to available memory
- Latent dimension: **64**
- Dropout: approximately **0.1**, if used
- Early stopping based on validation performance/loss
- Reproducible random seeds

---

## 12. Baseline Models

To determine whether the learned deep representation provides meaningful benefits, it is compared with simpler approaches.

### Baseline 1: Raw 500-D Features

Use the standardized 500-dimensional input directly with:

- kNN
- SVM

### Baseline 2: Dimensionality Reduction

Apply:

- PCA
- optionally UMAP

followed by:

- kNN
- SVM

### Baseline 3: Vanilla Autoencoder

Use the 64-dimensional embedding learned by the plain autoencoder.

### Proposed Model

Use the 64-dimensional embedding learned by:

```text
1D CNN Encoder
+
Reconstruction Loss
+
Classification Loss
+
Center Loss
```

---

## 13. Embedding Evaluation

The quality of the latent representation is evaluated independently from the classifier.

### 13.1 5-NN Probe

The learned 64-dimensional embeddings are frozen and evaluated using a 5-NN classifier.

### 13.2 Linear Probe

A simple linear classifier is also trained on the frozen embeddings.

This provides an additional evaluation of whether genre information is present in the latent representation.

### 13.3 Silhouette Score

The silhouette score is used as a secondary measure of clustering quality.

### 13.4 Latent-Space Visualization

The latent space is visualized using methods such as:

- PCA
- t-SNE
- optionally UMAP

Points are colored according to genre.

---

## 14. Track-Level Evaluation

The model initially predicts the genre of individual 1-second clips.

Track-level prediction is also performed.

For a song containing multiple clips:

```text
Clip 1 → Genre probabilities
Clip 2 → Genre probabilities
Clip 3 → Genre probabilities
          ↓
Average probability vectors
          ↓
     Final genre
```

The final genre is the class with the highest average probability.

Both **clip-level and track-level** metrics are reported.

---

## 15. Evaluation Metrics

The following metrics are reported:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Metrics are reported at:

1. Clip level
2. Track level

For final evaluation, 5-fold **StratifiedGroupKFold** is used with track ID as the grouping variable.

Results are reported as:

\[
Mean \pm Standard\ Deviation
\]

where appropriate.

---

## 16. Ablation Study

An ablation study evaluates the contribution of center loss.

### Model A — Without Center Loss

\[
L =
\gamma L_{reconstruction}
+
(1-\gamma)L_{CE}
\]

### Model B — Full Model

\[
L =
\gamma L_{reconstruction}
+
(1-\gamma)L_{CE}
+
\lambda L_{center}
\]

The comparison includes:

- Classification accuracy
- F1-score
- 5-NN embedding accuracy
- Linear probe accuracy
- Silhouette score
- Reconstruction error
- Latent-space visualization

Claims about improvement from center loss are made only if supported by the experimental results.

---

## 17. Hyperparameter Experiment Visualization

The 27 experiments are visualized using a plot of:

```text
Y-axis → 5-NN Validation Accuracy
X-axis → Validation Reconstruction Error
```

The reconstruction constraint:

\[
1.5\times Plain\ AE\ Error
\]

is shown on the plot.

This illustrates the trade-off between retaining information and learning useful genre separation.

---

## 18. Complete Experimental Pipeline

```text
                    GTZAN Dataset
                         ↓
               Select 4 Genres
                         ↓
        Classical / Country / Disco / Hip-Hop
                         ↓
                Track-Level Split
                         ↓
                  1-sec Clips
                         ↓
                 Raw Waveform
                         ↓
               Average Pooling
                         ↓
                     500-D
                         ↓
                  Standardization
                         ↓
             ┌─────────────────────┐
             │                     │
             ↓                     ↓
       Vanilla AE            Proposed Model
                                  ↓
                             1D CNN Encoder
                                  ↓
                              64-D Embedding
                                  ↓
                ┌─────────────────┼─────────────────┐
                ↓                 ↓                 ↓
             Decoder          Classifier       Center Loss
                ↓                 ↓
             500-D            4 Genres
          Reconstruction
```

---

## 19. Experiment Workflow

### Phase 1 — Dataset and Preprocessing

- Load GTZAN
- Select four genres
- Create track-level split
- Generate 1-second clips
- Apply average pooling
- Generate 500-dimensional inputs
- Standardize data

### Phase 2 — Baselines

Implement:

- Raw 500-D + kNN
- Raw 500-D + SVM
- PCA + classifier
- Vanilla autoencoder

### Phase 3 — Proposed Model

Implement:

- 1D CNN encoder
- 64-dimensional latent representation
- Decoder
- Genre classifier
- Center loss
- Combined loss

### Phase 4 — Hyperparameter Search

Run all 27 combinations of:

- \(\gamma\)
- \(\lambda\)
- \(\alpha\)

### Phase 5 — Model Selection

Apply the reconstruction constraint and select the best model according to validation 5-NN accuracy.

### Phase 6 — Ablation

Compare:

```text
Reconstruction + Cross-Entropy
vs.
Reconstruction + Cross-Entropy + Center Loss
```

### Phase 7 — Final Evaluation

Run:

- 5-fold StratifiedGroupKFold
- Clip-level metrics
- Track-level metrics
- Embedding evaluation
- Confusion matrices
- Latent-space visualizations

---

## 20. Expected Outputs

### Tables

1. Dataset distribution
2. Baseline comparison
3. 27 hyperparameter experiments
4. Final model metrics
5. Ablation results
6. Cross-validation results

### Figures

1. Model architecture
2. Training/validation loss curves
3. Reconstruction examples
4. Hyperparameter Pareto plot
5. Confusion matrix
6. Latent-space visualization
7. Baseline vs proposed model comparison

---

## 21. Limitations

### Limited Number of Genres

Only four genres are used, so the conclusions may not generalize to all musical genres.

### Raw Waveform Compression

Average pooling significantly reduces the temporal resolution of the original waveform. Important high-frequency information may therefore be lost.

### GTZAN Limitations

GTZAN may contain repeated artists, duplicate tracks, and labeling issues.

A track-level split prevents direct clip leakage but does not completely eliminate artist overlap.

### Dataset Size

The number of independent tracks is relatively small, making generalization an important concern.

### Different Experimental Setup From the Original Paper

The project uses a different genre subset and a different splitting strategy from the original paper.

Therefore, numerical results should not be presented as directly comparable to the original paper.

### Center-Loss Contribution

Center loss should not be claimed to improve performance without experimental evidence from the ablation study.

---

## 22. Final Research Question

> **Can a CNN-based autoencoder trained directly on raw audio learn a compact 64-dimensional latent representation that captures useful musical genre information, and does adding center loss improve the organization and usefulness of this latent space?**

---

## 23. Key Difference From the Original Paper

The original paper uses a **Deep Softmax Autoencoder** to jointly learn genre embeddings and classification from raw audio. Its final model uses a 64-dimensional encoding and combines autoencoder reconstruction with genre classification.

The present project retains the core idea while introducing controlled modifications:

| Original Paper | Present Project |
|---|---|
| Classical, Jazz, Metal, Pop | **Classical, Country, Disco, Hip-Hop** |
| Clip-level splitting | **Track-level splitting** |
| Deep autoencoder | **Lightweight 1D CNN encoder + decoder** |
| 64-D embedding | **64-D embedding** |
| Reconstruction + classification | **Reconstruction + classification + center loss** |
| Limited evaluation | **Baselines + probes + clustering + track-level metrics** |
| No center-loss ablation | **Center-loss ablation** |
| Original experimental setup | **Systematic 27-run hyperparameter search** |

The project therefore extends the core idea of the original work while keeping the problem manageable and focused on learning meaningful latent representations from raw audio.
