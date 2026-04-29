# Differentially Private Time-Series GAN
## for Privacy-Preserving Medical Data Generation

---

## Slide 1: The Privacy Dilemma in Healthcare
- **Value**: Medical time-series data (ECG, vitals) is crucial for predictive modeling.
- **Problem**: Sharing this data violates patient privacy (HIPAA/GDPR).
- **Current Solutions**: Anonymization fails for high-dimensional time-series (linkage attacks).
- **Our Goal**: Generate synthetic functional data that protects individual privacy with mathematical guarantees.

---

## Slide 2: Generative Adversarial Networks (GANs)
- **Concept**: A Generator creates fake data; a Discriminator tries to distinguish fake from real.
- **TimeGAN**: Specialized architecture preserving spatial and temporal dynamics via Embedding and Supervisor networks.
- **Vulnerability**: Standard TimeGANs can "memorize" and leak exact patient sequences.

---

## Slide 3: Differential Privacy (DP) Framework
- **Definition**: An algorithm is DP if removing a single patient’s data doesn't significantly change the output distribution.
- **DP-SGD**: We inject calibrated Gaussian noise into the gradients during training.
- **Privacy Budget ($\epsilon, \delta$)**: $\epsilon$ measures privacy loss. Lower $\epsilon$ = higher privacy, but lower data utility.

---

## Slide 4: Proposed Architecture: DP-TimeGAN
1. **Data Ingestion**: Sequence rolling of Vitals (HR, BP, SpO2).
2. **Autoencoder Phase**: Maps complex sequences to simpler latent space.
3. **Differentially Private Discriminator**: Trained using Opacus with DP-SGD.
4. **Generator**: Inherits privacy guarantee via the post-processing property of DP.

---

## Slide 5: Evaluation & Utility Metrics
- **TSTR (Train on Synthetic, Test on Real)**: Evaluates if models trained on synthetic data perform well in the real world.
- **Similarity**: Measured via RMSE and Wasserstein distance.
- **Results**: DP-TimeGAN ($\epsilon=1.0$) shows a minor reduction in utility while providing robust protection against membership inference attacks.

---

## Slide 6: Interactive Dashboard & Codebase
- **Streamlit App**: Real-time logging of privacy budget vs. fidelity trade-offs.
- **Modularity**: Codebase supports standard LSTMs, TimeGAN, and DP-TimeGAN natively.
- **Future Work**: Scaling to Federated Hospital settings.
