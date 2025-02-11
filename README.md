# DiTTo-TTS: Diffusion Transformers for Scalable Text-To-Speech without domain-specific factors

**Implemented by:** Abdelkrim Halimi, Walid Ghenait, Melissa Dahlia Attabi  
*Original paper by:* Keon Lee, Dong Won Kim, Jaehyeon Kim, Seungjun Chung, Jaewoong Cho

## Introduction
Latent Diffusion Models (LDMs) have shown high performance in various tasks such as image, audio, and video generation. When applied to Text-To-Speech (TTS), these models require domain-specific factors to ensure proper temporal alignment between text and speech. However, this dependency complicates data preparation and limits model scalability.

DiTTo-TTS introduces a novel approach to overcome these limitations while achieving high performance. This method is based on a Diffusion Transformer (DiT) architecture and integrates a speech length predictor.

---

## Model Components

### Speech Length Predictor
- Predicts the total length of the audio signal for a given text.
- The encoder transforms the text bidirectionally.
- The decoder takes encoded audio (NAC) as input and applies a causal mask.
- Cross-attention between encoded text and audio enables length prediction.
- Trained separately using cross-entropy loss.

<img src="https://github.com/user-attachments/assets/29878190-513b-4c80-928b-0b2f527ee2aa" alt="Speech Length Predictor" style="float: right; width: 300px; margin: 0 0 10px 10px;" />

### Neural Audio Codec (NAC)
Encodes audio signals into latent representations aligned with the text, quantizes them, and then decodes them.

**Components:**
- Encoder
- Vector Quantizer
- Decoder
- Language Model

<img src="https://github.com/user-attachments/assets/56c86287-6b97-41cb-84c7-666ac7c427db" alt="Neural Audio Codec" style="float: right; width: 300px; margin: 0 0 10px 10px;" />

**Loss Function:**
\[
L(\psi) = L_{NAC}(\psi) + \lambda L_{LM}(\psi), \quad L_{LM}(\psi) = -\log p_{\phi}(x|f(z_{speech}))
\]

### Diffusion Model
- Generates speech from textual representations \( z_{text} \) and audio \( z_{speech} \) using a diffusion process.

**Loss Function:**
\[
L_{\text{diffusion}} = \mathbb{E}_{t \sim \mathcal{U}(1,T), \epsilon \sim \mathcal{N}(0,I)} \left[ \| v^{(t)} - v_{\theta}(z^{(t)}, x, t) \|^2 \right]
\]

<img src="https://github.com/user-attachments/assets/48d2a465-bc5c-4148-ab3a-ff59b7343693" alt="Diffusion Model" style="float: right; width: 300px; margin: 0 0 10px 10px;" />

### Audio-Text Pipeline
- **Dataset:** MLS Librispeech – 10,000 selected French audio samples with text transcriptions.
- **Preprocessing:**
  - **Text:** Tokenization (GPT2, ByT5).
  - **Audio:** Resampled to 24kHz.
- **Audio Signal Reconstruction:** BigVGAN.

<img src="https://github.com/user-attachments/assets/377e3dde-0e1d-42ca-a7e5-d1a1dae3ad6c" alt="Audio-Text Pipeline" style="float: right; width: 300px; margin: 0 0 10px 10px;" />

---

For more details, refer to the original research or documentation.
