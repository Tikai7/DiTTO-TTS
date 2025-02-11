# DiTTo-TTS: Diffusion Transformers for Scalable Text-To-Speech without domain-specific factors

**Implemented by:** Abdelkrim Halimi, Walid Ghenait, Melissa Dahlia Attabi  
*Original paper by:* Keon Lee, Dong Won Kim, Jaehyeon Kim, Seungjun Chung, Jaewoong Cho

## Introduction
Latent Diffusion Models (LDMs) have shown high performance in various tasks such as image, audio, and video generation. When applied to Text-To-Speech (TTS), these models require domain-specific factors to ensure proper temporal alignment between text and speech. However, this dependency complique la préparation des données et limite l'évolutivité du modèle.

DiTTo-TTS propose une approche novatrice pour contourner ces limitations tout en obtenant de hautes performances. Cette méthode repose sur une architecture Diffusion Transformer (DiT) et intègre un prédicteur de longueur de parole.

---

## Model Components

### Speech Length Predictor

<div style="display: flex; align-items: flex-start;">
  <div style="flex: 1; padding-right: 10px;">
    - Prédit la longueur totale du signal audio pour un texte donné.<br>
    - L'encodeur transforme le texte de manière bidirectionnelle.<br>
    - Le décodeur prend en entrée l'audio encodé (NAC) et applique un masque causal.<br>
    - L'attention croisée entre le texte encodé et l'audio permet la prédiction de la longueur.<br>
    - Entraîné séparément à l'aide d'une fonction de perte par entropie croisée.
  </div>
  <div style="width: 300px;">
    <img src="https://github.com/user-attachments/assets/29878190-513b-4c80-928b-0b2f527ee2aa" alt="Speech Length Predictor" style="width:100%;" />
  </div>
</div>

### Neural Audio Codec (NAC)

<div style="display: flex; align-items: flex-start;">
  <div style="flex: 1; padding-right: 10px;">
    Encode les signaux audio en représentations latentes alignées avec le texte, les quantifie, puis les décode.<br><br>
    **Components :**
    - Encoder<br>
    - Vector Quantizer<br>
    - Decoder<br>
    - Language Model<br><br>
    **Loss Function :**
    
    \[
    \mathcal{L}(\psi) = \mathcal{L}_{\text{NAC}}(\psi) + \lambda\, \mathcal{L}_{\text{LM}}(\psi)
    \]
    
    \[
    \mathcal{L}_{\text{LM}}(\psi) = -\log p_{\phi}\Bigl(x \,\Bigl|\, f\bigl(z_{\text{speech}}\bigr)\Bigr)
    \]
  </div>
  <div style="width: 300px;">
    <img src="https://github.com/user-attachments/assets/56c86287-6b97-41cb-84c7-666ac7c427db" alt="Neural Audio Codec" style="width:100%;" />
  </div>
</div>

### Diffusion Model

<div style="display: flex; align-items: flex-start;">
  <div style="flex: 1; padding-right: 10px;">
    Génère la parole à partir des représentations textuelles \( z_{\text{text}} \) et audio \( z_{\text{speech}} \) via un processus de diffusion.<br><br>
    **Loss Function :**
    
    \[
    \mathcal{L}_{\text{diffusion}} = \mathbb{E}_{t \sim \mathcal{U}(1,T),\, \epsilon \sim \mathcal{N}(0,I)} \!\left[ \left\| v^{(t)} - v_{\theta}\Bigl(z^{(t)}, x, t\Bigr) \right\|^2 \right]
    \]
  </div>
  <div style="width: 300px;">
    <img src="https://github.com/user-attachments/assets/48d2a465-bc5c-4148-ab3a-ff59b7343693" alt="Diffusion Model" style="width:100%;" />
  </div>
</div>

### Audio-Text Pipeline

<div style="display: flex; align-items: flex-start;">
  <div style="flex: 1; padding-right: 10px;">
    - **Dataset :** MLS Librispeech – 10 000 échantillons audio français sélectionnés avec leurs transcriptions.<br>
    - **Preprocessing :**<br>
      &nbsp;&nbsp;&nbsp;&nbsp;**Text :** Tokenization (GPT2, ByT5).<br>
      &nbsp;&nbsp;&nbsp;&nbsp;**Audio :** Remappé à 24 kHz.<br>
    - **Audio Signal Reconstruction :** BigVGAN.
  </div>
  <div style="width: 300px;">
    <img src="https://github.com/user-attachments/assets/377e3dde-0e1d-42ca-a7e5-d1a1dae3ad6c" alt="Audio-Text Pipeline" style="width:100%;" />
  </div>
</div>

---

For more details, refer to the original research or documentation.
