import torch

class BaseConfig:
    """
        Base configuration class for shared settings.
    """
    # Audio settings
    SAMPLE_RATE = 24000  # Hz
    MIN_AUDIO_DURATION = 10  # seconds
    MAX_AUDIO_DURATION = 20  # seconds
    
    # Training settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    BETAS = [0.9, 0.999]

    # Data settings
    TRAIN_PATH = "/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/data/mls_french_opus/train"
    TEST_PATH = "/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/data/mls_french_opus/test"
    DEV_PATH = "/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/data/mls_french_opus/dev"

    # TRAIN_PATH = "C:/Cours-Sorbonne/M2/UE_DEEP/AMAL/Projet/data/mls_french_opus/mls_french_opus/train"
    # TEST_PATH = "C:/Cours-Sorbonne/M2/UE_DEEP/AMAL/Projet/data/mls_french_opus/mls_french_opus/test"
    # DEV_PATH = "C:/Cours-Sorbonne/M2/UE_DEEP/AMAL/Projet/data/mls_french_opus/mls_french_opus/dev"
    
    @staticmethod
    def display_common():
        """Display common configuration settings."""
        print("############# Base Settings: #############")
        print("\nAudio Settings:")
        print(f"  Sample Rate: {BaseConfig.SAMPLE_RATE} Hz")
        print(f"  Min Audio Duration: {BaseConfig.MIN_AUDIO_DURATION} seconds")
        print(f"  Max Audio Duration: {BaseConfig.MAX_AUDIO_DURATION} seconds")
        print("\nTraining Settings:")
        print(f"  Betas: {BaseConfig.BETAS}")
        print(f"  Device: {BaseConfig.DEVICE}")
        print("\nData Settings:")
        print(f"  Train Path: {BaseConfig.TRAIN_PATH}")
        print(f"  Test Path: {BaseConfig.TEST_PATH}")
        print(f"  Dev Path: {BaseConfig.DEV_PATH}")

class ConfigNAC(BaseConfig):
    """
        Configuration for NAC model.
    """
    MODEL_NAME = "NAC"  
    LAMBDA_FACTOR = 0.1
    NB_SAMPLES = 10000
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4
    MAX_TOKEN_LENGTH = 1024

    @staticmethod
    def display():
        """Display NAC configuration."""
        BaseConfig.display_common()
        print("\n############# NAC Settings: #############")

        print("\nModel Settings:")
        print(f"  Model Name: {ConfigNAC.MODEL_NAME}")
        print(f"  Lambda Factor: {ConfigNAC.LAMBDA_FACTOR}")
        print(f"  Number of Samples: {ConfigNAC.NB_SAMPLES}")
        print(f"  Batch Size: {ConfigNAC.BATCH_SIZE}")
        print(f"  Learning Rate: {ConfigNAC.LEARNING_RATE}")
        print(f"  Epochs: {ConfigNAC.EPOCHS}")
        print(f"  Token length for GPT2: {ConfigNAC.MAX_TOKEN_LENGTH}")

class ConfigSLP(BaseConfig):
    """
        Configuration for SLP model.
    """
    MODEL_NAME = "SLP"  
    EMBEDDING_DIM = 1472  # Audio embedding dimension
    NUM_LAYERS = 1  # Number of model layers
    NHEAD = 1  # Number of attention heads
    NB_CLASSES = int(BaseConfig.MAX_AUDIO_DURATION - BaseConfig.MIN_AUDIO_DURATION + 1)
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 8
    NB_SAMPLES = 10000
    MAX_TOKEN_LENGTH = 128

    @staticmethod
    def display():
        """Display SLP configuration."""
        BaseConfig.display_common()
        print("\n############# SLP Settings: #############")
        print("\nModel Settings:")
        print(f"  Model Name: {ConfigSLP.MODEL_NAME}")
        print(f"  Embedding Dimension: {ConfigSLP.EMBEDDING_DIM}")
        print(f"  Number of Layers: {ConfigSLP.NUM_LAYERS}")
        print(f"  Number of Attention Heads: {ConfigSLP.NHEAD}")
        print(f"  Number of Classes: {ConfigSLP.NB_CLASSES}")
        print(f"  Number of Samples: {ConfigSLP.NB_SAMPLES}")
        print(f"  Batch Size: {ConfigSLP.BATCH_SIZE}")
        print(f"  Learning Rate: {ConfigSLP.LEARNING_RATE}")
        print(f"  Epochs: {ConfigSLP.EPOCHS}")
        print(f"  Token length for ByT5: {ConfigSLP.MAX_TOKEN_LENGTH}")


class ConfigDiT(BaseConfig):
    """
        Configuration for DiT (Diffusion Transformer) model.
    """
    MODEL_NAME = "DiT"  # Model name

    # Model architecture (Section 3.3 of the paper)
    HIDDEN_DIM = 768  # Hidden dimension of the transformer
    NUM_LAYERS = 12  # Number of DiT blocks (12 for base model, 24 for large)
    NUM_HEADS = 12  # Number of attention heads
    TIME_DIM = 256  # Dimension of time embeddings
    TEXT_EMBED_DIM = 768  # Dimension of text embeddings (matches ByT5/SpeechT5)
    LATENT_DIM = 128  # Dimension of latent space (Mel-VAE output)

    # Diffusion process (Section 3.1 of the paper)
    DIFFUSION_STEPS = 1000  # Number of diffusion steps (T in the paper)
    NOISE_SCHEDULE = "cosine"  # Noise schedule (cosine as per the paper)
    NOISE_SCALE_SHIFT = 0.3  # Scale-shift for noise scheduler
    CLASSIFIER_FREE_GUIDANCE_SCALE = 5.0  # CFG scale for inference

    # Training settings (Section 4 of the paper)
    EPOCHS = 100  # Number of training epochs
    LEARNING_RATE = 1e-4  # Learning rate
    BATCH_SIZE = 32  # Batch size
    GRADIENT_ACCUMULATION_STEPS = 2  # Gradient accumulation steps
    WARMUP_STEPS = 1000  # Learning rate warmup steps

    # Data settings
    MAX_TOKEN_LENGTH = 512  # Maximum token length for text
    MAX_LATENT_LENGTH = 320  # Maximum latent length for audio (Mel-VAE output)

    # Loss weights (Section 3.2 of the paper)
    LM_LOSS_WEIGHT = 0.1  # Weight for language modeling loss
    RECONSTRUCTION_LOSS_WEIGHT = 1.0  # Weight for reconstruction loss

    @staticmethod
    def display():
        """Display DiT configuration."""
        BaseConfig.display_common()
        print("\n############# DiT Settings: #############")

        print("\nModel Architecture Settings:")
        print(f"  Model Name: {ConfigDiT.MODEL_NAME}")
        print(f"  Hidden Dimension: {ConfigDiT.HIDDEN_DIM}")
        print(f"  Number of Layers: {ConfigDiT.NUM_LAYERS}")
        print(f"  Number of Attention Heads: {ConfigDiT.NUM_HEADS}")
        print(f"  Time Embedding Dimension: {ConfigDiT.TIME_DIM}")
        print(f"  Text Embedding Dimension: {ConfigDiT.TEXT_EMBED_DIM}")
        print(f"  Latent Dimension: {ConfigDiT.LATENT_DIM}")

        print("\nDiffusion Process Settings:")
        print(f"  Diffusion Steps: {ConfigDiT.DIFFUSION_STEPS}")
        print(f"  Noise Schedule: {ConfigDiT.NOISE_SCHEDULE}")
        print(f"  Noise Scale-Shift: {ConfigDiT.NOISE_SCALE_SHIFT}")
        print(f"  Classifier-Free Guidance Scale: {ConfigDiT.CLASSIFIER_FREE_GUIDANCE_SCALE}")

        print("\nTraining Settings:")
        print(f"  Epochs: {ConfigDiT.EPOCHS}")
        print(f"  Learning Rate: {ConfigDiT.LEARNING_RATE}")
        print(f"  Batch Size: {ConfigDiT.BATCH_SIZE}")
        print(f"  Gradient Accumulation Steps: {ConfigDiT.GRADIENT_ACCUMULATION_STEPS}")
        print(f"  Warmup Steps: {ConfigDiT.WARMUP_STEPS}")

        print("\nData Settings:")
        print(f"  Max Token Length: {ConfigDiT.MAX_TOKEN_LENGTH}")
        print(f"  Max Latent Length: {ConfigDiT.MAX_LATENT_LENGTH}")

        print("\nLoss Weights:")
        print(f"  LM Loss Weight: {ConfigDiT.LM_LOSS_WEIGHT}")
        print(f"  Reconstruction Loss Weight: {ConfigDiT.RECONSTRUCTION_LOSS_WEIGHT}")DDI