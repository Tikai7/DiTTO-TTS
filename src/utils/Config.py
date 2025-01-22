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
    # TRAIN_PATH = "/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/data/mls_french_opus/train"
    # TEST_PATH = "/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/data/mls_french_opus/test"
    # DEV_PATH = "/tempory/M2-DAC/UE_DEEP/AMAL/DiTTO-TTS/data/mls_french_opus/dev"

    TRAIN_PATH = "C:/Cours-Sorbonne/M2/UE_DEEP/AMAL/Projet/data/mls_french_opus/mls_french_opus/train"
    TEST_PATH = "C:/Cours-Sorbonne/M2/UE_DEEP/AMAL/Projet/data/mls_french_opus/mls_french_opus/test"
    DEV_PATH = "C:/Cours-Sorbonne/M2/UE_DEEP/AMAL/Projet/data/mls_french_opus/mls_french_opus/dev"
    
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
    BATCH_SIZE = 8
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
