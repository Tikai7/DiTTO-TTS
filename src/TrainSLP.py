import torch
import torch.nn as nn 

from model.SpeechLP import SLP

from utils.Config import Config
from utils.MLS import MLSDataset
from utils.Trainer import Trainer

from torch.utils.data import DataLoader


print(Config.display())

# No need on windows
# Processing.remove_metadata_from_audio_folder(Config.TRAIN_PATH+"/"+"audio", Config.TRAIN_PATH+"/"+"audio_clean",)
# Processing.remove_metadata_from_audio_folder(Config.TEST_PATH+"/"+"audio", Config.TEST_PATH+"/"+"audio_clean",)
# Processing.remove_metadata_from_audio_folder(Config.DEV_PATH+"/"+"audio", Config.DEV_PATH+"/"+"audio_clean",)

train_set = MLSDataset(
    data_dir=Config.TRAIN_PATH,
    max_text_token_length=Config.MAX_TOKEN_LENGTH,
    sampling_rate=Config.SAMPLE_RATE,
    nb_samples=Config.NB_SAMPLES
)

val_set = MLSDataset(
    data_dir=Config.DEV_PATH,
    max_text_token_length=Config.MAX_TOKEN_LENGTH,
    sampling_rate=Config.SAMPLE_RATE,
)


test_set = MLSDataset(
    data_dir=Config.TEST_PATH,
    max_text_token_length=Config.MAX_TOKEN_LENGTH,
    sampling_rate=Config.SAMPLE_RATE,
)

train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=MLSDataset.collate_fn)
val_loader = DataLoader(val_set, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=MLSDataset.collate_fn)
test_loader = DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=MLSDataset.collate_fn)


model = SLP(Config.NB_CLASSES, Config.NHEAD ,Config.NUM_LAYERS)
model = model.to(Config.DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW

trainer = Trainer()
trainer.set_model(model, name="SLP")\
    .set_criterion(criterion)\
    .set_optimizer(optimizer)\
    .fit(
        train_data=train_loader, validation_data=val_loader, 
        epochs=Config.EPOCHS, learning_rate=Config.LEARNING_RATE, checkpoint_interval=1        
    )