import torch
import torch.nn as nn 

from model.SpeechLP import SLP

from utils.Config import Config
from utils.MLS import MLSDataset
from utils.Trainer import Trainer

from torch.utils.data import DataLoader, random_split


print(Config.display())

# No need on windows
# Processing.remove_metadata_from_audio_folder(Config.TRAIN_PATH+"/"+"audio", Config.TRAIN_PATH+"/"+"audio_clean",)
# Processing.remove_metadata_from_audio_folder(Config.TEST_PATH+"/"+"audio", Config.TEST_PATH+"/"+"audio_clean",)
# Processing.remove_metadata_from_audio_folder(Config.DEV_PATH+"/"+"audio", Config.DEV_PATH+"/"+"audio_clean",)

dataset = MLSDataset(
    data_dir=Config.TRAIN_PATH,
    max_text_token_length=Config.MAX_TOKEN_LENGTH,
    sampling_rate=Config.SAMPLE_RATE,
)

train_set, val_set = random_split(dataset, [Config.TRAIN_RATIO, Config.VAL_RATIO])

train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=MLSDataset.collate_fn)
val_loader = DataLoader(val_set, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=MLSDataset.collate_fn)

model = SLP(Config.MAX_AUDIO_DURATION, Config.NHEAD ,Config.NUM_LAYERS).to(Config.DEVICE)
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