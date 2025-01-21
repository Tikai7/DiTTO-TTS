import torch
import torch.nn as nn 

from model.NeuralAudioCodec import NAC

from utils.Config import ConfigNAC
from utils.MLS import MLSDataset
from utils.Trainer import Trainer

from tqdm import tqdm
from torch.utils.data import DataLoader


print(ConfigNAC.display())

# No need on windows
# Processing.remove_metadata_from_audio_folder(ConfigNAC.TRAIN_PATH+"/"+"audio", ConfigNAC.TRAIN_PATH+"/"+"audio_clean",)
# Processing.remove_metadata_from_audio_folder(ConfigNAC.TEST_PATH+"/"+"audio", ConfigNAC.TEST_PATH+"/"+"audio_clean",)
# Processing.remove_metadata_from_audio_folder(ConfigNAC.DEV_PATH+"/"+"audio", ConfigNAC.DEV_PATH+"/"+"audio_clean",)

train_set = MLSDataset(
    data_dir=ConfigNAC.TRAIN_PATH,
    max_text_token_length=ConfigNAC.MAX_TOKEN_LENGTH,
    sampling_rate=ConfigNAC.SAMPLE_RATE,
)

val_set = MLSDataset(
    data_dir=ConfigNAC.DEV_PATH,
    max_text_token_length=ConfigNAC.MAX_TOKEN_LENGTH,
    sampling_rate=ConfigNAC.SAMPLE_RATE,
)


test_set = MLSDataset(
    data_dir=ConfigNAC.TEST_PATH,
    max_text_token_length=ConfigNAC.MAX_TOKEN_LENGTH,
    sampling_rate=ConfigNAC.SAMPLE_RATE,
)

train_loader = DataLoader(train_set, batch_size=ConfigNAC.BATCH_SIZE, shuffle=True, collate_fn=MLSDataset.collate_fn)
val_loader = DataLoader(val_set, batch_size=ConfigNAC.BATCH_SIZE, shuffle=True, collate_fn=MLSDataset.collate_fn)
test_loader = DataLoader(test_set, batch_size=ConfigNAC.BATCH_SIZE, shuffle=True, collate_fn=MLSDataset.collate_fn)


model = NAC(ConfigNAC.LAMBDA_FACTOR)
model = model.to(ConfigNAC.DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW


def train(self, train_loader):
    losses = 0
    self.model.train()

    for batch in tqdm(train_loader):
        batch["text"]["input_ids"] = batch["text"]["input_ids"].to(self.device)
        batch["text"]["attention_mask"] = batch["text"]["attention_mask"].to(self.device)

        text = batch["text"]
        audio = batch["audio"].to(self.device)

        output = self.model(text, audio)
        loss = output.total_loss

        losses += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    return losses / len(train_loader), {"accuracy" : -1}


def validation(self, validation_loader):
    losses = 0
    self.model.eval()

    for batch in tqdm(validation_loader):
        batch["text"]["input_ids"] = batch["text"]["input_ids"].to(self.device)
        batch["text"]["attention_mask"] = batch["text"]["attention_mask"].to(self.device)
        
        text = batch["text"]
        audio = batch["audio"].to(self.device)

        output = self.model(text, audio)
        loss = output.total_loss
        
        losses += loss.item()

    return losses / len(train_loader), {"accuracy" : -1}


trainer = Trainer()
trainer.set_model(model, name=ConfigNAC.MODEL_NAME)\
    .set_criterion(criterion)\
    .set_optimizer(optimizer)\
    .set_custom_functions(train_func=train, validation_func=validation)\
    .fit(
        train_data=train_loader, validation_data=val_loader, 
        epochs=ConfigNAC.EPOCHS, learning_rate=ConfigNAC.LEARNING_RATE, checkpoint_interval=1        
    )