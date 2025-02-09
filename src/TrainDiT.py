import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.DiTTO import DiTTO
from utils.Config import ConfigDiTTO  
from utils.MLS import MLSDataset
from utils.Trainer import Trainer

print(ConfigDiTTO.display())


train_set = MLSDataset(
    data_dir=ConfigDiTTO.TRAIN_PATH,
    max_text_token_length=ConfigDiTTO.MAX_TOKEN_LENGTH,
    sampling_rate=ConfigDiTTO.SAMPLE_RATE,
    tokenizer_model="gpt2"
)

val_set = MLSDataset(
    data_dir=ConfigDiTTO.DEV_PATH,
    max_text_token_length=ConfigDiTTO.MAX_TOKEN_LENGTH,
    sampling_rate=ConfigDiTTO.SAMPLE_RATE,
    tokenizer_model="gpt2"
)

train_loader = DataLoader(train_set,
                          batch_size=ConfigDiTTO.BATCH_SIZE,
                          shuffle=True,
                          collate_fn=MLSDataset.collate_fn)

val_loader = DataLoader(val_set,
                        batch_size=ConfigDiTTO.BATCH_SIZE,
                        collate_fn=MLSDataset.collate_fn)

model = DiTTO(
    hidden_dim=ConfigDiTTO.HIDDEN_DIM,
    num_layers=ConfigDiTTO.NUM_LAYERS,
    num_heads=ConfigDiTTO.NUM_HEADS,
    time_dim=ConfigDiTTO.TIME_DIM,
    text_dim=ConfigDiTTO.TEXT_EMBED_DIM,
    diffusion_steps=ConfigDiTTO.DIFFUSION_STEPS,
).to(ConfigDiTTO.DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW

def train(self, train_loader):
    self.model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        # Prepare inputs
        latents = batch['audio'].to(self.device)
        text_emb = batch['text']['input_ids'].to(self.device)
        padding_mask = batch['padding_mask_audio'].to(self.device)

        # Sample random time steps
        t = torch.randint(0, ConfigDiTTO.DIFFUSION_STEPS, (latents.size(0),),
                          device=self.device).long()
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        noisy_latents = self.model.q_sample(
            latents, t, noise)

        # Predict noise
        noise_pred = self.model(noisy_latents, text_emb, t, padding_mask)

        loss = self.criterion(noise_pred, noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader), {}


def validation(self, val_loader):
    self.model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            latents = batch['audio'].to(self.device)
            text_emb = batch['text']['input_ids'].to(self.device)
            padding_mask = batch['padding_mask_audio'].to(self.device)

            t = torch.randint(0, ConfigDiTTO.DIFFUSION_STEPS, (latents.size(0),),
                              device=self.device).long()

            noise = torch.randn_like(latents)
            noisy_latents = self.model.q_sample(latents, t, noise)
            noise_pred = self.model(noisy_latents, text_emb, t, padding_mask)

            loss = self.criterion(noise_pred, noise)
            total_loss += loss.item()

    return total_loss / len(val_loader), {}


trainer = Trainer()
trainer.set_model(model, name=ConfigDiTTO.MODEL_NAME)\
    .set_criterion(criterion)\
    .set_optimizer(optimizer)\
    .set_custom_functions(train_func=train, validation_func=validation)\
    .fit(
        train_data=train_loader,
        validation_data=val_loader,
        epochs=ConfigDiTTO.EPOCHS,
        learning_rate=ConfigDiTTO.LEARNING_RATE,
        checkpoint_interval=1
)
