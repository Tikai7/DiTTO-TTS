import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


from components.DiT import DiT
from utils.Config import ConfigDiT  
from utils.MLS import MLSDataset
from utils.Trainer import Trainer

# DiT-specific configuration
print(ConfigDiT.display())


# Initialize datasets
train_set = MLSDataset(
    data_dir=ConfigDiT.TRAIN_PATH,
    max_text_token_length=ConfigDiT.MAX_TOKEN_LENGTH,
    sampling_rate=ConfigDiT.SAMPLE_RATE,
    tokenizer_model="gpt2"
)

val_set = MLSDataset(
    data_dir=ConfigDiT.DEV_PATH,
    max_text_token_length=ConfigDiT.MAX_TOKEN_LENGTH,
    sampling_rate=ConfigDiT.SAMPLE_RATE,
    tokenizer_model="gpt2"
)

# Create data loaders
train_loader = DataLoader(train_set,
                          batch_size=ConfigDiT.BATCH_SIZE,
                          shuffle=True,
                          collate_fn=MLSDataset.collate_fn)

val_loader = DataLoader(val_set,
                        batch_size=ConfigDiT.BATCH_SIZE,
                        collate_fn=MLSDataset.collate_fn)

# Initialize DiT model
model = DiT(
    hidden_dim=ConfigDiT.HIDDEN_DIM,
    num_layers=ConfigDiT.NUM_LAYERS,
    num_heads=ConfigDiT.NUM_HEADS,
    time_dim=ConfigDiT.TIME_DIM,
    text_dim=ConfigDiT.TEXT_EMBED_DIM,
    diffusion_steps=ConfigDiT.DIFFUSION_STEPS,
).to(ConfigDiT.DEVICE)

# Diffusion loss function


class DiffusionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, noise_pred, noise_true):
        return torch.mean((noise_pred - noise_true) ** 2)


criterion = DiffusionLoss()
optimizer = torch.optim.AdamW


def train(self, train_loader):
    self.model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        # Prepare inputs
        latents = batch['audio'].to(self.device)
        text_emb = batch['text']['input_ids'].to(self.device)
        padding_mask = batch['padding_mask_audio'].to(self.device)

        # Diffusion process
        # 1. Sample random time steps
        t = torch.randint(0, ConfigDiT.DIFFUSION_STEPS, (latents.size(0),),
                          device=self.device).long()

        # 2. Add noise to latents
        noise = torch.randn_like(latents)
        noisy_latents = self.model.q_sample(
            latents, t, noise)

        # 3. Predict noise
        noise_pred = self.model(noisy_latents, text_emb, t)

        # 4. Calculate loss
        loss = self.criterion(noise_pred, noise)

        # Optimize
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

            t = torch.randint(0, ConfigDiT.DIFFUSION_STEPS, (latents.size(0),),
                              device=self.device).long()

            noise = torch.randn_like(latents)
            noisy_latents = self.model.q_sample(latents, t, noise)
            noise_pred = self.model(noisy_latents, text_emb, t, padding_mask)

            loss = self.criterion(noise_pred, noise)
            total_loss += loss.item()

    return total_loss / len(val_loader), {}


# Initialize trainer
trainer = Trainer()
trainer.set_model(model, name=ConfigDiT.MODEL_NAME)\
    .set_criterion(criterion)\
    .set_optimizer(optimizer)\
    .set_custom_functions(train_func=train, validation_func=validation)\
    .fit(
        train_data=train_loader,
        validation_data=val_loader,
        epochs=ConfigDiT.EPOCHS,
        learning_rate=ConfigDiT.LEARNING_RATE,
        checkpoint_interval=1
)
