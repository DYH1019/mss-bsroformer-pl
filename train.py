# import debugpy; debugpy.connect(('localhost', 1234))

import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
import torch
import torch.nn.functional as F
import sys
import os
import tqdm
import soundfile as sf
import glob
import wandb
import torchaudio

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.utils_metric import new_sdr
from utils.callbacks import ValidationProgressBar
import augment
from data_module import WaveBleedingDataModule
import sys
sys.path.append('/23SA01/codes/Music-Source-Separation-BSRoFormer-pl/')

# print(sys.path)
from models.bs_roformer.bs_roformer import BSRoformer
from separate import sep_weighted_avg_shift

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

PROJECT_NAME = "mss_bs_roformer_pl"

class TrainModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        # print(config.model)
        self.model = BSRoformer(**config.model)

        # data augment
        augments = [
            augment.Shift(),
            augment.FlipChannels(),
            augment.FlipSign(),
            augment.Scale(),
            augment.Remix(),
        ]
        self.source_name = ["drums", "bass", "other", "vocals"]
        self.train_idx = -1
        self.augment = torch.nn.Sequential(*augments)

    def forward(self, mix):
        x = self.model(mix)
        return x

    def training_step(self, batch, batch_idx):

        sources = self.augment(batch)
        mix = sources.sum(dim=1)
        # print(mix.shape, sources.shape)
        multi_stft_loss = self.model(mix, sources[:, 3, ...])
        # print(estimate.shape)

        # time domain loss
        # source = sources[:, self.train_idx, ...].unsqueeze(1)
        # loss_t = F.l1_loss(estimate, source)
        # print(estimate.shape, source.shape)

        # total_loss = loss_t
        self.log('train_loss', multi_stft_loss, prog_bar=True, on_epoch=True)    
        return multi_stft_loss
    
    def validation_step(self, batch, batch_idx):
        
        mix, index = batch
        mix = mix.to(self.device)
        # print(mix.device)
        estimate = sep_weighted_avg_shift(self.model, mix, self.config.training.seg_len)
        # print('estimate.shape', estimate.shape)

        instr = self.source_name[self.train_idx]
        pbar_dict = {}
        path_list = os.listdir(self.config.training.valid_root_dir)
       
        # if instr != 'other' or self.hparams.config.training.other_fix is False:
        track, _ = torchaudio.load(os.path.join(self.config.training.valid_root_dir, path_list[index]) + f'/{instr}.wav')
        track = track.to(self.device)
        # else:
        #     # other is actually instrumental
        #     track, _ = sf.read(folder + f'/vocals.wav')
        #     track = mix.cpu().numpy() - track
        # [1, 1, 2, L] -> [b, s, c, T]

        references = torch.unsqueeze(track, axis=0).unsqueeze(0)
        # print('references.shape',references.shape)
        # estimates = torch.unsqueeze(estimate, axis=0)
        # print(estimate.shape)

        val_loss = F.l1_loss(estimate, references)
        sdr_val = new_sdr(references, estimate)[0]
        single_val = torch.tensor([sdr_val]).to(self.device)
        self.log('val_F1_loss', val_loss, prog_bar=True, on_epoch=True)
        self.log(f'val_sdr_{instr}', single_val, prog_bar=True, on_epoch=True)
        pbar_dict[f'sdr_{instr}'] = sdr_val

        return pbar_dict

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.config.training.optim.lr,
                                      )
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40000, gamma=0.9)
        # scheduler = CosineAnnealingLR(optimizer, T_max=self.config.optim.lr_scheduler.t_max)

        return [optimizer], [scheduler]


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load('/23SA01/codes/Music-Source-Separation-BSRoFormer-pl/configs/train_config_bs_roformer.yaml')
    seed = config.training.random_seed
    instr = config.training.instruments

    seed_everything(seed)

    # if torch.cuda.is_available():
    #     num_gpus = torch.cuda.device_count()
    #     for i in range(num_gpus):
    #         print(f"Using GPU: {torch.cuda.get_device_name(i)} (Index: {i})")
    # else:
    #     print("No GPU available, using CPU.")

    # Prepare for DataModule.
    dm = WaveBleedingDataModule(config)
    dm.setup()

    # Wandb Logger
    wandb.login(key=config.wandb.api_key)
    wandb_logger = WandbLogger(project=PROJECT_NAME, config=config)

    # Callbacks for checkpointing and learning rate monitoring
    checkpoint_callback = ModelCheckpoint(
        monitor='val_sdr_vocals',
        dirpath='checkpoints/',
        filename='mss_bs_roformer_pl_vocals-{epoch:02d}-{val_sdr:.2f}',
        save_top_k=3,
        mode='max',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Exponential Moving Averaging (EMA)
    ema = ValidationProgressBar()

    # Use SimpleProfiler or AdvancedProfiler
    simple_profiler = SimpleProfiler()
    # advanced_profiler = AdvancedProfiler(filename="advanced_profile.txt")

    # Train the model
    model = TrainModule(config)
    trainer = Trainer(
        check_val_every_n_epoch=1,
        accelerator="auto", devices=8,
        strategy="ddp_find_unused_parameters_false",
        accumulate_grad_batches=config.training.grad_accumulate,
        max_epochs=config.training.num_epochs,
        num_sanity_val_steps=0,
        precision='16-mixed',  # Mixed Precision Training
        logger=wandb_logger,
        profiler=simple_profiler,
        callbacks=[checkpoint_callback, lr_monitor, ema]
    )

    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)

