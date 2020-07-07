# best.pth
import torch
from torch import nn
import torch.nn.functional as F

class conv2_autoencoder(nn.Module):
    def __init__(self):
        super(conv2_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            
            nn.Conv2d(3, 6, 3, 1, 1),            # [batch, 6, 32, 32]
            # nn.Conv2d(6, 12, 4, 2, 1),           # [batch, 12, 32, 32]
            nn.Conv2d(6, 12, 3, 1, 1),           # [batch, 12, 32, 32]
            nn.MaxPool2d(2, 2, 0),                # [batch, 12, 16, 16]
            nn.BatchNorm2d(12),
            nn.Dropout(0.5),
            nn.ReLU(),

            # nn.Conv2d(12, 24, 4, 2, 1),           # [batch, 24, 16, 16]
            nn.Conv2d(12, 24, 3, 1, 1),           # [batch, 24, 16, 16]
            nn.MaxPool2d(2, 2, 0),                # [batch, 24, 8, 8]
            nn.BatchNorm2d(24),
            nn.Dropout(0.5),
            nn.ReLU(),

            # nn.Conv2d(24, 48, 4, 2, 1),           # [batch, 48, 8, 8]
            nn.Conv2d(24, 48, 3, 1, 1),           # [batch, 48, 8, 8]
            nn.MaxPool2d(2, 2, 0),                 # [batch, 48, 4, 4]
            nn.BatchNorm2d(48),
            nn.Dropout(0.5),
            nn.ReLU(),

            # nn.Conv2d(48, 96, 4, 2, 1),           # [batch, 96, 4, 4]
            nn.Conv2d(48, 96, 3, 1, 1),           # [batch, 96, 4, 4]
            nn.MaxPool2d(2, 2, 0),                  # [batch, 96, 2, 2]
            nn.BatchNorm2d(96),
            nn.Dropout(0.5),
            nn.ReLU(),

        )
        self.decoder = nn.Sequential(
            
            # nn.ConvTranspose2d(96, 48, 4, 2, 1),           # [batch, 48, 2, 2]
            nn.ConvTranspose2d(96, 48, 3, 1, 1),           # [batch, 48, 2, 2]
            nn.Upsample(scale_factor=2, mode='nearest'),    # [batch, 48, 4, 4]
            nn.BatchNorm2d(48),
            nn.Dropout(0.2),
            nn.ReLU(),

            # nn.ConvTranspose2d(48, 24, 4, 2, 1),          # [batch, 24, 4, 4]
            nn.ConvTranspose2d(48, 24, 3, 1, 1),          # [batch, 24, 4, 4]
            nn.Upsample(scale_factor=2, mode='nearest'),    # [batch, 24, 8, 8]
            nn.BatchNorm2d(24),
            nn.Dropout(0.2),
            nn.ReLU(),
                  
            # nn.ConvTranspose2d(24, 12, 4, 2, 1),           # [batch, 12, 8, 8]
            nn.ConvTranspose2d(24, 12, 3, 1, 1),           # [batch, 12, 8, 8]
            nn.Upsample(scale_factor=2, mode='nearest'),    # [batch, 12, 16, 16]
            nn.BatchNorm2d(12),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.ConvTranspose2d(12, 6, 3, 1, 1),            # [batch, 6, 16, 16]
            # nn.ConvTranspose2d(6, 3, 4, 2, 1),            # [batch, 3, 16, 16]
            nn.ConvTranspose2d(6, 3, 3, 1, 1),            # [batch, 3, 16, 16]
            nn.Upsample(scale_factor=2, mode='nearest'),    # [batch, 3, 32, 32]
            nn.BatchNorm2d(3),
            nn.Dropout(0.2),
            nn.Tanh(),
        )

        # self.dense1 = nn.Sequential(
        #     nn.Linear(96*2*2, 96),
        #     nn.ReLU(),

        #     nn.Linear(96, 32),
        # )

        # self.dense2 = nn.Sequential(
        #     nn.Linear(32, 96),
        #     nn.ReLU(),

        #     nn.Linear(96, 96*2*2),
        # )


    def forward(self, x):
        x = self.encoder(x)
        # x = x.view(x.size()[0], -1)
        # x = self.dense1(x)
        # x = self.dense2(x)
        # x = x.view(x.size()[0], 96, 2, 2)
        x = self.decoder(x)
        return x