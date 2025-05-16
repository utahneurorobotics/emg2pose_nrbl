import torch.nn as nn
import pytorch_lightning as pl
from emg2pose.lightning import Emg2PoseModule

class ExtendedEmg2PoseNet(pl.LightningModule):
    def __init__(self, config, extra_hidden_dim=128, new_output_dim=None):
        super().__init__()
        pretrained_model = Emg2PoseModule.load_from_checkpoint(config.checkpoint,
                            network=config.network,
                            optimizer=config.optimizer,
                            lr_scheduler=config.lr_scheduler,
        )

        pretrained_output_dim = 40  
        # Example: add a new hidden layer and a new output layer
        # Assume pretrained model's output is a flat vector
        self.first_layer = nn.Linear(48, 16)
        self.relu1 = nn.ReLU()
        self.pretrained = pretrained_model
        # Freeze pretrained weights
        for param in self.pretrained.parameters():
            param.requires_grad = False
        
        self.last_layer = nn.Linear(pretrained_output_dim, extra_hidden_dim)
        self.relu2 = nn.ReLU()
        self.new_output = nn.Linear(40, 12)

    def forward(self, batch, provide_initial_pos=False):
        # Get features from the frozen model
        features = self.pretrained.forward(batch, provide_initial_pos)[0]  # [0] for preds
        x = self.extra_layer(features)
        x = self.relu(x)
        x = self.new_output(x)
        # Return in the same tuple format as before
        return x, features, batch["no_ik_failure"]