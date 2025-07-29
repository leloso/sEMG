import torch.nn as nn
import numpy as np
import torch
import ptwt
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.functional as F
import pytorch_lightning as pl


class FCBlock(nn.Module):
    """
    Fully connected block with ReLU activation and dropout.
    """
    def __init__(self, in_dim, out_dim, out_classes=None):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )
        if out_classes is not None:
            self.block.add_module('output', nn.Linear(out_dim, out_classes))

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, dropout=0.3):
        super().__init__()

        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, 
                                           padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, 
                                           padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)
        self.match_dim = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # crop right padding for causality
        out = self.conv2(self.drop1(self.relu1(out)))
        out = out[:, :, :x.size(2)]  # crop right padding for causality
        out = self.drop2(self.relu2(out))
        return out + self.match_dim(x)
    

class BaseModel(pl.LightningModule):
    def __init__(self, in_channels, num_classes: int, sequence_length: int = 60, loss = nn.CrossEntropyLoss()):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.loss = loss
        self.sequence_length = sequence_length

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement this method.")

    def training_step(self, batch, batch_idx):
        samples, labels = batch
        outputs = self(samples)

        train_loss = self.loss(outputs, labels)
        
        self.log('train_loss', train_loss, on_step=True, on_epoch=False) # Log training loss per step
        return train_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        samples, labels = batch
        outputs = self(samples)
        val_loss = self.loss(outputs, labels)

        self.log('val_loss', val_loss, add_dataloader_idx=True)

        preds = torch.argmax(outputs, dim=1)

        acc = (preds==labels).float().mean()

        self.log('val_accuracy', acc, on_step=False, on_epoch=True, 
                add_dataloader_idx=True)
        
        return {'val_accuracy': acc}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        samples, labels = batch
        outputs = self(samples)
       
        test_loss = self.loss(outputs, labels)
        self.log('test_loss', test_loss, add_dataloader_idx=True)

        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        self.log('test_acc', acc)

        return {"test_loss": test_loss, "test_acc": acc}


class CNN(BaseModel):
    def __init__(self, in_channels, num_classes, loss):
        super().__init__(in_channels=in_channels, num_classes=num_classes, loss = loss)
        self.conv_block1 = ConvBlock(in_channels, 64)
        self.conv_block2 = ConvBlock(64, 128, kernel_size=5)
        self.dropout = nn.Dropout()

        self.fc = FCBlock(in_dim=128, out_dim=128, out_classes=num_classes)

    def forward(self, x):
        x = self.conv_block1(x)     
        x = self.conv_block2(x)
        x = self.dropout(x)

        #global average pooling
        x = torch.mean(x, dim=2)  # Average over time dimension
        x = self.fc(x)
        
        return x


class CNN_GRU(BaseModel):
    def __init__(self, in_channels, num_classes, loss):
        super().__init__(num_classes, loss)

        self.feature_dim = 128

        self.conv_block1 = ConvBlock(in_channels, 64)
        self.conv_block2 = ConvBlock(64, self.feature_dim, kernel_size=5)

        self.dropout = nn.Dropout()

        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.feature_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        self.fc = FCBlock(in_dim=self.feature_dim, out_dim=self.feature_dim, out_classes=num_classes)

    def forward(self, x):
        x = self.conv_block1(x)  # (B, 64, L1)
        x = self.conv_block2(x)  # (B, 128, L2)
        
        x = self.dropout(x)

        # Reshape for GRU: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)  # (B, T=out_length, C=128)

        # GRU: output shape = (B, T, hidden_size), last timestep used for classification
        gru_out, _ = self.gru(x)

        x = gru_out[:, -1, :]

        logits = self.fc(x)
        return logits


class CNN_BiGRU(BaseModel):
    def __init__(self, in_channels, num_classes, loss):
        super().__init__(num_classes, loss)

        self.conv_block1 = ConvBlock(in_channels, 64)
        self.conv_block2 = ConvBlock(64, 128, kernel_size=5)

        self.dropout = nn.Dropout()

        self.feature_dim = 128  # Output channels from last conv

        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.feature_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc1 = FCBlock(self.feature_dim, self.feature_dim)

        self.fc2 = FCBlock(in_dim=2 * self.feature_dim, out_dim= 2 * self.feature_dim, out_classes=num_classes)

    def forward(self, x):
        x = self.conv_block1(x)  # (B, 64, L1)
        x = self.conv_block2(x)  # (B, 128, L2)
        x = self.dropout(x)

        # Reshape for GRU: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)  # (B, T=out_length, C=128)

        x = self.fc1(x)

        # GRU: output shape = (B, T, 2*hidden_size), we'll use last timestep
        gru_out, _ = self.gru(x)
        x = gru_out[:, -1, :]  # last time step

        logits = self.fc2(x)

        return logits


class CNN_LSTM(BaseModel):
    def __init__(self, in_channels, num_classes, loss):
        super().__init__(num_classes, loss)

        self.hidden_size = 128

        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
        )

        self.lstm = nn.LSTM(
            input_size=128,  # Output channels from last conv
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )

        self.fc1 = FCBlock(in_dim=self.hidden_size, out_dim = 2*self.hidden_size)
        self.fc2 = FCBlock(in_dim=2*self.hidden_size, out_dim = 64, out_classes=num_classes)

    def forward(self, x):

        x = self.convs(x)
        # Reshape for GRU: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)  # (B, T=input_length, C=128)

        # LSTM: output shape = (B, T, hidden_size), we'll use last timestep
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # last time step

        logits = self.fc2(self.fc1(x))

        return logits


class TCN(BaseModel):

    def __init__(self, in_channels=8,  num_classes=7, dropout=0.2, loss = None):
        super().__init__(in_channels=in_channels, num_classes=num_classes, loss=loss)
        self.block1 = TCNBlock(in_channels, 32, kernel_size=3, dilation=1, padding=2, dropout=dropout)
        self.block2 = TCNBlock(32, 32, kernel_size=3, dilation=2, padding=4, dropout=dropout)
        self.block3 = TCNBlock(32, 64, kernel_size=3, dilation=4, padding=8, dropout=dropout)
        self.block4 = TCNBlock(64, 128, kernel_size=3, dilation=8, padding=16, dropout=dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):   
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # global average pooling for embedding
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return x
    
# SE Block
class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: [B, C, T]
        b, c, t = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(b, c)  # [B, C]
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1)
        return x * y  # broadcast to [B, C, T]

# Convolutional block with SE
class ConvSEBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.se(F.relu(self.bn(self.conv(x))))
        return x


class SingleBranch(nn.Module):
    """ Time-domain branch of MESTNet.
    Processes the time-domain sEMG signals.
    """

    def __init__(self, in_channels=8):
        super().__init__()
        self.block1 = ConvSEBlock(in_channels, 64, 3)
        self.block2 = ConvSEBlock(64, 128, 5)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        # x: [B, 8, 60]
        x = self.pool(self.block2(self.block1(x)))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # [B, T, C]
        return self.fc(x)  # [B, T, 128]


# Final MESTNet
class MESTNet(BaseModel):
    def __init__(self, 
            in_channels: int, 
            scales: int, 
            num_classes: int, 
            wavelet: str = 'morl', 
            loss = None,
        ):
        super().__init__(in_channels=in_channels, num_classes=num_classes, loss=loss)

        self.time_branch = SingleBranch(in_channels=in_channels)
        self.freq_branch = SingleBranch(in_channels=in_channels*scales)
        self.bigru = nn.GRU(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(256, num_classes)

        self.scales = np.arange(1, scales+1)
        self.wavelet = wavelet

    def _get_cwt(self, x):
        
        b, c, t = x.shape

        cwt_out, _ = ptwt.cwt(x.view(b*c, t), self.scales, self.wavelet) # (scales, batch_size*channels, timestep) output

        # TODO: check correct layout of return
        cwt_out = cwt_out.view(self.scales[-1], b, c, t).permute(1, 0, 2, 3).reshape(b, -1, t) # ()

        return cwt_out.to(dtype=torch.float32)
    
    def forward(self, source_input, target_input=None):
        # Input: time-freq merged -> (channels+freq_scales, timesteps)

        t = self.time_branch(input)   # [B, T, 128]
        f = self.freq_branch(self._get_cwt(input))  # [B, T, 128]

        gru_out, _ = self.bigru(torch.cat((t, f), dim=2)) # [B, T, 256]

        logits = self.classifier(gru_out[:, -1, :]) # use only last timestep output for projection
        
        return logits    # [B, num_classes]

    def training_step(self, batch, batch_idx):
        source_sample, source_label = batch

        # 1. Forward pass for source and target
        source_logits = self(source_sample)

        # 2. Calculate Classification Loss (on source data)
        
        loss = self.loss(source_logits, source_label)
        self.log('cross_ent_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    # pure prediction-label cross entropy / accuracy
    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        source_sample, source_label = batch # No need for target_sample in validation for classification metric
        source_logits = self(source_sample)
        val_loss = self.loss(source_logits, source_label)
        
        preds = torch.argmax(source_logits, dim=1)
        acc = (preds == source_label).float().mean()

        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        return val_loss


class MESTNet_DA(MESTNet):
    def __init__(self, 
            in_channels: int, 
            scales: int, 
            num_classes: int, 
            loss: nn.Module,
            wavelet: str = 'morl', 
            da_loss = None,
        ):

        super().__init__(in_channels=in_channels, scales=scales, num_classes=num_classes, wavelet=wavelet, loss=loss)

        self.da_loss = da_loss

    def _get_embedding(self, input):

        t = self.time_branch(input)   # [B, T, 128]
        f = self.freq_branch(self._get_cwt(input))  # [B, T, 128]

        gru_out, _ = self.bigru(torch.cat((t, f), dim=2)) # [B, T, 256]

        return gru_out[:, -1, :] # [B, 256]

    def forward(self, source_input, target_input=None):
        # Input: time-freq merged -> (channels+freq_scales, timesteps)

        source_feats = self._get_embedding(source_input)
        logits = self.classifier(source_feats)

        
        target_feats = self._get_embedding(target_input)
        return logits, source_feats, target_feats
        
    def training_step(self, batch, batch_idx):

        source_sample, source_label, target_sample = batch
            # 1. Forward pass for source and target
        source_logits, source_features, target_features = self(source_sample, target_input=target_sample)

        # 2. Calculate Classification Loss (on source data)
        
        classification_loss = self.loss(source_logits, source_label)

        self.log('cross_ent_loss', classification_loss, on_step=True, on_epoch=True, prog_bar=True)

        da_loss = self.da_loss(source_features, target_features)
        
        self.log('train_da_loss', da_loss, on_step=True, on_epoch=True, prog_bar=True)

        total_loss = classification_loss + self.C * da_loss
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss



if __name__ == "__main__":

    from loss import MMDLoss
    import torch.nn as nn

    da_loss = MMDLoss()

    model = MESTNet(in_channels=8, num_classes=7, loss=nn.CrossEntropyLoss(), wavelet='morl', scales=32)

    mestnet = MESTNet_DA(in_channels=8, num_classes=7, wavelet='morl', scales=32, da_loss=da_loss, loss=nn.CrossEntropyLoss())

    input = torch.rand(size=(256, 8, 60))
    target_input = torch.rand(size=(256, 8, 60))

    output = model(input)

    print(output.shape)