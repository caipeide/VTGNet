import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FeatExtractor(nn.Module):
    def __init__(self, feature_size):
        super(FeatExtractor, self).__init__()
        model = models.mobilenet_v2(pretrained=True)
        modules = list(model.children())[:-1]
        self.cnn = nn.Sequential(*modules)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, feature_size)
        )
                
    def forward(self, images):
        features = self.cnn(images)
        features = features.mean([2, 3])
        features = self.classifier(features)

        return features


class TrajGenerator(nn.Module):
    def __init__(self, feature_size):
        super(TrajGenerator, self).__init__()

        self.info_layer = nn.Sequential(
            nn.Linear(3,32),

            nn.Linear(32,64),

            nn.Linear(64,128)
        )

        self.feature_layer = nn.Sequential(
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128,66)
        )
        
        self.lstm_features = nn.LSTM( feature_size + 128, hidden_size=256, num_layers=3, batch_first=True)

        self.layer_uncertainty = nn.Sequential(
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128,66)
        )

        self.layer_attention = nn.Sequential(
            nn.Linear((feature_size + 128) * 12, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.Linear(128, 12),
            nn.BatchNorm1d(12),
            nn.Softmax(dim=1)
        )
        
    def forward(self, features, info_history):
        
        info_history = info_history.type(torch.FloatTensor).cuda()
        info_history = self.info_layer(info_history)

        packed = torch.cat([features,info_history],dim=2)
        attentions = self.layer_attention(packed.reshape(-1, (128+512)*12))
        
        packed = torch.cat([attentions.reshape(-1, 12, 1) * features, info_history], dim=2)

        features,_ = self.lstm_features(packed)
        log_var = self.layer_uncertainty(torch.sum(attentions.reshape(-1, 12, 1) * features, dim=1))

        features = features[:,-1,:]
        outputs = self.feature_layer(features)

        return outputs, log_var, attentions