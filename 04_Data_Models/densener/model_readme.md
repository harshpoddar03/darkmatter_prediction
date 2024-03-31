# Tested Models

## DenseNet201  (240329)

- Data: Augmented Data
- Preprocessing: None
- Model: DenseNet201
- Epochs: 10

- Train Accuracy:
- Validation Accuracy: 

```python

class DenseNet201(nn.Module):
    
    def __init__(self, n_classes):
        super(DenseNet201, self).__init__()
        self.transfer_learning_model = timm.create_model("densenet201", pretrained=True, in_chans=1)
        
        for param in self.transfer_learning_model.parameters():
            param.requires_grad = True
            
        self.classifier = nn.Sequential(
            nn.Linear(1920 * 4 * 4, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.33),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.33),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, x):
        x = self.transfer_learning_model.forward_features(x)
#         print(x.shape)
        x = x.view(-1, 1920 * 4 * 4)
        x = self.classifier(x)
        
        return x


```