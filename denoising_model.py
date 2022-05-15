import torch.nn as nn

class denoising_model(nn.Module):
    def __init__(self):
        super(denoising_model,self).__init__()
        self.encoder= nn.Sequential(
            nn.Linear(28* 28,256),
            nn.ReLU(True),
            nn.Linear(256,128),
            nn.ReLU(True),
            nn.Linear(128,64),
            nn.ReLU(True)

        )
        self.decoder= nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(True),
            nn.Linear(128,256),
            nn.ReLU(True),
            nn.Linear(256,28* 28),
            nn.Sigmoid(),
        )
    def forward(self,x ):
        x= self.encoder(x)
        x= self.decoder(x)
        return x
