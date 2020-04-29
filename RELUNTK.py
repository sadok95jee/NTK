import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

def kernel_test( x, xprime):
    norm_prod = np.linalg.norm(x) * np.linalg.norm(xprime)
    align = np.dot(x , xprime)/norm_prod
    return (norm_prod * ( align/np.pi * (np.pi - np.arccos(align) ) + 1/np.pi * (align * ( np.arccos(-align) ) + np.sqrt(1-align**2) )) )
class NTKtest(nn.Module):
    def __init__(self, nin , nhidden , nout ):
        super(NTKtest, self).__init__()
        self.fc1 = nn.Linear(nin , nhidden , bias= False)
        self.fc1.weight  = nn.Parameter(torch.randn(nhidden,nin))
        self.fc2 = nn.Linear(nhidden , nout , bias= False)
        self.fc2.weight = nn.Parameter(torch.randn(nout,nhidden))

    def forward(self , x , nhidden ):
        x = self.fc1(x)
        x = F.relu(x)

        return self.fc2(x)* torch.sqrt(torch.tensor(2/nhidden))


    def Empiricaldynamique(self,x, xprime , nhidden):
        output = self.forward(x , nhidden , )
        outputprime = self.forward(xprime , nhidden)
        
        xgrad = ( torch.autograd.grad(output , self.parameters() )   )
        xprimegrad = ( torch.autograd.grad(outputprime , self.parameters()   ) )

        Tangent_dynamic = torch.tensor([0] , dtype = torch.float)
        for i in range(len(xgrad)):
            Tangent_dynamic += torch.sum(xgrad[i]*xprimegrad[i])
        
        return Tangent_dynamic

        #return torch.dot(torch.autograd.grad(self.forward(x , nhidden) , x) , torch.autograd.grad(self.forward(xprime , nhidden) ,xprime))
                
def train( model, train_data, optimizer, target , nhidden ):
    model.train()
    optimizer.zero_grad()
    output = model.forward(train_data , nhidden )
    loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    return loss


       

