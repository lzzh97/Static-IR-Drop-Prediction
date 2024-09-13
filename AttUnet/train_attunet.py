import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from model import AttUNet  # Assuming model.py defines this
from DataLoad_normalization import load_real, load_fake, load_real_original_size

# Argument parser to handle --phase argument
parser = argparse.ArgumentParser(description='Train AttUNet for static IR drop prediction')
parser.add_argument('--phase', type=str, choices=['pretrain', 'finetune'], required=True, help='Phase of training: pretrain or finetune')
args = parser.parseArgs()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
batch_size = 8  # Defined based on your provided dataloaders
num_epochs_pretrain = 450
num_epochs_finetune = 600
learning_rate = 0.0005
learning_rate_min = 0.00001
dropout_pretrain = 0.3
dropout_finetune = 0.1
lambda_custom_loss = 2

# Custom loss function as described in the paper
class CustomLoss(nn.Module):
    def __init__(self, lambda_value=2):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        loss = torch.where(pred >= target, self.mse_loss(pred, target), self.lambda_value * self.mse_loss(pred, target))
        return loss.mean()

# Define model, optimizer, scheduler, and loss function
def build_model(dropout_rate, phase):
    model = AttUNet(dropout_rate=dropout_rate)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Scheduler only for finetune phase
    if phase == 'finetune':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs_finetune, eta_min=learning_rate_min)
    else:
        scheduler = None  # No scheduler for pretrain (constant learning rate)
    
    criterion = CustomLoss(lambda_value=lambda_custom_loss)
    return model, optimizer, scheduler, criterion

# Load data based on phase (pretrain or finetune)
def load_data(phase):
    if phase == 'pretrain':
        datapath_fake = '../data/fake-circuit-data-plus/'
        dataset_fake = load_fake(datapath_fake)
        dataloader_fake = torch.utils.data.DataLoader(dataset=dataset_fake, batch_size=batch_size, shuffle=True)
        return dataloader_fake

    elif phase == 'finetune':
        datapath_real = '../data/real-circuit-data-plus/'
        dataset_real = load_real(datapath_real, mode='train', testcase=[])
        dataloader_real = torch.utils.data.DataLoader(dataset=dataset_real, batch_size=batch_size, shuffle=True)
        return dataloader_real

# Training function
def train_model(model, optimizer, scheduler, criterion, train_loader, num_epochs, phase):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
        
        # Only step scheduler for finetune phase
        if scheduler:
            scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

# Main function to handle pretraining and fine-tuning
def main():
    if args.phase == 'pretrain':
        print("Starting pretraining...")
        model, optimizer, scheduler, criterion = build_model(dropout_pretrain, 'pretrain')
        train_loader = load_data('pretrain')
        model = train_model(model, optimizer, scheduler, criterion, train_loader, num_epochs_pretrain, 'pretrain')
        # Save the pretrained model
        torch.save(model.state_dict(), 'attunet_pretrained.pth')

    elif args.phase == 'finetune':
        print("Starting fine-tuning...")
        model, optimizer, scheduler, criterion = build_model(dropout_finetune, 'finetune')
        model.load_state_dict(torch.load('attunet_pretrained.pth'))  # Load pretrained model
        train_loader = load_data('finetune')
        model = train_model(model, optimizer, scheduler, criterion, train_loader, num_epochs_finetune, 'finetune')
        # Save the fine-tuned model
        torch.save(model.state_dict(), 'attunet_finetuned.pth')

if __name__ == '__main__':
    main()
