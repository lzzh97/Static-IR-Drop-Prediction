import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from model import AttUNet  # Assuming model.py defines this
from DataLoad_normalization import load_real, load_fake, load_real_original_size
from sklearn.metrics import f1_score
from model import VCAttUNet
from skimage.transform import resize


# Argument parser to handle --model argument
parser = argparse.ArgumentParser(description='Evaluate AttUNet model for static IR drop prediction')
parser.add_argument('--model', type=str, required=True, help='Path to the model to evaluate')
args = parser.parse_args()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
datapath_test = '../data/hidden-real-circuit-data-plus/'

# Load the test data (original size)
dataset_test_original_size = load_real_original_size(datapath_test, mode='train', testcase=[])
dataloader_test_original_size = torch.utils.data.DataLoader(dataset=dataset_test_original_size, batch_size=1, shuffle=False)


# Resize function using skimage.transform.resize
def resize_image(image, target_size=(512, 512)):
    # Assuming `image` is a numpy array (height, width)
    resized_image = resize(image, target_size, preserve_range=True, anti_aliasing=True)
    return resized_image

# Function to resize the output back to original size
def resize_output_to_original(output, original_size):
    # Resize the 512x512 output back to the original size for comparison
    return resize(output, original_size, preserve_range=True, anti_aliasing=True)

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    total_mae = 0
    total_f1_score = 0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Inputs and targets are assumed to be of original size
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get the original size of the input/target
            original_size = inputs.shape[-2:]  # Height, Width of the original image
            
            # Resize input to 512x512
            inputs_resized = torch.from_numpy(resize_image(inputs.cpu().numpy(), (512, 512))).float().to(device)
            
            # Model prediction (512x512)
            outputs_resized = model(inputs_resized)
            
            # Resize output back to the original size
            outputs_original_size = torch.from_numpy(resize_output_to_original(outputs_resized.cpu().numpy(), original_size)).float().to(device)
            
            # Compute MAE (Mean Absolute Error)
            mae = torch.mean(torch.abs(outputs_original_size - targets))
            total_mae += mae.item()

            # Compute F1 score (thresholding the outputs)
            output_thresholded = (outputs_original_size > 0.5).float()  # Adjust threshold as per need
            target_thresholded = (targets > 0.5).float()

            output_flat = output_thresholded.cpu().numpy().flatten()
            target_flat = target_thresholded.cpu().numpy().flatten()

            f1 = f1_score(target_flat, output_flat)
            total_f1_score += f1

            num_samples += 1

            print(f'Batch {batch_idx + 1}/{len(dataloader)} - MAE: {mae:.4f}, F1: {f1:.4f}')

    # Calculate average metrics
    avg_mae = total_mae / num_samples
    avg_f1 = total_f1_score / num_samples

    print(f'\nAverage MAE: {avg_mae:.4f}')
    print(f'Average F1 Score: {avg_f1:.4f}')

# Main evaluation script
def main():
    # Load the trained model from the provided path
    model_path = args.model
    model = AttUNet(dropout_rate=0.1)  # Adjust dropout as per trained model
    
    try:
        model.load_state_dict(torch.load(model_path))  # Load the model
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model = model.to(device)

    # Start evaluation
    print("Starting evaluation...")
    evaluate_model(model, dataloader_test_original_size)

if __name__ == '__main__':
    main()
