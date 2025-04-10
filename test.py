from PIL import Image
import torch
import torchvision.transforms as transforms
from FeedforwardNN import FeedforwardNN
from torchvision.transforms import ToPILImage

# Load the trained model
model = FeedforwardNN()
model.load_state_dict(torch.load("mnist_feedforward_model.pth"))
model.eval()  # Set the model to evaluation mode

# Preprocess the new image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
        transforms.Resize((28, 28)),                 # Resize to 28x28 pixels
        transforms.ToTensor()                        # Convert to tensor (no normalization)
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.view(1, 28*28)  # Flatten the image and add batch dimension
    return image

# Save the preprocessed image
def save_preprocessed_image(image_tensor, save_path="preprocessed_image.png"):
    to_pil = ToPILImage()
    image = image_tensor.view(1, 28, 28)  # Reshape to (1, 28, 28) for single-channel image
    image = to_pil(image)  # Convert tensor to PIL image
    image.save(save_path)
    print(f"Preprocessed image saved to {save_path}")

# Path to the new image
image_path = "one.png"  # Replace with the actual path to your image
image_tensor = preprocess_image(image_path)

save_preprocessed_image(image_tensor, "preprocessed_image.png")

# Predict the number
with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)

print(f"The model predicts the number is: {predicted.item()}")