from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from FeedforwardNN import FeedforwardNN
from torchvision.transforms import ToPILImage
from PIL import ImageOps
from PIL import ImageFilter

# Load the trained model
model = FeedforwardNN()
model.load_state_dict(torch.load("mnist_feedforward_model.pth"))
model.eval()  # Set the model to evaluation mode

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
        transforms.Resize((28, 28), interpolation=Image.NEAREST),  # Resize directly to 28x28
        transforms.ToTensor(),                       # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))         # Normalize to mean 0.5 and std 0.5
    ])
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors
    # Skip sharpening and thresholding to retain more detail
    image = transform(image)
    image = image.view(1, 28*28)  # Flatten the image and add batch dimension
    return image

def preprocess_image_minimal(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
        transforms.Resize((28, 28)),  # Resize directly to 28x28
        transforms.ToTensor(),                       # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))         # Normalize to mean 0.5 and std 0.5
    ])
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors
    image = image.point(lambda x: 255 if x > 200 else 0, '1')  # Set non-black pixels to white
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
image_path = "three.png"  # Replace with the actual path to your image
image_tensor = preprocess_image_minimal(image_path)

save_preprocessed_image(image_tensor, "preprocessed_image.png")

# # Predict the number
with torch.no_grad():
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)

# ==== test from dataset ====
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
#     transforms.Resize((28, 28)), 
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))  # Normalize to mean 0.5 and std 0.5
# ])

# mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# test_image, test_label = mnist_test[3]  # Get the first test image and label
# test_image = test_image.view(1, 28*28)  # Flatten the image

# save_preprocessed_image(test_image, "test.png")


# # Predict the number
# with torch.no_grad():
#     output = model(test_image)
#     _, predicted = torch.max(output, 1)

# ================

print(f"The model predicts the number is: {predicted.item()}")