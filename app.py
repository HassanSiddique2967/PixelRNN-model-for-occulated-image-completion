import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn


# ConvLSTM Cell Definition
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, hidden_dim * 4, kernel_size, padding=padding)

    def forward(self, x, hidden):
        h_cur, c_cur = hidden

        combined = torch.cat([x, h_cur], dim=1)
        conv_output = self.conv(combined)

        i, f, o, g = torch.chunk(conv_output, 4, dim=1)

        i = torch.sigmoid(i)      # Input gate
        f = torch.sigmoid(f)      # Forget gate
        o = torch.sigmoid(o)      # Output gate
        g = torch.tanh(g)         # Cell state

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# Optimized PixelRNN Model Definition
class OptimizedPixelRNN(nn.Module):
    def __init__(self):
        super(OptimizedPixelRNN, self).__init__()

        # Encoder: Deeper for better feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),  # [B, 64, 128, 128]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, 64, 64]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # [B, 256, 32, 32]
            nn.ReLU()
        )

        # ConvLSTM
        self.convlstm = ConvLSTMCell(input_dim=256, hidden_dim=256, kernel_size=3)

        # Decoder: PixelShuffle for smoother upscaling
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.PixelShuffle(2),  # [B, 128, 64, 64]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.PixelShuffle(2),  # [B, 64, 128, 128]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.PixelShuffle(2),  # [B, 32, 256, 256]
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Ensure output is in range [0, 1]
        )

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        # Encoding
        x = self.encoder(x)  # [B, 256, 32, 32]

        # Initialize hidden and cell states
        h, c = (torch.zeros(batch_size, 256, 32, 32, device=device),
                torch.zeros(batch_size, 256, 32, 32, device=device))

        # ConvLSTM step
        h, c = self.convlstm(x, (h, c))

        # Decoding
        x = self.decoder(h)  # [B, 3, 256, 256]
        return x


# Streamlit interface setup
st.title("PixelRNN Image Completion")

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OptimizedPixelRNN().to(device)
model.load_state_dict(torch.load("optimized_pixelrnn.pth", map_location=device))
model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Function to process and display images
def process_and_display_image(uploaded_image):
    # Open the uploaded image
    image = Image.open(uploaded_image).convert("RGB")
    occluded_image = transform(image).unsqueeze(0).to(device)

    # Generate reconstructed image
    with torch.no_grad():
        reconstructed_image = model(occluded_image)

    # Clamp values to [0, 1]
    reconstructed_image = torch.clamp(reconstructed_image, 0, 1)
    
    # Move tensors to CPU for visualization
    occluded_image = occluded_image.cpu()
    reconstructed_image = reconstructed_image.cpu()

    # Display results
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # Display the occluded image
    axs[0].imshow(occluded_image.squeeze(0).permute(1, 2, 0))
    axs[0].set_title("Occluded Image")
    axs[0].axis("off")

    # Display the reconstructed image
    axs[1].imshow(reconstructed_image.squeeze(0).permute(1, 2, 0))
    axs[1].set_title("Reconstructed Image")
    axs[1].axis("off")

    # Show the plot
    st.pyplot(fig)

# Upload image section
uploaded_image = st.file_uploader("Upload an Occluded Image", type=["jpg", "jpeg", "png"])

# When an image is uploaded
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.write("Generating the reconstructed image...")

    # Process and display the uploaded image
    process_and_display_image(uploaded_image)
