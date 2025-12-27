import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np

# ==========================================
# MODEL ARCHITECTURE FOR INFERENCE
# ==========================================

class FrequencyBranch(nn.Module):
    def __init__(self):
        super(FrequencyBranch, self).__init__()
        # Input size: 160x160 -> sau FFT vẫn giữ nguyên kích thước không gian
        # Tuy nhiên ta chỉ lấy Magnitude spectrum
        
        # Simple CNN cho Frequency Domain
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 160 -> 80
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 80 -> 40
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # Global Average Pooling
            nn.Flatten()
        )
        
    def forward(self, x):
        # x shape: [B, 3, H, W]
        # Chuyển sang Grayscale: [B, 1, H, W]
        # Công thức RGB to Gray chuẩn: 0.299R + 0.587G + 0.114B
        x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        x_gray = x_gray.unsqueeze(1)
        
        # FFT transform
        # fft2 trả về số phức, ta lấy magnitude (abs)
        # Cộng 1e-8 để tránh log(0)
        fft = torch.fft.fft2(x_gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.log(torch.abs(fft_shift) + 1e-8)
        
        # Đưa qua CNN
        out = self.conv_block(magnitude)
        return out

class CombinedModel(nn.Module):
    def __init__(self, num_classes=1): # Sigmoid output cho Binary Classification
        super(CombinedModel, self).__init__()
        
        # 1. RGB Branch (EfficientNet-B0)
        # Sử dụng weights thay cho pretrained=True (cũ)
        weights = EfficientNet_B0_Weights.DEFAULT
        self.rgb_branch = efficientnet_b0(weights=weights)
        
        # Bỏ lớp classifier cuối cùng của EfficientNet
        # EfficientNet-B0 features dim = 1280
        self.conv_branch = nn.Sequential(
            *list(self.rgb_branch.children())[:-1], 
            nn.Flatten()
        )
        self.rgb_dim = 1280
        
        # 2. Frequency Branch
        self.freq_branch = FrequencyBranch()
        self.freq_dim = 128 # Output từ conv_block
        
        # 3. Fusion & Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.rgb_dim + self.freq_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, num_classes)
            # Không có Sigmoid/Softmax ở đây vì dùng BCEWithLogitsLoss khi train
            # Khi predict sẽ thêm Sigmoid sau
        )
        
    def forward(self, x):
        rgb_feat = self.conv_branch(x)
        freq_feat = self.freq_branch(x)
        
        # Concatenate features
        combined = torch.cat((rgb_feat, freq_feat), dim=1)
        
        out = self.classifier(combined)
        return out

# Nếu cần chạy test thử file này độc lập thì mới dùng đoạn dưới
if __name__ == "__main__":
    print("Testing CombinedModel architecture...")
    model = CombinedModel()
    dummy_input = torch.randn(1, 3, 160, 160)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")