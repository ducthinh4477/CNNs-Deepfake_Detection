import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# ==========================================
# 1. CẤU HÌNH & ĐƯỜNG DẪN
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Tự động phát hiện môi trường Kaggle hay Local/Colab
if os.path.exists('/kaggle/input/dire-deepfake-dataset/dire/train'):
    BASE_DIR = '/kaggle/input/dire-deepfake-dataset/dire'
    TRAIN_DIR = os.path.join(BASE_DIR, 'train')
    VAL_DIR   = os.path.join(BASE_DIR, 'val')
    OUTPUT_DIR = '/kaggle/working'
else:
    # Nếu chạy Colab, hãy thay đường dẫn này
    TRAIN_DIR = '/content/drive/MyDrive/Dataset/train'
    VAL_DIR   = '/content/drive/MyDrive/Dataset/val'
    OUTPUT_DIR = './'

BATCH_SIZE = 64
EPOCHS = 10
IMG_SIZE = 160

# ==========================================
# 2. DATASET (XỬ LÝ CẤU TRÚC LỒNG NHAU)
# ==========================================
class DIREDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] 
        
        if not os.path.exists(root_dir):
            print(f"Error: Path not found {root_dir}")
            return

        print(f"Scanning {root_dir}...")
        # Cấu trúc: Dataset (LSUN/CelebA) -> Method (Real/ADM/StyleGAN...) -> Image
        for dataset_name in os.listdir(root_dir):
            ds_path = os.path.join(root_dir, dataset_name)
            if not os.path.isdir(ds_path): continue
            
            for method_name in os.listdir(ds_path):
                method_path = os.path.join(ds_path, method_name)
                if not os.path.isdir(method_path): continue
                
                # Label: Real = 0.0, Fake = 1.0
                label = 0.0 if method_name.lower() == 'real' else 1.0
                
                for img_name in os.listdir(method_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(method_path, img_name), label))
        
        print(f"Loaded {len(self.samples)} images.")
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.float32)
        except:
            return self.__getitem__((idx + 1) % len(self))

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
class FrequencyBranch(nn.Module):
    def __init__(self, output_size=128):
        super(FrequencyBranch, self).__init__()
        # Input: 3 channels * 2 (Amp+Phase) * H * W
        self.flat_features = 3 * IMG_SIZE * IMG_SIZE * 2
        
        self.net = nn.Sequential(
            nn.Linear(self.flat_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, img):
        # FFT transform
        f_transform = torch.fft.fft2(img)
        f_transform_shifted = torch.fft.fftshift(f_transform)
        # Concatenate Amplitude & Phase
        features = torch.cat((torch.abs(f_transform_shifted), torch.angle(f_transform_shifted)), dim=1)
        # Flatten
        features = features.view(features.size(0), -1) 
        return self.net(features)

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.freq_branch = FrequencyBranch()
        
        self.conv_branch = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Thay classifier cuối của EfficientNet
        self.conv_branch.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 128)
        )
        
        # Kết hợp
        self.fusion = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1) # Output 1 giá trị (Logit)
        )

    def forward(self, x):
        freq = self.freq_branch(x)
        conv = self.conv_branch(x)
        combined = torch.cat((freq, conv), dim=1)
        return self.fusion(combined)

# ==========================================
# 4. TRAINING & VALIDATION LOOP
# ==========================================
def run_training():
    # 1. Setup Data
    train_dataset = DIREDataset(TRAIN_DIR, transform=transform)
    valid_dataset = DIREDataset(VAL_DIR, transform=transform)
    
    if len(train_dataset) == 0: return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Setup Model
    model = CombinedModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    best_val_acc = 0.0
    print("\n STARTING TRAINING...")

    for epoch in range(EPOCHS):
        # --- TRAIN PHASE ---
        model.train()
        train_loss = 0
        train_correct = 0
        total_train = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.unsqueeze(1) # Shape (N, 1)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == targets).sum().item()
            total_train += targets.size(0)
            
            pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / total_train

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Valid]", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.unsqueeze(1)
                
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == targets).sum().item()
                total_val += targets.size(0)

        avg_val_loss = val_loss / len(valid_loader)
        val_acc = val_correct / total_val

        # --- LOGGING & SAVING ---
        print(f"Epoch {epoch+1}: "
              f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")

        # Chỉ lưu model tốt nhất (dựa trên Val Accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"New Best Model Saved! (Acc: {best_val_acc:.4f})")
            
        # Lưu checkpoint mỗi epoch (tuỳ chọn)
        # torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'last_model.pth'))

if __name__ == "__main__":
    run_training()