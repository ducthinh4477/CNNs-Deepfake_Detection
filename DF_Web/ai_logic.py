import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from model import MyNet # Import class từ file model.py
from model2 import CombinedModel # Import CombinedModel từ model2.py
import os


class DeepfakeAI:
    """
    AI Engine for Deepfake Detection
    Separates image processing for model inference vs UI display
    Supports multiple model architectures: MyNet, CombinedModel
    """
    
    # Constants for model input size (used for inference only)
    MODEL_INPUT_SIZE = (224, 224)
    # CombinedModel uses 160x160 input
    COMBINED_MODEL_INPUT_SIZE = (160, 160)
    # Constants for analysis visualization (preserves aspect ratio better)
    ANALYSIS_SIZE = (512, 512)
    
    def __init__(self, model_path, model_type='MyNet'):
        """
        Initialize DeepfakeAI with specified model
        
        Args:
            model_path: Path to the model weights file (.pth)
            model_type: Type of model architecture ('MyNet' or 'CombinedModel')
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model_path = model_path
        
        # Set input size based on model type
        if model_type == 'CombinedModel':
            self.input_size = self.COMBINED_MODEL_INPUT_SIZE
        else:
            self.input_size = self.MODEL_INPUT_SIZE
        
        self.model = self._load_model(model_path, model_type)
        
        # Transform ONLY for model inference - không ảnh hưởng đến UI display
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_model(self, path, model_type='MyNet'):
        """
        Load model based on architecture type
        
        Args:
            path: Path to model weights
            model_type: 'MyNet' or 'CombinedModel'
        """
        # Create model instance based on type
        if model_type == 'CombinedModel':
            model = CombinedModel()
            print(f"🧠 Sử dụng kiến trúc: CombinedModel (EfficientNet + Frequency)")
        else:
            model = MyNet()
            print(f"🧠 Sử dụng kiến trúc: MyNet (Custom CNN)")
        
        abs_path = os.path.abspath(path)
        print(f"🔄 Đang tìm file trọng số (.pth) tại: {abs_path}")
        
        if not os.path.exists(path):
            print(f"❌ LỖI: Không tìm thấy file tại đường dẫn '{path}'")
            print(f"📁 Thư mục code đang chạy tại: {os.getcwd()}")
            try:
                print(f"📄 Các file đang có ở đây: {os.listdir(os.getcwd())}")
            except:
                pass
        else:
            try:
                state_dict = torch.load(path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"✅ Đã nạp Model '{model_type}' thành công!")
            except Exception as e:
                print(f"❌ LỖI MODEL: File tồn tại nhưng nạp thất bại. Chi tiết: {e}")
        
        model.to(self.device)
        model.eval()
        return model

    # ========== IMAGE PROCESSING UTILITIES ==========
    
    def _prepare_for_model(self, pil_image):
        """
        Prepare image for model inference (resize to MODEL_INPUT_SIZE)
        This is ONLY for prediction - NOT for display
        """
        return self.transform(pil_image).unsqueeze(0).to(self.device)
    
    def _prepare_for_analysis(self, pil_image, preserve_aspect=True):
        """
        Prepare image for analysis visualization
        Optionally preserves aspect ratio to prevent distortion on UI
        
        Args:
            pil_image: PIL Image input
            preserve_aspect: If True, pad image to square before resize
            
        Returns:
            numpy array ready for analysis
        """
        img_np = np.array(pil_image)
        
        if preserve_aspect:
            # Pad to square to preserve aspect ratio
            h, w = img_np.shape[:2]
            max_dim = max(h, w)
            
            # Create square canvas
            if len(img_np.shape) == 3:
                canvas = np.zeros((max_dim, max_dim, img_np.shape[2]), dtype=img_np.dtype)
            else:
                canvas = np.zeros((max_dim, max_dim), dtype=img_np.dtype)
            
            # Center the image
            y_offset = (max_dim - h) // 2
            x_offset = (max_dim - w) // 2
            canvas[y_offset:y_offset+h, x_offset:x_offset+w] = img_np
            img_np = canvas
        
        # Resize for analysis
        return cv2.resize(img_np, self.ANALYSIS_SIZE)
    
    def _to_grayscale(self, img_np):
        """Convert numpy image to grayscale"""
        if len(img_np.shape) == 3:
            return cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        return img_np
    
    # ========== PREDICTION ==========

    def predict(self, pil_image):
        """
        Run deepfake prediction on image
        Uses MODEL_INPUT_SIZE internally - does NOT affect original image
        
        Supports both architectures:
        - MyNet: 2-class output with softmax
        - CombinedModel: 1-logit output with sigmoid
        
        Returns:
            pred: 0 (Real) or 1 (Fake)
            conf: Confidence score (0-1)
            img_tensor: Tensor used for prediction
        """
        img_tensor = self._prepare_for_model(pil_image)
        with torch.no_grad():
            output = self.model(img_tensor)
            
            if self.model_type == 'CombinedModel':
                # CombinedModel outputs single logit: sigmoid for probability
                # Output > 0.5 = Fake (1), Output <= 0.5 = Real (0)
                prob = torch.sigmoid(output).item()
                pred = 1 if prob > 0.5 else 0
                conf = prob if pred == 1 else (1 - prob)
            else:
                # MyNet outputs 2 classes: softmax for probabilities
                probs = F.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                pred = pred.item()
                conf = conf.item()
                
        return pred, conf, img_tensor

    # ========== HEATMAP ANALYSIS ==========

    def generate_heatmap(self, pil_image, preserve_aspect=True):
        """
        Generate Grad-CAM (Gradient-weighted Class Activation Mapping) heatmap overlay
        
        Grad-CAM uses gradients flowing into the final convolutional layer (conv4)
        to understand which regions of the image are important for prediction.
        
        Args:
            pil_image: Original PIL Image (not resized)
            preserve_aspect: Preserve aspect ratio for UI display
            
        Returns:
            RGB numpy array of heatmap overlay
        """
        # Prepare original image for visualization (separate from model inference)
        img_cv = self._prepare_for_analysis(pil_image, preserve_aspect)
        
        # Ensure RGB format for overlay
        if len(img_cv.shape) == 2:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
        
        # ========== GRAD-CAM IMPLEMENTATION ==========
        
        # Storage for features and gradients from hooks
        features = []
        gradients = []
        
        def forward_hook(module, input, output):
            """Hook to capture feature maps from target layer"""
            features.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            """Hook to capture gradients flowing back through target layer"""
            gradients.append(grad_output[0])
        
        # Register hooks on the appropriate layer based on model type
        # MyNet: conv4 outputs feature maps of size 14x14 from 224x224 input
        # CombinedModel: use the last conv layer of EfficientNet (conv_branch.features)
        if self.model_type == 'CombinedModel':
            target_layer = self.model.conv_branch.features[-1]
        else:
            target_layer = self.model.conv4
            
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        try:
            # Prepare image tensor for model (requires grad for backprop)
            img_tensor = self._prepare_for_model(pil_image)
            img_tensor.requires_grad_(True)
            
            # Forward pass (without no_grad to allow gradient computation)
            self.model.eval()
            output = self.model(img_tensor)
            
            # Get the score for backward pass
            if self.model_type == 'CombinedModel':
                # CombinedModel: single logit output
                class_score = output[0, 0]
            else:
                # MyNet: 2-class output, use predicted class
                pred_class = output.argmax(dim=1).item()
                class_score = output[0, pred_class]
            
            # Backward pass to compute gradients
            self.model.zero_grad()
            class_score.backward()
            
            # Get the captured features and gradients
            # features[0] shape: (1, 256, 14, 14) - from conv4 output
            # gradients[0] shape: (1, 256, 14, 14) - gradients w.r.t conv4 output
            feature_maps = features[0].detach()
            grads = gradients[0].detach()
            
            # ========== COMPUTE GRAD-CAM ==========
            
            # Global Average Pooling of gradients to get channel weights
            # Shape: (1, 256, 14, 14) -> (1, 256, 1, 1) -> (256,)
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
            
            # Weighted combination of feature maps
            # weights: (1, 256, 1, 1), feature_maps: (1, 256, 14, 14)
            # cam: (1, 256, 14, 14) -> sum over channels -> (1, 1, 14, 14) -> (14, 14)
            cam = torch.sum(weights * feature_maps, dim=1).squeeze()
            
            # Apply ReLU to keep only positive influences
            cam = F.relu(cam)
            
            # Convert to numpy for OpenCV processing
            cam = cam.cpu().numpy()
            
            # Normalize CAM to 0-1 range
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max - cam_min > 1e-8:  # Avoid division by zero
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)
            
            # Resize CAM to match the analysis image size
            cam_resized = cv2.resize(cam, self.ANALYSIS_SIZE)
            
            # Convert to 8-bit for colormap application
            heatmap = np.uint8(255 * cam_resized)
            
            # Apply JET colormap (red = high activation, blue = low)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            
            # Overlay heatmap on original image
            # 0.6 weight for original, 0.4 for heatmap
            overlay = cv2.addWeighted(img_cv, 0.6, heatmap_rgb, 0.4, 0)
            
            return overlay
            
        except Exception as e:
            print(f"⚠️ Grad-CAM generation failed: {e}")
            # Fallback: return original image with a subtle overlay indicating error
            fallback = cv2.addWeighted(img_cv, 0.8, np.zeros_like(img_cv), 0.2, 0)
            return fallback
            
        finally:
            # Always remove hooks to prevent memory leaks
            forward_handle.remove()
            backward_handle.remove()

    # ========== FOURIER FREQUENCY ANALYSIS ==========

    def generate_fourier_analysis(self, pil_image, preserve_aspect=True):
        """
        Generate Fourier Frequency Domain Analysis (Magnitude Spectrum)
        
        Uses numpy.fft to transform image to frequency domain with enhanced
        visualization using percentile-based contrast stretching.
        
        The magnitude spectrum shows frequency distribution:
        - Center = Low frequencies (overall brightness, gradual changes)
        - Edges = High frequencies (sharp edges, fine details, potential artifacts)
        
        Deepfake artifacts often show unusual patterns in high-frequency regions,
        such as grid-like patterns from upsampling or periodic artifacts from GAN.
        
        Args:
            pil_image: Original PIL Image (any size)
            preserve_aspect: Preserve aspect ratio for UI display
            
        Returns:
            RGB numpy array of magnitude spectrum visualization
        """
        try:
            # Prepare image for analysis (NOT for model)
            img_cv = self._prepare_for_analysis(pil_image, preserve_aspect)
            gray = self._to_grayscale(img_cv)
            
            # Validate input size
            if gray.shape[0] < 8 or gray.shape[1] < 8:
                raise ValueError("Image too small for FFT analysis")
            
            # Apply 2D Discrete Fourier Transform using numpy.fft (vectorized)
            f_transform = np.fft.fft2(gray.astype(np.float64))
            
            # Shift zero-frequency component to center
            f_shift = np.fft.fftshift(f_transform)
            
            # Calculate magnitude spectrum with improved log-scale
            # Using log1p for numerical stability: log1p(x) = log(1 + x)
            magnitude = np.abs(f_shift)
            magnitude_spectrum = np.log1p(magnitude)
            
            # ========== PERCENTILE-BASED CONTRAST STRETCHING ==========
            # Instead of simple min-max which is sensitive to outliers,
            # use percentile clipping for robust normalization
            
            # Clip extreme values using 2nd and 98th percentiles
            p_low, p_high = np.percentile(magnitude_spectrum, [2, 98])
            
            # Clip values outside percentile range
            magnitude_clipped = np.clip(magnitude_spectrum, p_low, p_high)
            
            # Normalize clipped values to 0-1 range
            if p_high - p_low > 1e-8:
                magnitude_normalized = (magnitude_clipped - p_low) / (p_high - p_low)
            else:
                magnitude_normalized = np.zeros_like(magnitude_clipped)
            
            # Apply gamma correction for better mid-tone visibility
            # gamma < 1 brightens mid-tones, revealing more detail
            gamma = 0.7
            magnitude_gamma = np.power(magnitude_normalized, gamma)
            
            # Convert to 8-bit
            magnitude_uint8 = np.uint8(255 * magnitude_gamma)
            
            # Apply VIRIDIS colormap for scientific visualization
            fourier_color = cv2.applyColorMap(magnitude_uint8, cv2.COLORMAP_VIRIDIS)
            
            # Convert BGR to RGB for display
            fourier_rgb = cv2.cvtColor(fourier_color, cv2.COLOR_BGR2RGB)
            
            return fourier_rgb
            
        except Exception as e:
            print(f"⚠️ Fourier analysis failed: {e}")
            # Return black image on error
            return np.zeros((*self.ANALYSIS_SIZE, 3), dtype=np.uint8)
    
    def generate_fourier_highpass(self, pil_image, cutoff_freq=30, preserve_aspect=True):
        """
        Generate High-Pass Filtered Fourier Analysis using Gaussian Highpass Filter (GHPF)
        
        Uses the mathematical Gaussian Highpass Filter:
            H(u,v) = 1 - exp(-D²(u,v) / (2 * D₀²))
        
        where D(u,v) is the distance from center and D₀ is the cutoff frequency.
        This avoids ringing artifacts that occur with ideal (sharp cutoff) filters.
        
        High-frequency components reveal:
        - Edge information and fine details
        - Compression artifacts (JPEG blocking)
        - GAN upsampling artifacts (grid patterns)
        - Splicing boundaries
        
        Args:
            pil_image: Original PIL Image
            cutoff_freq: Cutoff frequency D₀ for Gaussian filter (default 30)
                         Lower values = more aggressive filtering (more edges visible)
            preserve_aspect: Preserve aspect ratio for UI display
            
        Returns:
            RGB numpy array of high-pass filtered visualization
        """
        try:
            # Prepare image for analysis
            img_cv = self._prepare_for_analysis(pil_image, preserve_aspect)
            gray = self._to_grayscale(img_cv)
            
            # Validate input size
            if gray.shape[0] < 8 or gray.shape[1] < 8:
                raise ValueError("Image too small for FFT analysis")
            
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            
            # Apply Fourier Transform
            f_transform = np.fft.fft2(gray.astype(np.float64))
            f_shift = np.fft.fftshift(f_transform)
            
            # ========== GAUSSIAN HIGHPASS FILTER (GHPF) ==========
            # Create distance matrix from center using vectorized operations
            # D(u,v) = sqrt((u - M/2)² + (v - N/2)²)
            u = np.arange(rows).reshape(-1, 1) - crow  # Column vector
            v = np.arange(cols).reshape(1, -1) - ccol  # Row vector
            
            # Distance from center (vectorized, no loops)
            D = np.sqrt(u**2 + v**2)
            
            # Gaussian Highpass Filter: H(u,v) = 1 - exp(-D² / (2 * D₀²))
            # This provides smooth transition without ringing artifacts
            D0 = cutoff_freq
            H = 1.0 - np.exp(-(D**2) / (2.0 * (D0**2)))
            
            # Apply high-pass filter in frequency domain
            f_shift_filtered = f_shift * H
            
            # Inverse FFT to get filtered image
            f_ishift = np.fft.ifftshift(f_shift_filtered)
            img_filtered = np.fft.ifft2(f_ishift)
            img_filtered = np.abs(img_filtered)
            
            # ========== ENHANCED NORMALIZATION FOR EDGE VISIBILITY ==========
            # Use percentile-based clipping to handle outliers
            p_low, p_high = np.percentile(img_filtered, [1, 99])
            img_clipped = np.clip(img_filtered, p_low, p_high)
            
            # Normalize to 0-1
            if p_high - p_low > 1e-8:
                img_normalized = (img_clipped - p_low) / (p_high - p_low)
            else:
                img_normalized = np.zeros_like(img_clipped)
            
            # Apply slight gamma correction to enhance subtle edges
            gamma = 0.8
            img_gamma = np.power(img_normalized, gamma)
            
            # Convert to uint8
            img_uint8 = np.uint8(255 * img_gamma)
            
            # Apply INFERNO colormap - better for edge visualization
            # (perceptually uniform, good contrast, easier on eyes than HOT)
            result_color = cv2.applyColorMap(img_uint8, cv2.COLORMAP_INFERNO)
            result_rgb = cv2.cvtColor(result_color, cv2.COLOR_BGR2RGB)
            
            return result_rgb
            
        except Exception as e:
            print(f"⚠️ Fourier highpass analysis failed: {e}")
            # Return black image on error
            return np.zeros((*self.ANALYSIS_SIZE, 3), dtype=np.uint8)
    
    def generate_fourier_phase(self, pil_image, preserve_aspect=True):
        """
        Generate Fourier Phase Spectrum Analysis
        
        Phase information captures structural information of the image.
        Manipulation often disrupts natural phase patterns.
        
        Args:
            pil_image: Original PIL Image
            preserve_aspect: Preserve aspect ratio for UI display
            
        Returns:
            RGB numpy array of phase spectrum visualization
        """
        # Prepare image for analysis
        img_cv = self._prepare_for_analysis(pil_image, preserve_aspect)
        gray = self._to_grayscale(img_cv)
        
        # Apply Fourier Transform
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        
        # Calculate phase spectrum
        phase_spectrum = np.angle(f_shift)
        
        # Normalize from [-π, π] to [0, 255]
        phase_normalized = ((phase_spectrum + np.pi) / (2 * np.pi) * 255)
        phase_normalized = phase_normalized.astype(np.uint8)
        
        # Apply colormap for visualization
        phase_color = cv2.applyColorMap(phase_normalized, cv2.COLORMAP_TWILIGHT)
        phase_rgb = cv2.cvtColor(phase_color, cv2.COLOR_BGR2RGB)
        
        return phase_rgb
    
    def generate_azimuthal_average(self, pil_image, preserve_aspect=True):
        """
        Generate Azimuthal Average of Power Spectrum
        Optimized using Vectorization (np.bincount) instead of For-loop
        """
        # 1. Prepare image
        img_cv = self._prepare_for_analysis(pil_image, preserve_aspect)
        gray = self._to_grayscale(img_cv)
        
        # 2. Compute Power Spectrum
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        # Sử dụng Power Spectrum (Năng lượng) thay vì Magnitude
        power_spectrum = np.abs(f_shift) ** 2
        
        # 3. Compute Azimuthal Average (Optimized logic)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Tạo lưới tọa độ
        y, x = np.ogrid[:rows, :cols]
        # Tính khoảng cách từ tâm (bán kính r)
        r = np.sqrt((x - ccol)**2 + (y - crow)**2).astype(int)
        
        # Dùng bincount để tính tổng năng lượng tại mỗi bán kính r (Thay cho vòng lặp for)
        # weights là giá trị năng lượng tại pixel đó
        tbin = np.bincount(r.ravel(), weights=power_spectrum.ravel())
        nr = np.bincount(r.ravel())
        
        # Tính trung bình: Tổng năng lượng / Số điểm ảnh tại bán kính r
        radial_profile = tbin / (nr + 1e-10)
        
        # 4. Log scale for visualization/features (Log of Average)
        radial_profile_log = np.log(radial_profile + 1)
        
        # --- Visualization part (Giữ nguyên để hiển thị trên web) ---
        spectrum_vis = 20 * np.log(np.abs(f_shift) + 1)
        spectrum_vis = cv2.normalize(spectrum_vis, None, 0, 255, cv2.NORM_MINMAX)
        spectrum_vis = spectrum_vis.astype(np.uint8)
        spectrum_color = cv2.applyColorMap(spectrum_vis, cv2.COLORMAP_MAGMA) # Dùng MAGMA đẹp hơn JET
        spectrum_rgb = cv2.cvtColor(spectrum_color, cv2.COLOR_BGR2RGB)
        
        return {
            'spectrum_image': spectrum_rgb,
            'profile_data': radial_profile_log
        }