import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
from model import MyNet # Import class từ file model.py
import os


class DeepfakeAI:
    """
    AI Engine for Deepfake Detection
    Separates image processing for model inference vs UI display
    """
    
    # Constants for model input size (used for inference only)
    MODEL_INPUT_SIZE = (224, 224)
    # Constants for analysis visualization (preserves aspect ratio better)
    ANALYSIS_SIZE = (512, 512)
    
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        # Transform ONLY for model inference - không ảnh hưởng đến UI display
        self.transform = transforms.Compose([
            transforms.Resize(self.MODEL_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_model(self, path):
        model = MyNet()
        
        abs_path = os.path.abspath(path)
        print(f"🔄 Đang tìm file trọng số (.pth) tại: {abs_path}")
        
        if not os.path.exists(path):
            print(f"LỖI TO: Không tìm thấy file tại đường dẫn '{path}'")
            print(f"Thư mục code đang chạy tại: {os.getcwd()}")
            try:
                print(f"📄 Các file đang có ở đây: {os.listdir(os.getcwd())}")
            except:
                pass
        else:
            try:
                state_dict = torch.load(path, map_location=self.device)
                model.load_state_dict(state_dict)
                print("Đã nạp Model thành công!")
            except Exception as e:
                print(f"LỖI MODEL: File tồn tại nhưng nạp thất bại. Chi tiết: {e}")
        
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
        
        Returns:
            pred: 0 (Real) or 1 (Fake)
            conf: Confidence score (0-1)
            img_tensor: Tensor used for prediction
        """
        img_tensor = self._prepare_for_model(pil_image)
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
        return pred.item(), conf.item(), img_tensor

    # ========== HEATMAP ANALYSIS ==========

    def generate_heatmap(self, pil_image, preserve_aspect=True):
        """
        Generate Grad-CAM style heatmap overlay
        
        Args:
            pil_image: Original PIL Image (not resized)
            preserve_aspect: Preserve aspect ratio for UI display
            
        Returns:
            RGB numpy array of heatmap overlay
        """
        # Prepare for analysis (separate from model inference)
        img_cv = self._prepare_for_analysis(pil_image, preserve_aspect)
        
        # Ensure RGB format
        if len(img_cv.shape) == 2:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2RGB)
        
        # Generate heatmap (Placeholder - implement Grad-CAM for real use)
        # TODO: Implement actual Grad-CAM based on model architecture
        heatmap = np.random.randint(0, 255, self.ANALYSIS_SIZE, dtype=np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        overlay = cv2.addWeighted(img_cv, 0.6, heatmap_rgb, 0.4, 0)
        return overlay

    # ========== FOURIER FREQUENCY ANALYSIS ==========

    def generate_fourier_analysis(self, pil_image, preserve_aspect=True):
        """
        Generate Fourier Frequency Domain Analysis (Magnitude Spectrum)
        
        Uses numpy.fft to transform image to frequency domain.
        The magnitude spectrum shows frequency distribution:
        - Center = Low frequencies (overall brightness, gradual changes)
        - Edges = High frequencies (sharp edges, fine details, potential artifacts)
        
        Deepfake artifacts often show unusual patterns in high-frequency regions.
        
        Args:
            pil_image: Original PIL Image (any size)
            preserve_aspect: Preserve aspect ratio for UI display
            
        Returns:
            RGB numpy array of magnitude spectrum visualization
        """
        # Prepare image for analysis (NOT for model)
        img_cv = self._prepare_for_analysis(pil_image, preserve_aspect)
        gray = self._to_grayscale(img_cv)
        
        # Apply 2D Discrete Fourier Transform using numpy.fft
        f_transform = np.fft.fft2(gray.astype(np.float32))
        
        # Shift zero-frequency component to center
        f_shift = np.fft.fftshift(f_transform)
        
        # Calculate magnitude spectrum (log scale for better visualization)
        # Adding 1 to avoid log(0)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        # Normalize to 0-255 range for display
        magnitude_spectrum = cv2.normalize(
            magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX
        )
        magnitude_spectrum = magnitude_spectrum.astype(np.uint8)
        
        # Apply VIRIDIS colormap for scientific visualization
        fourier_color = cv2.applyColorMap(magnitude_spectrum, cv2.COLORMAP_VIRIDIS)
        
        # Convert BGR to RGB for Streamlit display
        fourier_rgb = cv2.cvtColor(fourier_color, cv2.COLOR_BGR2RGB)
        
        return fourier_rgb
    
    def generate_fourier_highpass(self, pil_image, cutoff_radius=30, preserve_aspect=True):
        """
        Generate High-Pass Filtered Fourier Analysis
        
        Suppresses low frequencies and highlights high-frequency components
        that may indicate manipulation artifacts (compression, splicing, etc.)
        
        Args:
            pil_image: Original PIL Image
            cutoff_radius: Radius of low-frequency suppression (default 30)
            preserve_aspect: Preserve aspect ratio for UI display
            
        Returns:
            RGB numpy array of high-pass filtered visualization
        """
        # Prepare image for analysis
        img_cv = self._prepare_for_analysis(pil_image, preserve_aspect)
        gray = self._to_grayscale(img_cv)
        
        # Apply Fourier Transform
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        
        # Create high-pass filter mask (suppress center = low frequencies)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create circular mask
        mask = np.ones((rows, cols), dtype=np.float32)
        y, x = np.ogrid[:rows, :cols]
        distance_from_center = np.sqrt((x - ccol)**2 + (y - crow)**2)
        mask[distance_from_center <= cutoff_radius] = 0
        
        # Apply Gaussian smoothing to mask edge for smoother transition
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        # Apply high-pass filter
        f_shift_filtered = f_shift * mask
        
        # Inverse FFT to get filtered image
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_filtered = np.fft.ifft2(f_ishift)
        img_filtered = np.abs(img_filtered)
        
        # Normalize and convert to uint8
        img_filtered = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX)
        img_filtered = img_filtered.astype(np.uint8)
        
        # Apply HOT colormap to highlight anomalies
        result_color = cv2.applyColorMap(img_filtered, cv2.COLORMAP_HOT)
        result_rgb = cv2.cvtColor(result_color, cv2.COLOR_BGR2RGB)
        
        return result_rgb
    
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
        
        Computes radially averaged power spectrum - useful for detecting
        GAN-generated images which often show characteristic frequency patterns.
        
        Args:
            pil_image: Original PIL Image
            preserve_aspect: Preserve aspect ratio
            
        Returns:
            dict with 'spectrum_image' (RGB array) and 'profile_data' (1D array)
        """
        # Prepare image
        img_cv = self._prepare_for_analysis(pil_image, preserve_aspect)
        gray = self._to_grayscale(img_cv)
        
        # Compute power spectrum
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shift = np.fft.fftshift(f_transform)
        power_spectrum = np.abs(f_shift) ** 2
        
        # Compute azimuthal average
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create radial coordinate array
        y, x = np.ogrid[:rows, :cols]
        r = np.sqrt((x - ccol)**2 + (y - crow)**2).astype(int)
        
        # Compute radial profile
        max_r = min(crow, ccol)
        radial_profile = np.zeros(max_r)
        for i in range(max_r):
            mask = (r == i)
            if mask.sum() > 0:
                radial_profile[i] = power_spectrum[mask].mean()
        
        # Log scale for better visualization
        radial_profile_log = np.log(radial_profile + 1)
        
        # Create visualization image
        spectrum_vis = 20 * np.log(np.abs(f_shift) + 1)
        spectrum_vis = cv2.normalize(spectrum_vis, None, 0, 255, cv2.NORM_MINMAX)
        spectrum_vis = spectrum_vis.astype(np.uint8)
        spectrum_color = cv2.applyColorMap(spectrum_vis, cv2.COLORMAP_MAGMA)
        spectrum_rgb = cv2.cvtColor(spectrum_color, cv2.COLOR_BGR2RGB)
        
        return {
            'spectrum_image': spectrum_rgb,
            'profile_data': radial_profile_log
        }