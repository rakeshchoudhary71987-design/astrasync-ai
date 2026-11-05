import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline

# =====================================================================
# 1. METRICS & RIGOROUS TESTING FUNCTIONS
# =====================================================================

def generate_fractal_surface(shape=(128, 128), roughness=0.5):
    """Generates a non-smooth, rugged fractal terrain mimicking Mars topography."""
    w, h = shape
    terrain = np.zeros(shape)
    
    # Generate pink/fractal noise using a spectral power law 1/f^beta
    fx = np.fft.fftfreq(w).reshape(-1, 1)
    fy = np.fft.fftfreq(h).reshape(1, -1)
    f_norm = np.sqrt(fx**2 + fy**2)
    f_norm[0, 0] = 1 # Prevent division by zero
    
    beta = 1.5 + roughness  # Controls roughness slope
    spectral_filter = 1.0 / (f_norm ** beta)
    spectral_filter[0, 0] = 0
    
    white_noise = np.random.normal(size=shape)
    noise_fft = np.fft.fft2(white_noise) * spectral_filter
    terrain = np.real(np.fft.ifft2(noise_fft))
    
    # Normalize between 0 and 1
    return (terrain - np.min(terrain)) / (np.max(terrain) - np.min(terrain))

def calculate_metrics(original, reconstructed):
    """Calculates rigorous reconstruction error metrics."""
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((original - reconstructed) ** 2))
    
    # Peak Signal-to-Noise Ratio (PSNR)
    max_val = np.max(original)
    if rmse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(max_val / rmse)
        
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(original - reconstructed))
    
    return {"RMSE": rmse, "PSNR (dB)": psnr, "MAE": mae}

# =====================================================================
# 2. YOUR ORIGINAL CORE LOGIC (Maintained for structural testing)
# =====================================================================

class AstroSyncFlightEncoder:
    def __init__(self, sampling_step=4, bits=8):
        self.sampling_step = sampling_step
        self.bits = bits
        
    def encode_telemetry(self, raw_sensor_matrix):
        smoothed = gaussian_filter(raw_sensor_matrix, sigma=self.sampling_step / 2.0)
        skeleton = smoothed[::self.sampling_step, ::self.sampling_step]
        
        self.min_val, self.max_val = np.min(skeleton), np.max(skeleton)
        if self.max_val == self.min_val:
            quantized = np.zeros_like(skeleton, dtype=np.uint8)
        else:
            normalized = (skeleton - self.min_val) / (self.max_val - self.min_val)
            quantized = (normalized * ((2**self.bits) - 1)).astype(np.uint8)
            
        return quantized.ravel(), (self.min_val, self.max_val, skeleton.shape)

class AstroSyncFlightDecoder:
    def __init__(self, target_shape, sampling_step=4):
        self.target_shape = target_shape
        self.sampling_step = sampling_step
        self.full_y = np.arange(target_shape[0])
        self.full_x = np.arange(target_shape[1])

    def decode_and_reconstruct(self, payload_stream, metadata, missing_mask=None):
        min_val, max_val, skel_shape = metadata
        float_skel = payload_stream.astype(np.float64)
        if max_val != min_val:
            denom = payload_stream.max() if payload_stream.max() > 0 else 1
            float_skel = (float_skel / denom) * (max_val - min_val) + min_val
        
        skel_matrix = float_skel.reshape(skel_shape)
        
        if missing_mask is not None:
            bool_mask = missing_mask.reshape(skel_shape)
            if not np.all(bool_mask):
                padded = np.pad(skel_matrix, 1, mode='edge')
                avg_neighbors = (padded[:-2, 1:-1] + padded[2:, 1:-1] + 
                                 padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
                skel_matrix[~bool_mask] = avg_neighbors[~bool_mask]

        y_skel = np.arange(0, skel_shape[0] * self.sampling_step, self.sampling_step)
        x_skel = np.arange(0, skel_shape[1] * self.sampling_step, self.sampling_step)
        
        spline = RectBivariateSpline(y_skel, x_skel, skel_matrix, kx=3, ky=3)
        return spline(self.full_y, self.full_x)

# =====================================================================
# 3. ROBUST SIMULATION EXECUTION
# =====================================================================

if __name__ == "__main__":
    np.random.seed(42) # Replicable testing constraint
    matrix_shape = (128, 128)
    
    # Test 1: Real Rugged Fractal Martian Surface Simulation
    print("Generating non-smooth rugged fractal terrain...")
    true_surface = generate_fractal_surface(shape=matrix_shape, roughness=0.7)
    
    # Test 2: Add Real-World High-Frequency Sensor Noise
    noise_sigma = 0.03 
    sensor_noise = np.random.normal(0, noise_sigma, size=matrix_shape)
    raw_space_data = true_surface + sensor_noise
    
    # Pipeline Execution
    encoder = AstroSyncFlightEncoder(sampling_step=4, bits=8)
    decoder = AstroSyncFlightDecoder(target_shape=matrix_shape, sampling_step=4)

    payload, meta = encoder.encode_telemetry(raw_space_data)
    
    # AX.25 Frame loss packet corruption 
    packet_size = 16  
    num_packets = int(np.ceil(len(payload) / packet_size))
    packet_mask = np.random.choice([True, False], size=num_packets, p=[0.90, 0.10])
    mask = np.repeat(packet_mask, packet_size)[:len(payload)]
    
    corrupted_payload = payload.copy()
    corrupted_payload[~mask] = 0  

    # De-quantization and Reconstruction
    reconstructed_data = decoder.decode_and_reconstruct(corrupted_payload, meta, missing_mask=mask)

    # Metrics evaluation against the absolute ground truth surface
    performance = calculate_metrics(true_surface, reconstructed_data)
    
    # Bandwidth calculation
    original_size_bits = raw_space_data.size * 64
    transmitted_size_bits = len(payload[mask]) * 8  

    print("\n--- MARS SURFACE SIMULATION RECONSTRUCTION REPORT ---")
    print(f"Terrain Profile                : Rugged Fractal Non-Smooth Surface")
    print(f"Sensor Noise Level             : Gaussian (σ = {noise_sigma})")
    print(f"Packet Loss Rate               : 10% AX.25 Frame Failure")
    print(f"Bandwidth Saved                : {100 - (transmitted_size_bits / original_size_bits)*100:.2f}%")
    print("-" * 52)
    print("--- RECONSTRUCTION ERROR METRICS ---")
    for metric, value in performance.items():
        print(f"{metric:<30} : {value:.5f}")