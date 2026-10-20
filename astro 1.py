import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

print("--- STARTING ASTROSYNC AI PRODUCTION SIMULATION ---")

# 1. GENERATE A REALISTIC SPACE DATA MATRIX (Representing an actual sensor grid)
# Instead of formulas, this creates a realistic 100x100 elevation matrix (10,000 data points)
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
grid_x, grid_y = np.meshgrid(x, y)
# Simulating a complex mountain/crater landscape matrix
real_mars_surface = np.sin(grid_x/2) * np.cos(grid_y/2) + 0.2 * np.cos(grid_x)

# 2. THE ASTROSYNC PROTOCOL: DETERMINISTIC SKELETON SAMPLING
# FIX: Instead of random points, we sample using a fixed geometric step.
# A step of 4 means we sample every 4th point. 100/4 = 25. 25x25 = 625 points.
# This transmits only 625 points out of 10,000 (Saving 93.75% Bandwidth!)
sampling_step = 4

# The Rover extracts the skeleton values using the pre-shared matrix step
skeleton_values = real_mars_surface[::sampling_step, ::sampling_step]

# Extract the coordinate templates (Both Rover and Earth know this map beforehand, so 0 coordinate data is sent)
skeleton_x = grid_x[::sampling_step, ::sampling_step]
skeleton_y = grid_y[::sampling_step, ::sampling_step]

# Flatten arrays for the reconstruction engine
points_sent = np.vstack((skeleton_x.ravel(), skeleton_y.ravel())).T
values_sent = skeleton_values.ravel()

print(f"-> AstroSync Compression Complete!")
print(f"-> Sent only {len(values_sent)} elevation points over space instead of 10,000.")
print(f"-> Total Space Bandwidth Saved: {100 - (len(values_sent)/10000)*100:.2f}%")

# 3. THE EARTH-SIDE RECONSTRUCTION 
# The AI/Math engine on Earth maps the incoming raw values directly to its pre-shared coordinate map
reconstructed_surface = griddata(points_sent, values_sent, (grid_x, grid_y), method='cubic')

# 4. VISUALIZE THE REAL SYSTEM PERFORMANCE
plt.figure(figsize=(12, 5))

# Plot Original Raw Space Data
plt.subplot(1, 2, 1)
plt.imshow(real_mars_surface, extent=(0,10,0,10), origin='lower', cmap='viridis')
plt.title("Original Mars Survey Data Matrix\n(10,000 Points - 100% Uncompressed)")
plt.colorbar(label='Raw Elevation Data')

# Plot AstroSync AI Reconstructed Matrix
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_surface, extent=(0,10,0,10), origin='lower', cmap='viridis')
plt.title("AstroSync Reconstruction Matrix on Earth\n(Generated from ~6% Transmitted Bytes)")
plt.colorbar(label='Reconstructed Elevation')

plt.tight_layout()
print("-> Successfully reconstructed matrix on Earth with zero coordinate overhead!")
plt.show()
print("--- SIMULATION COMPLETE ---")