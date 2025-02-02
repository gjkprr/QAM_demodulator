import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
M = 4  # Order of QAM (e.g., 16-QAM)
N = 1000

symbols = np.random.randint(0, M, N)  # Random indices for QAM symbols
I_vals = 2 * (np.arange(np.sqrt(M)) - (np.sqrt(M) - 1) / 2)
Q_vals = 2 * (np.arange(np.sqrt(M)) - (np.sqrt(M) - 1) / 2)

# Map symbols to QAM_values
constellation = np.array([[i + 1j * q for q in Q_vals] for i in I_vals]).flatten()
modulated_signal = constellation[symbols]

# Ensure the directory exists before saving images
save_dir = f"dataset/test_set/{M}_QAM/"
os.makedirs(save_dir, exist_ok=True)

# Add noise
# SNR_dB_list = list(range(10, 30, 1))
SNR_dB_list = [10,12,15,20]
for index, snr_dB in enumerate(SNR_dB_list):
    # Compute signal power
    signal_power = np.mean(np.abs(modulated_signal) ** 2)    
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_dB / 10)    
    # Compute noise power
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(*modulated_signal.shape) + 1j * np.random.randn(*modulated_signal.shape))
    received_signal = modulated_signal + noise

    plt.title("Received Constellation")
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(received_signal), np.imag(received_signal), color='red', alpha=0.6, edgecolors='black')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # x-axis
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)  # y-axis
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    # Save the figure as a JPG file
    # Save the figure
    save_path = os.path.join(save_dir, f"Received_Constellation_{index}.jpg")
    plt.savefig(save_path, format="jpg", dpi=300)
