from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import butter, sosfilt, filtfilt, iirnotch



# --- 3. Sliding Window Function ---
def sliding_window(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Applies a sliding window to the signal.

    Args:
        signal (np.ndarray): The preprocessed signal data. Shape: (num_samples, num_channels)
        window_size (int): The size of each window (number of samples).
        stride (int): The step size between consecutive windows.

    Returns:
        np.ndarray: An array of windows. Shape: (num_windows, window_size, num_channels)
    """
    
    # Ensure signal is long enough for at least one window
    if len(signal) < window_size:
        print(f"  Warning: Signal length ({len(signal)}) is less than window_size ({window_size}). No windows generated.")
        return np.array([]) # Return empty array if no windows can be formed

    windows = np.array([signal[i:i+window_size] for i in range(0, len(signal) - window_size + 1, stride)])
    return windows


def preprocess_signal(signal: np.ndarray, config: dict) -> np.ndarray:
    """
    Applies preprocessing steps (sample removal, bandpass, notch filter) to the signal data.

    Args:
        signal (np.ndarray): The raw signal data (e.g., loaded directly from CSV).
                             Shape: (num_samples, num_channels)
        config (dict): A dictionary containing preprocessing parameters.

    Returns:
        np.ndarray: The processed and filtered signal data.
    """

    # Parameters from config
    samples_to_remove = config['preprocessing']['samples_to_remove']
    butter_order = config['preprocessing']['butter_order']
    butter_lowcut = config['preprocessing']['butter_lowcut']
    butter_highcut = config['preprocessing']['butter_highcut']
    fs = config['preprocessing']['fs']
    notch_freq = config['preprocessing']['notch_freq']
    notch_q = config['preprocessing']['notch_q']

    # Remove the first X samples
    data = signal[samples_to_remove:, :]

    # Apply Butterworth bandpass filter
    nyquist = 0.5 * fs
    low = butter_lowcut / nyquist
    high = butter_highcut / nyquist
    
    # Ensure cutoff frequencies are within valid range [0, 1] for nyquist scaling
    if not (0 < low < 1 and 0 < high < 1 and low < high):
        raise ValueError(f"Invalid Butterworth cutoff frequencies for fs={fs}: low={butter_lowcut}, high={butter_highcut}. "
                         f"Nyquist-normalized values: low_norm={low}, high_norm={high}. Must be 0 < low_norm < high_norm < 1.")

    sos = butter(butter_order, [low, high], btype='band', output='sos')
    processed_data = sosfilt(sos, data, axis=0)

    # Apply notch filter
    if notch_freq is not None:
        notch_w0 = notch_freq / nyquist
        if not (0 < notch_w0 < 1):
            raise ValueError(f"Invalid Notch frequency for fs={fs}: notch_freq={notch_freq}. "
                            f"Nyquist-normalized value: notch_w0={notch_w0}. Must be 0 < notch_w0 < 1.")

        b_notch, a_notch = iirnotch(notch_w0, notch_q) # fs is not needed if w0 is normalized
        processed_data = filtfilt(b_notch, a_notch, processed_data, axis=0)

    return processed_data


def filter_data_from_h5(
        data_file: str, 
        sessions: List[int] = None, 
        positions: List[int] = None,
        repetitions: List[int] = None, 
        subjects: List[int] = None):

    with h5py.File(data_file, 'r') as f:
            meta = {k: f['meta'][k][:] for k in f['meta']}
            mask = np.ones(len(meta['subject']), dtype=bool)

            if subjects != None:
                subjects = subjects if isinstance(subjects, list) else [subjects]
                subject_filter = np.isin(meta['subject'], subjects)
                mask &= subject_filter
            
            if sessions is not None:
                sessions = sessions if isinstance(sessions, list) else [sessions]
                session_filter = np.isin(meta['session'], sessions)
                mask &= session_filter

            if repetitions != None:
                repetitions = repetitions if isinstance(repetitions, list) else [repetitions]
                repetition_filter = np.isin(meta['repetition'], repetitions)
                mask &= repetition_filter
            
            if positions != None:
                positions = positions if isinstance(positions, list) else [positions]
                position_filter = np.isin(meta['position'], positions)
                mask &= position_filter

            indices = np.where(mask)[0]
    
    return indices

def display_results(accuracy_dict):

    model_names = list(accuracy_dict.keys())
    means = []
    stds = []

    for model in model_names:
        accuracies = list(accuracy_dict[model].values())
        means.append(np.mean(accuracies))
        stds.append(np.std(accuracies))

    # Create the bar plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(model_names, means, yerr=stds, capsize=10, 
                color='skyblue', alpha=0.7, edgecolor='navy', linewidth=1.2)
    
        # Customize the plot
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Model Performance Comparison\n(Mean ± Standard Deviation)', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)

    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Optional: Set y-axis limits for better visualization
    plt.ylim(0, max(means) + max(stds) + 0.05)

    # Show the plot
    plt.show()

    print("Model Performance Summary:")
    print("-" * 40)
    for model, mean, std in zip(model_names, means, stds):
        print(f"{model:15}: {mean:.3f} ± {std:.3f}")

