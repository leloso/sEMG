import os
import numpy as np
import h5py
import argparse


gesture_map = {
    "fist": 1,
    "middlefinger": 2,
    "two": 3,
    "hand": 4,
    "forefinger": 5,
    "varus": 6,
    "eversion": 7,
}


def parse_filename(filename):
    parts = filename.replace('.npz', '').split('_')
    position = int(parts[1][1:])
    repetition = int(parts[2][1:])
    label_str = parts[-1]
    label = gesture_map.get(label_str, 0)
    return position, repetition, label-1


def count_total_samples(data_dir):
    """First pass: count total individual samples across all NPZ files"""
    total_samples = 0
    for subject_idx in range(36):
        subject_dir = os.path.join(data_dir, f'h{subject_idx}')
        if not os.path.isdir(subject_dir):
            continue
        
        for session_name in os.listdir(subject_dir):
            session_path = os.path.join(subject_dir, session_name)
            if not os.path.isdir(session_path):
                continue
                
            for file_name in os.listdir(session_path):
                if not file_name.endswith('.npz'):
                    continue
                    
                file_path = os.path.join(session_path, file_name)
                npz = np.load(file_path)
                signals = npz['samples']  # shape: (batch_size, 8, 60)
                total_samples += signals.shape[0]  # Add batch_size to total
                
    return total_samples


def main():

    parser = argparse.ArgumentParser(description="Build HDF5 dataset from NPZ files.")
    parser.add_argument('--data_dir', type=str, default='processed_data', help='Directory containing subject/session/npz files')
    parser.add_argument('--hdf5_path', type=str, default='processed_data/emg_data.h5', help='Destination path for HDF5 file')
    args = parser.parse_args()

    data_dir = args.data_dir
    hdf5_path = args.hdf5_path
    
    print("Counting total samples present...")
    total_samples = count_total_samples(data_dir=data_dir)
    print(f"Found {total_samples} samples in total")

    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    # Assume sample shape is (8, 60)
    sample_shape = (8, 60)
    chunk_size = 256  # You can tune this

    with h5py.File(args.hdf5_path, 'w') as f:
        # Create datasets with known total size
        samples_ds = f.create_dataset('samples', shape=(total_samples, *sample_shape), 
                                    dtype='float32', 
                                    chunks=(256, *sample_shape),  # Chunk multiple samples together
                                    compression='gzip', shuffle=True)
        
        labels_ds = f.create_dataset('labels', shape=(total_samples,), 
                                    dtype='uint8', chunks=(chunk_size,))
        subjects_ds = f.create_dataset('meta/subject', shape=(total_samples,), 
                                    dtype='uint8', chunks=(chunk_size,))
        sessions_ds = f.create_dataset('meta/session', shape=(total_samples,), 
                                    dtype='uint8', chunks=(chunk_size,))
        positions_ds = f.create_dataset('meta/position', shape=(total_samples,), 
                                    dtype='uint8', chunks=(chunk_size,))
        repetitions_ds = f.create_dataset('meta/repetition', shape=(total_samples,), 
                                        dtype='uint8', chunks=(chunk_size,))
        
        sample_idx = 0
        
        # Second pass: actually load the data
        for subject_idx in range(36):
            subject_dir = os.path.join(args.data_dir, f'h{subject_idx}')
            if not os.path.isdir(subject_dir):
                continue
            print(f"Processing subject {subject_idx}")
            
            for session_name in os.listdir(subject_dir):
                session_path = os.path.join(subject_dir, session_name)
                if not os.path.isdir(session_path):
                    continue
                print(f"Processing session {session_name}")
                
                for file_name in os.listdir(session_path):
                    if not file_name.endswith('.npz'):
                        continue
                    
                    position, repetition, label = parse_filename(file_name)

                    file_path = os.path.join(session_path, file_name)
                    npz = np.load(file_path)
                    signals = npz['samples']  # shape: (batch_size, 8, 60)
                    
                    batch_size = signals.shape[0]
                    
                    # Store the entire batch at once (more efficient)
                    samples_ds[sample_idx:sample_idx + batch_size] = signals
                    labels_ds[sample_idx:sample_idx + batch_size] = label
                    subjects_ds[sample_idx:sample_idx + batch_size] = subject_idx
                    sessions_ds[sample_idx:sample_idx + batch_size] = int(session_name)
                    positions_ds[sample_idx:sample_idx + batch_size] = position
                    repetitions_ds[sample_idx:sample_idx + batch_size] = repetition
                    
                    sample_idx += batch_size
                        
    print(f'HDF5 file created at {hdf5_path} with {sample_idx} samples')

if __name__ == "__main__":
    main()