# Refactored Yokogawa DLM4000 waveform analysis script

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil

# ==============================
# Configuration
# ==============================
FOLDER_PATH = os.path.join('Data', '260518_IntBlaster_Jitter')
OUTPUT_DIR = os.path.join('output', '260518_IntBlaster_Jitter_output')
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANNEL_LIST = ['CH1', 'CH2']
records = []
DEBUG = True

# ==============================
# Yokogawa DLM4000 Metadata Extraction
# ==============================
def extract_yokogawa_metadata(path):
    """
    Extract metadata from Yokogawa DLM4000 CSV header.
    The header has a fixed structure with key-value pairs in double-quoted keys.
    Returns a dict with parsed metadata including model, sample rate,
    horizontal resolution (time step), horizontal offset, and block size.
    """
    meta = {
        'model': None,
        'sample_rate': None,
        'h_resolution': None,  # time step per sample (seconds)
        'h_offset': None,      # time offset of the first sample (seconds)
        'block_size': None,
        'trace_names': [],
        'header_size': 15,
    }

    with open(path, 'r', encoding='latin1', errors='ignore') as f:
        for i in range(20):
            line = f.readline().strip()
            if not line:
                continue

            # Parse key-value: "Key",value1,value2,...
            parts = [p.strip().replace('"', '') for p in line.split(',')]
            key = parts[0].strip()

            if key == 'Header Size' and len(parts) > 1:
                try:
                    meta['header_size'] = int(parts[1])
                except:
                    pass
            elif key == 'Model Name' and len(parts) > 1:
                meta['model'] = parts[1].strip()
            elif key == 'TraceName':
                meta['trace_names'] = [p.strip() for p in parts[1:] if p.strip()]
            elif key == 'BlockSize' and len(parts) > 1:
                try:
                    meta['block_size'] = int(parts[1])
                except:
                    pass
            elif key == 'SampleRate' and len(parts) > 1:
                try:
                    meta['sample_rate'] = float(parts[1])
                except:
                    pass
            elif key == 'HResolution' and len(parts) > 1:
                try:
                    meta['h_resolution'] = float(parts[1])
                except:
                    pass
            elif key == 'HOffset' and len(parts) > 1:
                try:
                    meta['h_offset'] = float(parts[1])
                except:
                    pass

    return meta

# ==============================
# Yokogawa CSV Reader
# ==============================
def read_yokogawa_csv(path):
    """
    Read a Yokogawa DLM4000 waveform CSV file.
    - Extracts metadata from the header
    - Reads the data block (after header)
    - Reconstructs the TIME axis from HOffset and HResolution
    - Returns (DataFrame with TIME/CH1/CH2 columns, metadata dict)
    """
    meta = extract_yokogawa_metadata(path)
    header_lines = meta['header_size'] + 1  # +1 for the blank line after header

    # Read data: Yokogawa format has a leading comma, so column 0 is empty
    df = pd.read_csv(path, skiprows=header_lines, header=None, encoding='latin1')

    # Map columns: col 0 = NaN (leading comma), col 1 = CH1, col 2 = CH2
    col_map = {}
    if 1 in df.columns:
        col_map[1] = 'CH1'
    if 2 in df.columns:
        col_map[2] = 'CH2'
    df = df.rename(columns=col_map)

    # Drop empty columns (col 0 and trailing col 3)
    df = df[['CH1', 'CH2']].copy()

    # Convert to numeric
    for ch in CHANNEL_LIST:
        if ch in df.columns:
            df[ch] = pd.to_numeric(df[ch], errors='coerce')

    # Reconstruct TIME axis: TIME[i] = HOffset + (i * HResolution)
    if meta['h_resolution'] is not None and meta['h_offset'] is not None:
        df['TIME'] = meta['h_offset'] + (df.index * meta['h_resolution'])
    else:
        raise ValueError(f"Missing HResolution or HOffset in metadata for {path}")

    df = df.dropna(subset=['TIME']).reset_index(drop=True)

    return df, meta

# ==============================
# Waveform Analysis (Adaptive Trigger)
# ==============================
def analyze_waveform(file_path, filename):
    if DEBUG:
        print(f"Processing: {filename}")

    df, meta = read_yokogawa_csv(file_path)

    # Dynamic midpoint threshold (50% of min-max range)
    thresholds = {}
    for ch in CHANNEL_LIST:
        if ch in df.columns:
            ch_min = df[ch].min()
            ch_max = df[ch].max()
            thresholds[ch] = (ch_min + ch_max) / 2.0

    # CH1: Find first rising edge crossing (reference trigger)
    rising_indices = {}
    for ch in CHANNEL_LIST:
        if ch in df.columns:
            idxs = df.index[df[ch] >= thresholds[ch]]
            rising_indices[ch] = idxs[0] if len(idxs) > 0 else None
        else:
            rising_indices[ch] = None

    if rising_indices['CH1'] is None:
        if DEBUG:
            print(f"  Skipping {filename}: CH1 trigger missing")
        return

    t_ref_idx = rising_indices['CH1']
    t_ref = df.loc[t_ref_idx, 'TIME']

    # CH2: Adaptive edge detection after CH1 trigger
    # Determine if CH2 starts HIGH or LOW at the CH1 trigger point
    ch2_at_trigger = df.loc[t_ref_idx, 'CH2']
    ch2_threshold = thresholds['CH2']
    starts_high = ch2_at_trigger >= ch2_threshold

    # Search for the first threshold crossing after the CH1 trigger
    ch2_after = df.loc[t_ref_idx:, 'CH2']
    if starts_high:
        # CH2 starts HIGH -> look for falling edge (first point below threshold)
        cross_idxs = ch2_after.index[ch2_after < ch2_threshold]
        edge_type = 'FALLING'
    else:
        # CH2 starts LOW -> look for rising edge (first point above threshold)
        cross_idxs = ch2_after.index[ch2_after >= ch2_threshold]
        edge_type = 'RISING'

    if len(cross_idxs) == 0:
        if DEBUG:
            print(f"  Skipping {filename}: CH2 trigger missing after CH1 (edge: {edge_type})")
        return

    t_val_idx = cross_idxs[0]
    t_val = df.loc[t_val_idx, 'TIME']
    delay = (t_val - t_ref) * 1e6  # Convert to microseconds

    records.append({
        'Filename': filename,
        'Channel': 'CH2',
        'Delay (us)': delay,
        'Edge': edge_type,
        'HResolution': meta['h_resolution'],
    })

    if DEBUG:
        print(f"  Delay: {delay:.3f} us ({edge_type} edge, HRes={meta['h_resolution']:.2e})")

    # Plot waveform
    plt.figure(figsize=(12, 6))
    for ch in CHANNEL_LIST:
        if ch in df.columns:
            line, = plt.plot(df['TIME'] * 1e6, df[ch], label=ch, alpha=0.7)

            # Plot trigger marker
            if ch == 'CH1':
                idx = t_ref_idx
            elif ch == 'CH2':
                idx = t_val_idx
            else:
                idx = None

            if idx is not None:
                plt.plot(df.loc[idx, 'TIME'] * 1e6, df.loc[idx, ch],
                         'o', color=line.get_color(), markersize=8)
                plt.annotate(f"{ch} Trigger",
                             (df.loc[idx, 'TIME'] * 1e6, df.loc[idx, ch]),
                             textcoords="offset points", xytext=(0, 10), ha='center')

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title(f"Waveform Analysis: {filename}\n(Delay: {delay:.3f} us, Edge: {edge_type})")
    plt.xlabel("Time (us)")
    plt.ylabel("Voltage (V)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename.replace('.csv', '_waveform.png')), dpi=150)
    plt.close()

# ==============================
# Main Execution
# ==============================
files = [f for f in os.listdir(FOLDER_PATH) if f.endswith('.csv')]
files.sort()

if DEBUG:
    print(f"Found {len(files)} CSV files in {FOLDER_PATH}")

for f in files:
    analyze_waveform(os.path.join(FOLDER_PATH, f), f)

# ==============================
# Result Processing
# ==============================
df_result = pd.DataFrame(records)

if not df_result.empty:
    # ==============================
    # Group Classification
    # ==============================
    # Group 1: Delay >= 1500 us (large delay group)
    # Group 2: Delay <= 100 us  (small delay group)
    GROUP1_THRESHOLD = 1500  # us
    GROUP2_THRESHOLD = 100   # us

    def classify_group(delay):
        if delay >= GROUP1_THRESHOLD:
            return 'Group1'
        elif delay <= GROUP2_THRESHOLD:
            return 'Group2'
        else:
            return 'Unclassified'

    df_result['Group'] = df_result['Delay (us)'].apply(classify_group)

    # Save overall summary
    df_result.to_csv(os.path.join(OUTPUT_DIR, 'summary_all.csv'), index=False)

    if DEBUG:
        group_counts = df_result['Group'].value_counts()
        print(f"\nGroup Classification:")
        for g, cnt in group_counts.items():
            print(f"  {g}: {cnt} files")

    # ==============================
    # Per-Group Analysis Helper
    # ==============================
    def generate_group_outputs(df_group, group_name, group_label):
        """Generate summary CSV, stats CSV, trend plot, and distribution plot for a group."""
        group_dir = os.path.join(OUTPUT_DIR, group_name)
        os.makedirs(group_dir, exist_ok=True)

        # Summary CSV
        df_group.to_csv(os.path.join(group_dir, f'summary_{group_name}.csv'), index=False)

        # Stats CSV
        df_stats = df_group.groupby('Channel')['Delay (us)'].agg(['mean', 'std', 'min', 'max'])
        df_stats['count'] = df_group.groupby('Channel').size()
        df_stats['range'] = df_stats['max'] - df_stats['min']
        df_stats['cv'] = df_stats['std'] / df_stats['mean']
        df_stats.to_csv(os.path.join(group_dir, f'stats_{group_name}.csv'))

        if DEBUG:
            print(f"\n[{group_label}] Statistics:")
            print(df_stats)

        # Trend Plot
        df_plot = df_group.sort_values('Filename')

        plt.figure(figsize=(20, 10))
        for ch in df_plot['Channel'].unique():
            ch_data = df_plot[df_plot['Channel'] == ch]
            plt.plot(ch_data['Filename'], ch_data['Delay (us)'],
                     marker='o', markersize=6, linestyle='-', label=ch, alpha=0.8)

            mean_val = ch_data['Delay (us)'].mean()
            plt.axhline(mean_val, color='red', linestyle='dashed', linewidth=1)
            plt.text(plt.xlim()[1] * 0.9, mean_val, f'Mean: {mean_val:.2f}')

        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.title(f"[{group_label}] Delay Time Trend (us)", fontsize=14)
        plt.xlabel("Filename", fontsize=10)
        plt.ylabel("Delay (us)", fontsize=12)
        plt.legend(['Delay time(CH1->CH2)'])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(group_dir, f'{group_name}_delay_trend.png'), dpi=150)
        plt.close()

        # Distribution Plot
        plt.figure(figsize=(10, 6))
        for ch in df_group['Channel'].unique():
            ch_data = df_group[df_group['Channel'] == ch]
            plt.hist(ch_data['Delay (us)'], bins='auto', alpha=0.7,
                     label=ch, edgecolor='black', density=False)

            mean_val = ch_data['Delay (us)'].mean()
            std_val = ch_data['Delay (us)'].std()
            plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1)
            plt.text(mean_val, plt.ylim()[1] * 0.9,
                     f'Mean: {mean_val:.3f} us\nStd: {std_val:.3f} us', color='red')

        plt.title(f"[{group_label}] Delay Time Distribution (Histogram)", fontsize=14)
        plt.xlabel("Delay (us)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.legend(['Delay time(CH1->CH2)'])
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(group_dir, f'{group_name}_delay_distribution.png'), dpi=150)
        plt.close()

        if DEBUG:
            print(f"  [{group_label}] Plots saved to {group_dir}/")

    # ==============================
    # Generate Group 1 Outputs (Delay >= 1500 us)
    # ==============================
    df_group1 = df_result[df_result['Group'] == 'Group1']
    if not df_group1.empty:
        generate_group_outputs(df_group1, 'Group1',
                               f'Group 1: Delay >= {GROUP1_THRESHOLD} us')
    else:
        print("  Group 1 is empty, skipping.")

    # ==============================
    # Generate Group 2 Outputs (Delay <= 100 us)
    # ==============================
    df_group2 = df_result[df_result['Group'] == 'Group2']
    if not df_group2.empty:
        generate_group_outputs(df_group2, 'Group2',
                               f'Group 2: Delay <= {GROUP2_THRESHOLD} us')
    else:
        print("  Group 2 is empty, skipping.")

    # ==============================
    # Overall Combined Trend Plot (both groups color-coded)
    # ==============================
    df_plot = df_result.sort_values('Filename')

    plt.figure(figsize=(30, 16))
    colors = {'Group1': '#e74c3c', 'Group2': '#2ecc71', 'Unclassified': '#95a5a6'}
    labels = {
        'Group1': f'Group 1 (>= {GROUP1_THRESHOLD} us)',
        'Group2': f'Group 2 (<= {GROUP2_THRESHOLD} us)',
        'Unclassified': 'Unclassified'
    }

    for group_name in ['Group1', 'Group2', 'Unclassified']:
        g_data = df_plot[df_plot['Group'] == group_name]
        if not g_data.empty:
            plt.plot(g_data['Filename'], g_data['Delay (us)'],
                     marker='o', markersize=6, linestyle='',
                     color=colors[group_name], label=labels[group_name], alpha=0.9)

    # Connect all points with a light line for trend visibility
    plt.plot(df_plot['Filename'], df_plot['Delay (us)'],
             linestyle='-', color='#bdc3c7', alpha=0.4, zorder=0)

    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.title("Delay Time Trend (us) - All Groups", fontsize=14)
    plt.xlabel("Filename", fontsize=10)
    plt.ylabel("Delay (us)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'summary_delay_trend_all.png'), dpi=150)
    plt.close()

    if DEBUG:
        print(f"\n  Overall trend plot saved to {os.path.join(OUTPUT_DIR, 'summary_delay_trend_all.png')}")

print("Done.")
