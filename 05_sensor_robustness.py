import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def ensure_output_directory(folder_name="figures"):
    """Ensure that the figures directory exists before saving plots."""
    os.makedirs(folder_name, exist_ok=True)


def evaluate_sensor_layout(mac_file, y_labels, dof_count, noise_level=0.0):
    """
    Evaluate structural damage classification accuracy under modal observability
    constraints and operational Gaussian noise injection.
    """
    if not os.path.exists(mac_file):
        print(f"Warning: File '{mac_file}' not found. Skipping layout.")
        return 0.0

    df_mac = pd.read_excel(mac_file)

    # Detect MAC feature columns flexibly (matching MAC1, MAC_1, mac1, etc.)
    all_mac_cols = [c for c in df_mac.columns if 'MAC' in str(c).upper()]
    
    # Sort columns by their numeric index if possible
    def extract_index(col_name):
        nums = ''.join(ch for ch in str(col_name) if ch.isdigit())
        return int(nums) if nums else 0

    all_mac_cols.sort(key=extract_index)

    # Physical Observability Constraint: Number of observable modes <= Number of active DOFs
    selected_features = all_mac_cols[:dof_count]

    if len(selected_features) < dof_count:
        print(f"Warning: '{mac_file}' has only {len(selected_features)} MAC columns; expected {dof_count}.")

    X = df_mac[selected_features].copy()

    # Operational Gaussian noise injection
    if noise_level > 0.0:
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, noise_level, X.shape)
        X = X + noise
        X = np.clip(X, 0.0, 1.0)

    # Stratified Train-Test split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_labels, test_size=0.20, stratify=y_labels, random_state=42
    )

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def main():
    label_file = 'Clustered_Results_new.xlsx'
    if not os.path.exists(label_file):
        print(f"Error: Target label dataset '{label_file}' not found.")
        sys.exit(1)

    df_labels = pd.read_excel(label_file)
    if 'Damage_Severity' in df_labels.columns:
        y = df_labels['Damage_Severity']
    elif 'Cluster_Label' in df_labels.columns:
        label_mapping = {0: 'Mild', 1: 'Moderate', 2: 'Severe'}
        y = df_labels['Cluster_Label'].map(label_mapping)
    else:
        print("Error: Severity target column missing in dataset.")
        sys.exit(1)

    # Sensor deployment configurations and active DOFs
    configurations = {
        '8-DOF\n(Full Benchmark)': ('MAC_8DOF.xlsx', 8),
        '4-DOF\n(EfI & MKE Optimal)': ('MAC_4DOF.xlsx', 4),
        '2-DOF\n(Under-instrumented)': ('MAC_2DOF.xlsx', 2)
    }

    layouts = list(configurations.keys())
    scores_noise_free = []
    scores_noisy = []

    print("--- Sensor Layout Classification Performance ---")
    for layout_title, (file_name, dof_count) in configurations.items():
        clean_name = layout_title.replace('\n', ' ')
        
        # 1. Ideal Noise-Free Baseline
        acc_clean = evaluate_sensor_layout(file_name, y, dof_count, noise_level=0.0)
        scores_noise_free.append(acc_clean * 100)

        # 2. Operational 5% Noise Environment
        acc_noisy = evaluate_sensor_layout(file_name, y, dof_count, noise_level=0.05)
        scores_noisy.append(acc_noisy * 100)

        print(f"{clean_name:<32} | Noise-Free: {acc_clean * 100:6.2f}% | 5% Noise: {acc_noisy * 100:6.2f}%")

    # Generate Academic Grouped Bar Chart
    ensure_output_directory("figures")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11

    x_positions = np.arange(len(layouts))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=300)

    rects1 = ax.bar(
        x_positions - bar_width / 2,
        scores_noise_free,
        bar_width,
        label='Ideal State (Noise-Free)',
        color='#4682b4',
        edgecolor='black',
        linewidth=0.8,
        zorder=3
    )
    rects2 = ax.bar(
        x_positions + bar_width / 2,
        scores_noisy,
        bar_width,
        label='Operational State (5% Noise)',
        color='#cd5c5c',
        edgecolor='black',
        linewidth=0.8,
        zorder=3
    )

    ax.set_ylabel('Classification Accuracy (%)', fontweight='bold')
    ax.set_title('Sensor Layout Robustness: Ideal vs. Operational Conditions', fontweight='bold', pad=15)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(layouts)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    # Function to annotate bar values
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f'{height:.1f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontweight='bold',
                fontsize=10
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    output_image = os.path.join("figures", "Sensor_Robustness_Comparative.png")
    fig.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nComparative robustness plot successfully saved: {output_image}")


if __name__ == "__main__":
    main()