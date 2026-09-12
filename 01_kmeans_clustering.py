import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def ensure_output_directory(folder_name="figures"):
    """Ensure that the figures directory exists before saving plots."""
    os.makedirs(folder_name, exist_ok=True)


def load_mac_data():
    """Load the modal assurance criterion (MAC) dataset from Excel."""
    excel_candidates = ['MAC_Results.xlsx', 'MAC_Sonuclari.xlsx', 'MAC_Dataset.xlsx']
    existing_files = [fname for fname in excel_candidates if os.path.exists(fname)]
    if not existing_files:
        raise FileNotFoundError(
            'Could not find MAC_Results.xlsx or MAC_Sonuclari.xlsx in the current directory.'
        )

    data_path = existing_files[0]
    df = pd.read_excel(data_path)

    # Normalize scenario column naming
    scenario_candidates = ['Scenario', 'Senaryo']
    scenario_col = next((c for c in df.columns if c in scenario_candidates), None)
    if scenario_col is None:
        raise ValueError('Could not find a scenario column named Scenario or Senaryo.')

    # Identify MAC feature columns
    mac_columns = [c for c in df.columns if 'MAC' in str(c).upper() and c != scenario_col]
    if not mac_columns:
        raise ValueError('Could not find MAC columns in the dataset.')

    df = df.copy()
    df.rename(columns={scenario_col: 'Scenario'}, inplace=True)
    df.set_index('Scenario', inplace=True)
    df = df[mac_columns]

    df.columns = [f'MAC_{i+1}' for i in range(len(df.columns))]
    return df


def evaluate_clusters(df, k_min=2, k_max=10):
    """Compute inertia and silhouette score across a range of K values."""
    inertia = []
    silhouette = []

    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(df)
        inertia.append(model.inertia_)
        if k > 1:
            silhouette.append(silhouette_score(df, labels))
        else:
            silhouette.append(np.nan)

    return inertia, silhouette


def plot_validation(inertia, silhouette, k_min=2, k_max=10, filename=os.path.join('figures', 'kmeans_validation.png')):
    """Plot the Elbow Method and Silhouette Score side by side."""
    ensure_output_directory('figures')
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11

    ks = list(range(k_min, k_max + 1))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Elbow curve
    axes[0].plot(ks, inertia, marker='o', color='tab:blue', linewidth=1.8)
    axes[0].set_title('Elbow Method for Optimal K', fontweight='bold')
    axes[0].set_xlabel('Number of clusters (k)')
    axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Silhouette curve
    axes[1].plot(ks, silhouette, marker='s', color='tab:orange', linewidth=1.8)
    axes[1].set_title('Silhouette Analysis for Optimal K', fontweight='bold')
    axes[1].set_xlabel('Number of clusters (k)')
    axes[1].set_ylabel('Mean Silhouette Score')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    return filename


def run_kmeans(df, n_clusters=3):
    """Fit K-Means clustering model and return cluster labels."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df)
    return labels, kmeans


def plot_clusters(df, labels, kmeans, cluster_names=None, filename=os.path.join('figures', 'kmeans_clusters.png')):
    """Generate a 2D PCA scatter plot showing cluster separation and centroids."""
    ensure_output_directory('figures')
    
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(df.values)
    centroids = pca.transform(kmeans.cluster_centers_)

    variance_ratios = pca.explained_variance_ratio_
    pc1_label = f'Principal Component 1 ({variance_ratios[0] * 100:.1f}% Variance)'
    pc2_label = f'Principal Component 2 ({variance_ratios[1] * 100:.1f}% Variance)'

    plot_df = pd.DataFrame(
        components,
        columns=['PC1', 'PC2'],
        index=df.index,
    )
    if cluster_names is not None:
        plot_df['Cluster'] = [cluster_names[int(label)] for label in labels]
    else:
        plot_df['Cluster'] = labels.astype(str)

    marker_map = {'Mild': 's', 'Moderate': 'o', 'Severe': '^'}
    color_map = {'Mild': 'tab:green', 'Moderate': 'tab:orange', 'Severe': 'tab:red'}

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11

    fig, ax = plt.subplots(figsize=(9.5, 6.5), dpi=300)

    for cluster_label, group in plot_df.groupby('Cluster'):
        ax.scatter(
            group['PC1'],
            group['PC2'],
            marker=marker_map.get(cluster_label, 'o'),
            color=color_map.get(cluster_label, 'tab:gray'),
            edgecolor='k',
            linewidth=0.6,
            s=60,
            label=cluster_label,
        )

    for i, centroid in enumerate(centroids):
        label = f'Centroid {i}'
        if cluster_names is not None:
            label = f"Centroid {i} ({cluster_names.get(i, 'Cluster')})"
        centroid_label = cluster_names.get(i) if cluster_names is not None else str(i)
        centroid_color = color_map.get(centroid_label, 'black')
        ax.scatter(
            centroid[0],
            centroid[1],
            s=250,
            marker='X',
            c=centroid_color,
            edgecolor='k',
            linewidth=1.2,
            label=label,
        )

    ax.set_title('K-Means Damage State Clusters on PCA-Reduced MAC Data', fontweight='bold')
    ax.set_xlabel(pc1_label)
    ax.set_ylabel(pc2_label)
    ax.legend(title='Damage Severity', loc='best', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.45)
    
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    return filename


def map_cluster_severity(df, labels):
    """Map each cluster to an SHM severity label based on mean MAC levels."""
    df_with_labels = df.copy()
    df_with_labels['cluster'] = labels
    cluster_mean = df_with_labels.groupby('cluster').mean().mean(axis=1)
    ordered_clusters = cluster_mean.sort_values(ascending=True).index.tolist()
    severity_labels = ['Severe', 'Moderate', 'Mild']
    return {cluster: severity_labels[idx] for idx, cluster in enumerate(ordered_clusters)}


def save_excel_safe(df, path):
    """Save the DataFrame to Excel, handling permission lock fallbacks."""
    try:
        df.to_excel(path, index=True)
        return path
    except PermissionError:
        fallback = os.path.splitext(path)[0] + '_fallback.xlsx'
        df.to_excel(fallback, index=True)
        return fallback


def main():
    print('[1/5] Loading MAC modal dataset...')
    df = load_mac_data()
    print(f'Successfully loaded dataset with shape: {df.shape}')

    print('[2/5] Evaluating optimal clusters (Elbow & Silhouette)...')
    inertia, silhouette = evaluate_clusters(df, k_min=2, k_max=10)
    validation_plot = plot_validation(inertia, silhouette, k_min=2, k_max=10)
    print(f'Validation diagnostic plot saved to: {validation_plot}')

    print('[3/5] Performing K-Means clustering (k=3)...')
    labels, kmeans_model = run_kmeans(df, n_clusters=3)
    df['Cluster_Label'] = labels

    severity_map = map_cluster_severity(df.drop(columns=['Cluster_Label']), labels)
    df['Damage_Severity'] = [severity_map[label] for label in labels]

    print('[4/5] Saving clustered dataset...')
    cluster_output = 'Clustered_Results_new.xlsx'
    saved_path = save_excel_safe(df, cluster_output)
    print(f'Clustered results saved to: {saved_path}')

    print('[5/5] Generating PCA cluster projection scatter plot...')
    cluster_plot = plot_clusters(
        df.drop(columns=['Cluster_Label', 'Damage_Severity']),
        labels,
        kmeans_model,
        cluster_names=severity_map,
    )
    print(f'Cluster visualization saved to: {cluster_plot}')

    print('\n--- Execution Summary ---')
    print('Scenarios per severity cluster:')
    print(df['Damage_Severity'].value_counts())


if __name__ == '__main__':
    main()