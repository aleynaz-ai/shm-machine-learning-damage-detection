import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_output_directory(folder_name="figures"):
    """Ensure that the figures directory exists before saving plots."""
    os.makedirs(folder_name, exist_ok=True)


def load_stiffness_data(excel_path="Clustered_Results_new.xlsx"):
    """
    Load element stiffness reduction (EI) data from clustered results.
    Generates a reproducible 80x10 synthetic matrix if file or columns are absent.
    """
    if os.path.isfile(excel_path):
        df = pd.read_excel(excel_path, sheet_name=0)
        
        # Standardize scenario indexing
        scenario_col = next(
            (c for c in df.columns if str(c).strip().lower() in ["scenario", "senaryo"]),
            None
        )
        if scenario_col is not None:
            df = df.set_index(scenario_col)

        # Extract stiffness reduction features (EI1 - EI10)
        ei_columns = [col for col in df.columns if str(col).upper().startswith("EI")]
        if ei_columns:
            print(f"Loaded {len(ei_columns)} stiffness parameters from {excel_path}.")
            return df[ei_columns]

    print("Stiffness columns not found in Excel. Generating representative 80x10 damage matrix...")
    element_names = [f"EI{i}" for i in range(1, 11)]
    scenario_names = [f"S{i}" for i in range(1, 81)]
    rng = np.random.default_rng(42)
    matrix = np.round(rng.uniform(0.0, 0.8, size=(80, 10)), 2)
    return pd.DataFrame(matrix, index=scenario_names, columns=element_names)


def plot_damage_heatmap(df, filename=os.path.join("figures", "Damage_Heatmap.png")):
    """Plot and export the structural damage distribution heatmap across scenarios."""
    ensure_output_directory("figures")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    fig, ax = plt.subplots(figsize=(8.5, 12), dpi=300)
    cbar_kws = {"label": "Damage Severity (Stiffness Reduction Ratio)", "shrink": 0.8}

    sns.heatmap(
        df,
        ax=ax,
        cmap="viridis",
        cbar=True,
        cbar_kws=cbar_kws,
        linewidths=0.5,
        linecolor="white",
        vmin=0.0,
        vmax=0.8,
    )

    ax.set_title("Distribution of Damage Location and Severity across Scenarios", fontsize=12, fontweight="bold")
    ax.set_xlabel("Structural Elements (EI1 - EI10)", fontsize=11)
    ax.set_ylabel("Damage Scenarios (S1 - S80)", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    ax.invert_yaxis()

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Damage Severity (Stiffness Reduction Ratio)", fontsize=11)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return filename


def main():
    print("[1/2] Loading structural stiffness loss parameters...")
    df_stiffness = load_stiffness_data()
    print(f"Data matrix ready with shape: {df_stiffness.shape}")

    print("[2/2] Generating damage distribution heatmap...")
    output_plot = plot_damage_heatmap(df_stiffness)
    print(f"Heatmap figure successfully saved to: {output_plot}")


if __name__ == "__main__":
    main()