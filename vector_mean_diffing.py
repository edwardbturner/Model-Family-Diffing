# %%
import os

import plotly.graph_objects as go  # type: ignore
import torch

# %%
# Configuration variables
DATA_TYPE = "answer"  # Can be "question" or "answer"
RANK = 8  # Can be 32 or 8


def load_and_process_data(rank, data_type):
    # Load activation data tensors
    activation_dir = f"activation_data/qwen2.5_14B_instruct_rank{rank}_mean_activations"

    # Load all .pt files in the directory
    activation_files = {
        "model_a_data_a": torch.load(os.path.join(activation_dir, "model-a_data-a_hs.pt")),
        "model_a_data_m": torch.load(os.path.join(activation_dir, "model-a_data-m_hs.pt")),
        "model_m_data_a": torch.load(os.path.join(activation_dir, "model-m_data-a_hs.pt")),
        "model_m_data_m": torch.load(os.path.join(activation_dir, "model-m_data-m_hs.pt")),
    }

    aa = activation_files["model_a_data_a"][data_type]
    am = activation_files["model_a_data_m"][data_type]
    ma = activation_files["model_m_data_a"][data_type]
    mm = activation_files["model_m_data_m"][data_type]

    mm_aa = {k: v - aa[k] for k, v in mm.items()}  # vary both
    mm_am = {k: v - am[k] for k, v in mm.items()}  # vary model (on misaligned data)
    mm_ma = {k: v - ma[k] for k, v in mm.items()}  # vary data (on misaligned model)

    return mm_aa, mm_am, mm_ma


def calculate_metrics(mm_aa, mm_am, mm_ma):
    # Calculate cosine similarity between vector pairs for each layer
    cos_sims = {}
    norms = {}
    for layer in range(48):  # 0 to 47 layers
        # Get vectors for this layer from each dictionary
        mm_aa_vec = mm_aa[layer]
        mm_am_vec = mm_am[layer]
        mm_ma_vec = mm_ma[layer]

        # Calculate cosine similarities
        cos_sims[layer] = {
            "mm_aa_vs_mm_am": torch.nn.functional.cosine_similarity(mm_aa_vec, mm_am_vec, dim=0),
            "mm_aa_vs_mm_ma": torch.nn.functional.cosine_similarity(mm_aa_vec, mm_ma_vec, dim=0),
            "mm_am_vs_mm_ma": torch.nn.functional.cosine_similarity(mm_am_vec, mm_ma_vec, dim=0),
        }

        # Calculate L2 norms
        norms[layer] = {
            "mm_aa": torch.norm(mm_aa_vec, p=2),
            "mm_am": torch.norm(mm_am_vec, p=2),
            "mm_ma": torch.norm(mm_ma_vec, p=2),
        }

    return cos_sims, norms


def create_plots(cos_sims, norms, rank, data_type):
    # Define legend names based on comments
    legend_names = {
        "mm_aa_vs_mm_am": "vary both vs vary model",
        "mm_aa_vs_mm_ma": "vary both vs vary data",
        "mm_am_vs_mm_ma": "vary model vs vary data",
        "mm_aa": "vary both",
        "mm_am": "vary model",
        "mm_ma": "vary data",
    }

    # Create Plotly figure for cosine similarities
    fig_cos = go.Figure()

    # Add traces for each pair
    layers = list(range(48))
    for pair_name in ["mm_aa_vs_mm_am", "mm_aa_vs_mm_ma", "mm_am_vs_mm_ma"]:
        similarities = [cos_sims[layer][pair_name].item() for layer in layers]
        fig_cos.add_trace(go.Scatter(x=layers, y=similarities, name=legend_names[pair_name], mode="lines+markers"))

    # Update layout for cosine similarity plot
    fig_cos.update_layout(
        title=f"Cosine Similarity Between Vector Pairs Across Layers (Rank {rank}, {data_type})",
        xaxis_title="Layer",
        yaxis_title="Cosine Similarity",
        hovermode="x unified",
        yaxis=dict(range=[-1, 1]),  # Cosine similarity ranges from -1 to 1
        showlegend=True,
        width=1200,  # Make the plot wider
        height=600,  # Keep the height proportional
    )

    # Create Plotly figure for norms
    fig_norm = go.Figure()

    # Add traces for each vector's norm
    for vec_name in ["mm_aa", "mm_am", "mm_ma"]:
        vec_norms = [norms[layer][vec_name].item() for layer in layers]
        fig_norm.add_trace(go.Scatter(x=layers, y=vec_norms, name=legend_names[vec_name], mode="lines+markers"))

    # Update layout for norm plot
    fig_norm.update_layout(
        title=f"L2 Norm of Vectors Across Layers (Rank {rank}, {data_type})",
        xaxis_title="Layer",
        yaxis_title="L2 Norm",
        hovermode="x unified",
        showlegend=True,
        width=1200,  # Make the plot wider
        height=600,  # Keep the height proportional
    )

    return fig_cos, fig_norm


# Process and plot for the specified rank and data type
mm_aa, mm_am, mm_ma = load_and_process_data(RANK, DATA_TYPE)
cos_sims, norms = calculate_metrics(mm_aa, mm_am, mm_ma)
fig_cos, fig_norm = create_plots(cos_sims, norms, RANK, DATA_TYPE)
fig_cos.show()
fig_norm.show()

# %%
