# %%
from neel.imports import *
import torch
from datasets import load_dataset
import pandas as pd
from tqdm.notebook import tqdm
import torch.nn.functional as F  # Import torch.nn.functional properly
import time
from datetime import timedelta
from IPython.display import display, HTML
import colorsys
import plotly.express as px

# %%
smol = HookedTransformer.from_pretrained_no_processing("gemma-2-2b")
big = HookedTransformer.from_pretrained_no_processing("gemma-2-9b")
# %%
# Load the Pile dataset
pile_dataset = load_dataset("NeelNanda/pile-10k", split="train")
# pile_sample = list(islice(pile_dataset, 10000))  # Get the first 10k examples

# %%
# Function to calculate KL divergence between model outputs
def calculate_kl_divergence(logits_p, logits_q):
    """
    Calculate KL(P||Q) for each token position
    logits_p: logits from model P (small)
    logits_q: logits from model Q (large)
    """
    # Convert logits to log probabilities
    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    
    # Calculate KL divergence for each position
    # KL(P||Q) = sum(P(x) * (log P(x) - log Q(x)))
    p = F.softmax(logits_p, dim=-1)
    kl = torch.sum(p * (log_p - log_q), dim=-1)
    
    return kl

# %%
# Function to get next token log probabilities
def get_next_token_logprob(logits, next_tokens):
    """Extract log probability of the actual next token"""
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather the log probs of the actual next tokens
    next_token_logprobs = torch.gather(log_probs, -1, next_tokens.unsqueeze(-1)).squeeze(-1)
    return next_token_logprobs

# %%
# Function to get context tokens as text
def get_context(tokens, position, context_size=5):
    """Get n tokens before and after the current position as text"""
    start = max(0, position - context_size)
    end = min(len(tokens), position + context_size + 1)
    
    before = tokens[start:position+1]
    after = tokens[position+1:end]
    
    before_text = smol.to_string(before)
    after_text = smol.to_string(after)
    
    return before_text, after_text

# %%
# Process the dataset and compute KL divergence - optimized version
results = []
max_tokens = 500  # Process first 500 tokens per document
batch_size = 8  # Process multiple documents at once
max_docs = 100  # Maximum number of documents to process (set to None for all)
save_every = 100  # Save intermediate results every N documents
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move models to the appropriate device
smol = smol.to(device)
big = big.to(device)

results = []
start_time = time.time()

pile_texts = pile_dataset["text"]

# Filter out texts shorter than 500 tokens more efficiently
filtered_texts = []
# Process in batches to speed up tokenization
for i in range(0, len(pile_texts), batch_size):
    batch = pile_texts[i:i+batch_size]
    # Tokenize all texts in the batch at once
    batch_tokens = [smol.to_tokens(text, truncate=False) for text in batch]
    # Filter based on length
    for j, tokens in enumerate(batch_tokens):
        if tokens.shape[1] >= 500:
            filtered_texts.append(batch[j])
    
    # Early stopping if we have enough documents
    if max_docs is not None and len(filtered_texts) >= max_docs:
        filtered_texts = filtered_texts[:max_docs]
        break

print(f"Filtered {len(pile_texts) - len(filtered_texts)} texts shorter than 500 tokens")

# Then take the requested number of documents
if max_docs is not None and len(filtered_texts) > max_docs:
    filtered_texts = filtered_texts[:max_docs]
    print(f"Taking first {max_docs} documents")

# %%
pile_texts = filtered_texts

# Process in batches
for batch_start in tqdm(range(0, len(pile_texts), batch_size)):
    batch_end = min(batch_start + batch_size, len(pile_texts))
    batch = pile_texts[batch_start:batch_end]
    batch_idx_offset = batch_start

    # Process each document in the batch
    for doc_idx, example in enumerate(batch):
        batch_idx = batch_idx_offset + doc_idx
        text = example

        # Tokenize the text
        tokens = smol.to_tokens(text, truncate=True).to(device)

        # Only process the first max_tokens
        if tokens.shape[1] > max_tokens:
            tokens = tokens[:, :max_tokens]

        # Ensure there's at least one token to predict from
        if tokens.shape[1] <= 1:
            continue

        # Get model outputs
        with torch.no_grad():
            smol_logits = smol(tokens)
            big_logits = big(tokens)

        # Calculate KL divergence for each position
        kl_divs = calculate_kl_divergence(smol_logits[:, :-1], big_logits[:, :-1])

        # Get next token log probabilities (for the actual next token)
        smol_next_token_logprobs = get_next_token_logprob(smol_logits[:, :-1], tokens[:, 1:])
        big_next_token_logprobs = get_next_token_logprob(big_logits[:, :-1], tokens[:, 1:])

        # --- Calculate Ranks ---
        # Get full log probs for all positions except the last
        smol_log_probs_all = F.log_softmax(smol_logits[0, :-1], dim=-1)
        big_log_probs_all = F.log_softmax(big_logits[0, :-1], dim=-1)
        # Get the actual next tokens
        actual_next_tokens = tokens[0, 1:]
        # --- End Rank Calculation Setup ---

        # Collect results for each token
        for pos in range(tokens.shape[1] - 1):  # -1 because we're predicting next tokens
            current_token_id = tokens[0, pos].item()
            next_token_id = tokens[0, pos+1].item()

            # --- Calculate Rank for this position ---
            smol_pos_log_probs = smol_log_probs_all[pos]
            big_pos_log_probs = big_log_probs_all[pos]
            smol_rank = (smol_pos_log_probs >= smol_pos_log_probs[next_token_id]).sum().item()
            big_rank = (big_pos_log_probs >= big_pos_log_probs[next_token_id]).sum().item()
            # --- End Rank Calculation for position ---

            before_text, after_text = get_context(tokens[0].cpu(), pos)

            results.append(
                {
                    "batch_idx": batch_idx,
                    "position": pos,
                    "token": current_token_id, # Current token ID
                    "current_str": smol.to_string(current_token_id),
                    "next_token": next_token_id, # Actual next token ID
                    "next_str": smol.to_string(next_token_id),
                    "kl_divergence": kl_divs[0, pos].item(),
                    "smol_logprob": smol_next_token_logprobs[0, pos].item(),
                    "smol_pred": smol.to_string(smol_logits[0, pos].argmax(dim=-1)),
                    "smol_actual_rank": smol_rank, # Add smol rank
                    "big_logprob": big_next_token_logprobs[0, pos].item(),
                    "big_pred": smol.to_string(big_logits[0, pos].argmax(dim=-1)),
                    "big_actual_rank": big_rank, # Add big rank
                    "context_before": before_text,
                    "context_after": after_text,
                }
            )

    # Optional: Add a progress update every few examples
    if batch_idx % 100 == 0:
        print(f"Processed {batch_idx} documents")

# %%
# Create final DataFrame with results
results_df = pd.DataFrame(results)
results_df["lp_diff"] = results_df["big_logprob"] - results_df["smol_logprob"]

# Display the first few rows
print("DataFrame created with rank columns:")
print(results_df.head())

# Save to CSV
results_df.to_csv('model_comparison_results.csv', index=False)
print(f"Total processing time: {timedelta(seconds=int(time.time() - start_time))}")
# %%

# Create 3D tensor from results
batch_size = max(results_df['batch_idx']) + 1
seq_length = max(results_df['position']) + 1
metrics = torch.zeros((4, batch_size, seq_length))

# Fill the tensor with values
for _, row in results_df.iterrows():
    b = int(row['batch_idx'])
    p = int(row['position'])
    metrics[0, b, p] = row['kl_divergence']
    metrics[1, b, p] = row['smol_logprob']
    metrics[2, b, p] = row['big_logprob']
    metrics[3, b, p] = row['big_logprob'] - row['smol_logprob']

# Create faceted plot
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'KL Divergence',
        'Small Model Log Prob',
        'Large Model Log Prob',
        'Log Prob Difference'
    )
)

# Helper function to add heatmap
def add_heatmap(data, row, col):
    return fig.add_trace(
        go.Heatmap(
            z=data,
            colorscale='Blues',
            showscale=True,
        ),
        row=row, col=col
    )

# Add the four heatmaps
add_heatmap(metrics[0].numpy(), 1, 1)
add_heatmap(metrics[1].numpy(), 1, 2)
add_heatmap(metrics[2].numpy(), 2, 1)
add_heatmap(metrics[3].numpy(), 2, 2)

# Update layout
fig.update_layout(
    height=800,
    width=1000,
    title_text="Model Comparison Metrics"
)

# Update axes labels
for i in range(1, 3):
    for j in range(1, 3):
        fig.update_xaxes(title_text="Position", row=i, col=j)
        fig.update_yaxes(title_text="Batch", row=i, col=j)

# fig.show()
# %%
from neel_plotly import *
AXIS_NAMES = ["KL Divergence", "Small Model Log Prob", "Large Model Log Prob", "Log Prob Difference"]
imshow(metrics[0], title=AXIS_NAMES[0], color_continuous_scale="Blues", zmin=0)
imshow(metrics[3], title=AXIS_NAMES[3])

# %%

sorted_lp_df = results_df.sort_values("lp_diff", ascending=False)
sorted_kl_df = results_df.sort_values("kl_divergence", ascending=False)
nutils.show_df(sorted_lp_df.head(20))
nutils.show_df(sorted_lp_df.tail(20))
nutils.show_df(sorted_kl_df.head(20))
# %%

# Function to process text columns with special characters for better visualization
def process_text_columns(df):
    """
    Replace spaces, newlines, and tabs in text columns with special characters
    for better visualization in notebooks.
    
    Uses the special characters defined in neel.utils:
    - SPACE = "·" for spaces
    - NEWLINE = "↩" for newlines
    - TAB = "→" for tabs
    """
    text_columns = ['current_str', 'next_str', 'context_before', 'context_after', 
                    'smol_pred', 'big_pred']
    
    processed_df = df.copy()
    
    for col in text_columns:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].apply(
                lambda s: s.replace(" ", nutils.SPACE)
                           .replace("\n", nutils.NEWLINE + "\n")
                           .replace("\t", nutils.TAB)
            )
    
    return processed_df

# Process the results DataFrame
processed_results_df = process_text_columns(results_df)

# Use the processed DataFrame for display
sorted_lp_df = processed_results_df.sort_values("lp_diff", ascending=False)
sorted_kl_df = processed_results_df.sort_values("kl_divergence", ascending=False)

# Display the results with processed text
print("=== Tokens with highest log prob difference (big - small) ===")
nutils.show_df(sorted_lp_df.head(20))

print("=== Tokens with lowest log prob difference (big - small) ===")
nutils.show_df(sorted_lp_df.tail(20))

print("=== Tokens with highest KL divergence ===")
nutils.show_df(sorted_kl_df.head(20))
# %%

# You can also use nutils.create_html to visualize the tokens with their values
# For example, to visualize tokens with their KL divergence:
top_kl_examples = sorted_kl_df.head(20)
nutils.create_html(
    top_kl_examples['next_str'].tolist(),
    top_kl_examples['kl_divergence'].tolist(),
    saturation=0.7
)

# Visualize tokens with their log prob difference
top_lp_diff_examples = sorted_lp_df.head(20)
nutils.create_html(
    top_lp_diff_examples['next_str'].tolist(),
    top_lp_diff_examples['lp_diff'].tolist(),
    saturation=0.7
)
# %%
batch_idx = 8
pos_idx = 56

def analyze_token_predictions(batch_idx, pos_idx, top_k=10):
    """
    Analyze token predictions for both models at a specific position.
    
    Args:
        batch_idx: The batch index to analyze
        pos_idx: The position index to analyze
        top_k: Number of top tokens to include from each model
        
    Returns:
        A DataFrame showing the top token predictions from both models
    """
    # Get the text from the specified batch
    text = pile_texts[batch_idx]

    # Tokenize the text
    tokens = smol.to_tokens(text, truncate=True).to(device)

    # Ensure we're within the max_tokens limit
    if tokens.shape[1] > max_tokens:
        tokens = tokens[:, :max_tokens]

    # Get the correct next token
    correct_token = tokens[0, pos_idx + 1].item()
    correct_token_str = smol.to_string([correct_token])

    # Get model outputs
    with torch.no_grad():
        smol_logits = smol(tokens)
        big_logits = big(tokens)

    # Get logits for the position we're interested in
    smol_pos_logits = smol_logits[0, pos_idx]
    big_pos_logits = big_logits[0, pos_idx]

    # Convert to log probabilities
    smol_log_probs = F.log_softmax(smol_pos_logits, dim=-1)
    big_log_probs = F.log_softmax(big_pos_logits, dim=-1)

    # Get top-k tokens for each model
    smol_topk_values, smol_topk_indices = torch.topk(smol_log_probs, top_k)
    big_topk_values, big_topk_indices = torch.topk(big_log_probs, top_k)

    # Calculate and print the total probability mass for top-k tokens
    smol_topk_probs = torch.exp(smol_topk_values)
    big_topk_probs = torch.exp(big_topk_values)

    print(f"Small model total probability for top {top_k} tokens: {smol_topk_probs.sum().item():.4f}")
    print(f"Big model total probability for top {top_k} tokens: {big_topk_probs.sum().item():.4f}")

    # Create a set of all tokens we want to include
    tokens_to_include = set([correct_token])
    tokens_to_include.update(smol_topk_indices.cpu().numpy())
    tokens_to_include.update(big_topk_indices.cpu().numpy())

    # Create a DataFrame with the results
    results = []

    # First add the correct token
    correct_smol_logprob = smol_log_probs[correct_token].item()
    correct_big_logprob = big_log_probs[correct_token].item()

    results.append({
        'token_id': correct_token,
        'token_str': correct_token_str.replace(" ", nutils.SPACE).replace("\n", nutils.NEWLINE),
        'smol_logprob': correct_smol_logprob,
        'big_logprob': correct_big_logprob,
        'is_correct': True,
        'smol_rank': (smol_log_probs >= correct_smol_logprob).sum().item(),
        'big_rank': (big_log_probs >= correct_big_logprob).sum().item(),
    })

    # Then add all other tokens
    for token_id in tokens_to_include:
        if token_id == correct_token:
            continue

        token_str = smol.to_string([token_id])
        smol_logprob = smol_log_probs[token_id].item()
        big_logprob = big_log_probs[token_id].item()

        results.append({
            'token_id': token_id,
            'token_str': token_str.replace(" ", nutils.SPACE).replace("\n", nutils.NEWLINE),
            'smol_logprob': smol_logprob,
            'big_logprob': big_logprob,
            'is_correct': False,
            'smol_rank': (smol_log_probs >= smol_logprob).sum().item(),
            'big_rank': (big_log_probs >= big_logprob).sum().item(),
        })

    # Create DataFrame and sort by big model's log probability
    df = pd.DataFrame(results)

    # Sort with correct token at top, then by big model's log probability
    df = pd.concat([
        df[df['is_correct']],
        df[~df['is_correct']].sort_values('big_logprob', ascending=False)
    ]).reset_index(drop=True)

    # Get context for display
    before_text, after_text = get_context(tokens[0].cpu(), pos_idx)
    before_text = before_text.replace(" ", nutils.SPACE).replace("\n", nutils.NEWLINE)
    after_text = after_text.replace(" ", nutils.SPACE).replace("\n", nutils.NEWLINE)

    print(f"Context before: {before_text}")
    print(f"Context after: {after_text}")
    print(f"KL divergence: {calculate_kl_divergence(smol_pos_logits.unsqueeze(0), big_pos_logits.unsqueeze(0)).item():.4f}")

    nutils.show_df(df)
    return df

# Test the function with the specified batch and position
token_predictions_df = analyze_token_predictions(batch_idx, pos_idx)
# nutils.show_df(token_predictions_df)

# %%
batch_idx = 1
text = pile_texts[batch_idx]
pos_idx = 381

def visualize_text_with_kl(batch_idx, highlight_pos=None, window_size=50, saturation=0.7):
    """
    Visualize a text with tokens colored by KL divergence between models.
    
    Args:
        batch_idx: The batch index of the text to visualize
        highlight_pos: Optional position to highlight/bold
        window_size: Number of tokens to show around the highlighted position
        saturation: Color saturation for the visualization
        
    Returns:
        HTML visualization of the text with tokens colored by KL divergence
    """
    # Get the text and filter results for this batch
    text = pile_texts[batch_idx]
    batch_results = results_df[results_df['batch_idx'] == batch_idx]
    
    # Tokenize the text
    tokens = smol.to_tokens(text, truncate=True).to(device)
    if tokens.shape[1] > max_tokens:
        tokens = tokens[:, :max_tokens]
    
    # Get token strings
    token_strings = [smol.to_string([tokens[0, i].item()]) for i in range(tokens.shape[1])]
    
    # Process tokens to show special characters
    processed_tokens = [t.replace(" ", nutils.SPACE).replace("\n", nutils.NEWLINE + "\n").replace("\t", nutils.TAB) for t in token_strings]
    
    # Get KL divergences for each position
    kl_values = batch_results['kl_divergence'].tolist()
    
    # If we have a highlight position, create a window around it
    if highlight_pos is not None:
        start_pos = max(0, highlight_pos - window_size // 2)
        end_pos = min(len(processed_tokens), highlight_pos + window_size // 2)
        
        # Slice the tokens and KL values to the window
        processed_tokens = processed_tokens[start_pos:end_pos]
        kl_values = kl_values[start_pos:min(end_pos-1, len(kl_values)-1)]
        
        # Adjust the highlight position for the window
        highlight_pos = highlight_pos - start_pos
    
    # Create HTML with optional highlighting
    html = ""
    for i, token in enumerate(processed_tokens):
        # Skip the last token as we don't have KL for it
        if i >= len(kl_values):
            # Add without coloring for the last token
            if i == highlight_pos:
                html += f'<span style="font-weight: bold; border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{token}</span>'
            else:
                html += f'<span style="border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{token}</span>'
            continue
            
        # Get the KL value for coloring
        kl_value = kl_values[i]
        
        # Scale the KL value for visualization
        scaled_value = min(kl_value * saturation, saturation)
        
        # Create RGB color based on KL value (blue for high KL)
        hue = 0.66  # Blue hue
        rgb_color = colorsys.hsv_to_rgb(hue, scaled_value, 1)
        hex_color = "#%02x%02x%02x" % (
            int(rgb_color[0] * 255),
            int(rgb_color[1] * 255),
            int(rgb_color[2] * 255),
        )
        
        # Add bold styling if this is the highlighted position
        if i == highlight_pos:
            html += f'<span style="background-color: {hex_color}; font-weight: bold; border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{token}</span>'
        else:
            html += f'<span style="background-color: {hex_color}; border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{token}</span>'
    
    # Display the HTML
    display(HTML(html))
    
    # If highlighting a position, also show the token predictions
    if highlight_pos is not None:
        actual_pos = highlight_pos if start_pos == 0 else highlight_pos + start_pos
        print(f"\nToken predictions at position {actual_pos}:")
        analyze_token_predictions(batch_idx, actual_pos)
    
    return html

# Test the visualization function
# Without highlighting
visualize_text_with_kl(batch_idx=1)

# With highlighting
visualize_text_with_kl(batch_idx=1, highlight_pos=pos_idx)
# %%
px.histogram(results_df, marginal="box", x="lp_diff").show()
px.histogram(results_df, marginal="box", x="kl_divergence").show()
# %%
print(f"{results_df['lp_diff'].sum()=}")
print(f"{results_df['lp_diff'].mean()=}")
line(results_df["lp_diff"].sort_values().cumsum() / results_df["lp_diff"].sum())
# %%
import copy
temp_df = copy.deepcopy(results_df[["lp_diff"]])
temp_df["abs_lp_diff"] = temp_df["lp_diff"].abs()
temp_df = temp_df.sort_values("abs_lp_diff", ascending=True)
temp_df["lp_diff_cumsum"] = (
    temp_df["abs_lp_diff"].cumsum() / temp_df["abs_lp_diff"].sum()
)
print(f"{temp_df['lp_diff'].std() / temp_df['lp_diff'].sum()=}")
px.scatter(temp_df, x="lp_diff", y="lp_diff_cumsum")

# %%
results_df["lp_diff_norm"] = results_df["lp_diff"] / results_df["lp_diff"].abs().sum()
# Create a 2D histogram (density heatmap)
fig_2d_hist = px.density_heatmap(
    results_df,
    x="big_logprob",
    y="smol_logprob",
    z="lp_diff_norm",          # Values to aggregate for color
    histfunc="sum",       # Aggregation function: sum
    nbinsx=100,            # Adjust number of bins as needed
    nbinsy=50,
    title="2D Histogram of Log Probabilities (Color = Sum of Log Prob Difference)",
    labels={
        "big_logprob": "Big Model Log Probability",
        "smol_logprob": "Small Model Log Probability",
        "color": "Sum of Log Prob Difference (Big - Small)" # Label for the color bar
    },
    color_continuous_scale="RdBu", # Use a diverging colorscale (Red-Blue)
    color_continuous_midpoint=0,
)

# Add a line y=x for reference (where log probs are equal)
min_val = min(results_df['big_logprob'].min(), results_df['smol_logprob'].min())
max_val = max(results_df['big_logprob'].max(), results_df['smol_logprob'].max())
fig_2d_hist.add_shape(
    type="line",
    x0=min_val, y0=min_val,
    x1=max_val, y1=max_val,
    line=dict(color="grey", width=1, dash="dash"),
)


fig_2d_hist.show()
# %%

# --- Filter the DataFrame based on the specified conditions ---

if 'big_actual_rank' in results_df.columns and 'smol_actual_rank' in results_df.columns:
    filtered_df = results_df[
        (results_df['big_actual_rank'] == 1) &
        (results_df['smol_actual_rank'] > 50)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning if modifying later

    print(f"\nFound {len(filtered_df)} tokens where the actual next token is rank 1 for Big model but >50 for Small model.")

    # Display the filtered results (using the processed text version for readability)
    if not filtered_df.empty:
        processed_filtered_df = process_text_columns(filtered_df)
        nutils.show_df(processed_filtered_df.sort_values('smol_actual_rank', ascending=False))
    else:
        print("No tokens matched the criteria.")
else:
    print("Rank columns not found in DataFrame. Calculation might have failed.")

# %%

# --- Visualize Context and Top Predictions for Filtered Examples ---

def visualize_context_with_lp_diff(batch_idx, end_position, saturation=0.7, max_abs_diff=None):
    """
    Visualize text context, coloring each token by the log prob difference
    assigned *to that token* during prediction.

    Args:
        batch_idx: The batch index of the text.
        end_position: The index of the token *after* the last token to display.
                      (i.e., display tokens 0 to end_position-1)
        saturation: Color saturation.
        max_abs_diff: Optional value to normalize colors against. If None, uses max abs diff in the sequence.
    """
    # Get the text and tokenize up to the end position
    text = pile_texts[batch_idx]
    tokens = smol.to_tokens(text, truncate=True).to(device)
    if tokens.shape[1] > max_tokens:
        tokens = tokens[:, :max_tokens]

    # Ensure end_position is valid
    end_position = min(end_position, tokens.shape[1])
    display_tokens = tokens[0, :end_position]

    # Get token strings and process them
    token_strings = [smol.to_string([t.item()]) for t in display_tokens]
    processed_tokens = [t.replace(" ", nutils.SPACE).replace("\n", nutils.NEWLINE + "\n").replace("\t", nutils.TAB) for t in token_strings]

    # Get the relevant lp_diff values from results_df
    # We need lp_diff for positions 0 to end_position-2 to color tokens 1 to end_position-1
    lp_diff_values = results_df[
        (results_df['batch_idx'] == batch_idx) &
        (results_df['position'] < end_position - 1) # Positions 0 to end_pos-2
    ]['lp_diff'].tolist()

    # Determine normalization factor for colors
    if max_abs_diff is None:
        if lp_diff_values:
            max_abs_diff = max(abs(v) for v in lp_diff_values) + 1e-6 # Avoid division by zero
        else:
            max_abs_diff = 1.0

    # Create HTML
    html = ""
    for i, token_str in enumerate(processed_tokens):
        if i == 0:
            # First token wasn't predicted, display without color
            html += f'<span style="border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{token_str}</span>'
        else:
            # Get lp_diff for predicting this token (stored at position i-1)
            lp_diff = lp_diff_values[i-1]

            # Scale the value (-1 to 1 range for color mapping)
            scaled_value = lp_diff / max_abs_diff * saturation
            scaled_value = max(-saturation, min(saturation, scaled_value)) # Clamp to [-saturation, saturation]

            # Map to color (Blue for positive diff/Big > Small, Red for negative diff/Small > Big)
            if scaled_value >= 0:
                hue = 0.66  # Blue
                sat = scaled_value
            else:
                hue = 0.0  # Red
                sat = -scaled_value

            rgb_color = colorsys.hsv_to_rgb(hue, sat, 1)
            hex_color = "#%02x%02x%02x" % (
                int(rgb_color[0] * 255),
                int(rgb_color[1] * 255),
                int(rgb_color[2] * 255),
            )

            html += f'<span style="background-color: {hex_color}; border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{token_str}</span>'

    display(HTML(html))

# Get the top 10 rows from the filtered DataFrame (sorted by smol_actual_rank descending)
if 'filtered_df' in locals() and not filtered_df.empty:
    # Use the original filtered_df to get numerical indices easily
    top_filtered_rows_indices = filtered_df.sort_values('smol_actual_rank', ascending=False).head(10).index
    top_filtered_rows_for_display = processed_filtered_df.loc[top_filtered_rows_indices] # Use processed for display info

    print("\n--- Context Visualization & Top Predictions for Filtered Examples ---")
    print("(Tokens colored by Log Prob Difference [Big - Small] for predicting *that* token)")
    print("(Blue = Big predicted higher prob, Red = Small predicted higher prob)")

    for index, row in top_filtered_rows_for_display.iterrows():
        batch_idx = int(row['batch_idx'])
        position = int(row['position']) # This is the position *before* the token of interest

        print(f"\nExample (Batch: {batch_idx}, Position: {position}, Next Token: '{row['next_str']}', Smol Rank: {int(row['smol_actual_rank'])}, Big Rank: {int(row['big_actual_rank'])})")

        # Visualize context up to and including the 'next_token'
        visualize_context_with_lp_diff(batch_idx, position + 2)

        # --- Calculate and Print Top 5 Predictions ---
        print(f"  Top 5 Predictions at Position {position}:")

        # Get the text and tokenize
        text = pile_texts[batch_idx]
        tokens = smol.to_tokens(text, truncate=True).to(device)
        if tokens.shape[1] > max_tokens:
            tokens = tokens[:, :max_tokens]

        # Rerun models to get logits for this specific position
        # (Could be optimized if logits were cached, but this is simpler)
        if position < tokens.shape[1] -1: # Ensure the position is valid
            with torch.no_grad():
                # Run up to the position of interest to get logits predicting the next token
                smol_logits_pos = smol(tokens[:, :position+1])[0, -1] # Get logits at the last position run
                big_logits_pos = big(tokens[:, :position+1])[0, -1]

            # Calculate log probs and probs for both models
            smol_log_probs = F.log_softmax(smol_logits_pos, dim=-1)
            big_log_probs = F.log_softmax(big_logits_pos, dim=-1)

            smol_probs = F.softmax(smol_logits_pos, dim=-1)
            big_probs = F.softmax(big_logits_pos, dim=-1)

            # Get Top 5 for Small Model
            smol_top_vals, smol_top_indices = torch.topk(smol_log_probs, 5)
            print("    Small Model:")
            for i in range(5):
                token_id = smol_top_indices[i].item()
                token_str = smol.to_string([token_id]).replace(" ", nutils.SPACE).replace("\n", nutils.NEWLINE)
                log_prob = smol_top_vals[i].item()
                prob = smol_probs[token_id].item() # Get prob using the index
                print(f"      {i+1}. '{token_str}' (LogProb: {log_prob:.3f}, Prob: {prob:.3f})")

            # Get Top 5 for Big Model
            big_top_vals, big_top_indices = torch.topk(big_log_probs, 5)
            print("    Big Model:")
            for i in range(5):
                token_id = big_top_indices[i].item()
                token_str = smol.to_string([token_id]).replace(" ", nutils.SPACE).replace("\n", nutils.NEWLINE)
                log_prob = big_top_vals[i].item()
                prob = big_probs[token_id].item() # Get prob using the index
                print(f"      {i+1}. '{token_str}' (LogProb: {log_prob:.3f}, Prob: {prob:.3f})")
        else:
             print("    Position out of bounds for prediction.")
        print("-" * 20) # Separator

else:
    print("\nFiltered DataFrame 'filtered_df' not found or is empty. Cannot visualize examples.")

# %%
