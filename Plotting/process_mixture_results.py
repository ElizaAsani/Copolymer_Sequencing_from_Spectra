import os
import ast

import pandas as pd
import h5py

def to_list(x):
    """Convert stringified list → Python list safely."""
    if isinstance(x, list):
        return x
    return ast.literal_eval(x)

def load_and_flatten(path, model_name):
    df = pd.read_csv(path)
    rows = []

    for mix_id, row in df.iterrows():
        preds = to_list(row["Predicted Sequences"])
        scores = to_list(row["Scores"])

        for i, (p, s) in enumerate(zip(preds, scores)):
            rows.append({
                "Mixture": mix_id,
                "Rank": i,
                f"{model_name}_Predicted": p,
                f"{model_name}_Score": round(s, 4)
            })

    return pd.DataFrame(rows)

def merge_all(folders, lamb):

    flattened_dfs = {}
    for spectrum, folder in folders.items():
        flattened_dfs[spectrum] = load_and_flatten(os.path.join(folder, f"lambda_{lamb}", "beam_search.csv"), spectrum)

    df = flattened_dfs["MS"].merge(flattened_dfs["NMR"], on=["Mixture", "Rank"]) \
           .merge(flattened_dfs["UV-Vis"],  on=["Mixture", "Rank"]) \
           .merge(flattened_dfs["All"],on=["Mixture", "Rank"])

    df = df[[
        "Mixture", 
        "NMR_Predicted", "NMR_Score",
        "MS_Predicted", "MS_Score",
        "UV-Vis_Predicted", "UV-Vis_Score",
        "All_Predicted", "All_Score"
    ]]

    return df

def load_and_flatten_targets(path):
    """Load target sequences from a file."""
    with h5py.File(path, "r") as f:
        sequences = [[s.decode('utf-8') for s in seqs] for seqs in f['sequence'][:]]
        ratios = f['ratio'][:]
        lambdas = f['lambda'][:]

    ids = []
    lambs = []
    seqs = []
    rats = []

    for i, (seq, ratio, lamb) in enumerate(zip(sequences, ratios, lambdas)):
        for s, r in zip(seq, ratio):
            ids.append(i)
            lambs.append(lamb)
            seqs.append(s)
            rats.append(r)

    return pd.DataFrame({"Mixture": ids, "Lambda": lambs, "Sequence": seqs, "Ratio": rats})

def calculate_mixture_stats(seqs, weights):
    """Calculate statistics for each mixture."""

    f_D = 0
    L = 0
    L_b = 0

    for seq, weight in zip(seqs, weights):
        if len(seq) == 0:
            print(f"ERROR: Empty sequence found, score: {weight}")
            continue

        if weight > 0:
            f_D += calcFracD(seq) * weight
            L += len(seq) * weight
            L_b += calcAvgBlockLength(seq) * weight

    return round(f_D, 6), round(L, 6), round(L_b, 6)

def get_weights(scores):
    """Calculate weights based on scores."""
    total = sum(scores)
    return [s/total for s in scores]

def calcAvgBlockLength(sequence):
    """Calculates average block length for a given copolymer sequence. 
    """
    numBlocks = 1    # counter for number of blocks
    
    # increment block counter every time a new block is encountered    
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i - 1]:
            numBlocks += 1
            
    return (len(sequence) / numBlocks)

def calcFracD(sequence):
    D = sequence.count('D')
    return D / len(sequence)

def process_mixture_results(mixture_folders, mixtures_file_template, lambdas):

    # rename mixture_folders['UV-Vis + MS + NMR'] to mixture_folders['All'] for clarity
    mixture_folders['All'] = mixture_folders.pop('UV-Vis + MS + NMR')
    all_folder = mixture_folders['All']

    for lamb in lambdas:
        out_folder = os.path.join(all_folder, f"lambda_{lamb}")

        results_df = merge_all(mixture_folders, lamb)
        results_df.to_csv(os.path.join(out_folder, "all_results.csv"), index=False)

        targets_df = load_and_flatten_targets(mixtures_file_template.format(lamb=lamb))
        targets_df.to_csv(os.path.join(out_folder, "targets.csv"), index=False)

        # write stats
        stats = {name: results_df.groupby("Mixture").apply(lambda x, name=name: calculate_mixture_stats(x[f"{name}_Predicted"], get_weights(x[f"{name}_Score"]))) 
                 for name in mixture_folders.keys()}
        target_stats = targets_df.groupby("Mixture").apply(lambda x: calculate_mixture_stats(x["Sequence"], x["Ratio"]))

        # write stats to csv
        METRICS = [(0, "f_D"), (1, "lengths"), (2, "block_lengths")]

        for metric_idx, filename in METRICS:
            df = pd.DataFrame({"Mixture": stats['All'].index})
            df['NMR'] = stats['NMR'].apply(lambda x: x[metric_idx])
            df['MS'] = stats['MS'].apply(lambda x: x[metric_idx])
            df['UV-Vis'] = stats['UV-Vis'].apply(lambda x: x[metric_idx])
            df['UV-Vis + MS + NMR'] = stats['All'].apply(lambda x: x[metric_idx])
            df['Target'] = target_stats.apply(lambda x: x[metric_idx])
            df.to_csv(os.path.join(out_folder, filename + ".csv"), index=False)

    # rename mixture_folders['All'] to mixture_folders['UV-Vis + MS + NMR']
    mixture_folders['UV-Vis + MS + NMR'] = mixture_folders.pop('All')
       
    return