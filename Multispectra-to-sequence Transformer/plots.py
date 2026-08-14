import pandas as pd
import matplotlib.pyplot as plt 

def plot_errors(targets, predictions, sequence_lengths, errors):
    """Function to visualize number of errors in generated sequences.
    """ 

    error_df = pd.DataFrame()
    error_df.insert(0, "target sequence", targets)
    error_df.insert(1, "predicted sequence", predictions)
    error_df.insert(2, "sequence_length", sequence_lengths)
    error_df.insert(3, "num_errors", errors)

    # plot count of number of errors 
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    error_idxs = error_df['num_errors'].value_counts().sort_index()
    axs[0].bar(error_idxs.index, error_idxs.values)
    axs[0].set_title("Count of Number of Errors")
    axs[0].set_xlabel("Number of Errors")
    axs[0].set_ylabel("Count")

    # plot avg number of errors vs. sequence length
    avg_error = error_df[['sequence_length', 'num_errors']].groupby('sequence_length').mean()
    axs[1].bar(avg_error.index, avg_error['num_errors'])
    axs[1].set_title("Avg. Number of Errors vs. Sequence Length")
    axs[1].set_xlabel("Sequence Length")
    axs[1].set_ylabel("Average Number of Errors")

    return error_df, fig