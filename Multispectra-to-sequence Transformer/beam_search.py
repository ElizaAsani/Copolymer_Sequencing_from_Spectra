"""
Code for beam search decoding:

beam_search()            : beam search + accuracy scoring for the
                            single-sequence test set
beam_search_mixtures()   : beam search for the mixtures dataset
_batched_beam_search()   : the shared core search loop used by both
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from SequenceEncoder import SeqDataset
from evaluate import get_accuracy
from decode import build_decoder_mask

def beam_search(model, test_dl, device='cpu', beam_width=5, alpha=0):
    # beam search
    target_sequences = []    # shape: (B)
    predicted_sequences = [] # shape: (B, beam_width)
    scores = []              # shape: (B, beam_width)
    dataset = test_dl.dataset.dataset
    max_len = dataset.model_max_sequence_length
    vocab_size = dataset.num_chars

    for _, data in enumerate(test_dl):

        encoder_input = {key: value.to(device) for key, value in data['encoder_input'].items()}
        sequence = data['sequence']
        B = len(sequence)
        
        with torch.no_grad():
            beams, batch_scores = _batched_beam_search(model, encoder_input, vocab_size, batch_size=B,
                                                      n_steps=max_len, beam_width=beam_width, alpha=alpha, device=device)

        target_sequences.extend(list(sequence))
        predicted_sequences.extend([[dataset.decodeSeq(seq.tolist()) for seq in row] for row in beams])
        scores.extend(batch_scores.tolist())

    # calculate accuracy
    hit_idxs = []
    errors_total = []
    num_seq = len(target_sequences)

    for target, predictions, _ in zip(target_sequences, predicted_sequences, scores):
        seq_acc, _, _, _, errors = get_accuracy(predictions, [target] * len(predictions))
        if seq_acc == 0:
            hit_idxs.append(-1)
        else:
            try:
                forward_hit_idx = predictions.index(target)
            except ValueError:
                forward_hit_idx = beam_width
            try:
                reverse_hit_idx = predictions.index(target[::-1]) 
            except ValueError:
                reverse_hit_idx = beam_width
            hit_idx = min(forward_hit_idx, reverse_hit_idx)          
            hit_idxs.append(hit_idx)   
        errors_total.append(errors)
    
    beam_df = pd.DataFrame({
        "Target Sequence": target_sequences,
        "Predicted Sequences": predicted_sequences,
        "Scores": scores,
        "Num Errors": errors_total
    })

    # calculate percent hits
    num_hits = sum([1 for hit_idx in hit_idxs if 0 <= hit_idx < beam_width])
    percent_hits = num_hits / num_seq * 100

    num_top_5_hits = sum([1 for hit_idx in hit_idxs if 0 <= hit_idx < 5])
    percent_top_5_hits = num_top_5_hits / num_seq * 100

    num_top_1_hits = sum([1 for hit_idx in hit_idxs if hit_idx == 0])
    percent_top_1_hits = num_top_1_hits / num_seq * 100

    return beam_df, percent_hits, percent_top_5_hits, percent_top_1_hits

def beam_search_mixtures(model, dataset, device='cpu', beam_width=5, alpha=0):
    # beam search for mixtures dataset, which has multiple sequences per spectra and lambda values. 
    B = len(dataset)
    target_sequences = [dataset[i]['sequence'] for i in range(B)]    # shape: (B, num_seqs)
    predicted_sequences = [] # shape: (B, beam_width)
    scores = []              # shape: (B, beam_width)
    ratios = [[round(rat, 6) for rat in dataset[i]['ratio'].tolist()] for i in range(B)]   # shape: (B, num_seqs)
    max_len = dataset.model_max_sequence_length
    vocab_size = dataset.num_chars
    
    for i in range(0, B, 32):
        with torch.no_grad():
            batch_size = min(32, B - i)
            encoder_input = {key: torch.stack([dataset[i]['encoder_input'][key] for i in range(i, i+batch_size)]).to(device)
                    for key in dataset[0]['encoder_input']}
            beams, batch_scores = _batched_beam_search(model, encoder_input, vocab_size, batch_size=batch_size,
                                                        n_steps=max_len, beam_width=beam_width, alpha=alpha, device=device)

        predicted_sequences.extend([[dataset.decodeSeq(seq.tolist()) for seq in row] for row in beams])
        scores.extend(batch_scores.tolist())

    beam_df = pd.DataFrame({
        "Target Sequence": target_sequences,
        "Predicted Sequences": predicted_sequences,
        "Scores": scores,
        "Ratios": ratios
    })

    return beam_df

def _batched_beam_search(model, encoder_input, vocab_size, batch_size, n_steps, beam_width, alpha, device='cpu'):
    """
    n_steps = sequence_length
    beam_width = number of sequences to keep at each step
    alpha = exponent for length normalization (0 = none, 1 = complete)
    sos_token, eos_token = start and end of sequence tokens

    Tensor sizes
    ----------
    encoder_input : {3x(batch_size, spec_length)}
    decoder_mask : (1, 1, i+1, i+1)
    decoder_input : (batch_size*beam_width, i+1)
    
    output_logits : (batch_size*beam_width, 1, output_size) # raw logits
    next_token : (batch_size*beam_width, 1)

    beam : (batch_size, beam_width, sequence_length) # for each sequence, for each beam, stores indices of characters
    scores : (batch_size, beam_width)                # scores of sequences, i.e. log probabilities of characters 
    finished : (batch_size, beam_width)              # boolean, whether the sequence is finished
    """

    sos_token = SeqDataset.sos_token
    eos_token = SeqDataset.eos_token
    pad_token = SeqDataset.pad_token
    
    # initialize outputs, scores, and finished
    beam = torch.empty(batch_size, beam_width, 0, dtype=torch.long).to(device)
    scores = torch.zeros((batch_size, beam_width)).to(device)

    # initialized finished flag and length tracker
    finished = torch.zeros(batch_size, beam_width, dtype=torch.bool).to(device)
    length = torch.ones(batch_size, beam_width, dtype=torch.long).to(device)    # start at one for first decoded token

    # initialize input characters to SOS token
    input_char = torch.full((batch_size*beam_width, 1), sos_token).to(device)    

    # expand encoder input to beam
    encoder_input = {key: value.repeat_interleave(beam_width, dim=0) for key, value in encoder_input.items()} # shape: {3x(B*K, spec_length)}
    
    for i in range(n_steps):
        # set up decoder input and mask
        decoder_input = torch.concat((input_char, beam.view(batch_size*beam_width, -1)), dim=1) # shape: (B*K, i+1)
        decoder_mask_subset = build_decoder_mask(decoder_input, device)

        # generate a raw logits, unflatten, and turn into scores
        output_logits = model.forward(encoder_input, decoder_input, decoder_mask_subset) # shape: (B*K, i+1, V)
        output_logits = output_logits[:,-1,:].detach().view(batch_size, beam_width, vocab_size) # shape: (B*K, V) -> (B, K, V)
        output_scores = F.log_softmax(output_logits, dim=-1)  

        # for first iteration, only allow first sequence to be expanded to avoid duplicates
        if i == 0:
            output_scores[:, 1:, :] = -float('inf') 

        # add scores to the current scores
        output_scores = output_scores + scores.unsqueeze(-1) # broadcasting: (B, K, V) + (B, K, 1) -> (B, K, V) 

        # mask finished sequence with original score on pad character and -inf on all other characters
        if torch.any(finished):
            finished_mask = finished.unsqueeze(-1) # (B, K, 1)
            output_scores = output_scores.masked_fill_(finished_mask, -float('inf'))
            output_scores[:, :, pad_token] = torch.where(condition=finished, input=scores, other=output_scores[:, :, pad_token])

        # length normalization
        norm_scores = output_scores / (length.unsqueeze(-1)**alpha) # broadcasting: (B, K, V) / (B, K, 1) -> (B, K, V)

        # flatten logits and scores for each sequence 
        output_scores = output_scores.view(batch_size, -1)  # shape: (B, K*V)
        norm_scores = norm_scores.view(batch_size, -1)      # shape: (B, K*V)

        # get top k scores and indices (i.e. top beam expansions for each sequence)
        _, top_k_indices = norm_scores.topk(beam_width, dim=-1) # shape: (B, K)
        beam_indices = top_k_indices // vocab_size 
        char_indices = top_k_indices % vocab_size  

        # collect and update top k sequences
        if i == 0:
            beam = char_indices.unsqueeze(-1)
        else:     
            beam = beam.gather(1, beam_indices.unsqueeze(-1).expand(batch_size, beam_width, i)) # shape: (B, K, i)
            beam = torch.cat((beam, char_indices.unsqueeze(-1)), dim=-1) # shape: (B, K, i+1)
        
        # update scores
        scores = output_scores.gather(1, top_k_indices)

        # update finished flag by searching for eos in all sequences 
        finished = finished.gather(1, beam_indices) # shape: (B, K)
        new_finished = (char_indices == eos_token) | (char_indices == pad_token)
        finished = finished | new_finished
        
        # update length tracker
        length = length.gather(1, beam_indices)
        length += (~finished).long() # only increment sequences that haven't finisehd

    # convert scores to probabilities
    scores = torch.exp(scores)

    return beam, scores

def plot_beam_histogram(beam_search_csv, threshold=1):
    # plot histogram of decodings
    import matplotlib.pyplot as plt
    import ast

    fig, axs = plt.subplots(2, 5, layout="constrained", figsize=(20,6))
    fig.suptitle("Decoding Distribution for Individual Copolymer Sequences", fontsize=16)
    style = {'facecolor': '#99CCFF', 'edgecolor': 'C0', 'linewidth': 3}
    style_2 = {'facecolor': '#9FD1AC', 'edgecolor': '#095911', 'linewidth': 3}
    
    # load dataset
    decodings = pd.read_csv(beam_search_csv)

    i = 0
    # enumerate through axes on plot
    for ax in axs.flatten():
        # get a sequence whose decodings are not certain
        continue_search = True
        while continue_search:
            score = ast.literal_eval(decodings['Scores'][i])
            if max(score) < threshold:
                continue_search = False
            else:
                i += 1
        continue_search = True
        
        # get decodings for single sequence
        seq = decodings['Target Sequence'][i]   
        predictions = decodings['Predicted Sequences'][i]
        scores = decodings['Scores'][i]
        num_errors = decodings['Num Errors'][i] 

        # convert predictions to array of strings
        predictions = ast.literal_eval(predictions)
        scores = np.round(ast.literal_eval(scores), 4)
        num_errors = ast.literal_eval(num_errors)

        # create dataframe of predictions, scores, and errors
        scores = {'Predictions': predictions, 'Scores': scores, 'Errors': num_errors}
        scores_df = pd.DataFrame(scores)
        scores_df = scores_df.sort_values(by='Errors')
        
        # generate text box of predictions
        textBox = "Predictions: \n"
        for pred_seq, score in zip(scores_df['Predictions'], scores_df['Scores']):
            if pred_seq == seq:
                textBox += f"*{pred_seq}: {score} \n"
            else:
                textBox += f"{pred_seq}: {score} \n"

        # plot histogram as bar chart
        container = ax.bar(scores_df['Predictions'], scores_df['Scores'], **style)
        # color the correct bar 
        try:
            correct_idx = scores_df['Predictions'].tolist().index(seq)
            container.patches[correct_idx].set(**style_2, label='correct')
        except ValueError:
            pass
        try:
            correct_idx = scores_df['Predictions'].tolist().index(seq[::-1]) # try reverse match
            container.patches[correct_idx].set(**style_2, label='correct')
        except ValueError:
            pass
        ax.set_title(f'Expected: {seq}')
        ax.set_ylabel('Score')       
        ax.set_xlabel('Number of Errors') 
        # set ticks to be the number of errors
        ax.set_xticks(np.arange(0, len(scores_df)), labels=scores_df['Errors'])
        ax.set_ylim(0, 1)
        ax.text(0.95, 0.95, textBox, transform=ax.transAxes, fontsize=8, horizontalalignment='right', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))        
    
        i += 1

    return fig