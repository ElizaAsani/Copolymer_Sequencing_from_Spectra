"""
Code for greedy inference
"""

import torch

from SequenceEncoder import SeqDataset

def build_decoder_mask(decoder_input, device):
    """Builds the (padding & causal) decoder mask for the current decoder_input.

    decoder_input  : (B, input_len) token ids
    returns        : (B, 1, input_len, input_len) boolean mask
    """
    input_len = decoder_input.size(1)
    causal_mask = SeqDataset.causal_mask(input_len).to(device)
    padding_mask = (decoder_input != SeqDataset.pad_token).unsqueeze(1).unsqueeze(1)
    return padding_mask & causal_mask.unsqueeze(0).unsqueeze(0)

def inference(model, test_dl, device='cpu'):
    """Greedy inference for a given model and test dataloader.
    Returns lists of target sequences, predicted sequences, sequence lengths, and lambdas.
    """
    target_sequences = []
    predicted_sequences = []
    sequence_lengths = []
    lambdas = []
    dataset = test_dl.dataset.dataset
    max_len = dataset.model_max_sequence_length

    for _, data in enumerate(test_dl):

        encoder_input = {key: value.to(device) for key, value in data['encoder_input'].items()}
        sequence = data['sequence']
        length = data['length']
        lambd = data['lambda']

        B = len(sequence)
        sos_token = SeqDataset.sos_token
        decoder_input = torch.full((B, 1), sos_token).to(device)
        input_len = decoder_input.size(1)
        finished = torch.zeros(B, dtype=torch.bool).to(device)

        with torch.no_grad():
            while input_len < max_len and not finished.all():
                # set up decoder mask for current input length
                decoder_mask_subset = build_decoder_mask(decoder_input, device)

                output = model.forward(encoder_input, decoder_input, decoder_mask_subset) # (B, input_len, d_vocab)
                next_token, _ = SeqDataset.reconstruct(output[:,-1,:]) # (B)

                # if sequence finished, set next token to pad; update finished flag
                next_token = torch.where(finished, SeqDataset.pad_token, next_token) 
                finished = finished | (next_token == SeqDataset.eos_token) 

                decoder_input = torch.concat([decoder_input, next_token.unsqueeze(-1)], dim=1)
                input_len = decoder_input.size(1)

        # convert targets and predictions to sequences and append to outer lists
        predicted_sequences.extend(dataset.decodeSeq(seq.tolist()) for seq in decoder_input)
        target_sequences.extend(sequence)
        sequence_lengths.extend(length.tolist())
        lambdas.extend(lambd.tolist())

    return target_sequences, predicted_sequences, sequence_lengths, lambdas
