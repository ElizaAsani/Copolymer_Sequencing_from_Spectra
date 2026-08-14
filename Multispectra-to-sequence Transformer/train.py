"""
Training Transformer 
"""
import matplotlib.pyplot as plt
import copy 

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from SequenceEncoder import SeqDataset

def train_model(model, train_dl, test_dl, epochs, d_model, chkpt_path, device='cpu', warmup=100, patience=25):
    train_loss = []
    validate_loss = []    
    patience_counter = 0

    # store best model weights
    best_val_loss = float('inf')
    best_val_model_wts = copy.deepcopy(model.state_dict())

    # define optimizer
    optimizer = Adam(model.parameters(), lr=1)
    
    def lr_lambda(step, warmup_steps=4000):
        if (step == 0):
            step = 1
        return d_model**(-0.5) * min(step**(-0.5), step*warmup_steps**(-1.5))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # define loss criterion
    criterion = CrossEntropyLoss(ignore_index=SeqDataset.pad_token, reduction='none')
    
    # enumerate epochs
    for epoch in range(epochs):
        # train
        model.train()
        running_train_loss = 0
        # enumerate mini batches 
        for _, data in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # move data to device
            encoder_input = {key: value.to(device) for key, value in data['encoder_input'].items()}
            decoder_input = data['decoder_input'].to(device)
            decoder_mask = data['decoder_mask'].to(device)
            sequence_lengths = data['length'].to(device)
            label = data['label'].to(device)
            # forward pass of model
            decoder_output = model(encoder_input, decoder_input, decoder_mask)
            # calculate batch-average loss
            loss = calculate_loss(criterion, decoder_output, label, sequence_lengths) 
            # append normalized loss to running total
            running_train_loss += loss.item()
            # backpropagate 
            loss.backward()
            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            # update model weights
            optimizer.step()
            # update learning rate
            scheduler.step()
        
        train_loss.append(running_train_loss / len(train_dl))

        # validate
        model.eval()
        running_test_loss = 0
        with torch.no_grad():
            for _, data in enumerate(test_dl):
                encoder_input = {key: value.to(device) for key, value in data['encoder_input'].items()}
                decoder_input = data['decoder_input'].to(device)
                decoder_mask = data['decoder_mask'].to(device)
                sequence_lengths = data['length'].to(device)
                label = data['label'].to(device)
                decoder_output = model(encoder_input, decoder_input, decoder_mask)
                loss = calculate_loss(criterion, decoder_output, label, sequence_lengths)
                running_test_loss += loss.item()

        validate_loss.append(running_test_loss / len(test_dl))

        # save best model weights   
        if validate_loss[-1] < best_val_loss:
            patience_counter = 0
            best_val_loss = validate_loss[-1]
            best_val_model_wts = copy.deepcopy(model.state_dict())
        elif epoch > warmup:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        if (epoch % 5 == 0):
            print(f"Epoch: {epoch}, Loss: {train_loss[-1]}" 
                  + f", Validation loss : {validate_loss[-1]}")
        if (epoch % 50 == 0) and (epoch > 0):
            torch.save(model.state_dict(), f"{chkpt_path}/epoch_{epoch}.pt")
        
    loss_fig, ax = plt.subplots()
    ax.set_title("Cross Entropy Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.plot(train_loss, label='train loss')  
    ax.plot(validate_loss, label='validate loss')
    ax.legend()
    
    # reload best weights
    model.load_state_dict(best_val_model_wts)

    return loss_fig
    
def calculate_loss(criterion, outputs, targets, sequence_lengths):
    """
    Tensor sizes
    outputs : (B, L, vocab_size) 
    targets : (B, L)
    """
    # transform outputs for CEL
    outputs = outputs.transpose(1, 2) # (B, L, vocab_size) --> (B, vocab_size, L)

    # sequence-averaged, batch-averaged reconstruction loss
    loss_per_token = criterion(outputs, targets.long()) # (B, L)
    loss_per_sequence = loss_per_token.sum(dim=1) / (sequence_lengths.float() + 1) # (B); add 1 to lengths for EOS
    loss = torch.mean(loss_per_sequence) # (1)

    return loss

def get_accuracy(outputs, inputs):
    """Calculates percent of sequences reconstructed exactly, percent of 
     characters reconstructed exactly, and percent of each sequence reconstructed"""
    
    num_sequences = len(inputs)
    seq_corr = 0
    char_corr = 0
    char_total = 0
    char_per_seq_acc = 0
    errors = []
    num_wrong_lengths = 0

    # loop through sequences
    for j in range(num_sequences):
        correct_chars, total_chars, length_diff = character_accuracy(outputs[j], inputs[j])
        # count number of errors
        num_errors = total_chars - correct_chars
        # check if sequence lengths are different
        if length_diff != 0:
            num_wrong_lengths += 1
            if length_diff > 0: # extra chars decoded
                num_errors += length_diff
        errors.append(num_errors)
        # add to running counts of total & correct characters
        char_corr += correct_chars
        char_total += total_chars
        # get character accuracy for sequence and add to running total
        char_acc = correct_chars / total_chars
        char_per_seq_acc += char_acc
        # check if sequence was reconstructed exactly
        if (num_errors == 0):
            seq_corr += 1

    # normalize 
    seq_acc = seq_corr / num_sequences
    char_acc = char_corr / char_total
    char_per_seq_acc = char_per_seq_acc / num_sequences

    return seq_acc, char_acc, char_per_seq_acc, num_wrong_lengths, errors

def character_accuracy(output, input):
    """Returns number of characters reconstructed correctly and 
    total characters for a given sequence"""

    sequence_length = min(len(input), len(output))
    correct_chars = 0
    correct_chars_rev = 0
    total_chars = len(input) 
    
    # check if strings are same length
    length_diff = len(output) - len(input)

    # loop through sequence_length
    for i in range(sequence_length):
        if (output[i] == input[i]):
            correct_chars += 1
        # check if reversed sequence is correct
        if (output[-i-1] == input[i]):
            correct_chars_rev += 1

    return max(correct_chars, correct_chars_rev), total_chars, length_diff