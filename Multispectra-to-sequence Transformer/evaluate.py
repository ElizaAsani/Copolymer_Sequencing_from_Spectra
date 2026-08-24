"""
Accuracy metrics for evaluating model performance.
"""

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