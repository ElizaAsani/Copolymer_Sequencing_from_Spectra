"""
Data structure to encode copolymer spectra and sequences. 
"""

import h5py
import torch
from torch.utils.data import Dataset, Subset
from collections import defaultdict

class SeqDataset(Dataset):
    """
    Class to read in spectra and sequences from CSV files and convert sequences to character index arrays.
    """ 
    special_tokens = ['SOS', 'EOS', 'PAD']

    sos_token = special_tokens.index('SOS')
    eos_token = special_tokens.index('EOS')
    pad_token = special_tokens.index('PAD')
    
    def __init__(self, path, scale={'uv_vis': False, 'nmr': False, 'ms': False}, mixtures=False):
        super().__init__()

        self.h5_path = path
        self.h5 = h5py.File(self.h5_path, 'r') 
        self.mixtures = mixtures

        # SPECTRA #   -- not stored, read in as needed -- 
        self.scale = scale
        self.spectra_datasets = {}
        self.spectra_length = {}

        for key in self.scale:
            self.spectra_datasets[key] = self.h5[key]
            self.spectra_length[key] = self.h5[key].attrs['points']

        # SEQUENCES #
        if not self.mixtures: 
            self.sequences = [s.decode('utf-8') for s in self.h5['sequence'][:]]  # read in as bytes and decode as strings
            self.sequence_lengths = [len(s) for s in self.sequences]
        else:
            self.sequences = [[s.decode('utf-8') for s in seqs] for seqs in self.h5['sequence'][:]]
            self.sequence_lengths = [[len(s) for s in seqs] for seqs in self.sequences]

        self.length = len(self.sequences)

        self.alphabet = SeqDataset.special_tokens + self.h5['sequence'].attrs['monomers'].tolist()
        self.num_chars = len(self.alphabet)

        self.max_sequence_length = self.h5['sequence'].attrs['max_length']   
        self.model_max_sequence_length = self.max_sequence_length + 1  

        self.causal_mask = SeqDataset.causal_mask(self.model_max_sequence_length)

        # LAMBDA - sequence correlation #
        if 'lambda' in self.h5:
            self.lambdas = self.h5['lambda'][:]
                       
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        spectra = {}

        for key in self.spectra_datasets:
            spectrum = torch.tensor(self.spectra_datasets[key][idx], dtype=torch.float32)
            if self.scale[key]:
                spectrum = SeqDataset.scale_by_spectrum(spectrum)
            spectra[key] = spectrum
        
        sequence = self.sequences[idx]
        encoding = torch.tensor(self.encodeSeq(sequence))
        sequence_length = self.sequence_lengths[idx]
        lambd = self.lambdas[idx]

        decoder_input = encoding[:-1]   # keep sos, pad; remove eos
        label = encoding[1:]            # keep eos, pad; remove sos
        
        padding_mask = (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0) # (1, 1, L)
        decoder_mask = padding_mask & self.causal_mask # (1, 1, L) & (L, L) --> (1, L, L)

        return {
            "encoder_input": spectra, # {(uv_vis_length), (nmr_length), (ms_length)}
            "decoder_input": decoder_input, # (L)
            "decoder_mask": decoder_mask, # (1, L, L)
            "label": label, # (L)
            "sequence": sequence,
            "length": sequence_length,
            "lambda": lambd
        }
    
    # ---- Spectra: scaling ---- # 
    @staticmethod
    def scale_by_spectrum(spectra):
        """
        Function to normalize spectra by spectrum so that each row has min=0 and max=1.     
        """
        row_mins = spectra.min(dim=-1, keepdim=True).values
        row_maxs = spectra.max(dim=-1, keepdim=True).values

        return (spectra - row_mins) / (row_maxs - row_mins)

    # ---- Sequences: encoding/decoding ---- # 
    def encodeSeq(self, seq):
        """
        Function to convert a string sequence to an indexed array with 
        dim (max_sequence_length)
        """
        encoded_seq = []

        # add start of sequence character
        encoded_seq.append(SeqDataset.sos_token)

        for char in seq:
            encoded_seq.append(self.alphabet.index(char))

        # add end of sequence character
        encoded_seq.append(SeqDataset.eos_token)
        
        # add padding if necessary
        padding = self.max_sequence_length + 2 - len(encoded_seq)
        encoded_seq.extend([SeqDataset.pad_token] * padding)
        
        return encoded_seq
    
    def decodeSeq(self, char_array): 
        """
        Function to convert a padded char index array with 
        dim (max_sequence_length) to a string sequence 
        """
        seq = ''
        for idx in char_array:
            char = self.alphabet[idx]
            if (char == 'SOS'):
                if len(seq) == 0:
                    continue
                else:
                    # incorrect termination - flag
                    seq += 'S'
                    break
            if (char == 'EOS'):
                break
            if (char == 'PAD'):
                # incorrect termination - flag
                seq += 'P'
                break
            seq += (char)

        return seq      
    
    @staticmethod
    def reconstruct(outputs):
        """
        Tensor sizes
        ----------
        outputs: (B, L, vocab_size)
        probs: (B, L, vocab_size)
        encodings: (B, L)
        """
        probs = torch.softmax(outputs, dim=-1)
        encodings = torch.argmax(outputs, dim=-1)
        
        return encodings, probs
    
    @staticmethod    
    def causal_mask(size):
        # 1 = can see, 0 = can't see; mask: 1 x query x key
        mask = torch.tril(torch.ones(size, size)).type(torch.bool)
        return mask
    
class SeqDatasetMixtures(SeqDataset):
    """
    Subclass of SeqDataset to read in mixtures data, which has multiple sequences per spectra and lambda values. 
    """
    def __init__(self, path, scale={'uv_vis': False, 'nmr': False, 'ms': False}):
        super().__init__(path, scale, mixtures=True)

        self.ratios = self.h5['ratio'][:]
    
    def __getitem__(self, idx):
        spectra = {}

        for key in self.spectra_datasets:
            spectrum = torch.tensor(self.spectra_datasets[key][idx], dtype=torch.float32)
            if self.scale[key]:
                spectrum = SeqDataset.scale_by_spectrum(spectrum)
            spectra[key] = spectrum
        
        sequence = self.sequences[idx]
        sequence_length = self.sequence_lengths[idx]
        ratio = self.ratios[idx]
        lambd = self.lambdas[idx]
        
        return {
            "encoder_input": spectra, # {(uv_vis_length), (nmr_length), (ms_length)}
            "sequence": sequence,
            "length": sequence_length,
            "ratio": ratio,
            "lambda": lambd
        }


def stratified_split_indices(sequence_lengths, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=0):
    """
    Stratified split by sequence length using only torch + Python.
    Returns: train_idx, val_idx, test_idx (torch.LongTensor)
    """

    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-8

    g = torch.Generator().manual_seed(seed)

    # group indices by length
    buckets = defaultdict(list)

    for idx, L in enumerate(sequence_lengths):
        buckets[int(L)].append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for L, idxs in buckets.items():
        idxs = torch.tensor(idxs, dtype=torch.long)

        # shuffle using torch permutation
        perm = torch.randperm(len(idxs), generator=g)
        idxs = idxs[perm]

        n = len(idxs)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)

        train_idx.extend(idxs[:n_train].tolist())
        val_idx.extend(idxs[n_train:n_train + n_val].tolist())
        test_idx.extend(idxs[n_train + n_val:].tolist())

    # final shuffle per split
    def shuffle_list(x):
        x = torch.tensor(x, dtype=torch.long)
        return x[torch.randperm(len(x), generator=g)]

    train_idx = shuffle_list(train_idx)
    val_idx = shuffle_list(val_idx)
    test_idx = shuffle_list(test_idx)

    return train_idx, val_idx, test_idx

def make_splits(dataset, train_frac, val_frac, test_frac, seed):
    train_idx, val_idx, test_idx = stratified_split_indices(dataset.sequence_lengths, train_frac=train_frac, 
                                                            val_frac=val_frac, test_frac=test_frac, 
                                                            seed=seed)

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)
    test_set  = Subset(dataset, test_idx)

    return train_set, val_set, test_set