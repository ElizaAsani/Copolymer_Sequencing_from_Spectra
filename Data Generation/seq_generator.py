"""
Copolymer Sequence Generation
"""

import argparse
import os
import csv
import random
import numpy as np

from seq_config import load_config

def parse_args():
    parser = argparse.ArgumentParser(description='Generate copolymer sequences.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    return parser.parse_args()

def run(cfg):
    if cfg.mode == "random":
        run_random(cfg)
    elif cfg.mode == "all":
        run_all(cfg)
    elif cfg.mode == "mixtures":
        run_mixtures(cfg)

def run_random(cfg):
    """mode: random - N unique sequences, lengths and lambda uniformly sampled"""

    sequences = set()
    rows = []

    while len(sequences) < cfg.num_sequences:
        # pick lambda from a uniform distr.
        lamb = random.uniform(-1, 0.99)
        lamb = np.round(lamb, 2)  

        # pick sequence length 
        length = random.randint(cfg.min_length, cfg.max_length)              
        seq = generateMarkovSequence(cfg.monomers, cfg.f_D, lamb, length)
        seq = canonicalizeSequence(seq)

        if seq not in sequences:
            sequences.add(seq)
            rows.append([seq, lamb])
    
    # create input file
    filename = f'seq_{cfg.min_length}-{cfg.max_length}.csv'
    
    # open and write to csv file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

def run_all(cfg):
    """mode: all - generate all sequences of lengths in range [min_length, max_length]"""

    os.makedirs(cfg.output_dir, exist_ok=True)

    for length in range(cfg.min_length, cfg.max_length + 1):
        # create input file
        filename = os.path.join(cfg.output_dir, f'all_seq_{length}.csv')

        # generate all sequences, canonicalize, and drop duplicates
        allSeq = generateAllSequences(cfg.monomers, length)
        allSeq = list(set([canonicalizeSequence(seq) for seq in allSeq]))

        # open and write to csv file
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for seq in allSeq:
                # calculate the theoretical sequence correlation
                lamb = calcLambda(seq)

                # write sequence and lambdato file
                writer.writerow([seq, lamb])  

def run_mixtures(cfg):
    """mode: mixtures - N mixtures of sequences of lengths in range [min_length, max_length]
        for each provided lambda"""

    os.makedirs(cfg.output_dir, exist_ok=True)

    for lamb in cfg.lambdas:
        # create input file
        filename = os.path.join(cfg.output_dir, f'mixtures_lamb{lamb}.csv')
        
        # if existing file, rewrite
        f = open(filename, 'w')
        f.write("")
        f.close()
        
        # open and write to csv file
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            for _ in range(cfg.num_mixtures):
                seqs = []
        
                for _ in range(len(cfg.ratios)):
                    # pick sequence length 
                    length = random.randint(cfg.min_length, cfg.max_length)

                    # generate sequence
                    seq = generateMarkovSequence(cfg.monomers, cfg.f_D, lamb, length)
                    seq = canonicalizeSequence(seq)
                    seqs.append(seq)
                    
                # write sequence to file
                writer.writerow([seqs, cfg.ratios, lamb])   
        
def generateMarkovSequence(monomers, f_D, lambd, length):
    """Generates a string representation of a polymer chain of given length
        with two monomers. The sequence is determined by lambd [-1, 1] 
        which describes the strength of sequence correlations in the chain: 
        lambd = -1 (alternating)
        lambd = 0 (random)
        lambd = 1 (block)
        """

    # calculate probabilities
    p_DA = f_D * (1 - lambd)        # prob of donor following acceptor
    p_AD = (1 - f_D) * (1 - lambd)  # prob of acceptor following donont

    sequence = random.choice(monomers)  # randomly choose first monomer

    # generate rest of sequence
    for i in range(0, length - 1):
        prob = random.random()      # generate random number

        # if current monomer is 'D', determine next monomer
        if sequence[i] == monomers[0]:
            if prob < p_AD:
                sequence += monomers[1]
            else:
                sequence += monomers[0]
        # if current monomer is 'A', determine next monomer
        else:
            if prob < p_DA:
                sequence += monomers[0]
            else:
                sequence += monomers[1]

    return sequence

def generateAllSequences(monomers, length):
    """Generates all possible sequences of a given length. 
    """
    sequences = []
    if length == 1:
        return monomers
    else:
        sequences = [a + b for a in monomers for b in generateAllSequences(monomers, length - 1)]

    return set(sequences)

def canonicalizeSequence(seq):
    """Cannonicalizes a linear sequence by selecting the orientation that 
    has the minimum lexicographical order.
    """
    return min(seq, seq[::-1])

def calcAvgBlockLength(sequence):
    """Calculates average block length for a given copolymer sequence. 
    """
    numBlocks = 1    # counter for number of blocks
    
    # increment block counter every time a new block is encountered    
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i - 1]:
            numBlocks += 1
            
    return (len(sequence) / numBlocks)   

def calcLambda(sequence):
    """Calculates the theoretical sequence correlation of a sequence using the inverse of 
       the average block length formula. 
    """
    avgBlockLength = calcAvgBlockLength(sequence)

    if (avgBlockLength == len(sequence)):
        return 1
    
    return round((1 - (2/avgBlockLength)), 2)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    run(cfg)

if __name__ == '__main__':
    main()
