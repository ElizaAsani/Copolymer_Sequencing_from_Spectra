"""
Copolymer Sequence Generation
"""

import random
import numpy as np
import csv

#------------Global Variables-----------#

monomers = ['D', 'A']       # donor and acceptor monomers

f_D = 0.5                   # fraction of donor monomers (0.5)
min_length = 3             # min polymer length (10)
max_length = 20             # max polymer length (50) 

numSequences = 10000         # number of sequences to generate (5000)

def main():
    
    # create input file
    filename = f'sequences_{min_length}-{max_length}.csv'
    
    # if existing file, rewrite
    f = open(filename, 'w')
    f.write("")
    f.close()
    
    # open and write to csv file
    with open(filename, 'w', newline='') as csvfile:
        
        writer = csv.writer(csvfile)
    
        for i in range(numSequences):

            # generate sequence based on random value of lambda and sequence length
            
            # pick lambda from a uniform distr.
            lamb = random.uniform(-1, 0.99)
            lamb = np.round(lamb, 2)  

            # pick sequence length 
            seq_length = random.randint(min_length, max_length)              
            seq = generateSequence(lamb, seq_length)
            seq = canonicalizeSequence(seq)
            
            # write sequence to file
            writer.writerow([seq, lamb])     

def allseq():

    for length in range(min_length, max_length+1):
        # create input file
        filename = f'Output/all/all_sequences_{length}.csv'
        
        # if existing file, rewrite
        f = open(filename, 'w')
        f.write("")
        f.close()

        # generate all sequences, canonicalize, and drop duplicates
        allSeq = generateAllSequences(length)
        allSeq = list(set([canonicalizeSequence(seq) for seq in allSeq]))

        # open and write to csv file
        with open(filename, 'w', newline='') as csvfile:
            
            writer = csv.writer(csvfile)
        
            for seq in allSeq:
                # calculate the theoretical sequence correlation
                lamb = calcLambda(seq)

                # write sequence to file
                writer.writerow([seq, lamb])     
        
def generateSequence(lambd, length):
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

def canonicalizeSequence(seq):
    """Cannonicalizes a linear sequence by selecting the orientation that 
    has the minimum lexicographical order.
    """
    return min(seq, seq[::-1])

def generateAllSequences(length):
    """Generates all possible sequences of a given length. 
    """
    sequences = []
    if length == 1:
        return monomers
    else:
        sequences = [a + b for a in monomers for b in generateAllSequences(length - 1)]

    return set(sequences)

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

if __name__ == '__main__':
    #main()
    allseq()