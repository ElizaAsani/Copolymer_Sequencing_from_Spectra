"""
Copolymer Sequence Generation
"""

import csv
from seq_generator import generateSequence, canonicalizeSequence

#------------Global Variables-----------#

monomers = ['D', 'A']       # donor and acceptor monomers

f_D = 0.5                   # fraction of donor monomers (0.5)
length = 10

lambdas = [-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]
ratios = [0.10] * 10        

num_mixtures = 100

def main():

    for lamb in lambdas:
        # create input file
        filename = f'mixtures_L{length}_lamb{lamb}.csv'
        
        # if existing file, rewrite
        f = open(filename, 'w')
        f.write("")
        f.close()
        
        # open and write to csv file
        with open(filename, 'w', newline='') as csvfile:
            
            writer = csv.writer(csvfile)

            for _ in range(num_mixtures):
                seqs = []
        
                for _ in range(len(ratios)):

                    # pick sequence length 
                    seq = generateSequence(lamb, length)
                    seq = canonicalizeSequence(seq)
                    seqs.append(seq)
                    
                # write sequence to file
                writer.writerow([seqs, ratios, lamb])    
        
if __name__ == '__main__':
    main()
