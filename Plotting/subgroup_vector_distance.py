import numpy as np

def subgroup_vector_distance(seq1, seq2, subgroup_type, distance_type):
    """Calculate the k-mer distance between two sequences based on their invariances."""
    v1 = np.array(calculateSubgroupVector(seq1, subgroup_type=subgroup_type))
    v2 = np.array(calculateSubgroupVector(seq2, subgroup_type=subgroup_type))
    if distance_type == 'euclidean':
        dist = euclidean_distance(v1, v2)
    elif distance_type == 'manhattan':
        dist = manhattan_distance(v1, v2)
    elif distance_type == 'exact':
        dist = np.array_equal(v1, v2)
    return dist

def euclidean_distance(v1, v2):
    """Calculate the Euclidean distance between two vectors."""
    return np.sqrt(np.sum((v1 - v2) ** 2))

def manhattan_distance(v1, v2):
    """Calculate the Manhattan distance between two vectors."""
    return np.sum(np.abs(v1 - v2))

def calculateSubgroupVector(seq, subgroup_type='all'):
    """Calculates the invariances for a given sequence. 
    """
    if subgroup_type == 'monomers':
        return countMonomers(seq)
    elif subgroup_type == 'dimers':
        return countDimers(seq)
    elif subgroup_type == 'trimers':
        return countTrimers(seq)
    elif subgroup_type == 'all':
        return countMonomers(seq) + countDimers(seq) + countTrimers(seq)
    elif subgroup_type == 'NMR':
        return countTrimers(seq) + countEndgroups(seq)
    elif subgroup_type == 'MS':
        return getFragments(seq)
    
    return []

def countMonomers(seq):
    A = seq.count('A')
    D = seq.count('D')
    return [A, D]

def countDimers(seq):
    AA = 0
    DD = 0
    AD = 0

    for i in range(len(seq) - 1):
        pair = seq[i:i+2]
        # check if pair is AA, AD + DA, or DD
        match pair:
            case 'AA':
                AA += 1
            case 'DD':
                DD += 1
            case 'AD' | 'DA':
                AD += 1

    return [AA, DD, AD]

def countTrimers(seq):
    AAA = 0
    DDD = 0
    AAD = 0
    DDA = 0
    ADA = 0
    DAD = 0
    
    for i in range(len(seq) - 2):
        triple = seq[i:i+3]
        # check if triple is AAA, DDD, AAD + DAA, ADD + DDA, ADA, or DAD
        match triple:
            case 'AAA':
                AAA += 1
            case 'DDD':
                DDD += 1
            case 'AAD' | 'DAA':
                AAD += 1
            case 'DDA' | 'ADD':
                DDA += 1
            case 'ADA':
                ADA += 1
            case 'DAD':
                DAD += 1

    return [AAA, DDD, AAD, DDA, ADA, DAD]

def countEndgroups(seq):
    AA = 0
    DD = 0
    AD = 0
    DA = 0

    left = seq[0:2]
    right = seq[-2:]
    match left:
        case 'AA':
            AA += 1
        case 'AD':
            AD += 1
        case 'DA':
            DA += 1
        case 'DD':
            DD += 1

    match right:
        case 'AA':
            AA += 1
        case 'AD':
            DA += 1
        case 'DA':
            AD += 1
        case 'DD':
            DD += 1
    
    return [AA, DD, AD, DA]

def getFragments(seq):
    
    monomers = {'D': 182, 'A': 131, '*': 15}

    def getFragmentMass(fragment):
        """Calculate the mass of a fragment"""
        return sum([monomers[mon] for mon in fragment])

    sequence = '*' + seq + '*'
    length = len(sequence)
    masses = [getFragmentMass(sequence)]
    for i in range(2, length - 2):
            masses.append(getFragmentMass(sequence[:i]))
            masses.append(getFragmentMass(sequence[i:]))

    masses = list(set(masses))
    masses.sort()  

    return masses