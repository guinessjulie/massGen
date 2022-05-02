import random

def crossover(genes1, genes2):
    pt = random.randint(0, len(genes1)-2)
    child = genes1[:pt] + genes2[pt:]
    return child