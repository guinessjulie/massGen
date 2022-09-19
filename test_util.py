def disp(genes,str=''):
   Util.plotGrid(LandGrid(genes, width, height),str)

def fits_to_probs(fits):
    uniques = list(set(fits))
    uniques_sum = sum(x for x in uniques)
    unique_probs = [{round(x, 6), x/uniques_sum} for x in fits]
    return unique_probs