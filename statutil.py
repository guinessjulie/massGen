import numpy as np

def get_descriptive(fitname, fitnesses, generation):
    data = [x.get(fitname) for x in fitnesses]
    mean = np.mean(data)
    max = np.max(data)
    min = np.min(data)
    std = np.std(data)
    values, counters = np.unique(data, return_counts = True)
    max_count =  counters[len(counters)-1]
    return {'gen': generation, 'mean':mean, 'max':max, 'min':min, 'max_count':max_count, 'std':std}


