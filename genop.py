import random
from fitness import Fitness
from util import Util,Pos
from landgrid import LandGrid
from collections import Counter
from statistics import mean
import numpy as np
import datetime

# change: normalized_fitness를 calcFitness로 옮겼음
def selection(pops, width, height, num_cells, fit_criteria = 'f(PAR)'):
    attr, fits, edges, fitnesses = calcFitness(pops,width, height, num_cells)
    normalized_fits = normalize_fitnesses(fitnesses) #todo: let's not do that
    matingPool = []
    tempMatingPool_originalId = [] # to trace initial pop
    fits_raw = [x.get(fit_criteria) for x in fitnesses]
    # norm_fits = Util.normalize_0to1(fits_raw)
    selected_idx = []
    while len(matingPool) <= len(pops) * 20:
        # for i, fitness in enumerate(fitnesses): #todo testing
        # for i, fitness  in enumerate(normalized_fits[fit_criteria]):
        for i, fitness in enumerate([x.get('f(PAR)') for x in fitnesses]):

            p = random.random()
            fit = fitness
            if p < fit:
                matingPool.append(pops[i])
                selected_idx.append(i)
    return matingPool


def calcFitness(pops, width, height, num_cells):
    fitnesses = []
    for i, individual in enumerate(pops):
        fitness = Fitness(individual, width, height, num_cells, i)
        fitnesses.append(fitness._fits)
        attr, fits, edges = fitness.build_attrs(individual, num_cells, i)
    # fitnesses = update_fitness(fitnesses) # todo: convert to normalized fitnesses=> the problem is that relatively very small number can become max 1.0 becuase it doesn't know the theoritical max value
    return attr, fits, edges, fitnesses

def update_fitness(fitnesses):
    selected_fit = selected_fitness_list(fitnesses, 'f(PAR)')
    #todo: convert to normalized finesses
    fpar = [fit['f(PAR)'] for fit in fitnesses]
    normed_fpar  = Util.normalize_0to1([fit['f(PAR)'] for fit in fitnesses])
    i = 0
    for v in fitnesses:
        v['f(PAR)'] = normed_fpar[i]
        i += 1
    return fitnesses

def selected_fitness_list(fitnesses, obj_fit):
    selected_fits =  [fit[obj_fit] for fit in fitnesses]
    return Util.normalize_0to1(selected_fits)

def normalize_fitnesses(fitnesses):
    normalized_fits = {}
    normalized_fits['f(PAR)'] = selected_fitness_list(fitnesses, 'f(PAR)')
    normalized_fits['f(FSH)'] = selected_fitness_list(fitnesses,'f(FSH)')
    normalized_fits['f(SSR)'] = selected_fitness_list(fitnesses,'f(SSR)')
    normalized_fits['f(VSymm)'] = selected_fitness_list(fitnesses,'f(VSymm)')
    normalized_fits['f(HSymm)'] = selected_fitness_list(fitnesses,'f(HSymm)')
    normalized_fits['f(AR)'] = selected_fitness_list(fitnesses,'f(AR)')
    return normalized_fits


def selectTwo(pops, width, height, num_cells, fit_criteria='f(PAR)'):
    attr, fits, edges, fitnesses = calcFitness(pops, width, height, num_cells)
    fpar = [f[fit_criteria] for f in fitnesses]
    pop_fits = np.sum(np.unique(fpar))
    maxfit = np.max(np.unique(fpar))
    minfit = np.min(np.unique(fpar))

    chrom_fits = np.unique(fpar)
    chrom_prob = [(chrom_fit -minfit)/ (maxfit - minfit) for chrom_fit in chrom_fits]
    fit_choice = [np.random.choice(chrom_fits, p=chrom_prob) for x in range(len(pops))]

    while len(matingPool) <= len(pops)*20:
        # for i, fitness in enumerate(fitnesses): #todo testing
        for i, fitness  in enumerate(normalized_fits[fit_criteria]):
            p = random.random()
            fit1 = fitness
            # fit1 = fitness.get('f(PAR)')
            # fit2 = fitness.get('f(FSH)')
            # fit = fit1*0.5 + fit2*0.5
            fit = fit1
            if p < fit:
                matingPool.append(pops[i])
                selected_idx.append(i)
    parent_index_count = Counter(selected_idx)


    return matingPool
    val_fits, cnt = np.unique(fit_choice, return_counts=True)
    valfit_cnts = dict(zip(val_fits, cnt))
    valfit_cnts[min(valfit_cnts)], valfit_cnts[max(valfit_cnts)] =0, valfit_cnts[max(valfit_cnts)] + valfit_cnts[min(valfit_cnts)]
    newpop = []

def selectOne(pops, width, height, num_cells, fit_criteria = 'f(PAR)'):
    attr, fits, edges, fitnesses = calcFitness(pops, width, height, num_cells)
    normalized_fits = normalize_fitnesses(fitnesses)
    fit_sum = sum([c[fit_criteria] for c in fitnesses])
    # selection_probability = [c[fit_criteria] / fit_sum for c in fitnesses ]
    # chosen_id = np.random.choice(len(pops), len(pops), p=selection_probability)
    # ids_counter = Counter(chosen_id)
    unique_fits = list(set([f[fit_criteria] for f in fitnesses]))
    unique_fits_str = ['%.5f'% x for x in unique_fits]
    unique_fit_probabilities = [ufit / sum(unique_fits) for ufit in unique_fits]
    unique_fits_str = ['%.5f' % x for x in unique_fits]
    uniq_prob = dict(zip(unique_fits_str, unique_fit_probabilities))
    unique_probabilities = dict(zip(unique_fits, unique_fit_probabilities)) #fitness_value - unique_fit_probabilities

    diff = max(unique_fits) - min(unique_fits)
    norm_prob = [(ufit - min(unique_fits)) / diff for ufit in unique_fits] #0에서 1사이의 unique fpar값
    norm_uniq_prob = [(ufit - min(unique_fits)) / diff for ufit in unique_fits]

    fpar = [f[fit_criteria] for f in fitnesses]

    [np.random.choice(np.array(range(len(uniq_prob))), p=[uniq_prob[fs] for fs in unique_fits_str]) for x in range(num_cells)] # 고유 값 (5개)에 대한 p에 따라 선택

    # selected = np.take(pops, chosen_id)

    #################################
    pop_fits = np.sum(np.unique(fpar))
    chrom_fits = np.unique(fpar)
    chrom_prob = [chrom_fit / pop_fits for chrom_fit in chrom_fits]
    fit_choice = [np.random.choice(chrom_fits, p=chrom_prob) for x in range(len(pops))]
    val_fits, cnt = np.unique(fit_choice, return_counts=True)
    valfit_cnts = dict(zip(val_fits, cnt))
    valfit_cnts[min(valfit_cnts)], valfit_cnts[max(valfit_cnts)] =0, valfit_cnts[max(valfit_cnts)] + valfit_cnts[min(valfit_cnts)]
    newpop = []

    for x in valfit_cnts:
        filter_selected = []
        for f in fpar:
            if f == x[0]:
                filter_selected.append(True)
            else:
                filter_selected.append(False)
        candidates = np.array(range(len(pops)))[filter_selected] # 각각의 교유값과 일치하는 chromosome 인덱스를 추출
        choices_idx = np.random.choice(candidates, x[1]) # 갯수만큼
        newpop = newpop + list(choices_idx)
        # 이 중에서 p 갯수만큼 축출

    for x in val_fits:
        selected = []
        while x[1] < len(selected): #  해당 갯수만큼 뽑는다.
            filter_selected = []
            for f in fpar:
                if f == x[0]:
                    filter_selected.append(True)
                else:
                    filter_selected.append(False)

            x[0] #  그 값이 ....
            np.random.choice()
            selected.append()
        newpop.append(selected)

    return newpop



def get_unassigned(genes, width, height):
    background = [Pos(x, y) for x in range(width) for y in range(height)]
    return list(set(background)-set(genes))

def mutate(genes, width, height, rate = 0.1):
    if(random.random() < rate):
        fullset = {Pos(x,y) for x in range(width) for y in range(height)}
        outsides = list(fullset - set(genes))
        bound_adjacent = Util.bound_adjacent(genes, width, height)
        exchange = random.choice(bound_adjacent)
        # exchange = random.choice(outsides)
        i = random.choice(range(len(genes)))
        genes[i] = exchange
    return genes



def mutate_old(genes, width, height, rate = 0.1):
    loc = random.randint(0, len(genes)-1)
    mut_axis = random.randint(0, 1)
    unassigned_genes = get_unassigned(genes, width, height)
    kodons = []
    pick = random.random()
    if (pick < rate):
        print(f'random pick: {pick}, mutation rate= {rate}' )
        if (mut_axis == 0):
            while not kodons:
                loc = random.randint(0, len(genes) - 1)
                kodons = [new_kodon  for new_kodon in unassigned_genes if new_kodon.x == genes[loc].x]
            mutated = kodons[random.randint(0, len(kodons)-1)]
            genes[loc] = mutated
        else:
            # target_string = genes[loc].y
            while not kodons:
                loc = random.randint(0, len(genes) - 1)
                kodons = [new_kodon  for new_kodon in unassigned_genes if new_kodon.y == genes[loc].y]
            mutated = kodons[random.randint(0, len(kodons)-1)]
            genes[loc] = mutated
    return genes

'''
교차의 새로운 전략: 교차위치까지는 첫번째 부모에서 가져오고, 교차 이후는 두번째 부모에서 가져오는데 만일 겹치는 경우 0부터 다시시작
'''
def crossoverFour(mating_pool, width, height, gene_size, mut_rate = 0.2):
    cp = random.randint(0,gene_size-1) #crosspoint
    # 부모 선택
    p1 = mating_pool[random.randint(0, len(mating_pool) - 1)].poses
    p2 = mating_pool[random.randint(0, len(mating_pool) - 1)].poses

    # candidate
    c1 = p1[:cp] #반쪽
    c2 = p2[cp:]
    child_genes = c1+c2
    i = 0
    while len(set(child_genes)) < gene_size:
        child_genes.append(p2[i])
        i += 1

    child_genes = list(set(child_genes))

    return child_genes


def crossoverTwo(pops, width, height, gene_size, mut_rate = 0.1):
    crosspoints = []
    p1 = pops[random.randint(0, len(pops)-1)]
    p2 = pops[random.randint(0, len(pops)-1)]
    #
    # print(f'p1:\n {p1}')
    # print(f'2:\n {p2}')

    crosspoints = set(p1.poses).intersection(set(p2.poses))
    num_lack = gene_size - len(crosspoints)
    child_genes = list(crosspoints)
    while len(child_genes) < gene_size:
        diff1 = list(set(p1.poses).difference(set(p2.poses)))
        diff2 = list(set(p2.poses).difference(set(p1.poses)))
        if diff1:
            child_genes.append(diff1.pop())
        elif diff2:
            child_genes.append(diff2.pop())
        if diff2:
            child_genes.append(diff2.pop())
    child = LandGrid(child_genes, width, height)
    # child.display_poses()
    return child



def crossoverThree(pops, width, height,gene_size, mut_rate=0.1):
    crosspoints = []
    # to see proportion of failed crossover and why
    num_size_not_matched = 0
    num_crosspoints_exists = 0
    num_more_than_one_component = 0

          
    while not crosspoints:
        p1 = pops[random.randint(0, len(pops)-1)].sorted_by_col()
        p2 = pops[random.randint(0, len(pops)-1)].sorted_by_col()
        while p1 == p2:
            p2 = pops[random.randint(0, len(pops)-1)].sorted_by_col()
        for i in range(gene_size):
            if p1[i] == p2[i]:
                crosspoints.append(i)

    # print(f'p1:, {p1}')
    # LandP1 = LandGrid(p1, width, height)
    # LandP1.display_poses()
    #
    # LandP2 = LandGrid(p2, width, height)
    # print(f'parent2: {p2}')
    # LandP2.display_poses()

    cross_pt_loc = random.randint(0, len(crosspoints)-1)
    pt = crosspoints[cross_pt_loc]
    
    child_genes = p2[:pt] + p1[pt:]
    # print(f'child:{child_genes}')
    child = LandGrid(child_genes, width, height)
    # child.display_poses()
    num_crosspoints_exists += 1

    child.display_poses()
    return child

def crossover_random_backup(pops, width, height, gene_size):
    genes1 = pops[random.randint(0, len(pops)-1)].poses
    genes2 = pops[random.randint(0, len(pops)-1)].poses
    middle_point = int(gene_size/2)
    child_genes = genes1[:middle_point] + genes2[middle_point:]
    return LandGrid(child_genes, width, height)

# 좀 더 단순화시켜본다.
def crossover_random(pops, width, height, gene_size):
    genes1 = pops[random.randint(1, len(pops)-1)].poses
    genes2 = pops[random.randint(1, len(pops)-1)].poses
    middle_point = int(gene_size/2)
    child_genes = genes1[:middle_point]
    candidates = Util.bound_adjacent(child_genes, width, height)
    overwraps = list(set(candidates).intersection(set(genes2)))
    needed_len = gene_size - len(child_genes)
    child_genes += overwraps[:needed_len]
    while (len(child_genes) < gene_size):
        boundary_adjs = Util.bound_adjacent(child_genes, width, height)
        needed_len = gene_size - len(set(child_genes))
        child_genes += random.sample(boundary_adjs, min(needed_len, len(boundary_adjs)))
        child_genes = list(set(child_genes))
    return child_genes


def crossover_random_0816(pops, width, height, gene_size):
    genes1 = pops[random.randint(1, len(pops)-1)].poses
    genes2 = pops[random.randint(1, len(pops)-1)].poses
    middle_point = int(gene_size/2)
    child_genes = genes1[:middle_point]
    candidates = Util.bound_adjacent(child_genes, width, height)
    p2_rest = genes2[middle_point:]
    overwraps = list(set(candidates).intersection(set(genes2)))
    needed_len = len(p2_rest)
    child_genes += random.sample(overwraps, min(needed_len, len(overwraps)))
    if (len(set(child_genes)) < gene_size):
        needed_len = gene_size - len(set(child_genes))
        child_genes += random.sample(candidates,  min(needed_len, len(candidates)))
    while len(set(child_genes)) < gene_size:
        new_bound_adjs = Util.bound_adjacent(child_genes, width, height)
        child_genes += random.sample(new_bound_adjs, min(gene_size-len(set(child_genes)), len(new_bound_adjs)))
    child_genes = list(set(child_genes))
    return child_genes

# 교차하는 위치에서
def crossover_overwrap(pops, width, height,gene_size, mut_rate=0.1, generation=0):

    index1 = random.randint(0,len(pops)-1)
    index2 = random.randint(0,len(pops)-1)
    genes1 = pops[index1].sorted_by_col()
    genes2 = pops[index2].sorted_by_col()

    #같은 위치에서 겹치는 셀
    same_cells_same_loc = []
    # todo: 이렇게 한다고 다양성이 증가되는 것이 아님
    # while genes1 == genes2: # if the same
    #     print('genes1==genes2')
    #     index1 = random.randint(0, len(pops) - 1)
    #     index2 = random.randint(0, len(pops) - 1)
    #     genes1 = pops[index1].sorted_by_col()
    #     genes2 = pops[index2].sorted_by_col()


    num_more_than_one_component = 0
    for i in range(gene_size):
        if genes1[i] == genes2[i]:
            same_cells_same_loc.append(i)
    #
    # if same_cells_same_loc: # 같은 위치에서 똑같은 경우, 해당 위치에서  Crossover
    #     print('same_cells_same_loc')
    #     cross_pt_loc = random.randint(0, len(same_cells_same_loc)-1)
    #     pt = same_cells_same_loc[cross_pt_loc]
    #     child_genes = genes1[:pt] + genes2[pt:]
    #     child = LandGrid(child_genes, width, height)


    else: # there is no 같은 위치에서 겹치는 셀이 없을 때
        # print('no-same-cells-same-loc')
        genes1 = pops[index1].sorted_by_col()
        genes2 = pops[index2].sorted_by_col()
        while not set(genes1).intersection(set(genes2)): #repeat if intersection is empty
            index1 = random.randint(0, len(pops) - 1)
            index2 = random.randint(0, len(pops) - 1)
            genes1 = pops[index1].sorted_by_col()
            genes2 = pops[index2].sorted_by_col()

        intersections  = list(set(genes1).intersection(set(genes2)))
        diff1 = set(genes1).difference(set(genes2))
        diff2 = set(genes2).difference(set(genes1))
        differnces = list(diff1.union(diff2))
        child_genes = list(intersections)
        bound_intersections = Util.bound_adjacent(intersections, width, height) #todo: 굳이 이렇게 할 필요성을 잘 모르겠으므로, 이 부분 다시 잘 봐서 없애자.
        adjs_overwrap_p1 = list(set(genes1).intersection(set(bound_intersections)))
        adjs_overwrap_p2 = list(set(genes2).intersection(set(bound_intersections)))
        child_genes += list(set(adjs_overwrap_p1 + adjs_overwrap_p2))

        while len(child_genes) < gene_size:
            if not adjs_overwrap_p1 or adjs_overwrap_p2:
                print('list is empty')
            first = adjs_overwrap_p1.pop(0)
            adjs_first = [x for x in Util.adjacent_four_way(first,width, height) if x in genes1 and x not in child_genes ]
            child_genes += adjs_first
            adjs_overwrap_p1 += adjs_first
            if not len(child_genes) >= gene_size:
                second = adjs_overwrap_p2.pop(0)
                adjs_second = [x for x in Util.adjacent_four_way(second,width, height) if x in genes2 and x not in child_genes ]
                child_genes += adjs_second
                adjs_overwrap_p2 += adjs_second

        while len(child_genes) > gene_size:
            child_genes.pop()

        child = LandGrid(child_genes, width, height)
        numcc = child.num_connected_component() # size가 같지만, 두 그릅인 경우
        if(numcc != 1): # size가 같고, 한덩어리일 경우에만 child를 return (crossover)
            print(f'Failed to crossover number of connected component of child: {numcc}')
            child = pops[index2] #todo mixed up with child and child_genes
            num_more_than_one_component += 1

    # print_parents(genes1, genes2, child, index1, index2, width, height)
    return child

def fillup_genes(cur_genes, gene_size, width, height):
    gap = gene_size -  len(set(cur_genes))
    all_locs = [Pos(x, y) for x in range(width) for y in range(height)]
    while len(set(cur_genes)) < gene_size:
        unselected_genes = [p for p in all_locs if p not in cur_genes]
        print(unselected_genes)
        picked = random.randint(0, len(unselected_genes)-1)
        # todo: if(unselected_genes[picked]) adjacency to cur_genes than add to cur_genes
        g = LandGrid(cur_genes, width, height)
        picked_genes = unselected_genes[picked] # list index out of range
        adjs = g.adjacency(picked_genes) #list index out of range
        if set(adjs).intersection(set(cur_genes)):
            cur_genes.append(unselected_genes[picked])
    return cur_genes

def crossover_org(pops, width, height,gene_size, mut_rate=0.1):
    index1 = random.randint(0,len(pops)-1)
    index2 = random.randint(0,len(pops)-1)
    pt = random.randint(0, gene_size - 2)

    genes1 = pops[index1].sorted_by_col()
    p1 = LandGrid(genes1, width, height)
    # print(f'parent1:, {genes1 }')
    # p1.display_poses()

    genes2 = pops[index2].sorted_by_col()
    p2 = LandGrid(genes2, width, height)
    # print(f'parent2: {genes2}')
    # p2.display_poses()

    parent1_last_pos = pops[index1].poses[pt-1]
    adjs = pops[index1].adjacency(parent1_last_pos)
    # print(adjs)

    child_genes = genes1[:pt] + genes2[pt:]
    child = LandGrid(child_genes, width, height).sorted_by_col()
    # print(f'child:{child_genes}')
    # child.display_poses()

    if len(child_genes) != gene_size:
        print(f'Failed to crossover: size of the child_gene {len(child_genes)} does not match gene_size {gene_size}')
        return pops[index1]

    child(child_genes, width, height)
    numcc = child.num_connected_component()
    if(numcc != 1): # size가 같고, 한덩어리일 경우에만 child를 return (crossover)
        print(f'Failed to crossover number of connected component of child: {numcc}')
        return pops[index2]

    print(f'child: number of connected component: {numcc}')
    return child

    # while pops[index2].poses[pt] not in adjs:
    #     pt = random.randint(0, gene_size - 2)
    #     child_genes = pops[index1].poses[:pt] + pops[index2].poses[pt:]
    #     print('True')
    #     p = random.random()
    #     if(p < mut_rate):
    #         child_genes = mutate(child_genes, width, height,mut_rate)
    # if not child_genes:
    #     return pops[index1]
    return LandGrid(child_genes, width, height)



    # for fit in fitnesses:
    #     pick = random.random()
    #     fitness = fit.get('f(PAR)')
    #     if(pick <= fitness):
    #         matingPool.push()
    # print(fitnesses._fits['f(BCR)'])
    #     if (fitness._fits['f(FSH)'] < 0.1):  # todo : print peculiar pattern
    #         pops.append(genes)
    #         Util.plotColorMesh(genes, fitness, str_settings)
    # return pops



def print_parents(genes1, genes2, child, index1, index2, width, height):
    p1 = LandGrid(genes1, width, height)
    print('p1\n')
    p1.display_poses()

    p2 = LandGrid(genes2, width, height)
    print('p2\n')
    p2.display_poses()

    print(f'p1={index1}, p2={index2}')
    child.display_poses()

def line304deletedcode(genes1, genes2, adjs_overwrap_p1,adjs_overwrap_p2, gene_size, width, height, child_genes):
    # genes1 = list(set(genes1) - set(adjs_overwrap_p1))
    # genes2 = list(set(genes2) - set(adjs_overwrap_p2))
    adjs_overwrap_p1_index = [genes1.index(x) for x in adjs_overwrap_p1]
    adjs_overwrap_p2_index = [genes2.index(x) for x in adjs_overwrap_p2]
    picked_overwrap_p1 = random.randint(0, len(adjs_overwrap_p1) - 1)
    picked_overwrap_p2 = random.randint(0, len(adjs_overwrap_p2) - 1)
    picked_adjs_overwrap1 = random.choice(adjs_overwrap_p1)
    picked_adjs_index1 = genes1.index(picked_adjs_overwrap1)
    picked_adjs_overwrap2 = random.choice(adjs_overwrap_p2)
    picked_adjs_index2 = genes2.index(picked_adjs_overwrap2)

    # if(picked_adjs_overwrap1.x > int( gene_size / 2)):
    needed_len = int((gene_size - len(child_genes)) / 2)
    child_genes += genes1[picked_adjs_index1: picked_adjs_index1 + needed_len + 1]
    child_genes += genes2[picked_adjs_index2 - needed_len:picked_adjs_index2]
    child1 = genes1[:picked_adjs_index1]
    child2 = genes2[:picked_adjs_index2]

    p1_size = int(gene_size / 2)
    p2_size = gene_size - p1_size
    child_genes = genes1[p1_size:]
    child_genes = child_genes + genes2[:p1_size]

    # pt = random.randint(0, gene_size - 2)
    # child_genes = list(set(genes1[:pt] + genes2[pt:]))
    # child = LandGrid(child_genes, width, height)
    print(f'child:{child_genes}')
    if len(child_genes) != gene_size:  # 겹치는 게 있어서 size가 다른경우
        print(f'Failed to crossover: size of the child_gene {len(child_genes)} does not match gene_size {gene_size}')
        child = pops[index1]  # todo fill up the insufficient gene
        child_genes = fillup_genes(child_genes, gene_size, width, height)
        num_size_not_matched += 1
        # return pops[index1]