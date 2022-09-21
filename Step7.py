import random
import unittest
import measureUtil
import util
from util import Util,Pos
from fitness import Fitness
from landgrid import LandGrid
from options import Options
import math
from collections import deque
import time
from genop import mutate, selected_fitnesses, selection, crossover_c, crossover_b, crossover_overwrap,crossover_random, calcFitness, crossoverFour, mutate, update_fitness, crossover_a
from datetime import datetime
from statistics import mean
import statutil

import matplotlib.pyplot as plt

def repeat(times):
    def repeatHelper(f):
        def callHelper(*args):
            for i in range(0, times):
                f(*args)
        return callHelper
    return repeatHelper



class GenArchiPlan(unittest.TestCase):
    config = Options()  # todo:  get rid of all config instance related functions and objects
    width = config.config_options('width')
    height =config.config_options('height')
    pop_size = config.config_options('population_size')
    str_settings = config.get_display_settings(width, height)
    # test all the case for massive data for statistics

    num_cells = Util.get_num_cells(width, height, config.config_options('required_faratio')) \
        if config.config_options('forcednalength') == 'yes' else config.config_options('numcell')
    opt_fit = config.config_options('opt_fit')

    def get_fitness(self, grid):
        fitness = Fitness(grid, self.width, self.height, self.num_cells)
        return fitness

    def test_for_paper(self):
        postfix_bydate = datetime.now().strftime('%m%d')
        targetpath = '.\\results' + postfix_bydate + '\\'
        pops = self.initial_population(self.pop_size, targetpath)  # todo : for f(ss) test
        stats = []
        filename_postfix = datetime.now().strftime('%d_%H_%M_%S.csv')
        fitness_filename = targetpath + 'fitness' + filename_postfix
        stat_filename = targetpath + 'stat' + filename_postfix
        ngen = self.config.config_options('ngeneration')
        for i in range(ngen):
            pops, stat = self.reproduce(pops, i, fitness_filename, targetpath)
            stats.append(stat)
        Util.saveStat(stat_filename, stats)

    def initial_population(self, pop_size, targetpath='.\\results\\'):
        start = datetime.now()
        numfig = self.config.config_options('numfig')
        init_method = self.config.config_options('init_method')[0]
        generate = {
            'random': self.generate_random,
            'a': self.generate_a, # selected from last position
            'b': self.generate_origin, # selected from random position
            'c': self.generate_by_block # refactored function name from generate_new_simple0914
        }
        pops = []
        popid = 0
        while len(pops) < pop_size:
            landgrid = generate[init_method](popid)
            landgrid.pop_id = popid
            popid += 1
            pops.append(landgrid)
        print(f'Elapsed Time for initial population: {(datetime.now() - start).total_seconds()}')

        start = datetime.now()

        # fig_text = '(a) Random assignment'
        # fig_text = '(b) random adjacent assignment of last position'
        # fig_text = '(c) all adjacent assignment of a selected position'
        fig_text = 'Initial Population'
        # Util.plotGridBatch(pops, numfig, rows,fig_text, omit_parents=True )
        random_pop = random.sample(pops, 50) if len(pops) > numfig else pops
        Util.plotGridOnlyRow(random_pop, int(len(random_pop) / 5), fig_text, omit_parents=True, targetpath=targetpath)
        print(f'Elapsed Time for drawing init pops: {(datetime.now() - start).total_seconds()}')
        return pops

    def generate_a(self, pop_id=0):  # expanding adjacent from last position
        # genes = Util.select_random_position(width, height)
        genes = self.choose_start_position()
        # grid = LandGrid(genes, width, height) #grid가 있어야 동적으로 gene 생성 가능
        curIdx = 0
        weights = []
        prob = [self.width / (self.width + self.height), self.height / (self.width + self.height)]
        while len(genes) < self.num_cells:
            last_pos = genes[len(genes) - 1]
            all_adjs = Util.adjacent_four_way(last_pos, self.width, self.height)
            gene_set = [x for x in all_adjs if x not in genes]

            i = 1
            while not gene_set:
                last_pos = genes[len(genes) - i]
                all_adjs = Util.adjacent_four_way(last_pos, self.width, self.height)
                gene_set = [x for x in all_adjs if x not in genes]
                i += 1

            vertical_sample = [pos for pos in gene_set if pos.x == last_pos.x]
            horiz_sample = [pos for pos in gene_set if pos.y == last_pos.y]
            weights = Util.build_weights(prob, gene_set, [horiz_sample, vertical_sample])
            if (len(gene_set) != len(weights)):
                print('not match')
            next_code = random.choices(gene_set, weights, k=1)[0]
            genes.append(next_code)

            # grid.update_positions(genes)
        grid = LandGrid(genes, self.width, self.height, pop_id)
        return grid

    def generate_origin(self, pop_id=0):  # method B
        genes = self.choose_start_position()
        # genes = list(Util.select_random_position(self.width, self.height))
        sample = []
        prob = [self.width / (self.width + self.height), self.height / (self.width + self.height)]

        while len(genes) < self.num_cells:
            while not sample:
                pickedIdx = random.choice(range(len(genes)))
                sample = list(set(Util.adjacent_four_way(genes[pickedIdx], self.width, self.height)) - set(genes))
                vertical_sample = [pos for pos in sample if pos.x == genes[pickedIdx].x]
                horiz_sample = [pos for pos in sample if pos.y == genes[pickedIdx].y]
                weights = Util.build_weights(prob, sample, [horiz_sample, vertical_sample])

            new_gene = random.choices(sample, weights, k=1)
            genes.append(new_gene[0])
            sample = []
        return LandGrid(genes, self.width, self.height)

    # def generate_new_simple0914(self,pop_id = 0): # Method C: expanding all cells adjacent to a random position
    def generate_by_block(self,pop_id = 0): # Method C: expanding all cells adjacent to a random position
        genes = self.choose_start_position()
        curIdx = 0
        while len(genes) < self.num_cells:
            gene_set = Util.available_adjacency(genes, curIdx, self.width, self.height)
            sample_size = min(self.num_cells - len(genes), len(gene_set))
            if sample_size > 0:
                genes.extend(random.sample(gene_set, sample_size))
            else:
                new_candidates = Util.bound_adjacent(genes, self.width, self.height)
                sample_size = min(self.num_cells - len(genes), len(new_candidates))
                genes.extend(random.sample(new_candidates, sample_size))
            curIdx += sample_size
            # grid.update_positions(genes)
        grid = LandGrid(genes, self.width, self.height, pop_id)
        return grid

    def choose_start_position(self):
        start_position = self.config.config_options('start_position')
        if 'middle' in start_position:
            genes = [Pos(int(self.width / 2), int(self.height / 2))]
        elif 'random' in start_position:
            genes = list(Util.select_random_position(self.width, self.height))
        else:
            genes = [Pos(int(self.width / 2), int(self.height / 2))]

        return genes

    def generate_random(self, popid):
        all_pos = {Pos(x, y) for x in range(self.width) for y in range(self.height)}
        genes = {random.choice(list(all_pos))}
        for x in range(self.num_cells - 1):
            not_in_genes = all_pos - genes
            chromosome = random.choice(list(not_in_genes))
            genes.add(chromosome)

        return LandGrid(list(genes), self.width, self.height, popid)

    def reproduce(self, pops, generation,fitness_filename, targetpath='./results/'):
        # ga parameters
        numfig = self.config.config_options('numfig')
        mutation_rate = self.config.config_options('mutationrate')
        matingpool_multiple = int(self.config.config_options('matingpool_multiple'))
        crossover_chance = self.config.config_options('crossoverchance')
        keep_best_rate = self.config.config_options('keep_best_rate') # how much you want to keep the best parent by rate
        mutate_option = self.config.config_options('mutate_option')[0]
        fit_option = self.config.config_options('fitoption')[0]
        crossover_method = self.config.config_options('crossover_method')[0]

        fitnesses = selected_fitnesses(pops, fit_option)
        probs = Util.fits_to_probs(fitnesses)
        new_pops, new_fitnesses, id = self.init_new_pops(pops)
        mating_pool = random.choices(pops, probs, k=matingpool_multiple*len(pops))

        crossover = {
            'a': crossover_a,
            'b': crossover_b,
            'c': crossover_c
        }
        while len(new_pops) < self.pop_size:
            if random.random() < crossover_chance :
                child_genes, p1, p2 = crossover[crossover_method](mating_pool, self.width, self.height, self.num_cells)
                # cc = Util.connected_component(child_genes, self.width, self.height)
                # while len(cc) > 1: #todo: when applied with random initialization method, too many cc raises the errors so just leave it now for the test
                #     child_genes, p1, p2 = crossover[crossover_method](mating_pool, self.width, self.height, self.num_cells)
                #     cc = Util.connected_component(child_genes, self.width, self.height)
            else:
                newchild = random.choice(mating_pool)
                child_genes = newchild.poses
                p1 =  p2 =   newchild.pop_id
            child_genes = mutate(child_genes, self.width, self.height, mutation_rate, mutate_option=mutate_option,
                                 fit_option=fit_option)

            child = LandGrid(child_genes, self.width, self.height, id, p1, p2)
            new_pops.append(child)
            fitness = child.fitness
            new_fitnesses.append(fitness._fits)
            id += 1

        Util.saveData(generation, new_pops, fit_option, new_fitnesses, fitness_filename, numfig, targetpath=targetpath)
        statdisc = statutil.get_descriptive(fit_option, new_fitnesses, generation)
        return new_pops, statdisc

    def init_new_pops(self, pops):
        #todo : testing for mutating only
        keep_best_rate = self.config.config_options('keep_best_rate')
        fit_option = self.config.config_options('fitoption')[0]
        keep_count = int(len(pops) * keep_best_rate)
        object_fits_tup = [(mass.pop_id, mass.fitness._fits[fit_option]) for mass in pops] # sorting에 유리
        selected_pops = sorted(object_fits_tup, key=lambda x: x[1])[-keep_count:]
        selected_pops_id = [x[0] for x in selected_pops]
        new_pops = [x for x in pops if x.pop_id in selected_pops_id] if keep_count else []
        new_fitnesses = [x.fitness._fits for x in pops if x.pop_id in selected_pops_id]
        id = 0
        for x in new_pops:
            x.p1 = x.pop_id
            x.p2 = -1
            x.pop_id = id
            new_fitnesses[id]['id'] = id
            id += 1
        # todo: end testing. chceck new_pops
        return new_pops, new_fitnesses, id

    def initial_population_save(self, pop_size, targetpath='.\\results\\'):
        start = datetime.now()
        pops = []
        popid = 0

        crossover = {
            'a': crossover_a,
            'b': crossover_b,
            'c': crossover_c
        }
        init_method = self.config.config_options('init_method')[0]
        generate = {
            'random': self.generate_random,
            'a': self.generate_gero,
            'b': self.generate_origin,
            'c': self.generate_by_block #generate_new_simple0914
        }
        while len(pops) < pop_size:
            landgrid = generate[init_method]
            # landgrid = self.generate_new(self.width, self.height, self.num_cells, len(pops))  # todo: for temp only 0825
            # landgrid = self.generate_origin(self.width, self.height, self.num_cells, len(pops))  # todo: for temp only 0825
            # landgrid = self.generate_gero(self.width, self.height, self.num_cells, len(pops)-1)  # todo: for temp only 0825
            # landgrid = self.generate_random()
            # landgrid = self.generate_new_simple0914(self.width, self.height, self.num_cells, len(pops))  # todo: for temp only 0825 # method C
            landgrid = self.generate_origin0909(self.width, self.height, self.num_cells, len(pops)) #method C
            # landgrid = self.generate_gero0909(self.width, self.height, self.num_cells, len(pops)-1)  # method A todo: for temp only 0825
            landgrid.pop_id = popid
            popid += 1
            pops.append(landgrid)
        print(f'Elapsed Time for initial population: {(datetime.now() - start).total_seconds()}')
        start = datetime.now()
        numfig = self.config.config_options('numfig')
        rows = int( numfig / 5)
        # fig_text = '(a) Random assignment'
        # fig_text = '(b) random adjacent assignment of last position'
        # fig_text = '(c) all adjacent assignment of a selected position'
        fig_text = 'Initial Population'
        # Util.plotGridBatch(pops, numfig, rows,fig_text, omit_parents=True )
        random_pop = random.sample(pops, 50) if len(pops) > numfig else pops
        Util.plotGridOnlyRow(random_pop, int(len(random_pop) / 5), fig_text, omit_parents=True, targetpath=targetpath)
        print(f'Elapsed Time for drawing init pops: {(datetime.now()-start).total_seconds()}')
        return pops


    def get_improvement(self, pops, fit_option, child, p1, p2):
        p1_fit = pops[p1].fitness._fits[fit_option]
        p2_fit = pops[p2].fitness._fits[fit_option]
        child_fit = child.fitness._fits[fit_option]

        child_from = None if  child_fit >= p1_fit and child_fit >= p2_fit else \
            p1 if p1_fit > p2_fit else p2
        return child if  child_from is None else \
            pops[p1] if p1_fit > p2_fit else pops[p2], child_from
            # child_fit >= p1_fit and child_fit >= p2_fit else \
            # pops[p1] if p1_fit > p2_fit else pops[p2]




    def reproduce_random(self, pops, generation, fitness_filename='fitness_sample.csv', stat_filename = 'stat_sample.csv'):
        mutation_rate = self.config.config_options('mutationrate')
        fit_option = self.config.config_options('fitoption')[0]
        mating_pool = selection(pops, self.width, self.height, self.num_cells, fit_option) #todo this is working version
        #parents2 = selectTwo(pops, self.width, self.height, self.num_cells)
        new_pops = []
        fitnesses = []
        while len(new_pops) < self.pop_size:
            child_genes = crossover_random (mating_pool, self.width, self.height, self.num_cells)
            # child = crossoverTwo (mating_pool, self.width, self.height, self.num_cells, mutation_rate)
            # child_genes = crossoverFour(mating_pool, self.width, self.height, self.num_cells) # todo: 8-15 changes
            child_genes = mutate(child_genes, self.width, self.height, mutation_rate)
            child = LandGrid(child_genes, self.width, self.height)
            new_pops.append(child)
            # fitness takes whole structure
            # fitness = self.get_fitness(child)
            fitness = child.fitness
            # print(fitness._fits)
            fitnesses.append(fitness._fits)
            # Util.plotColorMesh(child, fitness, self.str_settings)
        # fitnesses = update_fitness(fitnesses) #let's use original pa
        fig_title = 'Gen:'+str(generation)
        random_pop = random.sample(new_pops, 50)
        Util.plotGridOnlyRow(random_pop, int( len(random_pop) / 5), fig_title) #todo:  10 for pops= 100
        fitname = 'f(PAR)'
        mean_fitness = mean(x.get('f(PAR)') for x in fitnesses)
        Util.saveFitnessCsv(fitness_filename, fitnesses, generation, mean_fitness, fitname)
        statdisc = statutil.get_descriptive(fitname, fitnesses, generation)
        return new_pops, statdisc


    def initialize_adjs_available(self,width, height):
        adjs_available = {}


        for y in range(height):
            for x in range(width):
                valid_adjs = 4
                pos = Pos(x, y)
                if pos.x <= 0 or pos.x >= width - 1:
                    valid_adjs -= 1
                if pos.y <= 0 or pos.y >= height - 1:
                    valid_adjs -= 1
                adjs_available[Pos(x, y)] = valid_adjs
        return adjs_available


    def test_random_selection(self):
        print('Test Random Selection')
        init_pops = 100
        fit_criteria = 'f(PAR)'
        pops = self.initial_random_population( init_pops, fit_criteria)

        rows = int(len(pops)/5)
        Util.plotGridOnlyRow(pops, rows)
        # for plan in pops:
        #     fitness = self.get_fitness(plan)
        #     # Util.plotColorMesh(plan, fitness, self.str_settings)
        filename_postfix =  datetime.now().strftime('%Y%m%d%H%M%S.csv')
        fitness_filename = 'fitness' + filename_postfix
        stat_filename = 'stat'+filename_postfix
        stats = []
        for i in range(30):
            pops, stat = self.reproduce_random(pops, i, fitness_filename, stat_filename)
            stats.append(stat)
        Util.saveStat(stat_filename, stats)

    def initial_random_population(self, init_pops,fit_criteria):
        start = datetime.now()
        pops = []
        # while len(pops) < init_pops:
        while len(pops) < self.pop_size:
            genes = self.generate_random()
            pops.append(genes)

        numfig = self.config.config_options('numfig')
        rows = int( numfig / 5)
        fig_text = '(a) Initial population created by random assignment'
        Util.plotGridBatch(pops, numfig, rows, fig_text)

        # # todo: init_pop을 엄청나게 많이 한 다음에 여기서 필터링 하는 방법을 try하다가 index 가져오는 걸 실패해서 보류
        # attr, fits, edges, fitnesses = calcFitness(pops, self.width, self.height, self.num_cells)
        # fit_values = [x.get(fit_criteria) for x in fitnesses]
        # n = 10
        # filtered_pop_index = [(fit_values.index(x), fit_values) for x in
        #                       sorted(fit_values, reverse=True)][:n] # 값이 같을 때 index 값을 처음 걸 가져오네 제기랄

        print(f'Elapsed Time for initial population: {(datetime.now() - start).total_seconds()}')
        return pops


    def generate_new_simple(self, width, height, num_cells, pop_id = 0): # method A
        # grid = LandGrid(genes, width, height) #grid가 있어야 동적으로 gene 생성 가능
        genes = self.choose_start_position()

        curIdx = 0
        while len(genes) < num_cells:
            gene_set = Util.available_adjacency(genes, curIdx, width, height)
            sample_size = min(num_cells - len(genes), len(gene_set))
            if sample_size > 0:
                genes.extend(random.sample(gene_set, sample_size))
            else:
                new_candidates = Util.bound_adjacent(genes, width, height)
                sample_size = min(num_cells - len(genes), len(new_candidates))
                genes.extend(random.sample(new_candidates, sample_size))
            curIdx += sample_size
            # grid.update_positions(genes)
        grid = LandGrid(genes, width, height, pop_id)
        return grid

    def generate_new(self, width, height, num_cells, pop_id = 0):
        genes = list(Util.select_random_position(width, height))
        # grid = LandGrid(genes, width, height) #grid가 있어야 동적으로 gene 생성 가능
        curIdx = 0
        while len(genes) < num_cells:
            gene_set = Util.bound_adjacent(genes, width, height)
            sample_size = min(num_cells - len(genes), len(gene_set))
            if sample_size > 0:
                genes.extend(random.sample(gene_set, sample_size))
            else:
                new_candidates = Util.bound_adjacent(genes, width, height)
                sample_size = min(num_cells - len(genes), len(new_candidates))
                genes.extend(random.sample(new_candidates, sample_size))
            curIdx += sample_size
            # grid.update_positions(genes)
        grid = LandGrid(genes, width, height, pop_id)
        return grid


    def generate_gero(self, width, height, num_cells, pop_id = 0):
        # genes = Util.select_random_position(width, height)
        genes = self.choose_start_position()
        # grid = LandGrid(genes, width, height) #grid가 있어야 동적으로 gene 생성 가능
        curIdx = 0
        prob = [width / (width + height), height / (width + height)]
        while len(genes) < num_cells:
            last_pos = genes[len(genes)-1]
            all_adjs = Util.adjacent_four_way(last_pos, width, height)
            gene_set  = [x for x in all_adjs if x not in genes]
            if len(gene_set)  < 1:
                gene_set = Util.bound_adjacent(genes, width, height)
            next_code = random.choice(gene_set)
            genes.append(next_code)

            # grid.update_positions(genes)
        grid = LandGrid(genes, width, height, pop_id)
        return grid

    # def generate_parent(length, geneSet, get_fitness):
    #     genes = []
    #     while len(genes) < length:
    #         sampleSize = min(length - len(genes), len(geneSet))
    #         genes.extend(random.sample(geneSet, sampleSize))
    #     fitness = get_fitness(genes)
    #     return Chromosome(genes, fitness)

    def generate_new_0817(self, width, height, num_cells):
        genes = list(Util.select_random_position(width, height))
        grid = LandGrid(genes, width, height) #grid가 있어야 동적으로 gene 생성 가능
        curIdx = 0
        while len(genes) < num_cells:
            available_adjacency = grid.available_adjacency(genes[curIdx])
            if available_adjacency:
                choice = random.choice(available_adjacency)
                genes.append(choice)
            else: # if there are no available adajcency
                choice = random.choice(Util.bound_adjacent(genes, width, height))
                genes.append(choice)
            grid.update_positions(genes)
            curIdx += 1
        return grid

    def generate_origin_save(self, width, height, num_cells, pop_id = 0): # method B
        genes = list(Util.select_random_position(width, height))
        sample = []
        while len(genes) < num_cells:
            while not sample:
                pickedIdx = random.choice(range(len(genes)))
                sample = list(set(Util.adjacent_four_way(genes[pickedIdx], width, height)) - set(genes))
            new_gene = random.choice(sample)
            genes.append(new_gene)
            sample = []
        return LandGrid(genes, width, height)


#    def disp(genes):
#        Util.plotGrid(LandGrid(genes, width, height))


    def generate_new_save_save(self, width, height, num_cells, pop_id = 0):
        genes = list(Util.select_random_position(width, height)) #todo for random_creation
        # genes = [Pos(int(width/2), int(height/2))]
        grid = LandGrid(genes, width, height)

        currentIdx = len(genes) - 1
        adjs_occupied = {}
        adjs_available = self.initialize_adjs_available(width, height)
        genes_stack = deque([0])
        pickedIdx = currentIdx
        available_left = genes.copy()
        while len(genes) < num_cells:
            # adjs = grid.adjacency(genes[pickedIdx]) #changed now
            adjs = Util.adjacent_four_way(genes[pickedIdx], width ,height)
            nextIdx = currentIdx + 1
            available_adjs = [x for x in adjs if x not in genes] # adjs that is not in genes
            #DON'T DELETE FOLLOWING, TEST with faratio > 0.9 over
            #todo: to test removing following it is the case where all cells in the genes are occupied i.e there are no available cell in the genes
            # while not available_adjs:
            #     new_candidateIdx = genes_stack.pop()
            #     adjs = grid.adjacency(genes[new_candidateIdx])
            #     available_adjs = [x for x in adjs if x not in genes]
            #     pickedIdx = None #becuase there is no adjs for piced, so randomly pickedIdx in empty space, so there is no pickedIdx
            picked_adj = random.choice(available_adjs)

            if adjs_occupied.get(currentIdx) is None:
                adjs_occupied[currentIdx] = set()
            if adjs_occupied.get(nextIdx) is None:
                adjs_occupied[nextIdx] = set()

            genes.append(picked_adj)
            available_left.append(picked_adj)
            naver = set(x for x in grid.adjacency(picked_adj) if x in genes) #naver 는  genes에 있는 모든 adjs
            adjs_occupied[nextIdx] = adjs_occupied[nextIdx] | naver
            for naver_gene in naver: #이웃하는 노드들 in genes
                naver_loc = genes.index(naver_gene) #위치를 가져와서
                if not adjs_occupied[naver_loc].__contains__(picked_adj): #그 위치가 이미 다음 picked_adjs를 포함하지 않는다면, 즉 거기에서 유래되는 자식노드가 이미 점령되지 않았다면
                    adjs_occupied[naver_loc].add(picked_adj) # todo : check redundancy below
                    adjs_available[picked_adj] -= 1 #naver의 개수만큼 차감하는 거 맞아
                    if adjs_available[picked_adj] < 1:
                        available_left.remove(picked_adj)
                    adjs_available[naver_gene] -= 1
                    if adjs_available[naver_gene] < 1:
                        available_left.remove(naver_gene)

            # genes_stack.append(nextIdx)
            pickedIdx = genes.index(random.choice(available_left))
            currentIdx = nextIdx # todo: lets pick it form genes available
        grid = LandGrid(genes, width, height, pop_id)
        # print('grid\n',grid)
        # # todo: finish local functions of grid.py moving from fitness all thest
        # print('\nCreated Shape')
        # print(LandGrid(genes, width, height), '\n')  # Final Shape

        # print('grouped by row:' , grid.grouped_by_row())
        # print('grouped by col:' , grid.grouped_by_col())
        # adjGraph = grid.buildUndirectedGraph()
        # print(adjGraph)

        return grid

    def generate(self, width, height, num_cells):
        start = datetime.datetime.now()
        genes = [Pos(int(width/2), math.floor(height/2))] #todo: genes to genes
        grid = LandGrid(genes, width, height)

        no_adj_avaiables = []

        while len(set(genes)) < num_cells:

            # grid.display_genes(genes)
            pick_samples = [i for i in range(len(genes)) if i not in no_adj_avaiables] # 인접셀이 모두 점령된 경우눈 제외하고 선택
            pickIdx = random.choice(pick_samples)
            surr =[adj for adj in grid.adjacency(genes[pickIdx]) if adj not in genes] # grid.adjacency는 모든 인접셀을 구한다.

            if [i for i in surr if i not in genes]: # if surr is NOT empty
                genes = genes + random.sample(surr, 1)
            else: # surr is empty
                while not surr:
                    pick_samples = [i for i in range(len(genes)) if i not in no_adj_avaiables ]
                    pickIdx = random.choice(pick_samples)
                    surr = [i for i in grid.adjacency(genes[pickIdx]) if i not in genes]
                    if not surr:
                        no_adj_avaiables.append(pickIdx)
                    # print(pickIdx, surr, no_adj_avaiables)
                genes = genes + random.sample(surr, 1) #todo: 이걸 뒤로 빼면 나중에  pickIdx가 notempty 되었을 때의 값으로 대체될 거 같아
        grid.update_positions(genes)
        print(f'elapsed time generate() in ms: {start - datetime.datetime.now()}')
        # print('grid\n',grid)
        # # todo: finish local functions of grid.py moving from fitness all thest
        print('\nCreated Shape')
        print(LandGrid(genes, width, height), '\n') #  Final Shape

        # print('grouped by row:' , grid.grouped_by_row())
        # print('grouped by col:' , grid.grouped_by_col())
        # adjGraph = grid.buildUndirectedGraph()
        # print(adjGraph)

        return grid

    # build dict for adj occupied pair
    def create_population_new(self, width, height, num_cells, population_size, str_settings):
        pops = []
        fitnesses = []
        while len(pops) < population_size:
            # genes = self.generateNew(width, height, num_cells)  # todo 0511
            # genes = self.generate_new(width, height, num_cells)  # todo 0805
            genes=self.generate(width, height, num_cells)
            fitness = Fitness(genes, width, height, num_cells)
            fitnesses.append(fitness._fits)
            attr, fits, edges = fitness.build_attrs(genes,num_cells)
            # pops.append(genes)
            if (fitness._fits['f(FSH)'] < 0.1): #todo : print peculiar pattern
                pops.append(genes)
                Util.plotColorMesh(genes, fitness, str_settings)
        return pops
        # Util.plotGridOnlyl(pops) #todo 0511
        # Util.saveCsv('fitnesses.csv', fitnesses)

    def create_population_all(self, width, height, num_cells, population_size, str_settings):
    # building_line을 충족하지 못하는 것도 모두
        pops = []
        fitnesses = []
        while len(pops) < population_size:
            # genes = self.generateNew(width, height, num_cells)  # todo 0511
            genes = self.generate(width, height, num_cells)  # todo 0511
            fitness = Fitness(genes, width, height, num_cells)
            fitnesses.append(fitness._fits)
            attr, fits, edges = fitness.build_attrs(genes,num_cells)
            # pops.append(genes)
            if (fitness._fits['f(FSH)'] < 0.1): #todo : print peculiar pattern
                pops.append(genes)
                Util.plotColorMesh(genes, fitness, str_settings)
            # Util.plotColorMesh(genes, fitness, str_settings) # todo change to this whenever needed to print values
        Util.plotGridOnly(pops) #todo 0511
            # Util.plotColorMesh(genes, fitness, str_settings)
        #Util.saveCsv('fitnesses.csv', fitnesses)
    def create_population(self, width, height, num_cells, population_size, str_settings):
        pops = []
        while len(pops) < population_size:
        # for i in range(population_size):
            genes = self.generate(width, height, num_cells)
            fitness = Fitness(genes, width, height, num_cells)
            attr, fits, edges = fitness.build_attrs(genes,num_cells)
            # if fits['f(PAR)'] > 0.7:
            #         # and fits['sunlight hours'] > 8\
            #         # and fits['pa_ratio'] >= 0.7  :
            pops.append(genes)
            Util.plotColorMesh(genes, fitness, str_settings) # todo change to this whenever needed to print values
        Util.plotGridOnlyRow(pops,10) #todo to draw plot
        return pops

    def create_mating_pool(self, population, fitness_name):
        pass

    # def crossover(self, genes1, genes2):
    #     pt = random.randint(0, len(genes1) - 2)
    #     child = genes1[:pt] + genes2[pt:]
    #     return child


    #@repeat(10)
    # def test_generate(self):
    #     width = self.config.config_options('width')
    #     height = self.config.config_options('height')
    #     floorAreaRatio = self.config.config_options('required_faratio')
    #     num_cells = Util.get_num_cells(width, height, floorAreaRatio)
    #
    #     grid = self.generate(width, height, num_cells)
    #     tupPos = [tuple((p.x, p.y)) for p in grid.poses]

    def reproduce_save0920(self, pops, generation, fitness_filename):
        crossover_method = self.config.config_options('crossover_method')[0]
        mutation_rate = self.config.config_options('mutationrate')
        fit_option = self.config.config_options('fitoption')[0]
        mating_pool = selection(pops, self.width, self.height, self.num_cells, fit_criteria=fit_option) #todo this is working version
        #parents2 = selectTwo(pops, self.width, self.height, self.num_cells)
        new_pops = []
        fitnesses = []
        id = 0

        crossover = {
            'a': crossover_a,
            'b': crossover_b,
            'c': crossover_c
        }
        while len(new_pops) < self.pop_size:

            # child_genes, p1, p2 = crossover_random (mating_pool, self.width, self.height, self.num_cells)
            # child_genes, p1, p2 = crossover_a(mating_pool, self.width, self.height,self.num_cells)  # todo: comeback here 0907
            # child_genes, p1, p2 = crossover_b(mating_pool, self.width, self.height, self.num_cells)
            # child_genes, p1, p2 = crossover_c(mating_pool, self.width, self.height, self.num_cells)
            child_genes, p1, p2 = crossover[crossover_method](mating_pool, self.width, self.height, self.num_cells)
            child_genes = mutate(child_genes, self.width, self.height, mutation_rate)

            child = LandGrid(child_genes, self.width, self.height,id, p1, p2)
### todo:
### Don't delete Following.
### It is important to get fast local improvement

            # child, child_from = self.get_improvement(pops, fit_option, child,  p1, p2) # todo : when child is worse than take parents
            # if child_from is not None: # 두 부모 중 하나에게서 왔다. 그 부모를 모두 표시하라
            #     omit_parent = False
            #     child.p1 = child_from
            #     child.p2 = child_from
            #
            # elif child_from is None: # 두 부모에게서 왔다. 부모를 표시해라. omit 하지 말아라. False
            #     omit_parent = False


            new_pops.append(child)
            fitness = child.fitness
            fitnesses.append(fitness._fits)
            id += 1
            # Util.plotColorMesh(child, fitness, self.str_settings)
        fig_title = 'Generation #' + str(generation)
        # todo: delete when paper picture is done
        # random_pop = random.sample(new_pops, 10)
        # Util.plotGridOnlyRow(random_pop, int(len(random_pop) / 2), fig_title)  # todo:  10 for pops= 100

        # todo: plot 50 picture per page
        random_pop = random.sample(new_pops, 50)
        Util.plotGridOnlyRow(random_pop, int(len(random_pop) / 5), fig_title)  # todo:  10 for pops= 100
        # Util.plotGridOnlyRow(new_pops, int(len(new_pops) / 5), fig_title)  # todo:  10 for pops= 100
        fitname = fit_option
        mean_fitness = mean(x.get(fit_option) for x in fitnesses)
        Util.saveFitnessCsv(fitness_filename, fitnesses, generation, mean_fitness, fitname)
        statdisc = statutil.get_descriptive(fitname, fitnesses, generation)

        return new_pops, statdisc

    def test_selection(self):
        print('Test Selection')
        postfix_bydate = datetime.now().strftime('%m%d')
        targetpath = '.\\results' + postfix_bydate+'\\'
        pops = self.initial_population(self.pop_size, targetpath)  # todo : for f(ss) test
        # pops = self.initial_random_population(self.pop_size, 'f(PAR)')  # todo : for f(ss) test
        # for plan in pops:
        #     fitness = self.get_fitness(plan)
        #     # Util.plotColorMesh(plan, fitness, self.str_settings)
        stats=[]
        filename_postfix = datetime.now().strftime('%d_%H_%M_%S.csv')
        fitness_filename = targetpath+'fitness' + filename_postfix
        stat_filename =  targetpath+'stat' + filename_postfix
        ngen = self.config.config_options('ngeneration')
        for i in range(ngen):
           # pops, stat = self.reproduce(pops, i, fitness_filename) #todo: reproduce saved 0909
           pops, stat = self.reproduce(pops, i, fitness_filename, targetpath)
           # pops, stat = self.reproduce_random(pops, i, fitness_filename, stat_filename)
           stats.append(stat)
        Util.saveStat(stat_filename, stats)

if __name__ == '__main__':
    unittest.TestCase()
