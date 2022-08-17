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
from genop import selection, selectTwo, crossover_overwrap,crossover_random, calcFitness, crossoverFour, mutate, update_fitness
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

        # # todo: init_pop을 엄청나게 많이 한 다음에 여기서 필터링 하는 방법을 try하다가 index 가져오는 걸 실패해서 보류
        # attr, fits, edges, fitnesses = calcFitness(pops, self.width, self.height, self.num_cells)
        # fit_values = [x.get(fit_criteria) for x in fitnesses]
        # n = 10
        # filtered_pop_index = [(fit_values.index(x), fit_values) for x in
        #                       sorted(fit_values, reverse=True)][:n] # 값이 같을 때 index 값을 처음 걸 가져오네 제기랄

        print(f'Elapsed Time for initial population: {(datetime.now() - start).total_seconds()}')
        return pops


    def test_selection(self):
        print('Test Selection')
        pops = self.initial_population(self.pop_size)  # todo : for f(ss) test
        # for plan in pops:
        #     fitness = self.get_fitness(plan)
        #     # Util.plotColorMesh(plan, fitness, self.str_settings)
        stats=[]
        filename_postfix = datetime.now().strftime('%Y%m%d%H%M%S.csv')
        fitness_filename = 'fitness' + filename_postfix
        stat_filename = 'stat' + filename_postfix
        for i in range(30):
            # pops = self.reproduce(pops, i, fitness_filename)
           pops, stat = self.reproduce_random(pops, i, fitness_filename, stat_filename)
           stats.append(stat)
        Util.saveStat(stat_filename, stats)

    def reproduce(self, pops, generation, fitness_filename):
        mutation_rate = self.config.config_options('mutationrate')
        mating_pool = selection(pops, self.width, self.height, self.num_cells) #todo this is working version
        #parents2 = selectTwo(pops, self.width, self.height, self.num_cells)
        new_pops = []
        fitnesses = []
        while len(new_pops) < self.pop_size:
            # todo: crossover_overwrap은 십자 모양으로 일괄 수렴한다.
            # child = crossover_overwrap(mating_pool, self.width, self.height, self.num_cells, mutation_rate, generation)
            # child = crossoverTwo (mating_pool, self.width, self.height, self.num_cells, mutation_rate)
            # todo: 이 방법은 형태가 점점 복잡해지고, fitness가 더 적어진다.
            child = crossover_random (mating_pool, self.width, self.height, self.num_cells)

            new_pops.append(child)
            # fitness takes whole structure
            fitness = self.get_fitness(child)
            # print(fitness._fits)
            fitnesses.append(fitness._fits)
            # Util.plotColorMesh(child, fitness, self.str_settings)
        fig_title = 'Gen:' + str(generation)
        random_pop = random.sample(new_pops, 50)
        Util.plotGridOnlyRow(random_pop, int(len(random_pop) / 5), fig_title)  # todo:  10 for pops= 100
        fitname = 'f(PAR)'
        mean_fitness = mean(x.get('f(PAR)') for x in fitnesses)
        Util.saveFitnessCsv(fitness_filename, fitnesses, generation, mean_fitness, fitname)
        statdisc = statutil.get_descriptive(fitname, fitnesses, generation)

        Util.plotGridOnlyRow(new_pops, int( len(pops) / 5)) #todo:  10 for pops= 100
        fitname = 'f(PAR)'

        mean_fitness = mean(x.get('f(PAR)') for x in fitnesses)
        Util.saveFitnessCsv(fitness_filename, fitnesses, generation, mean_fitness, fitname)

        return new_pops, statdisc

    def reproduce_random(self, pops, generation, fitness_filename='fitness_sample.csv', stat_filename = 'stat_sample.csv'):
        mutation_rate = self.config.config_options('mutationrate')
        mating_pool = selection(pops, self.width, self.height, self.num_cells) #todo this is working version
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
            fitness = self.get_fitness(child)
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

    def initial_population(self, pop_size):
        start = datetime.now()
        pops = []
        while len(pops) < pop_size:
            genes = self.generate_new(self.width, self.height, self.num_cells)  # todo: for temp only 0825
            # genes = self.generate_random()
            pops.append(genes)
        print(f'Elapsed Time for initial population: {(datetime.now() - start).total_seconds()}')
        start = datetime.now()
        numfig = self.config.config_options('numfig')
        rows = int( numfig / 5)
        # Util.plotGridOnlyRow(pops, rows)
        Util.plotGridBatch(pops, numfig, rows)
        print(f'Elapsed Time for drawing init pops: {(datetime.now()-start).total_seconds()}')
        return pops


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

    def generate_random(self):
        all_pos = {Pos(x,y) for x in range(self.width) for y in range(self.height)}
        genes = {random.choice(list(all_pos))}
        for x in range(self.num_cells - 1 ):
            not_in_genes = all_pos - genes
            chromosome = random.choice(list(not_in_genes))
            genes.add(chromosome)
        return LandGrid(list(genes), self.width, self.height)

    def generate_new(self, width, height, num_cells):
        genes = list(Util.select_random_position(width, height))
        grid = LandGrid(genes, width, height) #grid가 있어야 동적으로 gene 생성 가능
        curIdx = 0
        while len(genes) < num_cells:
            gene_set = grid.available_adjacency(genes[curIdx])
            sample_size = min(num_cells - len(genes), len(gene_set))
            if sample_size > 1:
                genes.extend(random.sample(gene_set, sample_size))
            else:
                new_candidates = Util.bound_adjacent(genes, width, height)
                sample_size = min(num_cells - len(genes), len(new_candidates))
                genes.extend(random.sample(new_candidates, sample_size))
            curIdx += sample_size
            grid.update_positions(genes)
        return grid

    def generate_parent(length, geneSet, get_fitness):
        genes = []
        while len(genes) < length:
            sampleSize = min(length - len(genes), len(geneSet))
            genes.extend(random.sample(geneSet, sampleSize))
        fitness = get_fitness(genes)
        return Chromosome(genes, fitness)

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



    def generate_new_save(self, width, height, num_cells):
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
            adjs = grid.adjacency(genes[pickedIdx])
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
        grid.update_positions(genes)
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

if __name__ == '__main__':
    unittest.TestCase()