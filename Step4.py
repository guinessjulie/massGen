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
from genop import selection, selectTwo, crossover_overwrap, crossoverTwo
from datetime import datetime
from statistics import mean

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
    num_cells = Util.get_num_cells(width, height, config.config_options('required_faratio'))
    opt_fit = config.config_options('opt_fit')

    def initial_population(self):
        pops = []
        while len(pops) < self.pop_size:
            genes = self.generate_new(self.width, self.height, self.num_cells) # todo: for temp only 0825
            # genes = self.generate_random()
            pops.append(genes)
        return pops

    def get_fitness(self, grid):
        fitness = Fitness(grid, self.width, self.height, self.num_cells)
        return fitness

    def test_selection(self):
        pops = self.initial_population()  # todo : for f(ss) test
        rows = int( len(pops) / 5)
        Util.plotGridOnlyRow(pops, rows)
        for plan in pops:
            fitness = self.get_fitness(plan)
            # Util.plotColorMesh(plan, fitness, self.str_settings)
        fitness_filename = 'fitness' + datetime.now().strftime('%Y%m%d%H%M%S.csv')
        for i in range(10):
            pops = self.reproduce(pops, i, fitness_filename)

    def reproduce(self, pops, generation, fitness_filename):
        mutation_rate = self.config.config_options('mutationrate')
        mating_pool = selection(pops, self.width, self.height, self.num_cells) #todo this is working version
        #mating_pool2 = selectTwo(pops, self.width, self.height, self.num_cells)
        new_pops = []
        fitnesses = []
        while len(new_pops) < self.pop_size:
            child = crossover_overwrap(mating_pool, self.width, self.height, self.num_cells, mutation_rate, generation)
            # child = crossoverTwo (mating_pool, self.width, self.height, self.num_cells, mutation_rate)

            new_pops.append(child)
            # fitness takes whole structure
            fitness = self.get_fitness(child)
            # print(fitness._fits) #todo: use whenever you want to printout fitness value debug
            fitnesses.append(fitness._fits)
            # Util.plotColorMesh(child, fitness, self.str_settings)
        Util.plotGridOnlyRow(new_pops, int( len(pops) / 5)) #todo:  10 for pops= 100
        fitname = 'f(PAR)'
        mean_fitness = mean(x.get('f(PAR)') for x in fitnesses)
        Util.saveFitnessCsv(fitness_filename, fitnesses, generation, mean_fitness, fitname)

        return new_pops

    def _test_generateNew(self):

        print(self.str_settings)
        num_cells = Util.get_num_cells(self.width, self.height, self.config.config_options('required_faratio'))
        pops = self.create_population_new(self.width, self.height, num_cells, self.pop_size, self.str_settings)  # todo : for f(ss) test
        selection(pops, self.width, self.height, num_cells)
        print(child_grid)

    def _test_version1(self):
        width = self.config.config_options('width')
        height = self.config.config_options('height')
        pop_size = self.config.config_options('population_size')
        str_settings = self.config.get_display_settings(width, height)
        print(str_settings)
        num_cells = Util.get_num_cells(width, height, self.config.config_options('required_faratio'))
        pops = self.create_population(width, height, num_cells, pop_size, str_settings) #todo : for f(ss) test
        # pops = self.create_population_all(width, height, num_cells, pop_size, str_settings)
        # print('population size created',len(pops))
        # grid = self.generate(width, height, num_cells) # pos = fitness.floor = genes
        # grid2 = self.generate(width, height, num_cells)
        # childGenes = self.crossover(grid.poses, grid2.poses)
        # childGrid = Grid(childGenes, width, height)
        # fitness = self.get_fitness(grid, width, height, num_cells) # todo: revert no, no no no no fitness has already options so do not botgher the option here
        # Util.plotColorMesh(grid, fitness, str_settings)

    def don_test2(self):
        polygon_tiles = [(3, 0), (4, 0), (3, 1), (4, 1), (0, 2), (1, 2), (2, 2), (3, 2),
                         (4, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3)]
        outline = Util.trace(polygon_tiles)
        print(outline)

        # print(fitness)
        # perimeter = fitness.boundary_length()
        # southSide = fitness.south_view_ratio()
        #
        # cntSouth, cntNorth, cntWest, cntEast = fitness.south_view_ratio()
        # print('Fitness: ', perimeter, cntSouth,cntNorth, cntWest, cntEast)

    # change algorighm to speed up
    # only expend from last selected cell
    # the result is so random with alot of empty cell in the middle
    def generate_new_save(self, width, height, num_cells):
        genes = [Pos(int(width / 2), math.floor(height / 2))]  # todo: genes to genes
        grid = LandGrid(genes, width, height)
        pickedIdx = len(genes) -1
        no_adj_avaiables = []
        genes_available = genes.copy()
        tgenes = tuple(genes)
        adjs_occupied = {}
        genes_not_available = {}
        while len(genes) < num_cells:
            adjs = grid.adjacency(genes[pickedIdx])
            available_adjs = [x for x in adjs if x not in genes]# adjs that is not in genes
            if available_adjs :
                picked_adj = random.choice(available_adjs)
            else:
                all_cells = [Pos(x, y) for x in range(width) for y in range(height)]
                empty_cells = [x for x in all_cells if x not in genes]
                picked_adj = random.choice(empty_cells) # how do i choose adjs from one of genes

            if adjs_occupied.get(pickedIdx) is None:
                adjs_occupied[pickedIdx] = set()
            if adjs_occupied.get(pickedIdx+1) is None:
                adjs_occupied[pickedIdx+1] = set()

            naver = set(x for x in grid.adjacency(picked_adj) if x in genes)
            adjs_occupied[pickedIdx+1] = adjs_occupied[pickedIdx + 1] | naver

            adjs_occupied[pickedIdx].add(picked_adj)
            for val in naver:
                loc_adj = genes.index(val)
                adjs_occupied[loc_adj].add(picked_adj)
            genes.append(picked_adj)
            pickedIdx += 1 # todo: lets pick it form genes available


            # adjs_occupied[pickedIdx] = list(picked_adj) if not adjs_occupied else adjs_occupied[pickedIdx].append(picked_adj)
            # pickedIdx = [x for x in genes if not adjs_occupied[x]]:
            #
            # if available_adjs:
            #     pickedAdj = random.choice(available_adjs)
            #     genes.append(pickedAdj)
            #     pickedIdx = len(genes) - 1
            # else:
            #     genes_not_available+=adjs
            #     pickedIdx = [x for x in genes if not genes_not_available.contains(adjs)]

        grid.update_positions(genes)
        # print('grid\n',grid)
        # # todo: finish local functions of grid.py moving from fitness all thest
        print('\nCreated Shape')
        print(LandGrid(genes, width, height), '\n')  # Final Shape

        # print('grouped by row:' , grid.grouped_by_row())
        # print('grouped by col:' , grid.grouped_by_col())
        # adjGraph = grid.buildUndirectedGraph()
        # print(adjGraph)

        return grid

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
        genes = set()
        for x in range(self.num_cells):
            not_in_genes = all_pos - genes
            chromosome = random.choice(list(not_in_genes))
            genes.add(chromosome)
        return LandGrid(genes, self.width, self.height)

    # random하게 선택하게 해보자.
    def generate_new(self, width, height, num_cells):
        # genes = list(Util.select_random_position(width, height))
        genes = [Pos(int(width/2), int(height/2))]
        grid = LandGrid(genes, width, height)

        currentIdx = len(genes) - 1
        adjs_occupied = {}
        adjs_available = self.initialize_adjs_available(width, height)
        genes_stack = deque([0])
        pickedIdx = currentIdx
        available_left = genes.copy()
        start = time.time()
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

            genes_stack.append(nextIdx)
            pickedIdx = genes.index(random.choice(available_left))
            currentIdx = nextIdx # todo: lets pick it form genes available
            # start=time.time()
            # sample_available = [k for k in adjs_available if adjs_available.get(k) > 0]
            # sample_genes = [g for g in sample_available if g in genes]
            # pickedIdx = genes.index(random.choice(sample_genes))
            # print(f'two list seperate build takes: {time.time()-start}s')
            # start = time.time()
            # sample_genes = [sample_gene for sample_gene in [k for k in adjs_available if adjs_available.get(k)>0] if sample_gene in genes]
            # pickedIdx = genes.index(random.choice(sample_genes))
            # print(f'sample_genes build at once takes:{time.time()-start}s')
            # pickedIdx = genes.index(genes[picked_gene])
            # pickedIdx = random.choice(sample_genes)
            # adjs_available[currentIdx].add()
        # print(f'elapsed time: {time.time()-start}')
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

    def generate_new_saved_version2(self, width, height, num_cells):
        genes = [Pos(int(width / 2), math.floor(height / 2))]  # todo: genes to genes
        grid = LandGrid(genes, width, height)
        pickedIdx = len(genes) -1
        currentIdx = len(genes) -1
        adjs_occupied = {}
        genes_stack = deque()
        while len(genes) < num_cells:
            adjs = grid.adjacency(genes[currentIdx])
            nextIdx = currentIdx + 1
            available_adjs = [x for x in adjs if x not in genes]# adjs that is not in genes
            while not available_adjs:
                new_candidateIdx = genes_stack.pop()
                adjs = grid.adjacency(genes[new_candidateIdx])
                available_adjs = [x for x in adjs if x not in genes]
            picked_adj = random.choice(available_adjs)

            if adjs_occupied.get(currentIdx) is None:
                adjs_occupied[currentIdx] = set()
            if adjs_occupied.get(nextIdx) is None:
                adjs_occupied[nextIdx] = set()

            naver = set(x for x in grid.adjacency(picked_adj) if x in genes)
            adjs_occupied[nextIdx] = adjs_occupied[nextIdx] | naver
            adjs_occupied[currentIdx].add(picked_adj)
            for val in naver:
                loc_adj = genes.index(val)
                adjs_occupied[loc_adj].add(picked_adj)
            genes.append(picked_adj)
            genes_stack.append(currentIdx)
            currentIdx = nextIdx # todo: lets pick it form genes available

        grid.update_positions(genes)
        # print('grid\n',grid)
        # # todo: finish local functions of grid.py moving from fitness all thest
        print('\nCreated Shape')
        print(LandGrid(genes, width, height), '\n')  # Final Shape

        # print('grouped by row:' , grid.grouped_by_row())
        # print('grouped by col:' , grid.grouped_by_col())
        # adjGraph = grid.buildUndirectedGraph()
        # print(adjGraph)

        return grid
    def generate(self, width, height, num_cells):
        genes = [Pos(int(width/2), math.floor(height/2))] #todo: genes to genes
        grid = LandGrid(genes, width, height)

        no_adj_avaiables = []
        print('pickIdx', 'surr\t', 'no_adj_avaiables')
        while len(set(genes)) < num_cells:

            # grid.display_genes(genes)
            pick_samples = [i for i in range(len(genes)) if i not in no_adj_avaiables] # 인접셀이 모두 점령된 경우눈 제외하고 선택
            pickIdx = random.choice(pick_samples)
            surr =[adj for adj in grid.adjacency(genes[pickIdx]) if adj not in genes] # grid.adjacency는 모든 인접셀을 구한다.
            # surr =[adj for adj in grid.adjacency8(genes[pickIdx]) if adj not in genes]
            #print('grid.display_adjacent') #todo: for DEBUG
            #grid.display_adjacent(genes, genes[pickIdx], surr)#todo: for DEBUG
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
        # print('grid\n',grid)
        # # todo: finish local functions of grid.py moving from fitness all thest
        print('\nCreated Shape')
        print(LandGrid(genes, width, height), '\n') #  Final Shape

        # print('grouped by row:' , grid.grouped_by_row())
        # print('grouped by col:' , grid.grouped_by_col())
        # adjGraph = grid.buildUndirectedGraph()
        # print(adjGraph)

        return grid
    def generateOrigin(self, width, height, num_cells):
        genes = [Pos(int(width/2), math.floor(height/2))] #todo: genes to genes
        grid = LandGrid(genes, width, height)

        no_adj_avaiables = []

        while len(set(genes)) < num_cells:

            # grid.display_genes(genes)
            pick_samples = [i for i in range(len(genes)) if i not in no_adj_avaiables] # 인접셀이 모두 점령된 경우눈 제외하고 선택
            pickIdx = random.choice(pick_samples)
            surr =[adj for adj in grid.adjacency(genes[pickIdx]) if adj not in genes] #todo: 비교중 don't delete
            # surr =[adj for adj in grid.adjacency8(genes[pickIdx]) if adj not in genes]
            #print('grid.display_adjacent') #todo: for DEBUG
            #grid.display_adjacent(genes, genes[pickIdx], surr)#todo: for DEBUG
            if [i for i in surr if i not in genes]: # if surr is NOT empty
                genes = genes + random.sample(surr, 1)
            else: # surr is empty
                no_adj_avaiables.append(pickIdx)
                while not surr:
                    pick_samples = [i for i in range(len(genes)) if i not in no_adj_avaiables ]
                    pickIdx = random.choice(pick_samples)
                    surr = [i for i in grid.adjacency(genes[pickIdx]) if i not in genes]

                genes = genes + random.sample(surr, 1) #todo: 이걸 뒤로 빼면 나중에  pickIdx가 notempty 되었을 때의 값으로 대체될 거 같아
        grid.update_positions(genes)
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
            genes = self.generate_new(width, height, num_cells)  # todo 0511
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

    def generate2(self, width, height, num_cells): # printing debug working backup version 2022-03-16
        genes = [Pos(int(width/2), math.floor(height/2))] #todo: genes to genes
        grid = LandGrid(genes, width, height)
        no_adj_avaiables = []
        # surr = grid.adjacency(genes[0])
        print(set(genes), num_cells)
        while len(set(genes)) < num_cells:
            print('genes', genes)
            grid.display_genes(genes)
            # pickIdx = random.randint(0, len(genes)-1)
            pick_samples = [i for i in range(len(genes)) if i not in no_adj_avaiables] # 인접셀이 모두 점령된 경우눈 제외하고 선택
            pickIdx = random.choice(pick_samples)
            surr =[adj for adj in grid.adjacency(genes[pickIdx]) if adj not in genes]
            print('picked:', genes[pickIdx], 'surr:' , surr, 'genes:', genes)
            grid.display_adjacent(genes, genes[pickIdx], surr)
            if [i for i in surr if i not in genes]: # if surr is NOT empty
                # pick = random.sample(surr, 1)  # todo: 여기서 기존에 선택했던 걸 선택하면 안된다
                genes = genes + random.sample(surr, 1)
            else: # surr is empty
                print('surr is empty')
                no_adj_avaiables.append(pickIdx)
                while not surr:
                    pick_samples = [i for i in range(len(genes)) if i not in no_adj_avaiables ]
                    pickIdx = random.choice(pick_samples)
                    print('pickIdx excepted from adj not available', pickIdx)
                    #pickIdx = random.randint(0, len(genes)-1) # todo:
                    # surr = grid.adjacency(genes[pickIdx]) # todo: 여기 로직이 틀렸다. surr가 empty라는 것은 인접 셀이 모두 이미 genes에 있는 경우이다. 따라서 위에서 새로 pickIdx할 때, 현재 선택된 셀은 제외하고 선택해야 한다.
                    surr = [i for i in grid.adjacency(genes[pickIdx]) if i not in genes]

                genes = genes + random.sample(surr, 1) #todo: 이걸 뒤로 빼면 나중에  pickIdx가 notempty 되었을 때의 값으로 대체될 거 같아
                print('picked:', genes[pickIdx], 'surr:', surr, 'genes:', genes)
            # print('genes:', genes, 'surr after', surr, )
            # grid.display_adjacent(genes, genes[pickIdx], surr)

            # print(grid)
        print('Grid')
        print(LandGrid(genes, width, height), '\n')

        return genes


    def generate_(self , width, height, num_cells):

        genes = [Pos(int(width/2), int(height/2))] #todo: genes to genes
        grid = LandGrid(genes, width, height)
        surr = grid.adjacency(genes[0])
        print(set(genes), num_cells)
        while len(set(genes)) < num_cells:
            pickIdx = random.randint(0, len(genes)-1)
            surr = grid.adjacency(genes[pickIdx])
            pick = random.sample(surr, 1) # todo: 여기서 기존에 선택했던 걸 선택하면 안된다
            genes = genes + pick
        # adjacency_graph = UndirectedGraph(genes, width, height)

        print('Grid')
        print(LandGrid(genes, width, height), '\n')
        return genes


    def generate_multi(self, width, height, num_cells):
        for i in range(10):
            self.generate(width, height, num_cells)

if __name__ == '__main__':
    unittest.TestCase()