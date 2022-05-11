import random
import unittest
import measureUtil
import util
from util import Util, Pos
from fitness import Fitness
from grid import Grid
from options import Options
import math

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

    # test all the case for massive data for statistics
    def test(self):
        width = self.config.config_options('width')
        height = self.config.config_options('height')
        pop_size = self.config.config_options('population_size')
        str_settings = self.config.get_display_settings(width, height)
        print(str_settings)
        num_cells = Util.get_num_cells(width, height, self.config.config_options('required_faratio'))
        pops = self.create_population_all(width, height, num_cells, pop_size, str_settings)  # todo : for f(ss) test

    @repeat(1)
    def later_test1(self):
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
    def get_fitness(self, grid, width, height, num_cells):
        fitness = Fitness(grid, width, height, num_cells)
        return fitness
        # print(fitness)
        # perimeter = fitness.boundary_length()
        # southSide = fitness.south_view_ratio()
        #
        # cntSouth, cntNorth, cntWest, cntEast = fitness.south_view_ratio()
        # print('Fitness: ', perimeter, cntSouth,cntNorth, cntWest, cntEast)

    def generate(self, width, height, num_cells):
        genes = [Pos(int(width/2), math.floor(height/2))] #todo: genes to genes
        grid = Grid(genes, width, height)

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
        print(Grid(genes, width, height), '\n') #  Final Shape

        # print('grouped by row:' , grid.grouped_by_row())
        # print('grouped by col:' , grid.grouped_by_col())
        # adjGraph = grid.buildUndirectedGraph()
        # print(adjGraph)

        return grid

    # building_line을 충족하지 못하는 것도 모두
    def create_population_all(self, width, height, num_cells, population_size, str_settings):
        pops = []
        fitnesses = []
        while len(pops) < population_size:
            genes = self.generate(width, height, num_cells)
            fitness = Fitness(genes, width, height, num_cells)
            fitnesses.append(fitness._fits)
            attr, fits, edges = fitness.build_attrs(genes,num_cells)
            pops.append(genes)
            # if(fitness._fits['f(FSH)']<0.1): #todo : print peculiar pattern
            #     Util.plotColorMesh(genes, fitness, str_settings)
            # Util.plotColorMesh(genes, fitness, str_settings) # todo change to this whenever needed to print values
        # Util.plotGridOnlyl(pops)
        Util.saveCsv('fitnesses.csv', fitnesses)
    def create_population(self, width, height, num_cells, population_size, str_settings):
        pops = []
        while len(pops) < population_size:
        # for i in range(population_size):
            genes = self.generate(width, height, num_cells)
            fitness = Fitness(genes, width, height, num_cells)
            attr, fits, edges = fitness.build_attrs(genes,num_cells)
            if fits['Fulfill Building Line'] == 'Success':
                    # and fits['sunlight hours'] > 8\
                    # and fits['pa_ratio'] >= 0.7  :
                pops.append(genes)
                Util.plotColorMesh(genes, fitness, str_settings) # todo change to this whenever needed to print values
        # Util.plotGridOnlyl(pops) #todo to draw plot
        return pops

    def create_mating_pool(self, population, fitness_name):
        pass

    def crossover(self, genes1, genes2):
        pt = random.randint(0, len(genes1) - 2)
        child = genes1[:pt] + genes2[pt:]
        return child
    #@repeat(10)
    def test_generate(self):
        width = self.config.config_options('width')
        height = self.config.config_options('height')
        floorAreaRatio = self.config.config_options('required_faratio')
        num_cells = Util.get_num_cells(width, height, floorAreaRatio)

        grid = self.generate(width, height, num_cells)
        tupPos = [tuple((p.x, p.y)) for p in grid.poses]

    def generate2(self, width, height, num_cells): # printing debug working backup version 2022-03-16
        genes = [Pos(int(width/2), math.floor(height/2))] #todo: genes to genes
        grid = Grid(genes, width, height)
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
        print(Grid(genes, width, height), '\n')

        return genes


    def generate_(self , width, height, num_cells):

        genes = [Pos(int(width/2), int(height/2))] #todo: genes to genes
        grid = Grid(genes, width, height)
        surr = grid.adjacency(genes[0])
        print(set(genes), num_cells)
        while len(set(genes)) < num_cells:
            pickIdx = random.randint(0, len(genes)-1)
            surr = grid.adjacency(genes[pickIdx])
            pick = random.sample(surr, 1) # todo: 여기서 기존에 선택했던 걸 선택하면 안된다
            genes = genes + pick
        # graph = UndirectedGraph(genes, width, height)

        print('Grid')
        print(Grid(genes, width, height), '\n')
        return genes


    def generate_multi(self, width, height, num_cells):
        for i in range(10):
            self.generate(width, height, num_cells)

if __name__ == '__main__':
    unittest.TestCase()