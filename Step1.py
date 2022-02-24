import random
import unittest
import measureUtil
from util import Util, Pos
from fitness import Fitness
from grid import Grid


class GenArchiPlan(unittest.TestCase):
    def test1(self):
        width = 6
        height = 4
        floorAreaRatio = 0.3
        num_cells = Util.get_num_cells(width, height, floorAreaRatio)
        pos = self.generate(width, height, num_cells)
        self.get_fitness(pos, width, height, num_cells)

        # self.generate_multi(width, height, num_cells)

    def get_fitness(self, pos, width, height, num_cells):
        fitness = Fitness(pos, width, height, num_cells)
        print(fitness)
        # perimeter = fitness.boundary_length()
        # southSide = fitness.south_view_ratio()
        #
        # cntSouth, cntNorth, cntWest, cntEast = fitness.south_view_ratio()
        # print('Fitness: ', perimeter, cntSouth,cntNorth, cntWest, cntEast)

    def generate(self , width, height, num_cells):
        pos = [Pos(int(width/2), int(height/2))]
        grid = Grid(pos, width, height)
        surr = grid.adjacents(pos[0])
        print(set(pos), num_cells)
        while len(set(pos)) < num_cells:
            pickIdx = random.randint(0, len(pos)-1)
            surr = grid.adjacents(pos[pickIdx])
            pick = random.sample(surr, 1) # todo: 여기서 기존에 선택했던 걸 선택하면 안된다
            pos = pos + pick
        # graph = UndirectedGraph(pos, width, height)

        print('Grid')
        print(Grid(pos, width, height), '\n')
        return pos

    def generate_multi(self, width, height, num_cells):
        for i in range(10):
            self.generate(width, height, num_cells)

if __name__ == '__main__':
    unittest.TestCase()