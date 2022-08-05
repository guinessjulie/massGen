from util import Util, Pos
from collections import Counter
class LandGrid:
    def __init__(self, poses, width, height):
        self.grid = [['.']*width for _ in range(height)]
        self.poses = poses
        self.width = width
        self.height = height
        for p in poses:
            self.grid[p.y][p.x] = 'X'

    def init_grid(self):
        self.grid = [['.']* self.width for _ in range(self.height)]

    def __str__(self):
        return '\n'.join(' '.join(row) for row in self.grid)

    def __eq__(self, other):
        return Counter(self) == Counter(other)

    def adjacency(self, loc):
        return Util.adjacent_four_way(loc, self.width, self.height)

    def adjacency8(self, loc):
        return Util.adjacent_eight_way(loc, self.width, self.height)

    def display_adjacent(self, genes, pos, candid):
        self.update_positions(genes)
        for c in candid:
            self.grid[c.y][c.x] = 'A'
        self.grid[pos.y][pos.x] = 'N'
        print('\n'.join(' '.join(row) for row in self.grid))

    def display_genes(self, genes):
        self.grid = [['.'] * self.width for _ in range(self.height)]
        print(self.poses)
        for p in genes:
            self.grid[p.y][p.x] = 'X'
        print('\n'.join(' '.join(row) for row in self.grid))

    def display_poses(self):
        print('\n'.join(' '.join(row) for row in self.grid))

    def update_positions(self ,poses):
        self.init_grid()
        self.poses = poses
        for p in self.poses:
            self.grid[p.y][p.x] = 'X'

    def sorted_by_row(self):
        return sorted(self.poses, key=lambda item: (item.y, item.x))

    def sorted_by_col(self):
        return sorted(self.poses, key=lambda item:(item.x, item.y))

    def grouped_by_row(self):
        yset = set(map(lambda pos: pos.y, self.poses))
        return [[pos for pos in self.poses if pos.y==y] for y in yset]

    def grouped_by_col(self):
        xset = set(map(lambda pos:pos.x, self.poses))
        return[ [pos for pos in self.poses if pos.x ==x ] for x in xset]

    def num_connected_component(self):
        cc = self.connected_component()
        return len(cc)

    def buildUndirectedGraph(self):
        adjGraph = {}
        visited = set()
        for idx, curCell in enumerate(self.poses):
            visited.add(curCell)
            adjs = Util.adjacent_four_way(curCell, self.width, self.height)
            #         print('curCell', *curCell)
            #         print('adjs', *adjs)
            child = [adj for adj in adjs if adj in self.poses]
            adjGraph[curCell] = child

        return adjGraph

    def connected_component(self):
        visited = {}
        cc = []
        graph = self.buildUndirectedGraph()
        for node in graph:
            visited[node] = False
        for nodeId, node in enumerate(graph):
            if visited[node] == False:
                temp = []  # dsf algorithm
                cc.append(Util.dsf_util(visited, graph, node, temp))
        return cc

