from util import Util
import util

class Fitness:
    def __init__(self, floor, width, height, numCell):  # floor:[(x,y),...]
        self.floor = floor
        self.numCell = numCell
        self.width = width
        self.height = height
        adjGraph = self.buildUndirectedGraph()

        self.graph = adjGraph
        Util.printAdjGraph(adjGraph)
        self.Perimeter = self.boundary_length()
        self.SouthSideRatio = self.south_view_ratio()

    def __str__(self):
        return 'Fitness Perimeter:{0}, SouthSide:{1}'.format(self.Perimeter, self.SouthSideRatio)

    def golden_ratio(self):
        pass

    def boundary_length(self):  # 외피 사이즈
        # This is the Fitness Function
        cc = self.connected_component()
        Util.printCC(cc)

        insideWall = 0;
        for cell in self.graph:
            insideWall += len(self.graph[cell])
        return self.numCell * 4 - insideWall

    # 모든 이웃을 다 조사해서 south 이웃이 없으면 현재 셀이 south이므로 south 하나 증가
    def south_view_ratio(self):
        southSides = []
        northSides = []
        eastSides = []
        westSides = []
        for cell in self.graph:
            adj = self.graph[cell]
            southSides.extend([south for south in adj if cell.y < south.y])
            northSides.extend([north for north in adj if cell.y > north.y])
            westSides.extend([west for west in adj if cell.x > west.x])
            eastSides.extend([east for east in adj if cell.x < east.x])

        sz = len(self.graph)  # cell size
        totSouth = sz - len(southSides)
        totEast = sz - len(eastSides)
        totWest = sz - len(westSides)
        totNorth = sz - len(northSides)

        aspectRatio = (totSouth) / (totEast)

        # 담장이 있을 경우를 고려한다. 예를 들어 south에 경계담장이 있을 경우, 남쪽이 경계면과 닿는 면은 남쪽면에서 제외한다.
        # todo: option 처리를 어떻게 할 것인가를 고민한다.
        print('totSouth', totSouth)
        for cell in self.graph:
            print('cell.y: ', cell.y, 'self.height:', self.height)
            totSouth -= 1 if cell.y >= self.height - 1 else 0
            totNorth -= 1 if cell.y <= 0 else 0
            totWest -= 1 if cell.x <= 0 else 0
            totEast -= 1 if cell.x >= self.width - 1 else 0
        print('totSouth, North, West, East:', totSouth, totNorth, totWest, totEast)

        return (totSouth / (totEast + totWest + totNorth)), aspectRatio

    def buildUndirectedGraph(self):
        adjGraph = {}
        visited = set()
        for idx, curCell in enumerate(self.floor):
            visited.add(curCell)
            adjs = Util.adjacent_four_way(curCell, self.width, self.height)
            #         print('curCell', *curCell)
            #         print('adjs', *adjs)
            child = [adj for adj in adjs if adj in self.floor]
            adjGraph[curCell] = child

        return adjGraph

    def connected_component(self):
        visited = {}
        cc = []
        for node in self.graph:
            visited[node] = False
        for nodeId, node in enumerate(self.graph):
            if visited[node] == False:
                temp = []  # dsf algorithm
                cc.append(Util.dsf_util(visited, self.graph, node, temp))
        return cc
