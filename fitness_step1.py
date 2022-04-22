from util import Util

from options import Options
class Fitness:
    def __init__(self, floor, width, height, numCell):  # floor:[(x,y),...]
        self.floor = floor
        self.numCell = numCell
        self.width = width
        self.height = height
        # self.options = options # todo : move to ind function getIniConfig
        adjGraph = self.buildUndirectedGraph()

        self.graph = adjGraph
        Util.printAdjGraph(adjGraph)
        self.sideCells = self.calc_side_cells()
        self.Perimeter = self.boundary_length()
        self.SouthSideRatio = self.south_view_ratio()
        self.fulfill_distance()

    def __str__(self):
        return 'Fitness: Perimeter:{0}, SouthSideRatio:{1}, AspectRatio:{2}\n'.format(self.Perimeter, self.SouthSideRatio, self.aspect_ratio())

    def config_options(self, key):  # todo: duplicate move options
        config = Options()
        for section in config.get_sections():
            if config._options[section].get(key):
                vals = config.get_value(section, key).split(',')
                if section == 'Mass' or section == 'GAParams':  # only allow float or integer value in these section
                    return Util.tonumber(vals[0])
                else:
                    return vals


    def golden_ratio(self):
        return self.aspect_ratio() #todo: implement

    def boundary_length(self):  # 외피 사이즈
        # This is the Fitness Function
        cc = self.connected_component()
        Util.printCC(cc)

        insideWall = 0;
        for cell in self.graph:
            insideWall += len(self.graph[cell])
        return self.numCell * 4 - insideWall

    # 각 동서남북 방향에 면한 사이드 셀 갯수
    def calc_side_cells(self):
        southSides = []
        northSides = []
        eastSides = []
        westSides = []
        # todo: 이렇게 하면 안된다. 이건 모든 셀애 대해 각 방향별로 그루핑해서 동서남북별로 추가할 뿐이다.
        #  맨 끝 셀을 알려면 max 값을 구해야 한다.
        for cell in self.graph:
            adj = self.graph[cell]
            southSides.extend([south for south in adj if cell.y < south.y]) # adj 셀이 현재 셀 보다 밑에 있는 거 추가
            northSides.extend([north for north in adj if cell.y > north.y])
            westSides.extend([west for west in adj if cell.x > west.x])
            eastSides.extend([east for east in adj if cell.x < east.x])

        sz = len(self.graph)  # total number of cells size
        totSouth = sz - len(southSides)
        totEast = sz - len(eastSides)
        totWest = sz - len(westSides)
        totNorth = sz - len(northSides)
        return{
            'southSides':southSides,
            'northSides':northSides,
            'eastSides':eastSides,
            'westSides':westSides,
            'totSouth' : totSouth,
            'totEast' : totEast,
            'totWest' : totWest,
            'totNorth' : totNorth
        }
        # return [southSides, northSides, eastSides, westSides]
    def updateSideLength(self, sides):
        self.sideCells['totSouth'] = sides.get('south')
        self.sideCells['totNorth'] = sides.get('north')
        self.sideCells['totWest'] = sides.get('west')
        self.sideCells['totEast'] = sides.get('east')


    # 모든 이웃을 다 조사해서 south 이웃이 없으면 현재 셀이 south이므로 south 하나 증가
    def aspect_ratio(self):
        return (self.sideCells['totNorth']+self.sideCells['totNorth']) / (self.sideCells['totWest']+self.sideCells['totEast'])

    def south_view_ratio(self):
        # self.sideCells()
        # totEast, totWest, totSouth, totNorth = self.length_side_cells() # todo: merged with sideCells dic
        # aspectRatio = totSouth / totEast
        # todo => direct access
        totOpenEast = self.sideCells['totEast']
        totOpenWest = self.sideCells['totWest']
        totOpenSouth = self.sideCells['totSouth']
        totOpenNorth = self.sideCells['totNorth']
        print('totOpenEast', totOpenEast)
        # 담장이 있을 경우를 고려한다. 예를 들어 south에 경계담장이 있을 경우, 남쪽이 경계면과 닿는 면은 남쪽면에서 제외한다.
        # todo: option 처리를 어떻게 할 것인가를 고민한다.
        # todo: self.options has list of walls
        # todo: if all of them are there [south, east, west, north]
        print('totSouth', totOpenSouth)
        walls = self.config_options('wall_list')
        print('walls:', walls) # todo: move to wherever needed the walls list
        for cell in self.graph:
            print('cell.y: ', cell.y, 'self.height:', self.height)
            # totSouth -= 1 if cell.y >= self.height - 1 else 0
            totOpenSouth -= 1 if 'south' in walls and cell.y >= self.height -1 else 0
            totOpenNorth -= 1 if 'north' in walls and cell.y <= 0 else 0
            totOpenWest -= 1 if 'west' in walls and  cell.x <= 0 else 0
            totOpenEast -= 1 if 'east' in walls and cell.x >= self.width - 1 else 0
        print('Open Space: South, North, West, East:', totOpenSouth, totOpenNorth, totOpenWest, totOpenEast, self.aspect_ratio())

        return totOpenSouth / (totOpenEast + totOpenWest + totOpenNorth)

    def fulfill_distance(self):
        # mass = self.options['Mass']
        # fit = self.options['Fitness']
        adjacentLand = self.config_options('adjacent_land')
        roadSides = self.config_options('road_side')
        adjDistance = self.config_options('adjacent_distance')
        roadDistance = self.config_options('road_distance')
        print('adjacentLand:',adjacentLand, 'adj distance', adjDistance, 'roadSides:', roadSides, 'roadDistance:', roadDistance)

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
