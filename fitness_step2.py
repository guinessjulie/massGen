from util import Util
from grid import Grid

from options import Options
class Fitness:
    def __init__(self, grid, width, height, numCell):  # floor:[(x,y),...]
        self.grid= grid
        self.floor = grid.poses
        self.numCell = numCell
        self.width = width
        self.height = height
        # self.options = options # todo : move to ind function getIniConfig
        self.graph = grid.buildUndirectedGraph()

        Util.printAdjGraph(self.graph)

        # self.sideCells = self.calc_side_cells()
        self.Perimeter = self.boundary_length()
        # self.SouthSideRatio = self.south_view_ratio() #todo 0318 to reprogram
        self.fulfill_distance()
        print('call bound_box()')
        self.symmetry()

    def __str__(self):
        edges, aspect_ratio = self.side_cells()
        return 'Fitness: Perimeter:{0}, Edges:{1}, AspectRatio:{2}\n'.format(self.Perimeter, edges, aspect_ratio)
        # return 'Fitness: Perimeter:{0}, AspectRatio:{1}\n'.format(self.Perimeter, self.aspect_ratio())

    def config_options(self, key):  # todo: duplicate move options
        config = Options()
        for section in config.get_sections():
            if config._options[section].get(key):
                vals = config.get_value(section, key).split(',')
                if section == 'Mass' or section == 'GAParams':  # only allow float or integer value in these section
                    return Util.tonumber(vals[0])
                else:
                    return vals

    # todo: not yet used. get polygon coordinates
    # todo: to get polygon coordinates for num_vertices and etc
    def get_polygon_vertice(self):
        tupPos = [tuple((p.x, p.y)) for p in self.floor]
        return Util.trace(tupPos)

    def bound_box(self):
        newpos = Util.move_topleft((self.floor))
        print(newpos)
        print(Util.bounding_box(newpos))

    def symmetry(self):
        newpos = Util.move_topleft(self.floor)
        bb = Util.bounding_box(newpos)
        width = bb[1]+1
        height = bb[3]+1
        newgrid = Grid(newpos, width, height)
        print(newgrid)
        horz_diff = 0
        vertical_diff = 0
        i = 0
        k = height - 1

        while(i < width // 2): #todo: redundant in middle rows, change to after testing i < math.floor(width/2)
            for j in range(width):
                print(i, j, k, j)
                print(newgrid.grid[i][j], newgrid.grid[k][j])
                if(newgrid.grid[i][j] != newgrid.grid[k][j]):
                    horz_diff +=1
            i += 1
            k -= 1

        print('vertical symmetry')
        i = 0
        k = height - 1
        while (i < width // 2):

            # Checking each cell of a row.
            for j in range(height):
                print(j, i,j, k)
                print(newgrid.grid[j][i], newgrid.grid[j][k])
                # check if every cell is identical
                if (newgrid.grid[j][i] != newgrid.grid[j][k]):
                    vertical_diff += 1
            k -= 1
            i += 1
        vertical_symm = 1 - vertical_diff / len(self.floor)
        horz_symm = 1 - horz_diff / len(self.floor)

        print('Horizontal Symmetry:', horz_symm, 'vertical Symmetry:', vertical_symm)
        # i=0
        # k=M-1
        # while(i<M //2):
        #     for

    def golden_ratio(self):
        return self.aspect_ratio() #todo: implement

    def boundary_length(self):  # 외피 사이즈
        # This is the Fitness Function
        cc = self.grid.connected_component()
        Util.printCC(cc)

        insideWall = 0;
        for cell in self.graph:
            insideWall += len(self.graph[cell])
        return self.numCell * 4 - insideWall


    def side_cells(self):
        rows = self.grid.grouped_by_row()
        cols = self.grid.grouped_by_col()
        edges = {}
        edges['east'] = [max(row, key=lambda pos: pos.x) for row in rows]
        edges['west'] = [min(row, key=lambda pos: pos.x) for row in rows]
        edges['south'] = [max(col, key=lambda pos: pos.y) for col in cols]
        edges['north'] = [min(col, key=lambda pos: pos.y) for col in cols]
        aspect_ratio = len(edges['east'])/len(edges['north'])
        return edges, aspect_ratio

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

    # def buildUndirectedGraph(self): #todo: moving to grid
    #     adjGraph = {}
    #     visited = set()
    #     for idx, curCell in enumerate(self.floor):
    #         visited.add(curCell)
    #         adjs = Util.adjacent_four_way(curCell, self.width, self.height)
    #         #         print('curCell', *curCell)
    #         #         print('adjs', *adjs)
    #         child = [adj for adj in adjs if adj in self.floor]
    #         adjGraph[curCell] = child
    #
    #     return adjGraph

    # def connected_component(self): #todo moving to grid
    #     visited = {}
    #     cc = []
    #     for node in self.graph:
    #         visited[node] = False
    #     for nodeId, node in enumerate(self.graph):
    #         if visited[node] == False:
    #             temp = []  # dsf algorithm
    #             cc.append(Util.dsf_util(visited, self.graph, node, temp))
    #     return cc
