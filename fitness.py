from util import Util
from grid import Grid

from options import Options
import math

import constant
class Fitness:
    def __init__(self, grid, width, height, numCell):  # floor:[(x,y),...]
        attrs = {}
        fits = {}
        config = Options()
        self.config_options = lambda ky: config.config_options(ky)
        self.grid= grid
        self.graph = grid.buildUndirectedGraph()
        attrs['number of cells'] = numCell
        attrs['max cell width'] = width
        attrs['max cell height'] = height
        attrs['real max width'] = width *  config.config_options('cell_length')
        attrs['real max height'] = height *  config.config_options('cell_length')

        # Util.printAdjGraph(self.graph) #todo: for Debug

        self._attrs = attrs
        attrs['boundary_walls'], attrs['area'] , attrs['perimeter'], fits['pa_ratio']  = self.pa_ratio()
        fits['symmetry'] = self.get_symmetry()
        self.edges, aspect_ratio, south_side = self.side_cells()
        fits['south_side'] = south_side
        optimal_ratio = self.config_options('optimal_ratio')[0] # because it returns list
        optimal_ratio_value = constant.RATIOS[optimal_ratio]
        fits['aspect_ratio'] = aspect_ratio
        fits['optimal_ratio'] = aspect_ratio / optimal_ratio_value
        fits['solar_hour'] = self.get_daylight_hour()
        self._fits = fits
        sqmt = f'm\u00b2'
        postfix = ['', '', '', 'm','m', 'm', sqmt, 'm']
        Util.display_str_dict(attrs, 'Floor Property', postfix)
        Util.display_str_dict(fits, 'Fitness')

        # self.symmetry = self.get_symmetry()
        # self.fratio = min(aspect_ratio, self.golden_ratio) / max(aspect_ratio, self.golden_ratio)

    def __str__(self): #get_fitness 에서 print(fitness)를 지워서 이거 필요없음
        fits = self._fits
        config = lambda key : self.config_options(key)
        strFitness = f'Fitness: \n\
         1. Area to length of outer wall = {fits["pa_ratio"]:.4f} \n\
         2. Optimal Ratio({config("optimal_ratio")[0]}) = {fits["optimal_ratio"]:.4f} (Aspect ratio: {fits.get("aspect_ratio")})\n\
         3. Symmetry: (Vertical: {fits["symmetry"][0]:.2f}, Horizontal:{fits["symmetry"][1]:.2f})\n\
         4. South View Ratio: {fits["south_side"]:.4f}\n\
         5. Solar Hour: {fits["solar_hour"]}hours'
        return strFitness

    def pa_ratio(self):
        wall_length = self.config_options('cell_length')
        boundary_walls =  self.boundary_length()
        perimeter = boundary_walls * wall_length
        numCell = self._attrs['number of cells']
        area = numCell * ( wall_length**2 )
        return boundary_walls, area, perimeter, 16*area / perimeter**2

    def south_gap(self):
        # for max
        cell_length = self.config_options('cell_length')
        real_vertical_length = self._attrs['real max height']
        max_south_index = max(pos.y for pos in self.edges['south'])
        south_distance = real_vertical_length - ((max_south_index+1) * cell_length)
        return south_distance

    def get_daylight_hour(self):
        h = self.config_options('height_diff_south')
        d = self.config_options('adjacent_distance_south')
        d = d + self.south_gap()
        altitudes = []
        solar_data = {}
        with open('solar_elevation_hourly.txt', 'r') as fp:
            for line in fp:
                altitude = [float(x) for x in line.split('\t') if x.strip()]
                solar_data[int(altitude[0])] = altitude[1]
                altitudes.append(altitude[1])
        solar_hour = sum(1 for altitude in altitudes
                            if altitude > 0
                            and float(h)/math.tan(math.radians(altitude)) < d)
        return solar_hour

    # todo: not yet used. get polygon coordinates
    # todo: to get polygon coordinates for num_vertices and etc
    def get_polygon_vertice(self):
        tupPos = [tuple((p.x, p.y)) for p in self.grid.poses]
        return Util.trace(tupPos)

    def bound_box(self):
        newpos = Util.move_topleft((self.grid.poses))
        #print(newpos) #todo:recover
        print(Util.bounding_box(newpos)) # todo recover

    def get_symmetry(self):
        newpos = Util.move_topleft(self.grid.poses)
        bb = Util.bounding_box(newpos)
        width = bb[1]+1; height = bb[3]+1
        newgrid = Grid(newpos, width, height)

        horz_diff = 0; vertical_diff = 0;
        i = 0; k = height - 1

        # Horizontal Symmetry
        while(i < width // 2): #floor division operator
            for j in range(width):
                if(newgrid.grid[i][j] != newgrid.grid[k][j]):
                    horz_diff +=1
            i += 1; k -= 1

        # Vertical Symmetry
        i = 0; k = width - 1
        while (i < height // 2):
            # Checking each cell of a row.
            for j in range(height):
                if (newgrid.grid[j][i] != newgrid.grid[j][k]):
                    vertical_diff += 1
            k -= 1; i += 1

        vertical_symm = 1 - vertical_diff / len(self.grid.poses)
        horz_symm = 1 - horz_diff / len(self.grid.poses)

        return  vertical_symm, horz_symm,

    def boundary_length(self):  # 외피 사이즈
        # This is the Fitness Function
        cc = self.grid.connected_component()
        # Util.printCC(cc) # todo for DEBUG

        insideWall = 0;
        for cell in self.graph:
            insideWall += len(self.graph[cell])
        return self._attrs['number of cells'] * 4 - insideWall


    def side_cells(self):
        rows = self.grid.grouped_by_row()
        cols = self.grid.grouped_by_col()
        edges = {}
        edges['east'] = [max(row, key=lambda pos: pos.x) for row in rows]
        edges['west'] = [min(row, key=lambda pos: pos.x) for row in rows]
        edges['south'] = [max(col, key=lambda pos: pos.y) for col in cols]
        edges['north'] = [min(col, key=lambda pos: pos.y) for col in cols]
        aspect_ratio = len(edges['south']) / len(edges['east'])
        list_of_edges = [edges[i] for i in edges]
        set_edges = set([cell for cells in list_of_edges  for cell in cells]) #flattened=>set
        south_side_ratio =  len(edges['south'] ) / len(set_edges)
        return edges, aspect_ratio, south_side_ratio
