from util import Util
from grid import Grid

from options import Options
import math

import constant
class Fitness:
    def __init__(self, grid, width, height, numCell):  # floor:[(x,y),...]
        config = Options()
        self._width, self._height = width, height,
        self.config_options = lambda ky: config.config_options(ky)
        self._grid= grid
        self.graph = grid.buildUndirectedGraph()
        self._attrs, self._fits, self.edges = self.build_attrs(grid, numCell)
        Util.display_str_dict(self._fits, 'Fitness')

#먼저 init을 좀 정리하자.attr fit를 딴데서
    def build_attrs(self, gridpos, numCell):
        attrs = {}
        fits = {}
        config = Options()
        grid_by_col = gridpos.sorted_by_col()
        grid_by_row = gridpos.sorted_by_row()

        attrs['number of cells'] = numCell
        max_width = grid_by_col[numCell - 1].x - grid_by_col[0].x + 1
        max_height = grid_by_row[numCell - 1].y - grid_by_row[0].y + 1
        attrs['max width'] = max_width
        attrs['max height'] = max_height
        attrs['real max width'] = max_width *  config.config_options('cell_length')
        real_vertical_length = max_height *  config.config_options('cell_length')
        attrs['real max height'] = real_vertical_length
        attrs['boundary_walls'], attrs['area'], attrs['perimeter'], fits['pa_ratio'], attrs['real faratio'] = self.pa_ratio(numCell)

        fits['optimal fa_ratio'] = attrs['real faratio'] / self.config_options('required_faratio')
        fits['symmetry'] = self.get_symmetry()
        edges, aspect_ratio, south_side = self.side_cells()
        fits['south_side'] = south_side
        optimal_ratio = self.config_options('optimal_ratio')[0] # because it returns list
        optimal_ratio_value = constant.RATIOS[optimal_ratio]
        fits['aspect_ratio'] = aspect_ratio
        fits['optimal_ratio'] = aspect_ratio / optimal_ratio_value
        fits['daylight hour'] = self.get_daylight_hour(edges, real_vertical_length) # todo edges not set yet
        buildingends, lotends, fits['fulfill building line'] = self.fulfill_building_line(grid_by_row, grid_by_col)
        fits['setbacks'] = 'Failed' if fits['fulfill building line'] == 'Failed' else self.check_setbakcs(buildingends, lotends, grid_by_row, grid_by_col)
        return attrs, fits, edges
        # Util.printAdjGraph(self.graph) #todo: for Debug

# 따로 떼자 너무 복잡
    def fulfill_building_line(self, rows, cols):
        road_side= self.config_options('road_side')[0] #todo f
        road_width = self.config_options('road_width')
        cell_length = self.config_options('cell_length')
        setback_requirement = self.config_options('setback_requirement')
        setback_for_road_width = 0
        if(road_width < 4):
            setback_for_road_width = (4 - road_width) / 2

        plan_setbacks = {}
        required_setbacks = {}

        required_setbacks[road_side] = setback_for_road_width + setback_requirement
        buildingends = {}
        lotends = {}
        buildingends['south'] = lambda rows: rows[len(rows)-1].y
        buildingends['north'] = lambda rows: rows[0].y
        buildingends['east'] = lambda cols: cols[len(cols)-1].x
        buildingends['west'] = lambda cols:cols[0].x
        lotends['east'] = self._width - 1
        lotends['south'] = self._height - 1
        lotends['west'] = 0
        lotends['north'] = 0
        plan_setbacks[road_side] = abs(buildingends[road_side](rows) - lotends[road_side]) * cell_length #todo doing this now go back here

        # first test building_line
        if required_setbacks[road_side] < plan_setbacks[road_side]:
            return buildingends, lotends, 'Succeeded'
        else:
            return buildingends, lotends, 'Failed'



    def check_setbakcs(self, buildingends, lotends, rows, cols):
        cell_length = self.config_options('cell_length')
        setback_requirement = self.config_options('setback_requirement')
        plan_setbacks = {}
        required_setbacks = {}

        if abs(buildingends['south'](rows) - lotends['south'])*2 > setback_requirement  and \
           abs(buildingends['north'](rows) - lotends['north'])*2 > setback_requirement  and \
           abs(buildingends['east'](rows) - lotends['east'])*2 > setback_requirement and \
           abs(buildingends['west'](rows) - lotends['west'])*2 > setback_requirement:
            return 'Succeeded'
        else : return 'Failed'
        #
        # required_setbacks[road_side] = building_line_setback + setback_requirement
        # plan_setbacks[road_side] = (building_end_north - land_end_north ) * cell_length

        # 있는 경우에 Failed가 되었다면, 이미 전체가 다 Fail해서 더이상 조사하지 않고 리턴해버렸다.
        # None인데 아직 이 문장을 확인한다는 것은 있다는 거는 succeeded
        # if required_setbacks.get('south') is None: # road_side가 아닌 경우 대지내공지만 확인
        #     plan_setbacks['south'] ='Succeeded' \
        #         if (land_end_south - building_end_south) * 2 > setback_requirement \
        #         else 'Failed'
        # if required_setbacks.get('east') is None: #road_side가 아닌 경우 대지내공지만 확인
        #     plan_setbacks['east'] ='Succeeded' \
        #         if (land_end_east - building_end_east) * 2 > setback_requirement \
        #         else 'Failed'
        # if required_setbacks.get('north') is None: # road_side가 아닌 경우 대지내공지만 확인
        #     plan_setbacks['north'] ='Succeeded' \
        #         if (building_end_north - land_end_north ) * 2 > setback_requirement \
        #         else 'Failed'
        # if required_setbacks.get('west') is None: #road_side가 아닌 경우 대지내공지만 확인
        #     plan_setbacks['west'] ='Succeeded' \
        #         if (building_end_west - land_end_west) * 2 > setback_requirement \
        #         else 'Failed'
        #     return required_building_line_result, 'Failed'



        # second, test setback for all

        # return required_building_line_result, all

    def __str__(self): #get_fitness 에서 print(fitness)를 지워서 이거 필요없음
        fits = self._fits
        config = lambda key : self.config_options(key)
        strFitness = f'Fitness: \n\
         1. Area to length of outer wall = {fits["pa_ratio"]:.4f} \n\
         2. Optimal Ratio({config("optimal_ratio")[0]}) = {fits["optimal_ratio"]:.4f} (Aspect ratio: {fits.get("aspect_ratio")})\n\
         3. Symmetry: (Vertical: {fits["symmetry"][0]:.2f}, Horizontal:{fits["symmetry"][1]:.2f})\n\
         4. South View Ratio: {fits["south_side"]:.4f}\n\
         5. Solar Hour: {fits["daylight hour"]}hours'
        return strFitness

    def verify_setback_requirement(self):
        pass

    def pa_ratio(self, numCell):
        wall_length = self.config_options('cell_length')
        cell_area = wall_length ** 2 # area per cell
        boundary_walls =  self.boundary_length(numCell) #여기서
        perimeter = boundary_walls * wall_length
        # numCell = self._attrs['number of cells']
        floorarea = numCell * cell_area
        landarea = cell_area * self._width * self._height
        return boundary_walls, floorarea, perimeter, 16*floorarea / perimeter**2, floorarea/landarea

    #남향 방향으로 얼마나 빈 공간이 있느냐 하는 것
    def south_gap(self, edges, real_vertical_length):
        # for max
        cell_length = self.config_options('cell_length')
        # real_vertical_length = self._attrs['real max height'] # todo fixing bugs 04-26
        max_south_index = max(pos.y for pos in edges['south'])
        distance_south = (self._grid.height * cell_length) - ((max_south_index + 1) * cell_length) #남쪽 공간
        return distance_south

    def get_daylight_hour(self, edges, real_vertical_length):
        h = self.config_options('height_diff_south')
        d = self.config_options('adjacent_distance_south')
        d = d + self.south_gap(edges, real_vertical_length)
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
        tupPos = [tuple((p.x, p.y)) for p in self._grid.poses]
        return Util.trace(tupPos)

    def bound_box(self):
        newpos = Util.move_topleft((self._grid.poses))
        #print(newpos) #todo:recover
        print(Util.bounding_box(newpos)) # todo recover

    def get_symmetry(self):
        newpos = Util.move_topleft(self._grid.poses)
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

        vertical_symm = 1 - vertical_diff / len(self._grid.poses)
        horz_symm = 1 - horz_diff / len(self._grid.poses)

        return  vertical_symm, horz_symm,

    def boundary_length(self, numCell):  # 외피 사이즈
        # This is the Fitness Function
        cc = self._grid.connected_component()
        # Util.printCC(cc) # todo for DEBUG

        insideWall = 0;
        for cell in self.graph:
            insideWall += len(self.graph[cell])
        # return self._attrs['number of cells'] * 4 - insideWall
        return numCell * 4 - insideWall



    def side_cells(self):
        rows = self._grid.grouped_by_row()
        cols = self._grid.grouped_by_col()
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
