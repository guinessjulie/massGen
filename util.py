import configparser
import datetime
import re
import math
from collections.abc import Iterable
import matplotlib.pyplot as plt
import csv
import random
from statistics import mean
import matplotlib.gridspec as gridspec
class Pos:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return f"({self.x}, {self.y})"
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    def __lt__(self, other):
        return self.x < other.x and self.y < other.y
    def __repr__(self):
        return str(self)
    def __hash__(self):
        return self.x*1000+self.y
    def to_list(self):
        return list((self.x, self.y))

    def to_tuples(self):
        return (self.x, self.y)




def loadIniFile(inifile):
    options = configparser.ConfigParser()
    options.read(inifile)
    return options

class Util:
    @staticmethod
    def printAdjGraph(graph):
        print('Undirected Graph in Util.printAdjGraph\n')
        for cell in graph:
            print(cell, '=>', *[adj for adj in graph[cell]])

    @staticmethod
    def printCC(comps):
        for c in comps:
            print('connected compoonent:', *c)
            for adj in c:
                print('adj:', adj)



    def display_str_dict(attrs, title='', postfix=None, format=':.2f'):
        postfix= [''for _ in range(len(attrs))] if postfix == None else postfix
        new_line ='\n'
        str_result = f'{new_line}[{title}]{new_line}'
        i = 0
        for ky in attrs:
            if type(attrs[ky]) == float:
                str_result += f'{ky}: {attrs[ky]:.4f}{postfix[i]}{new_line}'
            elif isinstance(attrs[ky], Iterable):
                all_float = True
                for elm in attrs[ky]:
                    all_float &= type(elm) == float
                if all_float:
                    str_result += f'{ky}: ('
                    for elem in attrs[ky]:
                        str_result += f'{elem:.4f}{postfix[i]}, '
                    str_result = str_result[:-2]
                    str_result += f'){new_line}'
                else:
                    str_result += f'{ky}: {elm}{postfix[i]}{new_line}'
            else:
                str_result += f'{ky}: {attrs[ky]}{postfix[i]}{new_line}'
            i += 1
        print(str_result)

    def get_str_dict(attrs, title='', postfix=None, format=':.2f'):
        postfix= [''for _ in range(len(attrs))] if postfix == None else postfix
        new_line ='\n'
        str_result = f'{new_line}[{title}]{new_line}'
        i = 0
        for ky in attrs:
            if type(attrs[ky]) == float:
                str_result += f'{ky}: {attrs[ky]:.4f}{postfix[i]}{new_line}'
            elif type(attrs[ky]) == str:
                str_result += f'{ky}: {attrs[ky]}{new_line}'
            elif isinstance(attrs[ky], Iterable):
                all_float = True
                for elm in attrs[ky]:
                    all_float &= type(elm) == float
                if all_float:
                    str_result += f'{ky}: ('
                    for elem in attrs[ky]:
                        str_result += f'{elem:.4f}{postfix[i]}, '
                    str_result = str_result[:-2]
                    str_result += f'){new_line}'
                else:
                    str_result += f'{ky}: {elm}{postfix[i]}{new_line}'
            else:
                str_result += f'{ky}: {attrs[ky]}{postfix[i]}{new_line}'
            i += 1
        return str_result

    def plotGrid(grid):
        print('grid',grid)
        # for plot
        mat=[[0]*grid.width for _ in range(grid.height)]
        for cell in grid.poses:
            mat[cell.y][cell.x] = 1
        fig, ax = plt.subplots()
        plt.axis('off')
        ax.set_aspect(1)
        pltMass = ax.imshow(mat, cmap='GnBu', interpolation='nearest')
        plt.savefig('test.png')
        print(plt.axis())
        plt.show()

    def plotGridOnlyRow(pops, plotrows, fit_txt = ''):
        plotcols = math.ceil(len(pops) / plotrows)
        fig, axs = plt.subplots(plotcols,plotrows)
        plt.subplots_adjust(hspace=0.1, wspace=0.1)


        i = 0
        for land in pops:
            mat = [[0] * land.width for _ in range(land.height)]

            for cell in land.poses:
                # mat[cell.y][cell.x] = 1
                y = land.height - 1 - cell.y  # to reverse y value to transform screen coordinate to plot coordinate
                mat[y][cell.x] = 1
            plotcol, plotrow = divmod(i, plotrows)
            axs[plotcol, plotrow].pcolormesh(mat, cmap='gray_r', alpha=0.7, edgecolor='silver', linewidth=0)
            axs[plotcol, plotrow].set_xticks([])
            axs[plotcol, plotrow].set_yticks([])
            axs[plotcol, plotrow].text(0,20, fit_txt)
            axs[plotcol, plotrow].set_aspect(1.0)
            axs[plotcol, plotrow].set_title(str(i))
            i += 1


        filename = './results/conditional_'+ datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[4:]
        plt.savefig(filename)
        plt.show()

    def plotGridOnly(pops):
        plotrows = 5
        plotcols = math.ceil(len(pops) / plotrows)
        fig, axs = plt.subplots(plotcols,plotrows)
        plt.subplots_adjust(hspace=0.1, wspace=0.1)


        i = 0
        for land in pops:
            mat = [[0] * land.width for _ in range(land.height)]

            for cell in land.poses:
                # mat[cell.y][cell.x] = 1
                y = land.height - 1 - cell.y  # to reverse y value to transform screen coordinate to plot coordinate
                mat[y][cell.x] = 1
            plotcol, plotrow = divmod(i, plotrows)
            axs[plotcol, plotrow].pcolormesh(mat, cmap='gray_r', alpha=0.7, edgecolor='silver', linewidth=0)
            axs[plotcol, plotrow].set_xticks([])
            axs[plotcol, plotrow].set_yticks([])
            axs[plotcol, plotrow].set_aspect(1.0)
            i += 1


        filename = './results/conditional_'+ datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[4:]
        plt.savefig(filename)
        plt.show()

    def plotColorMesh(land, fitness,txt_setting):
        sqmt = f'm\u00b2'
        postfix_plan = ['', '', '', 'm', 'm', 'm', sqmt, 'm', '','','']

        txt_plan = Util.get_str_dict(fitness._attrs, 'Mass Plan',postfix_plan)
        txt_fitness = Util.get_str_dict(fitness._fits, 'Fitness' )
        mat=[[0]*land.width for _ in range(land.height)]

        for cell in land.poses:
            # mat[cell.y][cell.x] = 1
            y = land.height -1 - cell.y # to reverse y value to transform screen coordinate to plot coordinate
            mat[y][cell.x] = 1
        fig = plt.figure(constrained_layout = True)
        gs = fig.add_gridspec(6,3)
        gs.update(left=0.1, right=0.3, top=0.965, bottom=0.03, wspace=0.3, hspace=0.09)
        ax1 = fig.add_subplot(gs[:5,:2])

        # ax1.pcolormesh(mat,  cmap='gray_r', alpha=0.7, edgecolor='silver', linewidth = 0.01)
        ax1.pcolormesh(mat,  cmap='gray_r', alpha=0.7, edgecolor='silver', linewidth = 0)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_aspect(1.0)

        Util.set_gridplot_text(fig, gs[:5,2:], txt_setting) #todo reorganizing and test
        # Util.set_gridplot_text(fig,gs[2,0], txt_plan)
        # Util.set_gridplot_text(fig, gs[2,2],txt_fitness)
        Util.set_gridplot_text(fig,gs[5,:2], txt_plan)
        Util.set_gridplot_text(fig, gs[5,2:],txt_fitness)

        filename = './results/plot_'+ datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[4:]
        plt.savefig(filename)
        plt.show()

    @staticmethod
    def set_gridplot_text(fig, gsloc, attrs):
        ax=fig.add_subplot(gsloc)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.05, 0.05, attrs,
                # horizontalalignment='left',
                # verticalalignment='top',
                )
        ax.axis(False)

    @staticmethod
    def select_random_position(width, height):
        init_row = random.randint(0, height - 1)
        init_col = random.randint(0, width - 1)
        genes = [ Pos(init_col, init_row)]
        print(genes)
        return genes

    @staticmethod
    def saveCsv(filename, fits, generation=0, fit_value=0, fitname='fitness'):
        keys = fits[0].keys()
        with open(filename, 'a', newline='') as fp:
                dict_writer = csv.DictWriter(fp, keys)
                dict_writer.writeheader()
                dict_writer.writerows(fits)

        with open(filename, 'a', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['fitname', fit_value])

    @staticmethod
    def dsf_util(visited, graph, node, components):
        visited[node] = True
        components.append(node)
        for adj in graph[node]:
            if visited[adj] is False:
                components = Util.dsf_util(visited, graph, adj, components)
        return components


    # 인접셀 집합을 구한다. 대각선 방향 포함
    @staticmethod
    def adjacent_eight_way(loc, width, height):
        return [i for i in set(Pos(x + loc.x, y+loc.y)
                               for x in [-1, 1, 0] if 0<= x+loc.x < width
                               for y in [-1, 1, 0]  if 0<= y+loc.y < height
                               and loc != Pos(x+loc.x, y+loc.y)) ]


    # 동서남북 네 방향의 셀만 구한다. 동서남북 네 방향만
    @staticmethod
    def adjacent_four_way(loc, width, height):
        return [i for i in set(Pos(x+loc.x, y+loc.y)
                               for x in [ -1, 1, 0] if 0<=x+loc.x < width
                               for y in [-1, 1, 0] if 0 <= y+loc.y < height
                               and loc != Pos(x+loc.x, y+loc.y)
                               and abs(x) != abs(y)
                              )]
    @staticmethod
    def bound_adjacent(loc_list, width, height):
        bound_adjs = []
        for x in loc_list:
            for adj in Util.adjacent_four_way(x, width, height):
                if adj not in loc_list:
                    bound_adjs.append(adj)
        return list(set(bound_adjs))



    @staticmethod
    def all_cell(self, width, height):
        return [ i for i in list(Pos(x, y) for x in range(width) for y in range(height))]

    @staticmethod
    def move_topleft(positions):
        miny = min(positions, key=lambda pos:pos.y).y
        minx = min(positions, key=lambda pos: pos.x).x
        return [Pos(pos.x-minx, pos.y-miny) for pos in positions]


    @staticmethod
    def bounding_box(positions):
        # minx = min(positions, key=lambda pos:pos.x)
        # maxx = max(positions, key= lambda pos: pos.x)
        # miny = min(positions, key=lambda pos:pos.y)
        # maxy = max(positions, key= lambda pos: pos.y)
        # corners =  [Pos(minx.x, miny.y), Pos(maxx.x, miny.y), Pos(maxx.x, maxx.y), Pos(minx.x, maxy.y)]
        return [min(positions, key=lambda pos:pos.x).x,
                max(positions, key=lambda pos:pos.x).x,
                min(positions, key=lambda pos:pos.y).y,
                max(positions, key=lambda pos:pos.y).y,
                ]
        # return [min(corners, key=lambda pos:pos.x).x, max(corners, key=lambda pos:pos.x).x, min(corners)] #가로방향의 최소값을 가진 셀

    # 용적률에 해당하는 셀의 갯수를 구한다.
    @staticmethod
    def get_num_cells(width, height, floorAreaRatio):
        return int(width * height * floorAreaRatio)



    # area of unit cell
    @staticmethod
    def unit_area(unit_len):
        return unit_len * unit_len

    @staticmethod
    def total_area_cells(num_cells, unit_len):
        return num_cells * Util.unit_area(unit_len)

    @staticmethod
    def isfloat(s):
        match = re.match(r'^=?\d+(?:\.\d+)$', s)
        return match is not None

    @staticmethod
    def tofloat(s):
        return float(s) if Util.isfloat(s) else s

    @staticmethod
    def tonumber(s):
        return int(s) if s.isnumeric() else Util.tofloat(s)

    @staticmethod
    def normalize_0to1(raw):
        if max(raw) == min(raw):
            return [1 for i in range(len(raw))]
        return [(float(i)-min(raw))/(max(raw)-min(raw)) for i in raw]

    def neighbors(coord):
        for dir in (1, 0):
            for delta in (-1, 1):
                yield (coord[0] + dir * delta, coord[1] + (1 - dir) * delta)
    @staticmethod
    def get_angle(dir1, dir2):
        angle = math.acos(dir1[0] * dir2[0] + dir1[1] * dir2[1])
        cross = dir1[1] * dir2[0] - dir1[0] * dir2[1]
        if cross > 0:
            angle = -angle
        return angle

    @staticmethod
    def trace(p):
        if len(p) <= 1:
            return p

        # start at top left-most point
        pt0 = min(p, key=lambda t: (t[1], t[0]))
        dir1 = (0, -1)
        pt = pt0
        outline = [pt0]
        while True:
            pt_next = None
            angle_next = 10  # dummy value to be replaced
            dir_next = None

            # find leftmost neighbor
            for n in Util.neighbors(pt): #맨 top-left부터 시직 다음 인접셀을 찾아서
                if n in p: #인접셀이 셀집합에 포함되어 있으면,
                    dir2 = (n[0] - pt[0], n[1] - pt[1]) #인접셀 (4,1), 현재셀 pt가 (3,0)인 경우 dir2는 오른쪽으로 1(1,0) #n=(3,2)인 경우 dir2=(0,1)아래 방향
                    angle = Util.get_angle(dir1, dir2) #dir1은 위쪽(0,-1)  dir2=(1,0) #dir2=(0,1)인 경우 3.14159
                    if angle < angle_next: # angle = 1.5707 #두번쨰로 angle이 3.14169일때 angle_next가 앞의 값 1.5707이므로 만족못시킴
                        pt_next = n # 인접셀을 다음번에 검사
                        angle_next = angle #현재 angle 저장
                        dir_next = dir2 #현재 dir2를 다음 번에
            if angle_next != 0:
                outline.append(pt_next) #(3,1)에서 시작해서 인접셀 (4,1)을 구해서 더했다.
            else:
                # previous point was unnecessary
                outline[-1] = pt_next
            if pt_next == pt0:
                return outline[:-1]
            pt = pt_next #막 추가한 네이버를 pt로 대상으로 하고
            dir1 = dir_next #다음번 방향을 dir1 로 해서

class Stat:
    @staticmethod
    def list_mean(l):
        return mean(l)