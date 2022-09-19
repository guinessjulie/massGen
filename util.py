import configparser
import datetime
import re
import math
from collections.abc import Iterable
import matplotlib.pyplot as plt
import csv
import random
from statistics import mean
import os
import statutil
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
    def build_weights(probs, sample, div_sample):
        weights = []
        n = len(div_sample)
        for s in sample:
            if s in div_sample[0]:
                weights.append(probs[0])
            elif s in div_sample[1]:
                weights.append(probs[1])
        return weights

    @staticmethod
    def fits_to_probs(fits):
        uniques = list(set(fits))
        uniques_sum = sum(x for x in uniques)
        norm_uniques = [ x / uniques_sum for x in fits]
        norm_uniques = Util.normalize_0to1(norm_uniques)

        return norm_uniques


    @staticmethod
    def printCC(comps):
        for c in comps:
            print('connected compoonent:', *c)
            for adj in c:
                print('adj:', adj)

    @staticmethod
    def connected_component(loc_list, width, height):
        visited = {}
        cc = []
        graph = Util.buildUndirectedGraph(loc_list, width, height)
        for node in graph:
            visited[node] = False
        for nodeId, node in enumerate(graph):
            if visited[node] == False:
                temp = []  # dsf algorithm
                cc.append(Util.dsf_util(visited, graph, node, temp))
        return cc

    @staticmethod
    def buildUndirectedGraph(loc_list, width, height):
        adjGraph = {}
        visited = set()
        for idx, curCell in enumerate(loc_list):
            visited.add(curCell)
            adjs = Util.adjacent_four_way(curCell, width, height)
            child = [adj for adj in adjs if adj in loc_list]
            adjGraph[curCell] = child
        return adjGraph

    @staticmethod
    def grouped_by_row(positions):
        yset = set(map(lambda pos: pos.y, positions))
        return [[pos for pos in positions if pos.y==y] for y in yset]
    @staticmethod
    def grouped_by_col(positions):
        xset = set(map(lambda pos:pos.x, positions))
        return[ [pos for pos in positions if pos.x ==x ] for x in xset]

    @staticmethod
    def available_adjacency(genes, idx, width, height):
        all_adjs = Util.adjacent_four_way(genes[idx], width, height)
        return [x for x in all_adjs if x not in genes]

    @staticmethod
    def occupied_adjacency(icx, genes, width, height):
        all_adjs = Util.adjacent_four_way(icx,width,height)
        return [x for x in all_adjs if x in genes]



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

    @staticmethod
    def saveData(generation, new_pops, fit_option, fitnesses, fitness_filename, numfig = 50, targetpath='.\\results\\'):
        fig_title = 'Generation #' + str(generation)
        # todo: delete when paper picture is done
        # random_pop = random.sample(new_pops, 10)
        # Util.plotGridOnlyRow(random_pop, int(len(random_pop) / 2), fig_title)  # todo:  10 for pops= 100

        # todo: plot 50 picture per page

        random_pop = random.sample(new_pops, numfig) if len(new_pops) > numfig else new_pops
        # Util.plotGridOnlyRow(random_pop, int(len(random_pop) / 5), fig_title, omit_parents=False)  # todo:  10 for pops= 100
        Util.plotGridOnlyRow(random_pop, int(len(random_pop) / 5), fig_title, omit_parents=True, targetpath=targetpath, generation=generation)  # todo:  10 for pops= 100
        # Util.plotGridOnlyRow(new_pops, int(len(new_pops) / 5), fig_title)  # todo:  10 for pops= 100
        fitname = fit_option
        mean_fitness = mean(x.get(fit_option) for x in fitnesses)
        Util.saveFitnessCsv(fitness_filename, fitnesses, generation, mean_fitness, fitname)

    def plotGrid(grid, showTitle=False):
        print('grid',grid)
        # for plot
        mat=[[0]*grid.width for _ in range(grid.height)]
        for cell in grid.poses:
            mat[cell.y][cell.x] = 1
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.001, right=0.999, top=0.85, bottom=0)
        # fig.tight_layout()
        # plt.axis('off')
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 1.25
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        ax.set_aspect(1)
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('1')
        if showTitle:
            half = len(grid.poses) / 2
            title_rows = len(grid.poses) / half # max 8 개씩 출력
            if len(grid.poses) > 8:
                middle_pos = int(len(grid.poses) / title_rows) #2
                title = str(grid.poses[:middle_pos]) + '\n'
                title += str(grid.poses[middle_pos:])
            else:
                title = str(grid.poses)
            print(title)
            plt.title(title, pad=5, fontsize=16)
        pltMass = ax.imshow(mat, cmap='gray_r', interpolation='nearest')
        plt.savefig('test.png')
        print(plt.axis())
        plt.show()

    @staticmethod
    def createFolderIfNotExists(foldername):
        if not os.path.exists(foldername):
            os.makedirs(foldername, mode=0o777, exist_ok=False)

    def plotGridOnlyRow(pops, plotrows, fit_txt = '', omit_parents = False, targetpath= './results/Gen', generation=0):
        plotcols = math.ceil(len(pops) / plotrows)
        fig, axs = plt.subplots(plotcols,plotrows)
        # fig.set_size_inches(10, 4) #todo : for two row figure
        # plt.subplots_adjust(hspace=0.01, wspace=0.01)
        plt.subplots_adjust(hspace=0.1, wspace=0.1)


        i = 0
        for land in pops:
            mat = [[0] * land.width for _ in range(land.height)]
            if omit_parents:
                fig_title = f'{str(land.pop_id)}'
            else:
                fig_title = f'{str(land.pop_id)}\n{str(land.p1)}' if (land.p1 and land.p2 == -1) else f'{str(land.pop_id)}\n{str(land.p1)},{str(land.p2)}'

            # fig_title = f'{str(land.pop_id)}\n{land.p1},{land.p2}' if not omit_parents else f'{str(land.pop_id)}'
            for cell in land.poses:
                # mat[cell.y][cell.x] = 1
                y = land.height - 1 - cell.y  # to reverse y value to transform screen coordinate to plot coordinate
                mat[y][cell.x] = 1
            plotcol, plotrow = divmod(i, plotrows)
            axs[plotcol, plotrow].pcolormesh(mat, cmap='gray_r', alpha=0.7, edgecolor='silver', linewidth=0)
            axs[plotcol, plotrow].set_xticks([])
            axs[plotcol, plotrow].set_yticks([])
            # axs[plotcol, plotrow].text(0,20, fit_txt)
            axs[plotcol, plotrow].set_aspect(1.0)
            axs[plotcol, plotrow].set_title(fig_title, fontsize = 9)
            i += 1

        # filename = './results0916/pic'+ datetime.datetime.now().strftime('%Y%m%d-%H%M%S')[4:]

        Util.createFolderIfNotExists(foldername=targetpath)
        # filename = targetpath+'Gen'+str(generation) + '_'+datetime.datetime.now().strftime('%H%M%S')[4:]
        filename = targetpath+ datetime.datetime.now().strftime('%H%M') + '_Gen'+str(generation)
        # filename = './results0916/pic'+ datetime.datetime.now().strftime('%Y%m%d-%H%M%S')[4:]
        fig.suptitle(fit_txt, fontsize=12)
        plt.savefig(filename)
        plt.show()

    def plotGridBatch(pops, numfig, rows, fig_text = '', omit_parents = False):
        num_sheet = int(len(pops) / numfig)
        # fig_text = 'Initial Population'
        for i in range(num_sheet):
            # Util.plotGridOnlyRow(pops[i*numfig:i*numfig+numfig], rows, fig_text+" " + str(i))
            Util.plotGridOnlyRow(pops[i*numfig:i*numfig+numfig], rows, fig_text, omit_parents)

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

        filename = './results/gen_'+ datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")[4:]
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
        init_row = random.randint(1, height - 2)
        init_col = random.randint(1, width - 2)
        genes = [ Pos(init_col, init_row)]
        return genes

    @staticmethod
    def saveFitnessCsv(filename, fits, generation=0, fit_value=0, fitname='fitness'):
        keys = fits[0].keys()
        with open(filename, 'a', newline='') as fp:
                dict_writer = csv.DictWriter(fp, keys)
                dict_writer.writeheader()
                dict_writer.writerows(fits)

        with open(filename, 'a', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow([fitname, fit_value])

    @staticmethod
    def saveStat(filename, stats):
        keys = stats[0].keys()
        with open(filename, 'a', newline='') as fp:
            dict_writer =csv.DictWriter(fp, keys)
            dict_writer.writeheader()
            dict_writer.writerows(stats)

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