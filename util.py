import configparser
import re
import math

def loadIniFile(inifile):
    options = configparser.ConfigParser()
    options.read(inifile)
    return options

class Pos:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return f"({self.x}, {self.y})"
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    def __repr__(self):
        return str(self)
    def __hash__(self):
        return self.x*1000+self.y
    def to_tuples(self):
        return (self.x, self.y)

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

    import math

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

