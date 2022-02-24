

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
    def adjacent_eight_way(loc):
        return [i for i in set(Pos(x + loc.x, y+loc.y)
                               for x in [-1, 1, 0] if 0<= x+loc.x < self.width
                               for y in [-1, 1, 0]  if 0<= y+loc.y < self.height
                               and loc != Pos(x+loc.x, y+loc.y)) ]


    # 동서남북 네 방향의 셀만 구한다. 동서남북 네 방향만
    @staticmethod
    def adjacent_four_way(loc, width, height):
        return [i for i in set(Pos(x+loc.x, y+loc.y)
                               for x in [ -1, 1, 0] if 0<=x+loc.x < width
                               for y in [-1, 1, 0] if 0 <= y+loc.y < height
                               and loc != Pos(x+loc.x, y+loc.y) and abs(x) != abs(y)
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