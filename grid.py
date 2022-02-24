from util import Util



class Grid:
    def __init__(self, poses, width, height):
        self.grid = [['.']*width for _ in range(height)]
        self.poses = poses
        self.width = width
        self.height = height
        for p in poses:
            self.grid[p.y][p.x] = 'X'

    def __str__(self):
        return '\n'.join(' '.join(row) for row in self.grid)

    def adjacents(self, loc):
        return Util.adjacent_four_way(loc, self.width, self.height)