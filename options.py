import configparser
import os.path
from util import Util

class Options:
    def __init__(self):
        self.filename = './option.ini'
        if not os.path.isfile(self.filename):
            self.create_default_option() # if file is not exists
            print('file not exists. so created')
        self._options = configparser.ConfigParser()
        # self.create_default_option()

    def create_default_option(self):

        self._options['Fitness'] = { # todo: why Fitness? maybe Land Boundary
            'wall_list':'south,north',
            'road_side': 'south',
            'adjacent_land': 'east, west'
        }

        # only numbers are valid
        self._options['Mass'] = {
            'width': '5',
            'height': '6',
            'faRatio': '0.3',
            'cell_size':  '1000', #mm 1000mm = 1m
            'max_area': '150', # todo: now forget about this
            'min_area': '120', # todo: now forget about this
            'adjacent_distance' : '1.5',
            'road_distance': '4', # 2m 남쪽 앞 건물과의 간격
            'front_height': '6.8' #
        }

        self._options['GAParams']={
            'expandSize':'1',
            'numCross':'1',
            'mutationRate': '0.1',
            'nGeneration': '0',
            'nPopulation':'10',
            'crossoverChance': '0.5',
            'DNALength':'0',
        }

        self._options['Optional'] = { # todo: all the boolean options not used yet
            'forceDNALength': 'no',
            'selection': 'roulette',
        }

        with open(self.filename, 'w') as optionFile:
            self._options.write(optionFile)

    def read_option_file(self):
        self._options.read(self.filename)

    def get_sections(self):
        self._options.read(self.filename)
        return self._options.sections()

    def get_value(self, section, key):
        return self._options[section].get(key)

    def get_int_value(self, section, key):
        return int(self._options[section].get(key))

    def get_float_value(self, section, key):
        return float(self._options[section].get(key))

    def fitness_options(self):
        fitness = self._options['Fitness']
        fitOptions = {}

        for key in fitness:
            fitOptions[key] = fitness[key].split(',')
        return fitOptions

    def mass_options(self):
        mass = self._options['Mass']
        massOptions = {}
        for key in mass:
            if mass[key].isdigit():
                val = int(mass[key])
            else:
                val = float(mass[key])
            massOptions[key] = val
        return massOptions

    def mass_options_values(self, key):
        return self.mass_options(key)

    def param_options(self):
        param = self._options['GAParams']
        paramOptions = {}
        for key in param:
            paramOptions[key] = int(param[key]) if param[key].isdigit() else float(param[key])
        return paramOptions

    def optional_options(self):
        optional = self._options['Optional']
        optionalOptions = {}
        for key in optional:
            optionalOptions[key] = optional.get(key)
        return optionalOptions

    def get_options(self): # all at once
        return self.fitness_options(), self.mass_options(), self.param_options(), self.optional_options()

    # def to_float(self, str_val):
    #     # match = re.match(r'^-?\d+(?:\.\d+)$', str_val)
    #     # print(match)
    #     # if match is None :
    #     #     return None
    #     if Util.isfloat(str_val):
    #         return float(str_val)
    #
    def config_options(self, key):
        # config = Options()
        for section in self.get_sections():
            if self._options[section].get(key):
                vals = self.get_value(section, key).split(',')
                if section == 'Mass' or section == 'GAParams':  # only allow float or integer value in these section
                    return Util.tonumber(vals[0])
                else:
                    return vals

    def get_display_settings(self, width, height):
        config = lambda x:self.config_options(x)
        tostr = lambda title, x: f'{title}: {x}'
        cell_length = self.config_options('cell_length') # length of each side of the cell
        n_col, n_row = cell_length * width, cell_length * height

        settings = ['[Site Configuration]\n\n']
        settings += [f'Land Size: {n_col}m x {n_row}m']
        settings += [f'Real Cell Size: {cell_length}m x {cell_length}m']
        settings += [f'Legal Floor Area Ratio: {config("required_faratio")*100:.0f}%']
        settings += [f'Optimal Ratio: {config("optimal_ratio")[0]}']
        settings += [tostr('South Building Elevation Diff',  config('height_diff_south'))+'m']
        settings += [tostr('South Distance', config('south_gap'))+'m']
        # settings += [tostr('North Gap', config('north_gap'))+'m']
        # settings += [tostr('East Gap', config('east_gap'))+'m']
        # settings += [tostr('West Gap', config('west_gap'))+'m']
        settings += [tostr('Road Side', config('road_side'))]
        settings += [tostr('Road Width', config('road_width'))+'m']
        settings += [tostr('Setback Distance', config('setback_requirement'))+'m']
        settings += [tostr('Walls installed', config('wall_list'))]
        settings += [tostr('Wall Elevation', config('wall_height'))+'m']

        new_line = '\n'
        return f'{new_line}{new_line.join(settings)}'

