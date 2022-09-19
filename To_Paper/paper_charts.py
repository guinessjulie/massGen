import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


numeric_field = ['f(BCR)','f(PAR)', 'FSH(Sunlight Hours)','f(FSH)','f(SSR)','AR(Aspect Ratio)','f(AR)','f(VSymm)','f(HSymm)','f(CC)']
str_field = ['Fulfill Building Line', 'Setbacks']
fitfilename = 'fitness20220909160842.csv'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


numeric_field = ['f(BCR)','f(PAR)', 'FSH(Sunlight Hours)','f(FSH)','f(SSR)','AR(Aspect Ratio)','f(AR)','f(VSymm)','f(HSymm)','f(CC)']
str_field = ['Fulfill Building Line', 'Setbacks']
fitfilename = 'fitness20220909160842.csv'
def get_data(fitfilename):
    base_path = 'G:\\dev\\python\\2022project\\massGen\\To_Paper\\'
    folder = 'fsh_pop_500_gen_30_generate0909\\'
    filepath = base_path + folder + fitfilename
    return pd.read_csv(filepath, index_col='id')


def to_numeric_all(field_list, df):
    for field in field_list:
        df[field] = df.apply(pd.to_numeric, errors='coerce')
    return df



def split(df, n):
    gens = []
    skip = 0
    for i in range(20):
        start = i*n+skip
        end = start + n
        gen = df[start:end]
        gens.append(gen)
        skip += 2
    return gens


def main():
    df = get_data(fitfilename)
    df = df.drop('f(FSH)')
    df = to_numeric_all(numeric_field, df)
    print(df.info())
if __name__ == "__main__":
    main()