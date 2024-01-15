import os
from statistics import mean, stdev

import numpy as np


def tex_plot(df, x, y, identifier, method_col=r'$\bf{Test}$'):
    xnm = x
    if x == '# Environments':
        xnm = 'environments_spars03'
    if x == '# Samples':
        xnm = 'samples'
    if x == '# Variables':
        xnm = 'variables'
    if x == 'Edge density':
        xnm = 'edgeDensity'
    write_file = open(f"../results/tex{identifier}_{xnm}_{y}.csv", "w+")
    methods = df[method_col].unique()
    s = "X"
    for method in methods:
        s = s+"\t"+str(method)+"_"+str(y)+"\t"+str(method)+"_"+str(y)+"_std"+"\t"+str(method)+"_"+str(y)+"_cnf"


    #if x=="sparsity":
    #    x= 'Shift fraction' #.map(float) / df['n_variables'].map(float)
    for x_val in sorted(df[x].unique()):
        df_x = df[df[x] == x_val]
        xprint = x_val
        #if x == "sparsity":
        #    xprint = x_val * C_n # total number of mechanism changes
        s = s + "\n" + str(xprint)
        for method in methods:
            df_xm = df_x[df_x[method_col] == method]
            if y=="F1":
                df_xm[y] = df_xm[y].fillna(0)

            if y == 'RT':
                df_xm[y] = df_xm[y].fillna(0)
            if len(df_xm) > 1:
                mn = mean(df_xm[y])
                std = stdev(df_xm[y])
                cf =  round(1.96 * std / np.sqrt(len(df_xm[y])), 3)
                s = s + "\t"+str(round(mn, 3)) + "\t"+ str(round(std, 3))+ "\t" + str(round(cf, 3))
                print(xprint, method, "\t#", len(df_xm))
            else:
                if y == 'RT':
                    s = s + "\t0"+"\t0"+"\t0"
                else:
                    s = s + "\tna"+"\tna"+"\tna"

    write_file.write(s)
    write_file.flush()