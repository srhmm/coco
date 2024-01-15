### These experiments are adapted from the Sparse Shift implementation (see LICENSE)

#!/usr/bin/env python
# coding: utf-8


# In[1]:
import os
from exp_synthetic.tex_plots import tex_plot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[2]:

SIMPLIFIED = False

SAVE_FIGURES = False
RESULTS_DIR = '../results'


# In[3]:


EXPERIMENT = 'mec'
tag = '_plot'

df = pd.read_csv(f'{RESULTS_DIR}/paper_{EXPERIMENT}{tag}.csv', sep=',', engine='python')


# In[4]:


df = df.loc[df['Precision'].notna(), :]

df['Fraction of shifting mechanisms'] = df['sparsity'].map(float) / df['n_variables'].map(float)
df['F1'] = 2 * df['Recall'] * df['Precision'] / (df['Recall'] + df['Precision'])
df['RT']=df['RT'].replace(np.nan, 0)

x_var_rename_dict = {
    'sample_size': '# Samples',
    'Number of environments': '# Environments',
    'Fraction of shifting mechanisms': 'Shift fraction',
    'dag_density': 'Edge density',
    'n_variables': '# Variables',
}
plot_df = df.rename(
        x_var_rename_dict, axis=1
    ).rename(
        {'Method': r'$\bf{Test}$', 'Soft': r'$\bf{Score}$'}, axis=1
    ).replace(
        {
            'er': 'Erdos-Renyi',
            'ba': 'Hub',
            'PC (pool all)': 'Full PC (oracle)',
            'Full PC (KCI)': 'Pooled_PC_KCI',
            'Min changes (oracle)': 'MSS_oracle',
            'Min changes (KCI)': 'MSS_KCI',
            'Min changes (GAM)': 'MSS_GAM',
            'Min changes (Linear)': 'MSS_Linear',
            'Min changes (FisherZ)': 'MSS_FisherZ',
            'MC': 'MC',
            'linc_rff_gain' : 'LINC_rff_gain',
            'linc_rff_nogain' : 'LINC_rff',
            'linc_rff_clus' : 'LINC_clus',
            'linc_rff_ilp' : 'LINC_ILP',
            #'LINC_GP_nogain': 'LINC (gp)',
            #'LINC_RFF_nogain' : 'LINC (rff)',
            'linc_gp_nogain': 'LINC_gp',
            False: 'Hard',
            True: 'Soft',
        }
)

sns.set_context('paper')
grid_vars = list(x_var_rename_dict.values())[1:len(list(x_var_rename_dict.values()))+1]
metrics = ['Recall', 'Precision', 'F1', 'RT' ] #, 'shd', 'sid']
methods = ['Pooled_PC_KCI', 'MSS_KCI',  r'MC', 'LINC_rff',  'LINC_clus',  'LINC_ILP',  'LINC_gp' , 'globe', 'linc_rff_exhaustive'] #'LINC (rff, gain)','LINC (rff, nogain)','LINC (gp, nogain)']#,'LINC (rff)', 'LINC (gain)', 'LINC (gp)', ] #TODO LINC_gp

base_s = [1]
base_s_C =  [1/3]
base_C= [3]
base_var =[6]
base_n = [500]
base_p = [.3]
settings_plot_samp= {#_samp
'VaryingSamples':
     [{
    '# Variables': [3],
    'n_total_environments' : [3],
    'sparsity': base_s,
    'Edge density': base_p
    }, '# Samples'],}
settings_plot={#_clus_ctx
'VaryingEnvs':
     [{
    '# Variables': [12], #[3],
   'n_total_environments' : [5,10,20,50], #,25, 50
    'sparsity': base_s,
    '# Samples': base_n,
    'Edge density': base_p
    }, '# Environments'],}
settings_plot_vars = {
'VaryingVars':
     [{
    '# Variables': [ 15 , 17, 20,    25 ,  30],
    'n_total_environments' : base_C,
    'sparsity': base_s,
    '# Samples': base_n,
    'Edge density': base_p
    }, '# Variables'],}

settings_plot_base_ctx = {  #_base_ctx
    'VaryingEnvs':
        [{
            '# Variables': [6],
            'n_total_environments': [1,3,5,8,10],
            'sparsity': base_s,
            '# Samples': base_n,
            'Edge density': base_p
        }, '# Environments'], }

settings_plot_base_vars = {
'VaryingVars':
     [{
    '# Variables': [3,6,9,12],
    'n_total_environments' : [5],
    'sparsity': base_s,
    '# Samples': base_n,
    'Edge density': base_p
    }, '# Variables'],}
settings_plot_paper= {
    'VaryingSamples':
        [{
            '# Variables': base_var,
            'n_total_environments': base_C,
            "sparsity": base_s,
            "Edge density": base_p,
        }, '# Samples'],
    'VaryingDensity':
     [{
    '# Variables':base_var,
    'n_total_environments' : base_C,
    "sparsity": base_s,
    "# Samples":base_n
    }, 'Edge density'],
'VaryingEnvs':
     [{
    "# Variables":base_var,
    "sparsity": [1/3],
    "# Samples":base_n,
    "Edge density": base_p,
    }, '# Environments'],
    'VaryingVars':
     [{
    '# Variables': [3,6,9,12],
    'n_total_environments' : base_C,
    'sparsity': base_s,
    '# Samples': base_n,
    'Edge density': base_p
    }, '# Variables'],
 'VaryingSparsity':
      [{
     '# Variables': base_var,
     'n_total_environments' : base_C,
     'sparsity':[0, 1, 2, 3, 4, 5, 6],
     '# Samples':base_n,
     'Edge density': base_p
     }, 'sparsity'],

}
ax = 0

fig, axes = plt.subplots(
    len(metrics),
    len(grid_vars),
    sharey='row', sharex='col',
    figsize=(1.5*len(grid_vars), 3)
)
identifier=''
if not os.path.exists('../results/'):
    os.makedirs('../results/')

for row, metric in zip(axes, metrics):
    for var, name, ax in zip(grid_vars, settings_plot, row):
            columns, g_var = settings_plot[name]
            plot_df_ax = plot_df
            for col in columns:
                plot_df_ax = plot_df_ax.loc[plot_df_ax[col].isin(columns[col])]
            if g_var != '# Environments':
                plot_df_ax = plot_df_ax[ # IMPORTANT! otherwise average over all number of environments
                    plot_df_ax['# Environments'] == plot_df_ax['# Environments'].max()]

            if metric=="RT":
                ax.set( yscale="log")
            plot_df_ax = plot_df_ax[(plot_df_ax[r'$\bf{Test}$'].isin(methods))]
            plot_df_ax = pd.concat(
                (   plot_df_ax[(plot_df_ax[r'$\bf{Score}$'] == 'Hard') & ~(plot_df_ax[r'$\bf{Test}$'] == 'MSS_KCI')],
                    plot_df_ax[(plot_df_ax[r'$\bf{Score}$'] == 'Soft') & (plot_df_ax[r'$\bf{Test}$'] == 'MSS_KCI')],
                    plot_df_ax[plot_df_ax[r'$\bf{Test}$'] == 'LINC_rff'],plot_df_ax[plot_df_ax[r'$\bf{Test}$'] == 'linc_rff_exhaustive'],
                    plot_df_ax[plot_df_ax[r'$\bf{Test}$'] == 'LINC_clus'],plot_df_ax[plot_df_ax[r'$\bf{Test}$'] == 'LINC_ILP'],
                    plot_df_ax[plot_df_ax[r'$\bf{Test}$'] == 'LINC_gp'], plot_df_ax[plot_df_ax[r'$\bf{Test}$'] == 'globe'],
                ),
                ignore_index=True
            )

            # Create the tex files
            tex_plot(df=plot_df_ax, x=g_var, y=metric ,identifier='')

            #plot_df_ax.plot(x=g_var, y=metric, kind='scatter')
            palette = [2, 7, 6, 8, 9, 5, 3]
            # if SIMPLIFIED: palette = [2, 8]
            sns.lineplot(
                data=plot_df_ax,
                x=g_var,
                y=metric,
                hue=r'$\bf{Test}$',
                style=r'$\bf{Score}$',
                ax=ax,
                palette=[
                    sns.color_palette("tab10")[i]
                    for i in palette#[2, 7, 6, 8]
                ],
                legend='full',
                style_order=['Hard', 'Soft'],
                lw=2,
            )

            xmin = plot_df_ax[g_var].min()
            xmax = plot_df_ax[g_var].max()

            if xmax > 1:
                ax.set_xticks([
                    xmin,
                    int(xmin + (xmax - xmin) / 2),
                    xmax,
                ])
            else:
                ax.set_xticks([
                    np.round(xmin, 1),
                    np.round(xmin + (xmax - xmin) / 2 , 1),
                    np.round(xmax, 1),
                ])


leg_idx = 3

axes = np.concatenate(axes)

for i in range(len(axes)):
     if i == 0:
         ax.set_ylim(bottom=0)
         #TODO:
     #if i == 0:
     #    axes[i].set_xscale('log')
     #if i == 1:
         #axes[i].set_xticks(range(15))
     if i == leg_idx:
         axes[i].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
         plt.setp(axes[i].get_legend().get_title(), fontsize=22)
     else:
         try:
             axes[i].get_legend().remove()
         except:
             pass

plt.tight_layout()
plt.subplots_adjust(hspace=0.15)
#if SAVE_FIGURES:
#    plt.savefig(f'./figures/empirical_select_rates_er_others.pdf')
plt.show()


