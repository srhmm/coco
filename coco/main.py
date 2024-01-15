from experiments.exp_coco.examples import example_small
from experiments.exp_coco.reproduce_fig2 import reproduce_fig2
from experiments.exp_coco.reproduce_fig3 import reproduce_fig3
from experiments.exp_coco.reproduce_supporting import reproduce_supporting_sparse_shifts, \
    reproduce_supporting_emp_significance_power, reproduce_supporting_causal, reproduce_supporting_clustering
from experiments.exp_coco.results_coco import MethodType

if __name__ == "__main__":

    #Example
    #example_small(path='../experiments')

    #Figure 2
    #reproduce_fig2(path='../experiments')

    #Figure 3
    #reproduce_fig3(path='../data_cytometry')

    #Apx. Supporting Figures
    reproduce_supporting_clustering(path='../experiments')
    #todo reproduce_supporting_causal(path='../experiments')
    #reproduce_supporting_sparse_shifts(path='../experiments')
    #reproduce_supporting_emp_significance_power(path='../experiments')
