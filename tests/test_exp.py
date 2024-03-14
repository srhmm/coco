from experiments.examples import example_coco, example_coco_and_oracles, example_run, example_mini
from experiments.reproduce_fig2 import reproduce_fig2
from experiments.reproduce_fig3 import reproduce_fig3
from experiments.reproduce_supporting import reproduce_supporting_clustering, reproduce_supporting_causal, \
    reproduce_supporting_emp_significance_power, reproduce_supporting_sparse_shifts


def test_mini():
    example_mini()


def test_example_coco():
    example_coco()


def test_example_coco_and_oracles():
    example_coco_and_oracles()


def test_example_run():
    example_run("", testing=True)


def test_reproduce_fig2():
    reproduce_fig2("", testing=True)


def test_reproduce_fig3():
    reproduce_fig3()


def test_reproduce_supA():
    reproduce_supporting_clustering("", testing=True)
    reproduce_supporting_causal("", testing=True)


#def test_reproduce_supB():
#    reproduce_supporting_emp_significance_power("", testing=True)
#    reproduce_supporting_sparse_shifts("", testing=True)

