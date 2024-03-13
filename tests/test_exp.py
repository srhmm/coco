from experiments.examples import example_coco, example_coco_and_oracles, example_run, example_mini
from experiments.reproduce_fig2 import reproduce_fig2
from experiments.reproduce_fig3 import reproduce_fig3
from experiments.reproduce_supporting import reproduce_supporting_clustering


def test_mini():
    example_mini()

def test_example_coco():
    example_coco()

def test_example_coco_and_oracles():
    example_coco_and_oracles()

def test_example_run():
    example_run()

def test_reproducibility():
    reproduce_fig2("/out", testing=True)
    #reproduce_fig3("/out")

