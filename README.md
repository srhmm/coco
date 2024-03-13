## Identifying Confounding from Causal Mechanism Shifts (CoCo)
[![compat](https://github.com/srhmm/coco/actions/workflows/compat.yml/badge.svg)](https://github.com/srhmm/coco/actions/workflows/compat.yml)
[![exp](https://github.com/srhmm/coco/actions/workflows/exp.yml/badge.svg)](https://github.com/srhmm/coco/actions/workflows/exp.yml)
![Static Badge](https://img.shields.io/badge/python-%3E%3D3.8-blue?logo=python&label=python)
[![license](https://img.shields.io/github/license/machine-teaching-group/checkmate.svg)](https://github.com/srhmm/coco/blob/main/LICENSE)

### Setup
``` 
conda create -n "coco"
conda activate coco
pip install -r requirements.txt 
pip install -e . 
```

### Overview
- Examples (as below): `experiments/examples.py`
- Evaluation on synthetic data (Fig. 2): `experiments/reproduce_fig2.py`
- Evaluation on flow cytometry data (Fig. 3):  `experiments/reproduce_fig3.py`
- Empirical Analysis (Apx): `experiments/reproduce_supporting.py`


### Example
##### Sampling data in different contexts with latent variables and causal mechanism changes 
```
import numpy as np
from coco.dag_confounded import DAGConfounded
from coco.dag_gen import _random_nonlinearity
from coco.mi_sampling import Sampler

# parameters to test
n_nodes, n_confounders, n_contexts, n_samples = (5, 1, 5, 500)
n_shifts_observed, n_shifts_confounders = 1, 1 
fun_form = _random_nonlinearity()

# hyperparameters
seed = 42
FCI_INDEP_TEST = 'fisherz'

# Data Generation
sampler = Sampler()
dag = DAGConfounded(seed, n_contexts, n_nodes, n_confounders,  n_shifts_observed, n_shifts_confounders,
                    is_bivariate=False)

D, Dobs = dag.gen_data(seed, n_samples, _functional_form=fun_form, oracle_partition=True, noise_iv=False)

```
##### Discovering confounded pairs of variables

```
from coco.co_co import CoCo

coco_oZ = CoCo(D, dag.G.nodes, sampler=sampler, n_components=n_confounders, dag=dag)
coco = CoCo(D, dag.G.nodes, sampler=sampler, n_components=None, dag=dag, verbosity=1)

def _show(res, method, id):
	tp, fp, tn, fn, f1, _, _ = res
	print(f'\t{method}-{id}: \t(f1={np.round(f1, 2)})'
	      f'\t(tp={tp}, tn={tn}, fp={fp}, fn={fn})')
	      
# with known number of confounders
_show(coco_oZ.eval_estimated_edges(dag), coco_oZ, 'cfd')
      
# with unknown number of confounders
_show(coco.eval_estimated_edges(dag), coco, 'cfd')
```
##### Discovering causal edge directions under confounding
```
res = coco.eval_causal(dag)
res = res[0], res[1], res[2], res[3], res[4], 0, 0

_show(res, coco, 'caus')

```

##### CoCo with different oracles 

```
from coco.co_test_type import CoCoTestType, CoShiftTestType, CoDAGType

# CoCo with known causal directions 
coco_oG = CoCo(D, dag.G.nodes, sampler=sampler, n_components=None, 
                     dag_discovery=CoDAGType.SKIP, dag=dag, verbosity=1)
                               
_show(coco_oG.eval_estimated_edges(dag), coco_oG, 'cfd')   
 
     
# CoCo with known causal directions as well as partitions for each node
coco_oPi = CoCo(D, dag.G.nodes, sampler=sampler, n_components=None, 
                  shift_test=CoShiftTestType.SKIP,
                  dag_discovery=CoDAGType.SKIP, dag=dag, verbosity=1)
                                         
_show(coco_oPi.eval_estimated_edges(dag), coco_oPi, 'cfd')           
```
##### Comparison: Joint FCI

```
from experiments.method_types import MethodType 
from experiments.run_coco import run_fci

fci_jci, D, Dobs, seed = run_fci(dag,D,  Dobs, n_samples, fun_form, MethodType.FCI_JCI, FCI_INDEP_TEST, seed)

tp, fp, tn, fn, f1 = fci_jci.eval_confounded(dag, MethodType.FCI_JCI)
print(f'\tJFCI-cfd: \t(f1={np.round(f1,2)})\t(tp={tp}, tn={tn}, fp={fp}, fn={fn})')

tp, fp, tn, fn, f1 = fci_jci.eval_causal(dag, MethodType.FCI_JCI)
print(f'\tJFCI-caus: \t(f1={np.round(f1,2)})\t(tp={tp}, tn={tn}, fp={fp}, fn={fn})')
``` 

