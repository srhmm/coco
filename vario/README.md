### Discovering Invariant and Changing Mechanisms from Data

#### Vario Algorithms
- VarioPi: `causal_mechanism_search.py`, discovers a context partition for a target variable and its causes. The partition shows in which groups of contexts the causal mechanism is invariant and where it changes.
- Vario: `causal_variable_search.py`, discovers a partition and the causal parents for a target.
- VarioG: `causal_dag_search.py`, discovers partitions and causal parents for all variables. 


#### Quick Example
```
data, index_Y, true_dag, true_partitions = generate_data_from_DAG(n_nodes=5, n_contexts=5, n_samples_per_context=500,
                                                                partition_Y=[[0, 1, 2],[3,4]],
                                                                min_interventions=5,max_interventions=5, scale=False, verbose=True, seed=3) 
# Causal Mechanism Changes
oracle_partition, oracle_score, _ = causal_mechanism_search(data, index_Y, true_dag.parents_of(index_Y), greedy=False, verbose=True)

# Causal Parents and Mechanism Changes
estim_parents, estim_partition, estim_score, _ = causal_variable_search(data, index_Y,  [n for n in range(n_nodes) if not (n==index_Y)], greedy_mechanism_search=False, verbose=True)

```