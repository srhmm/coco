#### COCO's TODOs
### Notes

### Oracle validation

- case: true parents known
- does it work in the 2x oracle case where we know both parents and partition? 
    - **unconfd**: pi_i, pi_j sampled independently, 
  CO_i = [**1**{ pi_i[c1] =/= pi_i[c2]} for c2>c1]
    - **confd**: pi_i = intersect(pi_i, pi_z)
  CO_i = [**1**{ pi_i[c1] =/= pi_i[c2] OR  pi_z[c1] =/= pi_z[c2] } ]
- does this still work when we discover the partition using LINC/KCI?
- compare what happens for the **true** DAG in the case that there are confounders, vs. there are none. Is the overall MI for causal node pairs lower in the latter case?

### Causal Parents
#### Intuitions
- the DAG minimizing the MSS minimizes the pairwise correlations between the partition vectors; in the anticausal directions, partitions need to be tuned to fit the variables 
- the whole thing should work best if we use the true DAG, but then better for the MSS minimizer than inferior ones

#### Experiments
- does the above still work in the DAG that minimizes MSS?
- compare what happens for the DAG that minimizes MSS in the case that there are confounders, vs. there are none. Is the overall MI for causal node pairs lower in the latter case?
- compare what happens to noncausal DAGs

### multiple confounders 
- **idea**: if there is a single confounder for X1, X2, X3, their partitions can be explained by the **same** hidden partitions; if there are multiple confounders, with in turn independent partitions, those lead to more correlations
 
