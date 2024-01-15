
### LINC Experiments
```
cd exp_synthetic
pip install -r requirements.txt
```

#### Fig. 3
``` 
python plot_mec.py
```
#### Fig. 4
``` 
python plot_iid.py
```

#### MEC discovery
Specify the desired experimental settings in `settings.py`, the base settings shown in the paper are used by default. 

``` 
python run.py experiment --mec
```


Results will be in `mec.csv` and can be read using `plot_mec.py` (after editing in experimental settings used). 


#### DAG discovery 
Specify the desired experimental settings in `settings.py`. Here, the full DAG search is run by LINC or GLOBE, that is, without knowing the Markov Equivalence class.

``` 
python run.py experiment --mec
```

Results will be in `dag.csv` and can be read using `plot_dag.py` (after editing in experimental settings used). 

