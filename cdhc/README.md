# Deconfounded Causal Discovery
## Requirements
All python requirements are listed in `requirements.txt`. 
Further, the library `hexbin` is required to generate the Bayesian Signed Rank Test between GFCI and CDHC.
Further, to obtain the data for Fig. 2 (Bayesian Signed Rank Test to compare method performance), the library hexbin is further required.

## Configuration
Change the directory in which you want to store the results in the file `config.py`. By default, it will store the required files in the `data/` folder.

## Reproducing Results:
To reproduce the result figures in the paper, follow the steps below, separated by section.

### Synthetic (Sec. 6.1):
First run `python synthetic.py` to run both GFCI and CDHC on the synthetic data. Then run `python synthetic_plots.py` to generate the data required for the plots.

The files`alpha-f1.dat`, `alpha-conf.dat`, `conf-f1.dat` contain the data relevant for Fig. 1.
The data for Table 1 is in `xrs/f1-table.txt`.
The data for Fig. 2 is in `f1-comp.dat`.

### REGED (Sec 6.2):
Running `reged.py` will run CDHC on the data from `data/reged/` and store the relevant data in `xrs/reged-0.dat` and `xrs/reged-1.dat`.

### E. Coli SOS DNA Repair Network (Sec. 6.3)
Running `sos.py` stores the network structure of the inferred network in `xrs/sos-edges.txt`. Note that the inferred edges have to be compared with the suggested graph manually, as the graph was made manually.

### Plots
The files containing the code to recreate the plots in the publication in latex are in the `latex/` directory. Compile `paper.tex` to generate the plots from the result files generated above.  
If the data for any of the sections above has not yet been generated, the default files (corresponding to the files used for publication) in xrs/ will be used.

## Running on your own data
To run CDHC on your own data, run `python cdhc.py --in <input_file> --out <output_file>`. The input file should contain the data containing precisely the variables over which you want to infer the causal network with confounders. 
