
# Customize directories --------------------
path = "~/impls/coco/experiments/"

source("run_jci_sub.R")
 # Each experiment
for (subdir in list("data_jci/MI_ztest_SKIP_NLIN_500/5_0_1/")){ #TODO subdirs corresp to experiments: "data_jci/exp1", ...

  basefilename = paste0(path,  subdir)
  cat("Data read from ",basefilename)
  
  # Run JCI-FCI --------------------
  # Each seed/identifier
  for(identifier in 0:1){
    for (mode in list("obs" ,"pooled", "jci123"))
      run_jci(basefilename, identifier, "fci", mode, 1, subdir , seed = seed)
    
    # Run ICP --------------------

    #for (mode in list("sc" ,"mc" ))
    #  run_jci_icp(paste0(basefilename, seed), "icp", mode, 1, subdir , seed = seed)
  }
}

#TODO update:
for (subdir in list("data_jci/")){
                 #  list( "B_groups_shift/", c(1,3)),
                 #  list( "F_hard_interventions/", c(1)),
                 #   list( "G_2hard_interventions/", c(1,3)),
                  #  list("E_mgroups_changes/", c(1,4,5)))){
  
  subdir_nm = subdir

  cat("\n---------------\nJCI on ", subdir_nm,  "\n---------------\n")
  res=read_results_synthdata(mode =  "jci123", method = "fci", subdir_nm = subdir_nm, target_columns = target_columns)
  
}
