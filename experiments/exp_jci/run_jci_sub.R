# JCI/ICP wrapper -----------------
# To generate DAGs and run JCI on the synthetic data, this script was adapted from:

# Copyright (c) 2018-2020, Joris M. Mooij. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

run_jci = function( basefilename,
                       identifier, alg, mode, obsv_contexts, subdir,
                       systemVars = c(1,2,3,4,5),
                       contextVars = c(6,7,8,9),seed = 1,
                       acyclic = TRUE){
  
  
suppressMessages(library(rjson))
source('../../init.R',chdir=TRUE)

# parse command line arguments
#args <- commandArgs(trailingOnly = TRUE)

#basefilename<-args[1]
vb <- TRUE #TODO arg

# read metadata
metadatafile=paste0(basefilename,'D_', identifier, '.json')
metadata<-fromJSON(file=metadatafile)

systemVars<-metadata$SystemVars
contextVars<-metadata$ContextVars

pSysObs<-length(systemVars)
stopifnot( pSysObs > 0 )
pContext<-length(contextVars)
p <- pSysObs + pContext
obsContext <- matrix(0,1,pContext)
 
seed = metadata$seed
#ytarget = metadata$ytarget

#observational_group = metadata$true_observational_group_for_target
observational_contexts = c()
#if(length(observational_group)) 
#  observational_contexts = contextVars[1:observational_group]
#else
cid = as.numeric(obsv_contexts)
observational_contexts =  contextVars[cid] #,contextVars[args[5]])


#alg<-'fci'; 'icp'
#alg<-args[2]
#mode<-'obs', 'pooled', 'jci123'; 'sc', 'mc', 'scstar', 'mcstar'
#mode<-args[3]
#
#subdir<-args[5]

#subdir = "three_interactions" #"doubles_n6_1_50/" # "triples_n6_50_250/"

outfile<- paste0(basefilename, "R_", identifier, #paste("~/Vario/out/synthetic_out_jci_icp/", subdir, seed,  #"-y", ytarget,
                "_" ,alg, "_", mode)
if(vb) cat("\n------------------------")
if(vb) cat("\nReading from:", basefilename)
if(vb) cat("\nWriting to:", outfile)

#create = ifelse(!dir.exists(paste0("~/Vario/out/synthetic_out_jci_icp/", subdir)),
#                dir.create(paste0("~/Vario/out/synthetic_out_jci_icp/",subdir)) , FALSE)

#if(alg=="icp" & ( mode=="mcstar" | mode=="scstar"))
  # if(observational_group <2) stop("for ICPstar, need at least one observational context for Y besides the first one.")
if(vb) cat("\n------------------------\n", 
           "Method:",paste0(alg,"-",mode), "\n" )
if(vb) cat("  -Seed: ", seed, "\n", "  -SysVars: ", paste(systemVars,collapse=","), "\n","  -CtxVars: ", paste(contextVars,collapse=","), #paste0("  -Target Y:", ytarget), "\n",
            #"  -Partition of Y (if data gen by Vario): ",  paste0("(",paste(metadata$true_partition_group_sizes,collapse=""), ")"),
           "\n", "  -Contexts: ", paste0("(",paste(contextVars,collapse=""), ")"), "\n")
           # "  -Observational Contexts for Y: ", paste0("(",paste(obsv_contexts,collapse=","), ")"),
          # "\n")

miniter<-0
#if( !is.na(args[6]) ) {
#  miniter<-as.numeric(args[6])
#}
maxiter<-0
#if( !is.na(args[7]) ) {
#  maxiter<-as.numeric(args[7])
#}
alpha<-1e-2
#if( !is.na(args[8]) )
 # alpha<-as.numeric(args[8])
jcifci_test<-'gaussCIcontexttest'
#if( !is.na(args[9]) )
 # jcifci_test<-args[9]
doPDsep<-TRUE
#if( !is.na(args[10]) )
#  doPDsep<-(as.numeric(args[10]) != 0)
verbose<-0
#if( !is.na(args[11]) )
 # verbose<-(as.integer(args[11]))

###cat('run.R: running with arguments',args,'\n')

# read and preprocess data
#TODO header
 data<-read.csv(file=paste(basefilename,'D_', identifier, '.csv',sep=''),header=TRUE,sep=",")
 fixedGaps<- NULL #as.matrix(read.csv(file=paste(basefilename,'A_gaps_', identifier, '.csv',sep=''),header=FALSE,sep=","))
 fixedEdges<-NULL #as.matrix(read.csv(file=paste(basefilename,'A_edges_', identifier, '.csv',sep=''),header=FALSE,sep=","))

 fixedGaps <- NULL# sapply(as.data.frame(fixedGaps), as.logical)
 fixedEdges <- NULL# sapply(as.data.frame(fixedEdges), as.logical)
 A_true<-read.csv(file=paste(basefilename,'A_true_', identifier, '.csv',sep=''),header=TRUE,sep=",")

 # G <- as(skel@graph, "matrix")

# ####
# 
#  #add additional rows for one context variable that do not affect Y
#   data_icpstartwo= rbind(data_icpstar,data[which(data[, observational_contexts[1]]==1),])
# 
# non_observational = setdiff(contextVars, observational_contexts[1])
# data_icpstartwo = data_icpstartwo[,setdiff(1:ncol(data_icpstartwo),non_observational)]
# contextVars_icpstartwo = contextVars[1]

#write.csv(data_icpstartwo, file=paste0(basefilename, "-data_icpstartwo.csv"))
 
for( iter in miniter:maxiter ) {
  #cat('iter: ',iter,'\n')

set.seed(iter)
if( iter != 0 ) {
  subsamplefrac<-0.5
} else {
  subsamplefrac<-0.0
}

# start measuring time
start_time<-proc.time()[3]

# run causal-discovery
if( alg=='asd' ) { # run ASD
  if( (mode == 'obs' && pSysObs <= 6) ||
      (mode == 'pooled' && pSysObs <= 6) ||
      (mode == 'meta' && pSysObs <= 6) ||
      (mode == 'pikt' && pSysObs <= 6) ||
      (mode == 'jci123' && p <= 8) ||
      (mode == 'jci123kt' && p <= 8) ||
      (mode == 'jci13' && p <= 6) ||
      (mode == 'jci1' && p <= 6) ||
      (mode == 'jci1nt' && p <= 6) ||
      (mode == 'jci12' && p <= 6) ||
      (mode == 'jci12nt' && p <= 6) ||
      (mode == 'jci123-sc' && p <= 7) ||
      (mode == 'jci1-sc' && p <= 7) ||
      (mode == 'jci0' && p <= 6) ||
      (mode == 'jci0nt' && p <= 6) ||
      (mode == 'jci123io' && p <= 8)
     ) {
    if( iter != 0 )
      outfile<-paste(basefilename,'-',alg,'-',mode,'-',iter,sep='')
    else
      outfile<-paste(basefilename,'-',alg,'-',mode,sep='')
    indepfile<-paste(outfile,'.indep',sep='')

    if( mode=='pikt' ) # reason about perfect interventions
      extrafiles=c(file.path(asp_dir,'partial_comp_tree.pl'))
    else # can forget about perfect interventions
      extrafiles=c(file.path(asp_dir,'obs_comp_tree.pl'))
    if( acyclic ) { # use acyclic d-separation encoding
      if( is.null(metadata$sufficient) || !metadata$sufficient )
        extrafiles=c(extrafiles,file.path(asp_dir,'asd_acyclic.pl'))
      else
        extrafiles=c(extrafiles,file.path(asp_dir,'asd_acyclic_sufficient.pl'))
    } else { # cyclic, so run with sigma-separation (note: the model as a whole is not linear)
      extrafiles=c(extrafiles,file.path(asp_dir,'asd_sigma_cyclic.pl'))
    }

    if( mode=='jci123kt' ) {
      targetfile<-paste(outfile,'.targets',sep='')
      f=file(targetfile,'w')
      if( length(contextVars) > 0 )
        for( i in 1:length(contextVars) )
          for( j in 1:length(systemVars) ) {
            targets <- which(intToBits(metadata$targets[i])!=0)
            if( j %in% targets ) {
              cat(file=f,'edge(',contextVars[i]-1,',',systemVars[j]-1,').\n')
            } else {
              cat(file=f,':-edge(',contextVars[i]-1,',',systemVars[j]-1,').\n')
            }
          }
      close(f)
      extrafiles=c(extrafiles,targetfile)
    }

    result<-asd_wrapper(indepfile,data,systemVars,contextVars,alpha,verbose=verbose,subsamplefrac,mode,test='gaussCIcontexttest',obsContext,weightmul=1000,extrafiles,Itargets=metadata$Itargets)
    write.csv(result$arel,file=paste(outfile,'-arel.csv',sep=''),row.names=FALSE)
    write.csv(result$edge,file=paste(outfile,'-edge.csv',sep=''),row.names=FALSE)
    write.csv(result$conf,file=paste(outfile,'-conf.csv',sep=''),row.names=FALSE)
  } else
    cat('Skipping asd-',mode,' because unknown mode or p=',p,'\n',sep='')
} else{


  if( alg == 'fci' ) { # run FCI
  result<-fci_wrapper(data,#A,A_true,
  systemVars,contextVars,alpha,verbose=verbose,subsamplefrac,test=jcifci_test,mode,obsContext,doPDsep,
   fixedGaps = fixedGaps, fixedEdges = fixedEdges)

  # save results
  #ifelse(!dir.exists(outdir), dir.create(outdir), FALSE)
  #if( iter != 0 )
  #  outfile<-paste("~/Vario/out_relatedwork/", seed,  "-y", ytarget, "-" ,alg, "-", mode,"-", iter, sep='') #paste(basefilename,'-',alg,'-',mode,'-',iter,sep='')
  #else
  #  outfile<-paste("~/Vario/out_relatedwork/", seed,  "-y", ytarget, "-" ,alg, "-", mode,sep='') #paste(basefilename,'-',alg,'-',mode,sep='')


  
  save(result,file=paste(outfile,'-result.Rdata',sep=''))
  
  pag2graphviz(filename=paste(outfile,'-pag.dot',sep=''),p=result$p,contextVars=result$contextVars,labels=result$labels,pag=result$pag,mode=mode)
  mc2graphviz(filename=paste(outfile,'-arel.dot',sep=''),p=result$p,contextVars=result$contextVars,labels=result$labels,mc=result$arel)
  L2graphviz(filename=paste(outfile,'-mag.dot',sep=''),p=result$p,contextVars=result$contextVars,labels=result$labels,L=result$mag,mode=mode)
  write.csv(result$arel,file=paste(outfile,'-arel.csv',sep=''),row.names=FALSE)
  write.csv(result$edge,file=paste(outfile,'-edge.csv',sep=''),row.names=FALSE)
  write.csv(result$conf,file=paste(outfile,'-conf.csv',sep=''),row.names=FALSE)

  # demo: query an independence relation
  # cat('1 _||_ 11 | 2? ', !directed_reachable(1,11,c(2),c(),result$mag,verbose=0),'\n')
} else if( alg == 'icp' ) { # run ICP
  if(mode =='mc' | mode == 'sc'){
  if( mode == 'mc' )
    datamode<-'multiple'  
  else {
      if( mode == 'sc' )
        datamode<-'merge'
      else stop('icp mode?')
  }
  result<-icp_wrapper(data,systemVars,contextVars,alpha=alpha,verbose=verbose,subsamplefrac=subsamplefrac,datamode=datamode)
  }
  #######
  else{
    
    
    
    #for ICP, we also use a version with only those contexts where Y was not intervened upon - but this needs background knowledge of the context groups for Y.
    # and a version with only 2 contexts
    
    sm = apply(data[,contextVars], 1, function(i) return(sum(i)))
    data_icpstar = data[which(sm==0),]#start with observational rows (all system variables observational, all context vars 0)
    
    #take data rows for context variables specified(usually one,eg.observational_contexts=c(8) )
    for(obs_context in observational_contexts) #add additional rows for all context variables that do not affect Y
      data_icpstar = rbind(data_icpstar,data[which(data[,obs_context]==1),])
    
    #remove excess columns for other context vars
    non_observational = setdiff(contextVars, observational_contexts)
    data_icpstar = data_icpstar[,setdiff(1:ncol(data_icpstar),non_observational)]
    contextVars_icpstar = contextVars[1]#observational_contexts
    # 
    
    
    #TODO for debug:
    write.csv(data_icpstar, file=paste0(basefilename, paste0("-data_icpstar_",paste0(obsv_contexts,collapse="_"), ".csv")))
    
    outfile = paste0(outfile, "-", obsv_contexts)
    
    if( mode == 'mcstar' ) 
      result<-icp_wrapper(data_icpstar,systemVars,contextVars_icpstar,
                          alpha=alpha,verbose=verbose,subsamplefrac=subsamplefrac,datamode="multiple", hidden=FALSE)
    else {
      if( mode == 'scstar' )
        result<-icp_wrapper(data_icpstar,systemVars,contextVars_icpstar,
                            alpha=alpha,verbose=verbose,subsamplefrac=subsamplefrac,datamode="merge", hidden=FALSE)

      # else{
      #   if( mode == 'mcstartwo' ) result<-icp_wrapper(data_icpstartwo,systemVars,contextVars_icpstartwo,
      #                                              alpha=alpha,verbose=verbose,subsamplefrac=subsamplefrac,datamode="multiple")
      #   
      #     
      #   else{
      #   if(mode=='scstartwo') result<-icp_wrapper(data_icpstartwo,systemVars,contextVars_icpstartwo,
      #                                             alpha=alpha,verbose=verbose,subsamplefrac=subsamplefrac,datamode="merge")
      #   
      #   
         else{
           
           
           if(mode == "hicp-sc" |mode =="hicp-mc"){
             if(mode == "hicp-mc"){
               result<-icp_wrapper(data_icpstar,systemVars,contextVars_icpstar,
                                   alpha=alpha,verbose=verbose,subsamplefrac=subsamplefrac,datamode="multiple", hidden=TRUE)
             } else{
               result<-icp_wrapper(data_icpstar,systemVars,contextVars_icpstar,
                                   alpha=alpha,verbose=verbose,subsamplefrac=subsamplefrac,datamode="merge", hidden=TRUE)
             }
           }
           else 
             stop('icp mode?')
         }
        }
      } 
    }


 
 # if( iter != 0 )
#    outfile<- paste("~/Vario/out_relatedwork/", seed,  "-y", ytarget, "-" ,alg, "-", mode, "-", iter, sep='') #paste(basefilename,'-',alg,'-',mode,'-',iter,sep='')
#  else
   
  write.csv(result$arel,file=paste(outfile,'-arel.csv',sep=''),row.names=FALSE)
  save(result,file=paste(outfile,'-result.Rdata',sep=''))

  #save(result,file=paste("~/Vario/out_jci/", seed,  "-y",ytarget, "-" ,alg, "-", mode, '-result.Rdata',sep='')) #iter??
  
  mc2graphviz(filename=paste(outfile,'-arel.dot',sep=''),p=result$p,contextVars=result$contextVars,labels=result$labels,mc=result$arel)
 #else stop("alg? mode?")
  
}
#   
#   if( alg == 'lcd' ) { # run LCD
#   if( mode == 'mc' ) 
#     result<-lcd(data,systemVars,contextVars,alpha=alpha,verbose=verbose,subsamplefrac,datamode='multiple',test='gaussCItest',conservative=FALSE)
#   else if( mode == 'mccon' ) 
#     result<-lcd(data,systemVars,contextVars,alpha=alpha,verbose=verbose,subsamplefrac,datamode='multiple',test='gaussCItest',conservative=TRUE)
#   else if( mode == 'mcsct' )
#     result<-lcd(data,systemVars,contextVars,alpha=alpha,verbose=verbose,subsamplefrac,datamode='multiple',test='gaussCIsincontest',conservative=FALSE)
#   else if( mode == 'mcsctcon' )
#     result<-lcd(data,systemVars,contextVars,alpha=alpha,verbose=verbose,subsamplefrac,datamode='multiple',test='gaussCIsincontest',conservative=TRUE)
#   else if( mode == 'sc' )
#     result<-lcd(data,systemVars,contextVars,alpha=alpha,verbose=verbose,subsamplefrac,datamode='merge',test='gaussCIsincontest',conservative=FALSE)
#   else if( mode == 'sccon' )
#     result<-lcd(data,systemVars,contextVars,alpha=alpha,verbose=verbose,subsamplefrac,datamode='merge',test='gaussCIsincontest',conservative=TRUE)
#   else
#     stop('unknown lcd mode')
# 
#   if( iter != 0 )
#     outfile<-paste(basefilename,'-',alg,'-',mode,'-',iter,sep='')
#   else
#     outfile<-paste(basefilename,'-',alg,'-',mode,sep='')
#   save(result,file=paste(outfile,'-result.Rdata',sep=''))
#   write.csv(result$arel,file=paste(outfile,'-arel.csv',sep=''),row.names=FALSE)
#   mc2graphviz(filename=paste(outfile,'-arel.dot',sep=''),p=result$p,contextVars=result$contextVars,labels=result$labels,mc=result$arel > 0)
#   write.csv(result$conf,file=paste(outfile,'-conf.csv',sep=''),row.names=FALSE)
#   write.csv(result$edge,file=paste(outfile,'-edge.csv',sep=''),row.names=FALSE)
# } else if( alg == 'cif' ) { # run CIF
#   if( mode == 'mc' ) 
#     result<-cif(data,systemVars,contextVars,alpha=alpha,verbose=verbose,subsamplefrac,datamode='multiple',test='gaussCItest',conservative=FALSE,patterns=c(0,1))
#   else if( mode == 'mccon' ) 
#     result<-cif(data,systemVars,contextVars,alpha=alpha,verbose=verbose,subsamplefrac,datamode='multiple',test='gaussCItest',conservative=TRUE,patterns=c(0,1))
#   else if( mode == 'mcsct' ) 
#     result<-cif(data,systemVars,contextVars,alpha=alpha,verbose=verbose,subsamplefrac,datamode='multiple',test='gaussCIsincontest',conservative=FALSE,patterns=c(0,1))
#   else if( mode == 'mcsctcon' ) 
#     result<-cif(data,systemVars,contextVars,alpha=alpha,verbose=verbose,subsamplefrac,datamode='multiple',test='gaussCIsincontest',conservative=TRUE,patterns=c(0,1))
#   else if( mode == 'sc' )
#     result<-cif(data,systemVars,contextVars,alpha=alpha,verbose=verbose,subsamplefrac,datamode='merge',test='gaussCIsincontest',conservative=FALSE,patterns=c(0,1))
#   else if( mode == 'sccon' )
#     result<-cif(data,systemVars,contextVars,alpha=alpha,verbose=verbose,subsamplefrac,datamode='merge',test='gaussCIsincontest',conservative=TRUE,patterns=c(0,1))
#   else
#     stop('unknown cif mode')
# 
#   if( iter != 0 )
#     outfile<-paste(basefilename,'-',alg,'-',mode,'-',iter,sep='')
#   else
#     outfile<-paste(basefilename,'-',alg,'-',mode,sep='')
#   save(result,file=paste(outfile,'-result.Rdata',sep=''))
#   write.csv(result$arel,file=paste(outfile,'-arel.csv',sep=''),row.names=FALSE)
#   mc2graphviz(filename=paste(outfile,'-arel.dot',sep=''),p=result$p,contextVars=result$contextVars,labels=result$labels,mc=result$arel)
#   write.csv(result$conf,file=paste(outfile,'-conf.csv',sep=''),row.names=FALSE)
# } else if( alg == 'fisher' ) { # run Fisher's method
#   result<-fisher(data,systemVars,contextVars,alpha,verbose=verbose,subsamplefrac)
# 
#   if( iter != 0 )
#     outfile<-paste(basefilename,'-',alg,'-',iter,sep='')
#   else
#     outfile<-paste(basefilename,'-',alg,sep='')
#   save(result,file=paste(outfile,'-result.Rdata',sep=''))
#   write.csv(result$arel,file=paste(outfile,'-arel.csv',sep=''),row.names=FALSE)
#   mc2graphviz(filename=paste(outfile,'-arel.dot',sep=''),p=result$p,contextVars=result$contextVars,labels=result$labels,mc=result$arel > 0)
# } else
#   stop('Unknown algorithm')

# stop measuring time
stop_time<-proc.time()[3]

if( exists('outfile') ) {
  # write time to file
  cat(file=paste(outfile,'.runtime',sep=''),stop_time-start_time,'\n')
}

}



}

