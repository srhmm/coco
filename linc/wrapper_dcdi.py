from statistics import mean, variance, stdev

import numpy as np

from function_types import FunctionType
from intervention_types import IvType


#TODO: Make sure to run print_data() first, which prints to the datapaths used below.

def print_dcdi_commands(dcdi_dsf):
    print_commands(False,dcdi_dsf,False,False)
def print_jci_commands():
    print_commands(True,False,False,False)
def print_igsp_commands():
    print_commands(False, False, True, False)
def print_utigsp_commands():
    print_commands(False, False, False, True)

def print_commands(jci, dcdi_dsf, igsp, utigsp):
    C_n = 5
    D_n = 2000
    node_n = 5
    runs = 20
    seed = 42
    ct = 0

    for fun_type in [ FunctionType.GP_PRIOR ]:
        for iv_type in [IvType.SCALE, IvType.PARAM_CHANGE, IvType.CONST]:
            for iv_per_node in  [[0,0], [1,1], [1,2], [1,3]]:
                if iv_per_node[1]==1:
                    ivs = "iid"
                else:
                    ivs = "iv"+str(iv_per_node[1])

                path = "/data_synth/concat/"+str(fun_type)+"_"+str(iv_type)+"_"+ivs+"/"

                if fun_type == FunctionType.GP_PRIOR:
                    indepTest= 'kernelCItest'
                else:
                    indepTest= 'gaussCItest'
                indepTestGSP = "kci"
                for i in range(1, runs+1):
                    if jci:
                        print(f'python3 main_pc.py --data-path {path} --i-dataset {i} --exp-path exp --indep-test {indepTest} --knowledge unknown')
                    elif igsp:
                        print(f'python3 main.py --data-path  {path} --i-dataset {i}    --exp-path exp  --ci-test  {indepTestGSP} --model IGSP ')
                    elif utigsp:
                        print(f'python3 main.py --data-path  {path} --i-dataset {i}   --exp-path exp  --ci-test  {indepTestGSP} --model UTIGSP ')
                    elif dcdi_dsf:
                        print(f'mkdir expp{ct} \npython3 main.py --train --data-path {path} --num-vars 5 --i-dataset {i} --exp-path expp{ct} --model DCDI-DSF --intervention --intervention-type perfect --intervention-knowledge unknown --reg-coeff 0.5')
                    else:
                        print(f'mkdir expp{ct} \npython3 main.py --train --data-path {path} --num-vars 5 --i-dataset {i} --exp-path expp{ct}  --model DCDI-G --intervention --intervention-type perfect --intervention-knowledge unknown --reg-coeff 0.5')
                    ct = ct + 1


#TODO: make the following modifications in jci/main_pc.py to print their SID values to a file:
    #with open(os.path.join(opt.data_path,"jci_sid"), 'a') as f:
    #    print(sid, file=f)
    #with open(os.path.join(opt.data_path,"jci_shd"), 'a') as f:
    #    print(shd, file=f)
    #with open(os.path.join(opt.data_path,"jci_dag"), 'a') as f:
    #    print(dag, file=f)

def read_jci_results(method_name="dcdi-g",  #jci, utigsp, igsp
                     ivs_per_node=[[[0,0], [0,1], [0,2], [0,3]]],
                     ivtypes= [IvType.CONST, IvType.SCALE, IvType.PARAM_CHANGE],
                     shift=0,  box=False):
    runs = 20
    for fun_type in  [FunctionType.GP_PRIOR]:
        for iv_type in ivtypes:
            for iv_per_node in ivs_per_node:
                if iv_per_node[1] == 1:
                    ivs = "iid"
                else:
                    ivs = "iv" + str(iv_per_node[1])

                path = "/Users/sarah/linc/data_synth_500/concat/" + str(fun_type) + "_" + str(
                    iv_type) + "_" + ivs + "/"

                print("%", method_name, fun_type, iv_type, iv_per_node)
                shds = np.genfromtxt(path+method_name+"_shd", delimiter=',')
                sids = np.genfromtxt(path+method_name+"_sid", delimiter=',') # "igsp_sid"

                node_n = 5
                z = 1.96

                mn_shd, var_shd, dev_shd = mean(shds), variance(shds), stdev(shds)
                confint_shd = z * dev_shd / np.sqrt(len(shds))
                print("\tSHD avged:", round(mn_shd, 3), "\t-var:", round(var_shd, 3), "-std:",
                               round(dev_shd, 3), "-cf:", confint_shd)
                fct = node_n * (node_n)
                shd_norm = np.array([shd / fct for shd in shds])
                print("\tSHD scaled:", round(mean(shd_norm), 3), "\t-var:", round(variance(shd_norm), 3),
                               "-std:", round(stdev(shd_norm), 3), "-cf:",
                               round(z * stdev(shd_norm) / np.sqrt(len(shds)), 3))

                mn_sid, var_sid, dev_sid = mean(sids), variance(sids), stdev(sids)
                confint_sid = z * dev_sid / np.sqrt(len(sids))
                print("\n\tSID avged:", round(mn_sid, 3), "\t-var:", round(var_sid, 3), "-std:",
                               round(dev_sid, 3), "-cf:", round(confint_sid, 3))
                fct = node_n * (node_n - 1)
                sid_norm = np.array([sid / fct for sid in sids])
                print("\tSID scaled:", round(mean(sid_norm), 3), "\t-var:", round(variance(sid_norm), 3),
                               "-std:", round(stdev(sid_norm), 3), "-cf:",
                               round(z * stdev(sid_norm) / np.sqrt(len(sids)), 3))

                median = np.median(sid_norm)
                upper_quartile = np.percentile(sid_norm, 75)
                lower_quartile = np.percentile(sid_norm, 25)

                iqr = upper_quartile - lower_quartile
                upper_whisker = sid_norm[sid_norm <= upper_quartile + 1.5 * iqr].max()
                lower_whisker = sid_norm[sid_norm >= lower_quartile - 1.5 * iqr].min()
                if box:
                    print("\t", round(lower_whisker,3), "\t", round(lower_quartile, 3), "\t", round(median, 3), "\t",  round(upper_quartile, 3), "\t",  round(upper_whisker, 3))
                else:
                    i = iv_per_node[1]+shift
                    print("SID tex:\n\t("+str(i)+ ","+str(round(mean(sid_norm), 3))+")",
                         "\t+=("+str(i)+ ", "+str(round(  z * stdev(sid_norm) / np.sqrt(len(sids)), 3))+")",
                          "\t-=("+str(i)+ ", "+str(round(   z * stdev(sid_norm) / np.sqrt(len(sids)), 3))+")")

                median = np.median(shd_norm)
                upper_quartile = np.percentile(shd_norm, 75)
                lower_quartile = np.percentile(shd_norm, 25)

                iqr = upper_quartile - lower_quartile
                upper_whisker = shd_norm[shd_norm <= upper_quartile + 1.5 * iqr].max()
                lower_whisker = shd_norm[shd_norm >= lower_quartile - 1.5 * iqr].min()
                #print("\tSHD", round(lower_whisker,3), "\t", round(lower_quartile, 3), "\t", round(median, 3), "\t",  round(upper_quartile, 3), "\t",  round(upper_whisker, 3))
