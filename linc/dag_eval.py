import numpy as np
from cdt.metrics import SHD, SID
from statistics import mean, stdev, variance

def read_dag(file_nm,
             node_n,
             runs=20,
             path="",
             offset=0):
    """

    :param file_nm: name of subfolder in path/logs/dags/subfolder (output of out_naming.unique_name())
    :param run_repeats: j in subfolders path/dagfolder/i/dag_estim_j.npy (run_repeats in test_tree_search.py)
    :param path: OS path to dagfolder
    :param offset: for printing in tex
    :return: SID(path/dagfolder/i/true_j, path/dagfolder/i/estim_j)
    """
    ct = 0
    sids = [np.inf for _ in range(runs)]
    shds = [np.inf for _ in range(runs)]

    for id in range(1, runs + 1):
        Gestim = np.load(path+"logs/dags/"+file_nm+ + "estim_" + str(id) + ".npy")
        Gtrue = np.load(path+"logs/dags/"+file_nm+"/" + "true_" + str(id) + ".npy")
        sid = SID(Gtrue, Gestim)
        shd = SHD(Gtrue, Gestim)
        sids[ct] = sid.item()
        shds[ct] = shd
        ct = ct + 1

        print("Iterations: ", ct, "/", runs)
        z = 1.96
        mn_shd, var_shd, dev_shd = mean(shds), variance(shds), stdev(shds)
        confint_shd = z * dev_shd / np.sqrt(len(shds))
        print("\nSHD avged:", round(mn_shd, 3), "\t-var:", round(var_shd, 3), "-std:", round(dev_shd, 3),
                       "-cf:", round(confint_shd, 3))
        fct = node_n * (node_n)
        shd_norm = [shd / fct for shd in shds]
        print("SHD scaled:", round(mean(shd_norm), 3), "\t-var:", round(variance(shd_norm), 3), "-std:",
                       round(stdev(shd_norm), 3), "-cf:", round(z * stdev(shd_norm) / np.sqrt(len(shds)), 3))

        mn_sid, var_sid, dev_sid = mean(sids), variance(sids), stdev(sids)
        confint_sid = z * dev_sid / np.sqrt(len(sids))
        print("\nSID avged:", round(mn_sid, 3), "\t-var:", round(var_sid, 3), "-std:", round(dev_sid, 3),
                       "-cf:", round(confint_sid, 3))
        fct = node_n * (node_n - 1)
        sid_norm = [sid / fct for sid in sids]
        print("SID scaled:", round(mean(sid_norm), 3), "\t-var:", round(variance(sid_norm), 3), "-std:",
                       round(stdev(sid_norm), 3), "-cf:", round(z * stdev(sid_norm) / np.sqrt(len(sids)), 3))

        i = offset + 0.2
        print("\nTex:\n\t(" + str(i) + "," + str(round(mean(sid_norm), 3)) + ")",  # "\t-var:", round(variance(sid_norm), 3),
              "\t+=(" + str(i) + ", " + str(round(z * stdev(sid_norm) / np.sqrt(len(sids)), 3)) + ")",
              "\t-=(" + str(i) + ", " + str(round(z * stdev(sid_norm) / np.sqrt(len(sids)), 3)) + ")")

#TODO update folder names here
def read_dag_partitions(node_n,run_ids=[0],
             run_repeats=20,
             dagfolder="dags",
             path=""):
    raise Exception()
    """

    :param run_ids: list [i1, ...in] for subfolders path/dagfolder/i (run_id as in test_tree_search.py)
    :param run_repeats: j in subfolders path/dagfolder/i/dag_estim_j.npy (run_repeats in test_tree_search.py)
    :param dagfolder: dagfolder in test_tree_search.py
    :param path: path to dagfolder
    :return:
    """

    for it in run_ids:
        for id in range(1, run_repeats + 1):
            Gestim = np.load(path+dagfolder+"/"+str(it)+"/" + "estim_" + str(id) + ".npy")
            Gtrue = np.load(path+dagfolder+"/"+str(it)+"/" + "true_" + str(id) + ".npy")
            pitrue =np.genfromtxt(path+dagfolder+"/"+str(it)+"/" + "pi_true_" + str(id) + ".csv", delimiter=",")
            assert(node_n == len(pitrue))
            print("Nodes:", len(pitrue), "\nContexts:", len(pitrue[0]))
            for i in range(node_n):
                for j in range(node_n):
                    pi_i = pitrue[i]
                    pi_j = pitrue[j]
                    if (Gtrue[i][j] == 1):
                        print(Gestim[i][j], sum(pi_i))