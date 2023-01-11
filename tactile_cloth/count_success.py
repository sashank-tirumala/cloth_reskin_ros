from treelib import Node, Tree
import os
import numpy as np
import json
import pandas as pd
def create_tree(root_path):
    directory_tree = Tree()
    directory_tree.create_node("Root", root_path)
    cur_root= root_path
    cur_tree = []
    cur_tree.append(cur_root)
    while True:
        if(len(cur_tree) == 0):
            break
        for element in cur_tree:
            if os.path.isdir(element):
                dir_list = os.listdir(element)
                for dir_name in dir_list:
                    dir_path = element+"/"+dir_name
                    cur_tree.append(dir_path)
                    directory_tree.create_node(dir_path, dir_path, parent=element)
            cur_tree.remove(element)
    return directory_tree

def count_txt_under_node(directory_tree, node_id, txt_pattern):
    subtree = directory_tree.subtree(node_id)
    nodes = subtree.all_nodes()
    success_count = 0
    for node in nodes:
        if txt_pattern in node.identifier and not os.path.isdir(node.identifier):
            success_count+=1
    if(txt_pattern == "all"):
        node_list = filter(lambda x: os.path.isdir(x.identifier), subtree.children(node_id) )
        return len(node_list)
    return success_count

def get_list_of_attempts(directory_tree, node_id):
    subtree = directory_tree.subtree(node_id)
    children_node_list = subtree.children(node_id)
    attempts = []
    for child_node in children_node_list:
        if(os.path.isdir(child_node.identifier)):
            attempt= 0
            for grand_child in subtree.children(child_node.identifier):
                if "attempt" in grand_child.identifier:
                    attempt+=1
            attempts.append(attempt)
    return attempts


def create_dictionary(root):
    directory_tree = create_tree(root)
    node_list = directory_tree.children(root)
    node_list = filter(lambda x: "2002" and ("simple" or "complex") in x.identifier, node_list)
    res = dict()
    for node in node_list:
        ctr = count_txt_under_node
        nl = filter(lambda x: os.path.isdir(x.identifier), directory_tree.children(node.identifier))
        temp = []
        for cur_node in nl:
            temp.append( {cur_node.identifier.split("/")[-1]:{
            "number of trials": ctr(directory_tree, cur_node.identifier, "all"), 
            "success":ctr(directory_tree, cur_node.identifier, "success"), 
            "failed to grasp":ctr(directory_tree, cur_node.identifier, "fail_grasp"),
            "classifier did not predict correctly":ctr(directory_tree, cur_node.identifier, "fail_clf"),
            "number of attempts for each trial": get_list_of_attempts(directory_tree, cur_node.identifier),
            "average number of attempts": np.mean(get_list_of_attempts(directory_tree, cur_node.identifier))
            }}
            )
        #Code to pretty print
        name = node.identifier.split("/")[-1].split("_")
        name = name[-2]+"-"+name[-1]
        res[name] = temp
    return res

def get_df(root):
    columns = ("SR", "num of trials", "MA", "std attempts")
    rows = ("OL", "IC", "CL-TC", "R-TC")
    df = pd.DataFrame(np.zeros((4,4)), rows, columns)
    print(df)
    directory_tree = create_tree(root)
    node_list = directory_tree.children(root)
    node_list = filter(lambda x: "2002" and ("simple" or "complex") in x.identifier, node_list)
    res = dict()

    for node in node_list:
        ctr = count_txt_under_node
        nl = filter(lambda x: os.path.isdir(x.identifier), directory_tree.children(node.identifier))
        temp = []
        for cur_node in nl:
            name = cur_node.identifier.split("/")[-1]
            print(name)
            if("open" in name):
                df.loc["OL", "SR"] += ctr(directory_tree, cur_node.identifier, "success")
                df.loc["OL", "num of trials"] += ctr(directory_tree, cur_node.identifier, "all")
            elif("closed" in name):
                df.loc["CL-TC", "SR"] += ctr(directory_tree, cur_node.identifier, "success")
                df.loc["CL-TC", "num of trials"] += ctr(directory_tree, cur_node.identifier, "all")
            elif("random" in name):
                df.loc["R-TC", "SR"] += ctr(directory_tree, cur_node.identifier, "success")
                df.loc["R-TC", "num of trials"] += ctr(directory_tree, cur_node.identifier, "all")
            elif("image" in name):
                df.loc["IC", "SR"] += ctr(directory_tree, cur_node.identifier, "success")
                df.loc["IC", "num of trials"] += ctr(directory_tree, cur_node.identifier, "all")
        #Code to pretty print
    return df

def get_counts(experiment_path):
    trial_list = os.listdir(experiment_path)
    trial_count = len(trial_list)
    attempt_count= []
    success = 0
    fail_getgrasp = 0
    fail_liftgrasp = 0
    fail_grasp = 0
    fail_clf = 0
    for trial in trial_list:
        all_file_list = os.listdir(experiment_path+"/"+trial)
        attempt_list = filter(lambda x: "." not in x, all_file_list)
        attempt_count.append(len(attempt_list))
        for x in all_file_list:
            if "success" in x:
                success += 1
            if "fail_lostgrasp" in x or "fail_getgrasp" in x or "fail_nograsp" in x:
                fail_grasp +=1
            if "fail_getgrasp"  in x or "fail_nograsp" in x:
                fail_getgrasp +=1
            if "fail_lostgrasp" in x:
                fail_liftgrasp +=1

            if "fail_clf" in x:
                fail_clf +=1

    # return np.around(success*100./trial_count,2), np.around(fail_clf*100./trial_count,2), np.around(fail_getgrasp*100./trial_count,2), np.around(fail_liftgrasp*100./trial_count,2), attempt_count
    return np.around(success*100./trial_count,2), np.around(fail_clf*100./trial_count,2), np.around(fail_getgrasp*100./trial_count,2), np.around(fail_liftgrasp*100./trial_count,2), np.around(fail_grasp*100./trial_count,2), attempt_count

def get_all_counts(root_path):
    exp_names = filter(lambda x: "cfg" not in x,  os.listdir(root_path))
    columns = (
        "Success $\uparrow$",
        "Fail $\downarrow$ (Grasp)",
        "Fail $\downarrow$ \\ (Prediction)", 
        "Fail $\downarrow$ \\ (Lift)", 
        "Attempts $\downarrow$"
    )
    rows = (
        "Open-Loop", 
        "Random-Image",
        "Random-Tactile",
        "Feedback-Image",
        "Feedback-Tactile"
    )
    # df = pd.DataFrame(np.zeros((5,4)), rows, columns)
    df = pd.DataFrame(np.zeros((5,5)), rows, columns)
    ol_attempts = []
    ft_attempts = []
    rt_attempts = []
    ri_attempts= []
    fi_attempts = []
    for exp in exp_names:
        # success, fail_clf, fail_getgrasp, fail_liftgrasp, mean_attempts = get_counts(root_path + "/"+ exp)
        success, fail_clf, fail_grasp, fail_getgrasp, fail_liftgrasp, mean_attempts = get_counts(root_path + "/"+ exp)
        # print("fail_liftgrasp: "+str(fail_liftgrasp)+" "+str(exp))
        # print("fail_grasp: "+str(fail_grasp)+" "+str(exp))
        # print("fail_clf: "+str(fail_clf)+" "+str(exp))
        if("open" in exp):
            df.loc["Open-Loop", "Success $\uparrow$"]                    = success
            df.loc["Open-Loop", "Fail $\downarrow$ (Grasp)"]         = fail_getgrasp
            df.loc["Open-Loop", "Fail $\downarrow$ \\ (Prediction)"]              = fail_clf
            # df.loc["Open-Loop", "Fail $\downarrow$ \\ (Grasp)"]             = fail_grasp
            df.loc["Feedback-Tactile","Fail $\downarrow$ \\ (Lift)"]  = fail_liftgrasp
            ol_attempts                                                  = mean_attempts
            df.loc["Open-Loop", "Attempts $\downarrow$"] = str(np.around(np.mean(ol_attempts),2))+"$\pm$"+str(np.around(np.std(ol_attempts),1))
        elif("closedloop-tactile" in exp):
            df.loc["Feedback-Tactile","Success $\uparrow$"]              = success
            df.loc["Feedback-Tactile", "Fail $\downarrow$ (Grasp)"]  = fail_getgrasp
            df.loc["Feedback-Tactile", "Fail $\downarrow$ \\ (Prediction)"]              = fail_clf
            # df.loc["Feedback-Tactile", "Fail $\downarrow$ \\ (Grasp)"]             = fail_grasp
            df.loc["Feedback-Tactile","Fail $\downarrow$ \\ (Lift)"]  = fail_liftgrasp
            ft_attempts                                                  = mean_attempts
            df.loc["Feedback-Tactile", "Attempts $\downarrow$"] = str(np.around(np.mean(ft_attempts),2))+"$\pm$"+str(np.around(np.std(ft_attempts),1))
        elif("random-tactile" in exp):
            df.loc["Random-Tactile", "Success $\uparrow$"]               = success
            df.loc["Random-Tactile", "Fail $\downarrow$ (Grasp)"]   = fail_getgrasp
            df.loc["Random-Tactile", "Fail $\downarrow$ \\ (Prediction)"]         = fail_clf
            # df.loc["Random-Tactile", "Fail $\downarrow$ \\ (Grasp)"]        = fail_grasp
            df.loc["Random-Tactile", "Fail $\downarrow$ \\ (Lift)"]   = fail_liftgrasp
            rt_attempts                                                  = mean_attempts
            df.loc["Random-Tactile", "Attempts $\downarrow$"] = str(np.around(np.mean(rt_attempts),2))+"$\pm$"+str(np.around(np.std(rt_attempts),1))
        elif("random-image" in exp):
            df.loc["Random-Image","Success $\uparrow$"]                  = success
            df.loc["Random-Image", "Fail $\downarrow$ (Grasp)"]      = fail_getgrasp
            df.loc["Random-Image", "Fail $\downarrow$ \\ (Prediction)"]         = fail_clf
            # df.loc["Random-Image", "Fail $\downarrow$ \\ (Grasp)"]        = fail_grasp
            df.loc["Random-Image", "Fail $\downarrow$ \\ (Lift)"]     = fail_liftgrasp
            ri_attempts                                                  = mean_attempts
            df.loc["Random-Image", "Attempts $\downarrow$"] = str(np.around(np.mean(ri_attempts),2))+"$\pm$"+str(np.around(np.std(ri_attempts),1))
        elif("closedloop-image" in exp):
            df.loc["Feedback-Image", "Success $\uparrow$"]               = success
            df.loc["Feedback-Image", "Fail $\downarrow$ (Grasp)"]    = fail_getgrasp
            df.loc["Feedback-Image", "Fail $\downarrow$ \\ (Prediction)"]         = fail_clf
            # df.loc["Feedback-Image", "Fail $\downarrow$ \\ (Grasp)"]        = fail_grasp
            df.loc["Feedback-Image","Fail $\downarrow$ \\ (Lift)"]    = fail_liftgrasp
            fi_attempts                                                  = mean_attempts
            df.loc["Feedback-Image", "Attempts $\downarrow$"] = str(np.around(np.mean(fi_attempts),2))+"$\pm$"+str(np.around(np.std(fi_attempts),1))
        
    return df

if(__name__=="__main__"):
    root = "/media/ExtraDrive3/fabric_touch/paper_experiments/2022-02-26-00-03-53_simplepolicy_5class_knn"
    # root = "/media/ExtraDrive3/fabric_touch/paper_experiments/2022-02-26-01-36-12_simplepolicy_5class_knn_2layer"
    # root = "/media/ExtraDrive4/fabric_touch/paper_experiments/2022-02-26-20-56-15_simplepolicy_5class_knn_1layer"
    root="/media/ExtraDrive4/fabric_touch/paper_experiments/2022-02-26-22-05-57_simplepolicy_5class_knn_2layer"
    # root = "/media/ExtraDrive4/fabric_touch/paper_experiments/2022-02-27-02-15-50_simplepolicy_5class_agg_2layer"
    # root = "/media/ExtraDrive4/fabric_touch/paper_experiments/2022-02-27-16-10-31_simplepolicy_1layer_generalization"
    root = "/media/ExtraDrive4/fabric_touch/paper_experiments/cloths_combined"
    root = "/media/ExtraDrive4/fabric_touch/paper_experiments/2022-02-27-17-09-29_simplepolicy_1layer_generalization_towel2"
    # root = "/media/ExtraDrive4/fabric_touch/paper_experiments/2022-02-27-16-10-31_simplepolicy_1layer_generalization"

    # root = "/media/ExtraDrive4/fabric_touch/paper_experiments/2022-02-28-03-06-29_simplepolicy_2layer"

    # Thomas datasets for paper
    # root = "/media/ExtraDrive4/fabric_touch/paper_experiments/2022-02-26-20-56-15_simplepolicy_5class_knn_1layer"
    root = "/media/ExtraDrive4/fabric_touch/paper_experiments/2022-02-27-16-10-31_simplepolicy_1layer_generalization"
    # root = "/media/ExtraDrive4/fabric_touch/paper_experiments/2022-02-27-17-09-29_simplepolicy_1layer_generalization_towel2"
    # root = "/media/ExtraDrive4/fabric_touch/paper_experiments/2022-02-28-03-06-29_simplepolicy_2layer"
    # root = "/media/ExtraDrive4/fabric_touch/paper_experiments/2022-02-28-05-48-25_simplepolicy_2layer_generalization_whitetowel"
    # root = "/media/ExtraDrive4/fabric_touch/paper_experiments/2022-02-28-06-42-02_simplepolicy_2layer_generalization_patternedtowel"

    df = get_all_counts(root)
    print(df)
    print(df.to_latex(escape=False))

# \begin{tabular}{lrrrrr}
# \toprule
# {} &     Success Rate &    Fail (Grasp) &    Fail (pred) &   Mean Attempts \\
# \midrule
# Open-Loop        &   0.0 &   0.0 &   0.0 &  0.0 $\pm$   0.00 \\
# Random-Image     &   0.0 &   0.0 &   0.0 &  0.0 $\pm$   0.00 \\
# Random-Tactile   &   0.0 &   0.0 &   0.0 &  0.0 $\pm$   0.00 \\
# Feedback-Image   &  30.0 &  20.0 &  50.0 &  1.6 $\pm$   0.49 \\
# Feedback-Tactile &  80.0 &  20.0 &   0.0 &  2.3 $\pm$   0.78 \\

# \bottomrule
# \end{tabular}