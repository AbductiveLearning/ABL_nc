from clingo.symbol import Function, Number, parse_term, Tuple_
from clingo.control import Control
import time
import itertools
import numpy as np
from multiprocessing import Pool
from functools import partial


def _check_atom_consistence(hnd, target, truth):
    for m in hnd:
        if m.contains(target):
            if truth == True: # Exist at least one model that contains the target atom
                return True 
            else: # None of the model contains the target atom
                return False
    return not truth

# Check consistence given logic program and target (CWA)
def check_program_consistence(program_str, target_str=""):
    ctl = Control(["0"])
    ctl.add("base", [], program_str)
    ctl.ground([("base", [])])

    if len(target_str) == 0:
        return ctl.solve().satisfiable
    else:
        # Assume only one literal in target_str
        target_str = target_str.strip().replace(".","")
        if target_str.find("-") == -1:
            target = parse_term(target_str)
            truth = True
        else:
            target = parse_term(target_str.replace("-",""))
            truth = False
        with ctl.solve(yield_=True) as hnd:
            return _check_atom_consistence(hnd, target, truth)

# Check consistence given labels
def check_label_consistence(context_str, target_str, labels, kb_str, preds, prefix):
    label_str = labels2program(z_labels = labels, preds = preds, prefix = prefix)
    program_str = kb_str + "\n" + context_str + "\n" + label_str
    return check_program_consistence(program_str, target_str)

# Check consistence given batch of labels
def check_label_consistence_batch(kb_str, context_list, target_list, predict_labels_list, preds, prefix, learned_program=""):
    # consistence_list = []
    # for i in range(len(context_list)):
    #     if check_label_consistence(kb_str=kb_str+learned_program, context_str=context_list[i], target_str=target_list[i], labels=predict_labels_list[i], preds=preds, prefix=prefix):
    #         consistence_list.append(True)
    #     else:
    #         consistence_list.append(False)    
    partial_check_label_consistence = partial(check_label_consistence, kb_str = kb_str+learned_program, preds = preds, prefix = prefix)
    with Pool() as pool:
        consistence_list = pool.starmap(partial_check_label_consistence, zip(context_list, target_list, predict_labels_list))
    consistence_list = np.array(consistence_list)
    cnt, total = np.sum(consistence_list==True), len(consistence_list)
    cons_perc = cnt/total
    return cnt, cons_perc, consistence_list

# Convert the instance labels and attach to the program
# Ensure the z_labels are existing in the logic program
def labels2program(z_labels, preds, prefix, abducible_pred="abducible"):
    label_str = ""
    for i, label in enumerate(z_labels):
        argument = prefix + str(i)
        # assert(context_target_str.find(argument)!=-1)
        if label == -1: # Abducible
            atom = "%s(%s). "%(abducible_pred,argument)
        else:
            atom = "%s(%s). "%(preds[label],argument)
        label_str += atom
    return label_str

# Extract the abduced labels from the logic program models
def _models2label(hnd, labels, abducible_preds, prefix):
    abduced_label_list = []
    for m in hnd:
        # print(m)
        abduced_label = labels.copy()
        for i in range(len(abduced_label)):
            if abduced_label[i] == -1: # Change -1 to abduced labels, otherwise keep
                argument = prefix + str(i)
                for j, abducible_pred in enumerate(abducible_preds):
                    if m.contains(Function(abducible_pred, [Function(argument)])): # Atom in abduced result
                        abduced_label[i] = j
                        break # Only one atom
            assert(abduced_label[i] != -1)  # Assert change the -1 label after the loop
        abduced_label_list.append(abduced_label)
    return abduced_label_list

def _build_abduction_program(kb_str, context_str, target_str, label_str, abducible_preds):
    abduction_program = kb_str + "\n" + context_str + '\n' + target_str + "\n" + label_str
    # "1{king(X); queen(X); rook(X); bishop(X); knight(X); pawn(X)}1 :- abducible(X)."
    aggr_str = "\n1{%s}1 :- abducible(X).\n"%(";".join(["%s(X)"%pred for pred in abducible_preds]))
    abduction_program += aggr_str
    for pred in abducible_preds:
        abduction_program += "#show %s/1.\n"%(pred)
    return abduction_program

# Given the labels, where the abduced labels are set to -1, return the abduced labels list
def _try_abduce(kb_str, context_str, target_str, labels_for_abduce, abducible_preds, prefix):
    label_str = labels2program(z_labels=labels_for_abduce, preds=abducible_preds, prefix=prefix)
    abduction_program = _build_abduction_program(kb_str=kb_str, context_str=context_str, target_str=target_str, label_str=label_str, abducible_preds=abducible_preds)
    ctl = Control(["0"]) # All models
    ctl.add("base", [], abduction_program)
    ctl.ground([("base", [])])
    # print(ctl.solve(on_model=print))
    with ctl.solve(yield_=True) as hnd:
        abduced_labels = _models2label(hnd, labels_for_abduce, abducible_preds, prefix)
    return abduced_labels

def abduce_one(context_str, target_str, pred_labels, kb_str, abducible_preds, prefix):
    if check_label_consistence(context_str=context_str, target_str=target_str, labels=pred_labels, kb_str=kb_str, preds=abducible_preds, prefix=prefix):
        return [np.array(pred_labels)]
    ret = []
    for address_num in range(1, len(pred_labels) + 1):
        possible_faults_list = itertools.combinations(range(len(pred_labels)), address_num)
        for possible_fault_pos in possible_faults_list:
            labels_for_abduce = np.array(pred_labels)
            labels_for_abduce[list(possible_fault_pos)] = -1
            abduced_labels = _try_abduce(kb_str=kb_str, context_str=context_str, target_str=target_str, labels_for_abduce=labels_for_abduce, abducible_preds=abducible_preds, prefix=prefix)
            if len(abduced_labels) == 0:
                continue
            ret.extend(abduced_labels)
        if len(ret) > 0:
            break
    if len(ret) == 0:
        print("Abduction fail!!!")
        print(context_str, target_str, pred_labels) # TODO: Add a new class to each label?
        return [np.array(pred_labels)]
    return ret

def abduce_batch(kb_str, context_list, target_list, predict_labels_list, abducible_preds, prefix="_u", learned_program=""):
    partial_abduce_one = partial(abduce_one, kb_str = kb_str+learned_program, abducible_preds = abducible_preds, prefix = prefix)
    with Pool() as pool:
        ret = pool.starmap(partial_abduce_one, zip(context_list, target_list, predict_labels_list))
    return ret

if __name__ == "__main__":
    task = "less_than"
    task = "chess"
    task = "multiples_of_three"
    kb_file = task + "/target.lp"
    prefix = "_u"
    with open(kb_file, 'r') as f:
        kb_str = f.read()

    # Time test -- check_consistence
    if False:
        time_start = time.time()
        for i in range(1000):
            if task == "less_than":
                context_str = ""
                label_str = "four(_u0). five(_u1)."
                pred_labels = [4,5]
                if i%2==0:
                    target_str = "less(_u0,_u1)."
                else:
                    target_str = "-less(_u0,_u1)."
                abducible_preds = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
            elif task == "chess":
                context_str = "at((2, 5),_u0). at((5, 6),_u1). at((2, 1),_u2). at((2, 4),_u3). at((1, 7),_u4). at((7, 3),_u5). at((0, 0),_u6). at((4, 2),_u7). at((6, 3),_u8)."
                label_str = "pawn(_u0). queen(_u1). king(_u2). knight(_u3). rook(_u4). bishop(_u5). queen(_u6). rook(_u7). bishop(_u8)."
                pred_labels = [5,1,0,4,2,3,1,2,3]
                if i%2==0:
                    target_str = "sat."
                else:
                    target_str = "-sat."
                abducible_preds = ["king", "queen", "rook", "bishop", "knight", "pawn"]
            elif task == "multiple_of_three":
                context_str = ""
                label_str = "four(_u0). six(_u1)."
                pred_labels = [4,6]
                if i%2==0:
                    target_str = "even(_u0,_u1)."
                else:
                    target_str = "odd(_u0,_u1)."
                abducible_preds = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
            program_str = "\n".join([kb_str, context_str, label_str])
            result = check_program_consistence(program_str, target_str)
            print(result)
            result = check_label_consistence(context_str, target_str, pred_labels, kb_str, abducible_preds, prefix)
            print(result)
        time_end = time.time()
        print(time_end-time_start)

    # Time test -- abduction
    if True:
        if task == "less_than":
            context_str = ""
            target_str = "less(_u0,_u1)."
            pred_labels = [9,0]
            labels_for_abduce = [-1,0]
            abducible_preds = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        elif task == "chess":
            context_str = "at((2,3),_u0). at((1,1),_u1)."
            target_str =  "sat."
            pred_labels = [3,3]
            labels_for_abduce = [3,-1]
            abducible_preds = ["king", "queen", "rook", "bishop", "knight", "pawn"]
        elif task == "multiples_of_three":
            context_str = ""
            target_str = "new_1(_u0,_u1)."
            pred_labels = [7,8]
            labels_for_abduce = [-1,0]
            abducible_preds = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        
        context_list, target_list, predict_labels_list = [], [], []
        ret = []
        time_start = time.time()
        for i in range(10000):
            if task != "multiples_of_three":
                if i%2==0:
                    target_str = target_str.replace("-", "")
                else:
                    target_str = "-" + target_str.replace("-", "")
            elif task == "multiples_of_three":
                if i%3==0:
                    target_str = target_str.replace("new_1", "even_1")
                elif i%3==1:
                    target_str = target_str.replace("even_1", "odd_1")
                else:
                    target_str = target_str.replace("odd_1", "new_1")
            one_ret = abduce_one(context_str, target_str, pred_labels, kb_str, abducible_preds, prefix) 
            print(one_ret)
            input()
            # ret.append(one_ret)     
            context_list.append(context_str)
            target_list.append(target_str)
            predict_labels_list.append(pred_labels)
        ret = abduce_batch(kb_str=kb_str, context_list=context_list, target_list=target_list, predict_labels_list=predict_labels_list, abducible_preds=abducible_preds, prefix="_u", learned_program="")
        print(ret)

        time_end = time.time()
        print(time_end-time_start)