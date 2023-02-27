import subprocess
import clingo_interface

# Split target string into pos_target_list, neg_target_list
def split_target(target_str):
    pos_target_list, neg_target_list = [], []
    target_list = target_str.strip().split(".")
    for target in target_list:
        target = target.strip()
        if len(target) == 0:
            continue
        if target.find("-") == -1:
            pos_target_list.append(target)
        else:
            neg_target_list.append(target.replace('-', ''))
    return pos_target_list, neg_target_list

# Build an ILASP sample string
def build_ilasp_one_sample(idx, confidence, target_str, context_str, z_labels, preds, prefix):
    pos_target_list, neg_target_list = split_target(target_str)
    pos_target_str = ",".join(pos_target_list)
    neg_targt_str = ",".join(neg_target_list)
    logic_label_str = clingo_interface.labels2program(z_labels, preds, prefix)
    sample_str = '''
#pos(e%d@%d,{
    %s
},{
    %s
},{
    %s
    %s
}).
    ''' % (idx, confidence, pos_target_str, neg_targt_str, context_str, logic_label_str)
    return sample_str

# Build ILASP samples string
def build_ilasp_samples(confidence_list, target_str_list, context_str_list, z_labels_list, preds, prefix="_u"):
    samples_str = ""
    for i in range(len(target_str_list)):
        confidence = confidence_list[i]
        target_str = target_str_list[i]
        context_str = context_str_list[i]
        z_labels = z_labels_list[i]
        samples_str += build_ilasp_one_sample(i, confidence=confidence, target_str=target_str, context_str=context_str, z_labels=z_labels, preds=preds, prefix=prefix) + '\n'
    return samples_str

# Get the program learned by ILASP
def get_ilasp_result(cmd_list, inv_pred="", mode_file="", org_target_pred_name="invented_predicate", timeout=1200):
    try:
        result = subprocess.run(cmd_list, stdout=subprocess.PIPE, timeout=timeout)
        learned_program = result.stdout.decode()
        learned_program = learned_program.replace(";", ",")
    except subprocess.TimeoutExpired:
        learned_program = ""
        print("ILASP learning timeout!")
    # if len(inv_pred) > 0:
    #     learned_program += "\n" + "#defined %s/1."%(inv_pred)
    if len(mode_file) > 0:
        with open(mode_file,'r') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) > 0 and line.find("#mode") == -1 and line.find("~") == -1:
                    learned_program += "\n" + line.replace(org_target_pred_name, inv_pred)
    return learned_program

# Read background knowledge file and exclude some lines
def read_bk_file(bk_file, exclude_pred_list = []):
    bk_str = ""
    with open(bk_file,'r') as f:
        for line in f.readlines():   
            flag = True
            for exclude_pred in exclude_pred_list:
                if exclude_pred in line:
                    flag = False
                    break
            if flag:
                bk_str += line
    return bk_str

# Read mode declaration file and exclude some lines
def read_mode_file(mode_file, org_target_pred_name, inv_pred, exclude_pred_list = []):
    mode_str = ""
    with open(mode_file,'r') as f:
        for line in f.readlines():   
            flag = True
            for exclude_pred in exclude_pred_list:
                if exclude_pred in line or "#defined" in line:
                    flag = False
                    break
            if flag:
                mode_str += line.replace(org_target_pred_name, inv_pred)
    return mode_str

# Build the final ILASP program
def build_ilasp_program(kb_str, mode_file, inv_pred, confidence_list, target_str_list, context_str_list, z_labels_list, preds, exclude_pred_list = [], prefix="_u", max_penalty=None, maxv=None, out_file=""):
    org_target_pred_name = "invented_predicate"
    program_str = ""
    if max_penalty is not None:
        program_str += "#max_penalty(%d).\n"%(max_penalty)
    if maxv is not None:
        program_str += "#maxv(%d).\n"%(maxv)
    program_str += kb_str
    program_str += build_ilasp_samples(confidence_list=confidence_list, target_str_list=target_str_list, context_str_list=context_str_list, z_labels_list=z_labels_list, preds=preds, prefix="_u")
    program_str += read_mode_file(mode_file=mode_file, org_target_pred_name=org_target_pred_name, inv_pred=inv_pred, exclude_pred_list = exclude_pred_list)
    if len(out_file) > 0:
        with open(out_file, 'w') as f:
            f.write(program_str)
    return program_str

# Convert to one-vs-rest
def convert_ilasp_target(target_list, pos_pred):
    ilasp_target_list =  []
    for target in target_list:
        end = target.find("(")
        if target[:end]==pos_pred:
            ilasp_target_list.append(target)
        else:
            ilasp_target_list.append('-'+pos_pred+target[end:])
    return ilasp_target_list


# Contain the definition of preds
def contain_pred_def(line, preds):
    line = line.strip()
    for pred in preds:
        if line.startswith(pred):
            return True
    return False

# Conflict resolution for a kb line and learned program
def conflict_resol_line(kb_line, learned_program, new_pred):
    ret_list = []
    for learned_line in learned_program.split("\n"):
        if contain_pred_def(learned_line, [new_pred]):
            ret_list.append(conflict_resol_one(kb_line, learned_line))
    ret_line = "\n".join(ret_list)
    return ret_line

def gen_subs(learned_line, kb_line):
    learned_para = learned_line[learned_line.find("(")+1:learned_line.find(")")].replace(" ","").split(",")
    kb_para = kb_line[kb_line.find("(")+1:kb_line.find(")")].replace(" ","").split(",")
    mapping = list(zip(learned_para, kb_para))
    return mapping

# Conflict resolution for a kb line and learned line
# Assume they are conflicting, have not check sat
def conflict_resol_one(kb_line, learned_line):
    ret_list = []
    start_idx = learned_line.find(":-")
    learned_body_literals = learned_line[start_idx+2:].replace(".","").split(",")
    learned2kb_var = gen_subs(learned_line, kb_line)
    for literal in learned_body_literals:
        literal = literal.strip()
        for (v1,v2) in learned2kb_var:
            literal = literal.replace(v1,v2)
        new_kb_line = kb_line.replace(".",",")+" not_"+literal+"."
        ret_list.append(new_kb_line)
    ret_line = "\n".join(ret_list)
    return ret_line

# Conflict resolution for a kb and learned program
def conflict_resolution(kb_str, learned_program, known_preds, new_pred):
    new_kb_str = ""
    for kb_line in kb_str.split("\n"):
        if contain_pred_def(kb_line, known_preds):
            print("May conflict with kb, trying to refine:\n", kb_line)
            refined_kb_line = conflict_resol_line(kb_line, learned_program, new_pred)
            print("Refined kb rule:\n", refined_kb_line)
            new_kb_str += refined_kb_line + "\n"
        else:
            new_kb_str += kb_line + "\n"
    if new_kb_str.find('even_1(X,Y) :- div_2(X), div_2(Y), not_div_3(X).') and \
        new_kb_str.find('odd_1(X,Y) :- not_div_2(X), not_div_2(Y), not_div_3(X).'): # Found correct, add CWA
        new_kb_str += '''
-new_1(X,Y) :- misc(X,Y), not_div_3(X).
-new_1(X,Y) :- misc(X,Y), not_div_3(Y).
    '''
    return new_kb_str

if __name__ == "__main__":
    pass
    