import os
import argparse
import numpy as np
import random
import csv
import ast
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import torch
from NN_model_pytorch import LeNet
from NN_model_pytorch import Lenet_transform
from NN_model_pytorch import TorchDataset
import clingo_interface
import itertools
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pytod.models.lof import LOF
from pytod.utils.utility import validate_device
from ilasp_interface import build_ilasp_program, get_ilasp_result, read_bk_file, convert_ilasp_target, conflict_resolution
from similarity_calculator import nn_select_batch_abduced_result
from sklearn.model_selection import train_test_split



def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_device', default=1, type=int, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--task', choices=['less_than','chess','multiples_of_three'], default="less_than", type=str, help="task (less_than or chess")
    # parser.add_argument('--run', default=1, type=int, help='the run time for log file name, modify it every time' )
    parser.add_argument('--epochs', default=20, type=int, help='epochs for examples')
    # # Model
    # parser.add_argument('--num_pretrain_exs', default=0, type=int, help='number of examples for supervised pretraining')
    parser.add_argument('--num_pretrain_epoch', default=10, type=int, help='epoch for supervised pretraining')
    parser.add_argument('--pre_train_model_path', default=None, type=str, help="pre-train weights of CIFAR-10 for resnet")
    parser.add_argument('--nn_batch_size', default=256, type=int, help='batch size of nn training')
    parser.add_argument('--nn_fit_epoch', default=3, type=int, help='train epoch for nn update')
    # # Data
    parser.add_argument('--ex_each_batch', default=2048, type=int, help="the number of examples per batch in a epoch")
    # # Abduction
    parser.add_argument('--abduction_batch_size', default=128, type=int, help="number of examples used for each similarity-based abduction")
    parser.add_argument('--similar_coef', default=0.99, type=float, help="ratio of similarity scores, 1 means only similarity, 0 means only confidence, <0 means randomly selection")
    parser.add_argument('--beam_width', default=800, type=int, help="beam with for similarity-based abduction beam search")
    parser.add_argument('--backtrack_threshold', default=4, type=int, help="performance dropping times for backtracking abduction")
    parser.add_argument('--ilasp_timeout', default=1200, type=int, help="timeout for ILASP")
    args = parser.parse_args()
    return args

def get_label_dataset(imageset, known_class_list, label_rate = 1):
    known_label_idx_list = []
    for idx, data in enumerate(imageset):
        if data[1] in known_class_list:
            known_label_idx_list.append(idx)
    random.shuffle(known_label_idx_list)
    known_label_idx_list = known_label_idx_list[:int(label_rate*len(known_label_idx_list))]
    known_data = torch.utils.data.Subset(imageset, known_label_idx_list)
    return known_data

def split_list(for_split_list, splited_list):
    ret_list, cur_idx = [], 0
    for item in splited_list:
        ret_list.append(for_split_list[cur_idx : cur_idx + len(item)])
        cur_idx += len(item)
    assert (cur_idx == len(for_split_list))
    return ret_list

def get_images_list_dataset(imageset, images_list):
    image_idx_list = list(itertools.chain.from_iterable(images_list))
    dataset = torch.utils.data.Subset(imageset, image_idx_list)
    return dataset

def get_images_list_prediction(model, imageset, images_list, nn_batch_size):
    dataset = get_images_list_dataset(imageset, images_list)
    dataloader = DataLoader(dataset, batch_size=nn_batch_size, shuffle=False, num_workers=16, pin_memory=True)

    predict_probs, predict_features = model.predict(data_loader=dataloader)
    predict_probs, predict_features = predict_probs.cpu().numpy(), predict_features.cpu().numpy()
    predict_classes = predict_probs.argmax(axis=1)
    predict_classes_split_list = split_list(for_split_list=predict_classes, splited_list=images_list)
    predict_probs_split_list = split_list(for_split_list=predict_probs, splited_list=images_list)
    predict_fea_split_list = split_list(for_split_list=predict_features, splited_list=images_list)
    return predict_classes_split_list, predict_probs_split_list, predict_fea_split_list

def mix_label_list(predict_labels_list, new_class_prediction_list, new_class, cur_epoch):
    predict_labels = np.array(list(itertools.chain.from_iterable(predict_labels_list)))
    new_class_prediction = np.array(list(itertools.chain.from_iterable(new_class_prediction_list)))
    assert(len(predict_labels)==len(new_class_prediction))
    predict_labels[new_class_prediction==-1] = new_class
    mix_list = split_list(predict_labels, predict_labels_list)
    return mix_list

# If detect a new class and the example's mix label is inconsistent, then use the predicted label
def repair_mix_list(mix_list, consistence_list, predict_labels_list, new_class_prediction_list):
    for i in range(len(mix_list)):
        if (-1 in new_class_prediction_list[i]) and (consistence_list[i]==False):
            mix_list[i] = predict_labels_list[i]
    return mix_list

# Return the validation set's consistency
def validation_consistence(model, train_imageset, images_list, nn_batch_size, kb_str, context_list, target_list, abducible_preds, prefix, learned_program):
    predict_labels_list, _, _ = get_images_list_prediction(model, train_imageset, images_list, nn_batch_size)
    cnt, cons_perc, consistence_list = clingo_interface.check_label_consistence_batch(kb_str=kb_str, context_list=context_list, target_list=target_list, predict_labels_list=predict_labels_list, preds=abducible_preds, prefix=prefix, learned_program=learned_program)
    print("\nValidation Consistent count: %d/%d=%f"%(cnt, len(consistence_list), cons_perc))
    return cons_perc

def read_data_file(filename):
    context_list, target_list, images_list, labels_list = [], [], [], []
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            context, target, images, labels = row[0], row[1], ast.literal_eval(row[2]), ast.literal_eval(row[3])
            context_list.append(context)
            target_list.append(target)
            images_list.append(images)
            labels_list.append(labels)
    return context_list, target_list, images_list, labels_list

def train_model(args):
    task = args.task
    # run = args.run # The run time for log file name, modify it every time
    epochs = args.epochs # epochs for examples
    # num_pretrain_exs = args.num_pretrain_exs # number of examples for supervised pretraining
    num_pretrain_epoch = args.num_pretrain_epoch # epoch for supervised pretraining
    pre_train_model_path = args.pre_train_model_path #None # self-supervised weights of CIFAR-10 for resnet
    nn_batch_size = args.nn_batch_size # batch size of nn training
    nn_fit_epoch = args.nn_fit_epoch # train epoch for nn
    ex_each_batch = args.ex_each_batch # the number of examples per batch in a epoch
    abduction_batch_size = args.abduction_batch_size # number of examples used for each similarity-based abduction
    similar_coef = args.similar_coef # The ratio of similarity scores, 1 means only similarity, 0 means only confidence, <0 means randomly selection
    beam_width = args.beam_width # beam with for similarity-based abduction beam search
    backtrack_threshold = args.backtrack_threshold # performance dropping times for backtracking abduction
    ilasp_timeout = args.ilasp_timeout
    
    num_classes = 10
    prefix = "_u"
    data_file = task + "/data.csv"
    kb_file = task + "/target.lp"
    mode_file = task + "/mode_template.las"
    ilasp_path = './ILASP'
    clingo_path = '/usr/local/bin/clingo'
    max_penalty = 10000
    maxv, ml, max_rule_length = 3, 3, 3
    if task == "less_than":
        new_class = 3
        all_class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        known_class_list = all_class_list.copy()
        known_class_list.remove(new_class)
        inv_pred = 'new_1'
        org_preds = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    elif task == "chess":
        new_class = 4
        all_class_list = [0, 1, 2, 3, 4, 5]
        known_class_list = all_class_list.copy()
        known_class_list.remove(new_class)
        inv_pred = 'new_1'
        org_preds = ["king", "queen", "rook", "bishop", "knight", "pawn"]
    elif task == "multiples_of_three":
        all_class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        known_class_list = all_class_list
        inv_pred = 'new_1'
        known_pred_list = ['odd_1', 'even_1']
        org_preds = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    abducible_preds = org_preds.copy()
    if task == "less_than" or task == "chess":
        abducible_preds[new_class] = inv_pred
        exclude_pred_list = [org_preds[new_class]]
        ilasp_file = task + '/tmp/gen_learn_%d.las'%(new_class)
        use_new_class_detector = True
    elif task == "multiples_of_three":
        exclude_pred_list = [inv_pred]
        ilasp_file = task + '/tmp/gen_learn.las'
        use_new_class_detector = False
    ilasp_cmd_list = [ilasp_path, ilasp_file, '--version=4', '-ml=%d'%ml, '--max-rule-length=%d'%max_rule_length, '-nc', '-na', '-q']#, '--clingo', clingo_path]
    learned_program = ""

    # Read data and KB
    context_list, target_list, images_list, ground_labels_list = read_data_file(data_file)
    print("There are %d examples in total"%len(context_list))
    context_list, context_list_val, target_list, target_list_val, images_list, images_list_val, ground_labels_list, ground_labels_list_val = train_test_split(context_list, target_list, images_list, ground_labels_list, test_size=0.1)
    print("Split into %d training examples and %d validation examples"%(len(context_list), len(context_list_val)))
    kb_str = read_bk_file(kb_file, exclude_pred_list)

    # Model (CNN + new class detector)
    train_imageset = torchvision.datasets.MNIST(root='data', train=True, transform=Lenet_transform, download=True)
    test_imageset = torchvision.datasets.MNIST(root='data', train=False, transform=Lenet_transform, download=True)
    model = LeNet(num_class=num_classes, loss_criterion=torch.nn.CrossEntropyLoss(), batch_size=nn_batch_size).cuda()
    iforest = IsolationForest(n_estimators=1000, n_jobs=-1)
    lof = LocalOutlierFactor(novelty=True, metric='cosine', contamination=1/len(all_class_list), n_jobs=-1)
    new_class_detector = lof
    
    # Pretrain NN
    print("Pretraining CNN...")
    pretrain_label_rate = (0.05 if task == "multiples_of_three" else 1)
    train_known_data = get_label_dataset(train_imageset, known_class_list, pretrain_label_rate)
    print("Got %d pretain labeled data"%len(train_known_data))
    train_loader = DataLoader(train_known_data, batch_size=nn_batch_size, shuffle=False, num_workers=16, pin_memory=True)
    model.train_val(num_pretrain_epoch, is_train=True, data_loader=train_loader)
    # Test data loader
    test_all_data = get_label_dataset(test_imageset, all_class_list)
    test_loader = DataLoader(test_all_data, batch_size=nn_batch_size, shuffle=False, num_workers=16, pin_memory=True)
    # Test CNN
    image_test_loss, image_test_acc = model.train_val(1, False, data_loader=test_loader)
    cnn_acc_record_list = [image_test_acc]

    # Logger
    cons_perc_record_list = []
    cons_perc = validation_consistence(model, train_imageset, images_list_val, nn_batch_size, kb_str, context_list_val, target_list_val, abducible_preds, prefix, learned_program)
    cons_perc_record_list.append(cons_perc)
    
    for cur_epoch in range(1, epochs + 1):
        # Predict images' labels, probs, embeddings (splitted)
        print("Predicting images...")
        predict_labels_list, predict_probs_list, predict_fea_list = get_images_list_prediction(model, train_imageset, images_list, nn_batch_size)
        predict_embeddings = np.array(list(itertools.chain.from_iterable(predict_fea_list)))
        if use_new_class_detector:
            print("Detecting new class...")
            known_embeddings = model.predict(data_loader=train_loader)[1].cpu().numpy() # get known class embeddings
            new_class_detector.fit(known_embeddings)
            new_class_predictions = new_class_detector.predict(predict_embeddings)
            new_class_prediction_list = split_list(for_split_list=new_class_predictions, splited_list=images_list)
            mixed_label_list = mix_label_list(predict_labels_list, new_class_prediction_list, new_class, cur_epoch)
            if cur_epoch > 1:
                mixed_label_list = repair_mix_list(mix_list=mixed_label_list, consistence_list=consistence_list, predict_labels_list=predict_labels_list, new_class_prediction_list=new_class_prediction_list)
            use_new_class_detector = False
            print("---- Disable new class detector ----")
        else:
            mixed_label_list = predict_labels_list
        
        # Test consistence for mix prediction
        cnt, cons_perc, consistence_list = clingo_interface.check_label_consistence_batch(kb_str=kb_str, context_list=context_list, target_list=target_list, predict_labels_list=mixed_label_list, preds=abducible_preds, prefix=prefix, learned_program=learned_program)
        print("\nTraining mixed consistent count: %d/%d=%f"%(cnt, len(consistence_list), cons_perc))
        # for i in range(len(consistence_list)):
        #     if consistence_list[i]==False:
        #         print("Not consistent: ", ground_labels_list[i], predict_labels_list[i], new_class_prediction_list[i], mixed_label_list[i])
        
        # ILASP Learning
        confidence_list = [10 for i in range(len(context_list))]
        if task == "less_than" or task == "chess":
            ilasp_target_list = target_list
            ilasp_kb_str = kb_str
        elif task == "multiples_of_three":
            ilasp_target_list = convert_ilasp_target(target_list, inv_pred)
            ilasp_kb_str = read_bk_file(kb_file, exclude_pred_list+known_pred_list)
        program_str = build_ilasp_program(kb_str=ilasp_kb_str, mode_file=mode_file, inv_pred=inv_pred, confidence_list=confidence_list, target_str_list=ilasp_target_list, context_str_list=context_list, z_labels_list=mixed_label_list, preds=abducible_preds, exclude_pred_list=exclude_pred_list, prefix=prefix, max_penalty=max_penalty, maxv=maxv, out_file=ilasp_file)
        print("ILASP learning...")
        learned_program = get_ilasp_result(cmd_list=ilasp_cmd_list, inv_pred=inv_pred, mode_file=mode_file, timeout=ilasp_timeout)
        print("Learned program:\n", learned_program)

        # Check conflict rules (SAT)
        _, cons_perc, _ = clingo_interface.check_label_consistence_batch(kb_str=kb_str, context_list=[""], target_list=[""], predict_labels_list=[""], preds=abducible_preds, prefix=prefix, learned_program=learned_program)
        # Conflict Resolution
        if cons_perc < 1:
            print("\nConflict detected!\nStart conflict resolution...")
            refined_kb_str = conflict_resolution(kb_str, learned_program, known_pred_list, inv_pred)
        else:
            print("No conflict exists")
            refined_kb_str = kb_str
        # print(refined_kb_str)
        # Abduction
        print("\nAbducing labels...")
        abduced_list = clingo_interface.abduce_batch(kb_str=refined_kb_str, context_list=context_list, target_list=target_list, predict_labels_list=mixed_label_list, abducible_preds=abducible_preds, prefix=prefix, learned_program=learned_program)
        print("Selecting the best result according to similarity")
        if similar_coef >= 0:
            final_result = nn_select_batch_abduced_result(model=model, labeled_X=None, labeled_y=[], predict_probs_list=predict_probs_list, predict_fea_list=predict_fea_list, abduced_list=abduced_list, abduction_batch_size=abduction_batch_size, ground_labels_list=ground_labels_list, beam_width=beam_width, similar_coef=similar_coef)
        else:
            final_result = [res[random.randint(0,len(res)-1)] for res in abduced_list]

        epoch_abduce_correct_sign_cnt, epoch_sign_total = 0, 0
        batch_abduce_correct_sign_cnt, batch_sign_total = 0, 0
        for i in range(len(context_list)):
            cur_abduce_correct_cnt = sum(c1 == c2 for c1, c2 in zip(final_result[i], ground_labels_list[i]))
            batch_abduce_correct_sign_cnt += cur_abduce_correct_cnt
            batch_sign_total += len(ground_labels_list[i])

        epoch_abduce_correct_sign_cnt += batch_abduce_correct_sign_cnt
        epoch_sign_total += batch_sign_total

        # Retrain model
        final_result = np.concatenate(final_result, axis = 0)
        train_dataset = get_images_list_dataset(train_imageset, images_list)
        train_abduced_dataset = TorchDataset(images_np=None, label=final_result, transform=None, dataset=train_dataset)
        train_abduced_loader = DataLoader(train_abduced_dataset, batch_size=nn_batch_size, shuffle=False, num_workers=16, pin_memory=True)
        model.train_val(nn_fit_epoch, is_train=True, data_loader=train_abduced_loader)
        # Test CNN
        image_test_loss, image_test_acc = model.train_val(1, False, data_loader=test_loader)
        cnn_acc_record_list.append(image_test_acc)
        # If the consistency of validation set does not increase for several times, then backtrace to previous checkpoint
        cons_perc = validation_consistence(model, train_imageset, images_list_val, nn_batch_size, refined_kb_str, context_list_val, target_list_val, abducible_preds, prefix, learned_program)
        cons_perc_record_list.append(cons_perc)
        print("Perception acc:", cnn_acc_record_list)
        print("Reasoning acc:", cons_perc_record_list)
        if len(cons_perc_record_list)-1-np.argwhere(cons_perc_record_list==np.amax(cons_perc_record_list)).flatten()[-1] >= backtrack_threshold:
            print("---- Stop ----")
            break
    


if __name__ == "__main__":
    # Parameters
    args = arg_init()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    print(args)

    train_model(args)