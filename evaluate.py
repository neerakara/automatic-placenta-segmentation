# ===================================================
# evaluate using multiple models
# compute voxel-wise mean and variance of predictions
# compute dice (mean, ground truth)
# scatter plot of dice (mean, ground truth) v/s sum_over_all_voxels(variance)
# ===================================================
import util
from train_placenta import split_train_val
from unet_3d import UNet
import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import logging
import csv
import postprocess
import metrics
import utils_vis

HAUSDORFF_PCTILE = 95
IMG_DIR_NAME = 'volume'
LABEL_DIR_NAME = 'segmentation'
PAD_FACTOR = 16 # factor to make images divisible by.

# ======================
# function to plot variance in multiple predictions vs accuracy metrics
# ======================
def visualize_scatter_plots(save_dir,
                            arr_dice,
                            arr_hd,
                            arr_hd95,
                            arr_assd,
                            arr_var):

    utils_vis.plot_scatter(arr_var, arr_dice, 'dice', save_dir + '/var_vs_dice.png')
    utils_vis.plot_scatter(arr_var, arr_hd, 'hd', save_dir + '/var_vs_hd.png')
    utils_vis.plot_scatter(arr_var, arr_hd95, 'hd95', save_dir + '/var_vs_hd95.png')
    utils_vis.plot_scatter(arr_var, arr_assd, 'assd', save_dir + '/var_vs_assd.png')

    return 0

# ===================================================
# function to write obtained results to file
# ===================================================
def write_results_to_file(save_dir,
                          array_scores, # [num_subjects, num_models + 1, num_metrics]
                          array_variances, # num_subjects
                          array_names, # num_subjects
                          num_decimals):
    
    # visualize scatter plot for the ensemble predictions
    visualize_scatter_plots(save_dir,
                            array_scores[:,-1,0],
                            array_scores[:,-1,1],
                            array_scores[:,-1,2],
                            array_scores[:,-1,3],
                            array_variances)

    with open(os.path.join(save_dir,
                           'stats',
                           'stats_postprocess' + str(args.POST_PROCESS) + '.csv'), mode='w') as csv_file:
        
        csv_file = csv.writer(csv_file, delimiter=',')
        
        csv_file.writerow(['subj',
                           'dice',
                           'hausdorff',
                           'hausdorff_' + str(HAUSDORFF_PCTILE),
                           'assd',
                           'mean_bold_diff',
                           'variance',
                           'model'])
        
        # for each model
        for m in range(array_scores.shape[1]):
            csv_file.writerow(['--------', '--------', '--------', '--------', '--------', '--------', '--------', '--------'])
            if m == array_scores.shape[1] - 1:
                m_write = 'ensemble'
                var_mean = 'n/a'
                var_std = 'n/a'
            else:
                var_mean = 'n/a'
                var_std = 'n/a'
                m_write = 'model ' + str(m)

            # for each subject
            for j in range(array_names.shape[0]):
                csv_file.writerow([array_names[j],
                                   np.round(array_scores[j,m,0], num_decimals),
                                   np.round(array_scores[j,m,1], num_decimals),
                                   np.round(array_scores[j,m,2], num_decimals),
                                   np.round(array_scores[j,m,3], num_decimals),
                                   np.round(array_scores[j,m,4], num_decimals),
                                   np.round(array_variances[j], num_decimals),
                                   m_write])
        
            csv_file.writerow(['Average',
                                np.round(np.mean(array_scores[:,m,0]), num_decimals),
                                np.round(np.mean(array_scores[:,m,1]), num_decimals),
                                np.round(np.mean(array_scores[:,m,2]), num_decimals),
                                np.round(np.mean(array_scores[:,m,3]), num_decimals),
                                np.round(np.mean(array_scores[:,m,4]), num_decimals),
                                var_mean,
                                m_write])

            csv_file.writerow(['Std',
                                np.round(np.std(array_scores[:,m,0]), num_decimals),
                                np.round(np.std(array_scores[:,m,1]), num_decimals),
                                np.round(np.std(array_scores[:,m,2]), num_decimals),
                                np.round(np.std(array_scores[:,m,3]), num_decimals),
                                np.round(np.std(array_scores[:,m,4]), num_decimals),
                                var_std,
                                m_write])

        csv_file.writerow(['--------', '--------', '--------', '--------', '--------', '--------', '--------', '--------'])

    return 0

# ===================================================
# ===================================================
def save_preds_as_nii(save_dir_subject,
                      image,
                      label,
                      preds,
                      pred_suffix,
                      fn,
                      img_affine,
                      lbl_affine):
                
    # make directories
    if not os.path.exists(os.path.join(save_dir_subject, 'image')):
        os.makedirs(os.path.join(save_dir_subject, 'image'))
    if not os.path.exists(os.path.join(save_dir_subject, 'predicted_segmentation')):
        os.makedirs(os.path.join(save_dir_subject, 'predicted_segmentation'))
    if not os.path.exists(os.path.join(save_dir_subject, 'true_segmentation')):
        os.makedirs(os.path.join(save_dir_subject, 'true_segmentation'))
    
    # save 3d prediction to subject folder
    util.save_img(data = preds,
                  path = os.path.join(save_dir_subject, 'predicted_segmentation'),
                  fn = 'predicted_segmentation_' + pred_suffix + fn,
                  affine = img_affine)

    # save 3d image to subject folder
    util.save_img(data = image,
                  path = os.path.join(save_dir_subject, 'image'),
                  fn = fn,
                  affine = img_affine)

    # save 3d ground truth label to subject folder
    util.save_img(data = label,
                  path = os.path.join(save_dir_subject, 'true_segmentation'),
                  fn = 'true_segmentation_' + fn,
                  affine = lbl_affine)

    return 0

# ===================================================
# ===================================================
def compute_metrics_and_save_vis_for_this_pred(image,
                                               unnormalized_image,
                                               unpadded_unnormalized_image,
                                               label,
                                               unpadded_label,
                                               preds,
                                               unpadded_preds,
                                               voxelspacing,
                                               save_dir,
                                               subject,
                                               fn,
                                               affines,
                                               model_num):
    # ======================
    # compute metrics
    # ======================
    scores = metrics.compute_metrics(image = unnormalized_image,
                                     result = preds,
                                     reference = label,
                                     voxelspacing = voxelspacing)

    # ======================
    # save results as png files
    # ======================
    if args.SAVE_VIS == 1:

        metrics_str = 'dice: ' + str(round(scores[0], 3)) + \
                      ', hd: ' + str(round(scores[1], 3)) + \
                      ', hd95: ' + str(round(scores[2], 3)) + \
                      ', assd: ' + str(round(scores[3], 3))

        vis_suffix = '_' + subject + '_' + fn[:fn.find('.')]

        if model_num == -1: # ensemble prediction
            savepath = save_dir + '/vis_postprocess' + str(args.POST_PROCESS) + vis_suffix + '_ensemble.png'
        else: # prediction of a model "mdoel_num"
            savepath = savepath = save_dir + '/vis_postprocess' + str(args.POST_PROCESS) + vis_suffix + '_model' + str(model_num) + '.png'

        utils_vis.save_predictions_as_outlines(images = unpadded_unnormalized_image,
                                               preds = unpadded_preds,
                                               labels = unpadded_label,
                                               title = metrics_str,
                                               savepath = savepath)

    # ======================
    # save results as nii files
    # ======================
    if args.SAVE_NII == 1:
        
        if model_num == -1: # ensemble prediction
            pred_suffix =  '_ensemble_'
        else: # prediction of a model "mdoel_num"
            pred_suffix = '_model' + str(model_num) + '_'

        save_preds_as_nii(save_dir_subject = os.path.join(save_dir, subject),
                          image = image,
                          label = label,
                          preds = preds,
                          pred_suffix = pred_suffix,
                          fn = fn,
                          img_affine = affines[0],
                          lbl_affine = affines[1])

    return scores

# ==================================================================
# ==================================================================
def convert_soft_to_hard(probs):
    hard_seg = (probs > 0.5).astype(np.float32)
    if args.POST_PROCESS == 1:
        hard_seg = postprocess.postprocess_segmentation(seg = hard_seg)
    return hard_seg

# ==================================================================
# compute metrics and save visualizations
# this function gets soft predictions from multiple models
# ==================================================================
def compute_metrics_and_save_vis_for_this_image(multiple_predicted_probs,
                                                image,
                                                unnormalized_image,
                                                label,
                                                voxelspacing,
                                                img_pad_amnt,
                                                save_dir,
                                                subject,
                                                affines,
                                                fn):

    # unpad image and ground truth label
    unpadded_unnormalized_image = util.unpad_img(unnormalized_image, pad_amnt = img_pad_amnt)
    unpadded_label = util.unpad_img(img = label, pad_amnt = img_pad_amnt)
            
    scores_multiple_models = []

    # ======================
    # evals for individual models
    # ======================
    for model_num in range(multiple_predicted_probs.shape[-1]):

        this_model_predicted_seg = convert_soft_to_hard(probs = multiple_predicted_probs[..., model_num])

        scores_this_model = compute_metrics_and_save_vis_for_this_pred(image = image,
                                                                       unnormalized_image = unnormalized_image,
                                                                       unpadded_unnormalized_image = unpadded_unnormalized_image,
                                                                       label = label,
                                                                       unpadded_label = unpadded_label,
                                                                       preds = this_model_predicted_seg,
                                                                       unpadded_preds = util.unpad_img(img = this_model_predicted_seg, pad_amnt = img_pad_amnt),
                                                                       voxelspacing = voxelspacing,
                                                                       save_dir = save_dir,
                                                                       subject = subject,
                                                                       affines = affines,
                                                                       fn = fn,
                                                                       model_num = model_num)
        scores_multiple_models.append(scores_this_model)

    # ======================
    # evals for ensemble model
    # ======================
    # take voxel-wise mean and variance of multiple predictions
    predicted_probs_mean = np.mean(multiple_predicted_probs, axis=-1)
    predicted_probs_var = np.var(multiple_predicted_probs, axis=-1)
    # convert seg probabilities to binary prediction
    ensemble_predicted_seg = convert_soft_to_hard(probs = predicted_probs_mean)

    scores_ensemble = compute_metrics_and_save_vis_for_this_pred(image = image,
                                                                 unnormalized_image = unnormalized_image,
                                                                 unpadded_unnormalized_image = unpadded_unnormalized_image,
                                                                 label = label,
                                                                 unpadded_label = unpadded_label,
                                                                 preds = ensemble_predicted_seg,
                                                                 unpadded_preds = util.unpad_img(img = ensemble_predicted_seg, pad_amnt = img_pad_amnt),
                                                                 voxelspacing = voxelspacing,
                                                                 save_dir = save_dir,
                                                                 subject = subject,
                                                                 affines = affines,
                                                                 fn = fn,
                                                                 model_num = -1)
    scores_multiple_models.append(scores_ensemble)

    # ======================
    # visualize voxel-wise variance in predictions
    # ======================
    total_variance_across_image = np.sum(predicted_probs_var)

    return np.array(scores_multiple_models), total_variance_across_image
                                                                            
# ===================================================
# EVALUATION FUNCTION
# ===================================================
def evaluate(model,
             model_paths,
             test,
             device,
             save_dir):
    """
    Evaluates the given model on the given dataset. 

    Params:
    model: the model to evaluate
    test: the test dataset
    device: the device to run the evaluation on
    save_dir: the path to the directory where the results should be saved
    """
    
    # Set model to eval mode
    model.eval()

    # initialize lists for storing results of individual subjects
    list_scores = []
    list_var = []   
    list_names = []
    
    # required for computing distance based metrics
    voxel_to_mm = 3.0

    # num_decimals while writing results to file
    num_decimals = 3

    # counter used to stop eval after some number of evals
    image_counter = 1

    with torch.no_grad():
        
        for batch in test:
        
            # extract different elements of the batch
            logging.info("NEW BATCH")
            images = batch['img']['data']
            labels = batch['label']['data']
            fn = batch['fn']
            factor = batch['90_pct']
            low = batch['low']
            img_affine = batch['affine']
            label_affine = batch['label_affine']
            subj_name = batch['subj_name']
            pad_amnt = batch['pad_amnt']

            # move images and labels to gpu
            images = images.to(device, dtype = torch.float)
            labels = labels.to(device, dtype = torch.float)

            # load different models one by one and make predictions
            model_counter = 1
            for model_path in model_paths:
                # load this model
                model = util.load_checkpt(model_path, model)[0]
                # make prediction
                outputs = model(images)
                predicted_probs = nn.Sigmoid()(outputs)
                # move predictions to cpu
                predicted_probs = predicted_probs.to('cpu').numpy()
                predicted_probs = np.expand_dims(np.squeeze(predicted_probs), axis=-1)
                if model_counter == 1:
                    multiple_predicted_probs = predicted_probs
                else:
                    multiple_predicted_probs = np.concatenate((multiple_predicted_probs, predicted_probs), axis=-1)
                model_counter = model_counter + 1

            # move data back to cpu and to numpy, and remove redundant dims
            images = np.squeeze(images.to('cpu').numpy())
            labels = np.squeeze(labels.to('cpu').numpy())
            
            # assuming batch size is set to 1
            i=0
            subject = subj_name[i]
            label = labels
            image = images
            logging.info("IMAGE: {}".format(fn[i]))

            # evaluate predictions of each model individually, and the ensemble prediction
            scores, var = compute_metrics_and_save_vis_for_this_image(multiple_predicted_probs = multiple_predicted_probs,
                                                                      image = image,
                                                                      unnormalized_image = util.unnormalize_img(img = image,
                                                                                                                low = low[i].numpy(),
                                                                                                                high = factor[i].numpy()),
                                                                      label = label,
                                                                      voxelspacing = voxel_to_mm,
                                                                      img_pad_amnt = np.concatenate(pad_amnt[i].numpy(), axis=0),
                                                                      save_dir = save_dir,
                                                                      subject = subject,
                                                                      affines = [img_affine[i], label_affine[i]],
                                                                      fn = fn[i])

            # len(scores) --> 6
            # scores[i] --> np.array of shape 5
            # scores[i] = [dice, hd, hd95, assd, bold_mean]
            # append metrics of this subject to lists
            list_scores.append(scores)        
            list_names.append(fn[i])    
            list_var.append(var)  

            # increment image counter and move forward in life
            image_counter = image_counter + 1
            if args.test_with_few_images == 1 and image_counter > 2:
                break                                                       
        
    # write to file the dice scores per subject
    # also, plot variance in multiple predictions vs accuracy metrics
    write_results_to_file(save_dir,
                          np.array(list_scores), # [num_subjects, num_models + 1, num_metrics]
                          np.array(list_var), # num_subjects
                          np.array(list_names), # num_subjects
                          num_decimals)
    
# ===================================================
# MAIN FUNCTION
# ===================================================
def main(model_paths,
         save_dir,
         data_path,
         img_dir,
         label_dir,
         subj_folds,
         test_only=False):
    """
    sets up the model to be evaluated, and evaluates the model on both the train and test
    datasets. 

    Params:
    model_path: the path to the saved model file to be loaded in
    save_dir: the path to the directory where results should be saved
    data_path: the path to the directory where data is saved
    img_dir: subjdirectory for the images
    label_dir: subdirectory for the labels
    subj_folds: a dictionary with the text files for subject folds (train/val/test)
    test_only: bool. whether to only create a test set or not
    """
    
    pad_factor = PAD_FACTOR
    device = torch.device("cuda" if torch.cuda.is_available() 
                                 else "cpu")
    # device = torch.device("cpu")
        
    # ======================
    # load data
    # ======================
    data = split_train_val(data_dir = data_path,
                           img_dir = img_dir,
                           label_dir = label_dir,
                           transforms_string = "none",
                           data_split = subj_folds,
                           batch_size = 1,
                           pad_factor = pad_factor,
                           randomize_img_dataloader = False,
                           aug_severity = 0,
                           store_images = False,
                           test_only = test_only)

    train = data.train
    test = data.test
    val = data.val

    # ======================
    # define model
    # ======================
    model = UNet(1)
    model = model.to(device)

    if test_only == 0:
        
        # create dir to save predictions of training data
        logging.info(' TRAINING SET SIZE: ' + str(len(train)))
        save_dir_train = os.path.join(save_dir, 'train')
        try:
            os.makedirs(os.path.join(save_dir_train, 'stats'))
        except OSError as error:
            logging.info(error)
        
        # evaluate training data
        evaluate(model = model,
                 model_paths = model_paths,
                 test = train,
                 device = device,
                 save_dir = save_dir_train)             

        # create dir to save predictions of val data
        logging.info("START VAL")
        save_dir_val = os.path.join(save_dir, 'val')
        try:
            os.makedirs(os.path.join(save_dir_val, 'stats'))
        except OSError as error:
            logging.info(error)

        # evaluate val data
        evaluate(model = model,
                 model_paths = model_paths,
                 test = val,
                 device = device,
                 save_dir = save_dir_val)
 
    # create dir to save predictions of test data
    logging.info("START TEST")
    save_dir_test = os.path.join(save_dir, 'test')
    try:
        os.makedirs(os.path.join(save_dir_test, 'stats'))
    except OSError as error:
        logging.info(error)
    
    # evaluate test data
    evaluate(model = model,
             model_paths = model_paths,
             test = test,
             device = device,
             save_dir = save_dir_test)


# ===================================================
# BODY
# ===================================================
if __name__ == '__main__':

    # ======================
    # parse arguments
    # ======================
    parser = argparse.ArgumentParser(description = 'evaluate trained unet model')
    
    parser.add_argument('--models_basepath',
                        dest = 'models_basepath',
                        default = '/data/scratch/nkarani/projects/qcseg/models/existing_folds_new/', 
                        help = 'basepath where directories of individual runs exist')
    
    parser.add_argument('--model_name',
                        dest = 'model_name',
                        default = 'model_best')
    
    parser.add_argument('--data_path',
                        dest = 'data_path', 
                        default = '/data/vision/polina/projects/fetal/projects/placenta-segmentation/data/split-nifti-processed/')

    # NEW FOLDS EXIST HERE /data/vision/polina/projects/fetal/projects/placenta-segmentation/data/split-nifti-processed/
    # OLD FOLDS EXISTS HERE /data/vision/polina/projects/fetal/projects/placenta-segmentation/data/PIPPI_Data/split-nifti-processed/
    # You've trained multiple models with the new folds

    parser.add_argument('--eval_existing_folds',
                        type = int,
                        default = 1) # 1 / 0

    parser.add_argument('--test_only',
                        type = int,
                        default = 1) # 1 / 0

    parser.add_argument('--SAVE_NII',
                        type = int,
                        default = 1) # 1 / 0

    parser.add_argument('--SAVE_VIS',
                        type = int,
                        default = 1) # 1 / 0

    parser.add_argument('--POST_PROCESS',
                        type = int,
                        default = 0) # 1 / 0

    parser.add_argument('--test_with_few_images',
                        type = int,
                        default = 0) # 1 / 0
    
    args = parser.parse_args()
    
    # ======================
    # paths
    # ======================
    model_paths = [args.models_basepath + 'run0/model/' +  args.model_name + '.pt',
                   args.models_basepath + 'run1/model/' +  args.model_name + '.pt',
                   args.models_basepath + 'run2/model/' +  args.model_name + '.pt',
                   args.models_basepath + 'run3/model/' +  args.model_name + '.pt',
                   args.models_basepath + 'run4/model/' +  args.model_name + '.pt']
    save_path = args.models_basepath + 'results/' + args.model_name + '_output'
    img_dir = IMG_DIR_NAME
    label_dir = LABEL_DIR_NAME

    # ======================
    # define dict of subject names, 
    # depending on whether the evaluation has to be done on previously defined folds or not
    # ======================
    if args.eval_existing_folds == 1: 
        subj_folds = np.load(args.models_basepath + 'run0/model/model-folds.npy', allow_pickle = 'TRUE').item()
        test_only = args.test_only
    else:
        subj_folds = dict()
        # create a new dataset based on the data directory
        dir_list = util.listdir_nohidden_sort_numerical(path = args.data_path,
                                                        list_dir = True,
                                                        sort_digit = True)
        subj_folds['test'] = dir_list
        subj_folds['train'] = []
        subj_folds['val'] = []
        test_only = 1

    # ======================    
    # print folds
    # ======================
    logging.info(subj_folds)

    # ======================
    # Call main function
    # ======================
    main(model_paths = model_paths,
         save_dir = save_path,
         data_path = args.data_path,
         img_dir = img_dir,
         label_dir = label_dir,
         subj_folds = subj_folds,
         test_only = test_only)
