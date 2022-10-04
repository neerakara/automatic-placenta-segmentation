# import numpy as np
# import os
# import logging
# import torch
# import torch.nn as nn
# from unet_3d import UNet
# import torch.nn as nn
# import util
# from train_placenta import split_train_val
# import argparse
# import csv
# import postprocess
# import metrics
# from metrics import dice
# from medpy.metric.binary import assd as ASSD
# from medpy.metric.binary import hd as Hausdorff_Distance
# from medpy.metric.binary import hd95 as Hausdorff_Distance_95
# import utils_vis

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
from metrics import dice
from medpy.metric.binary import assd as ASSD
from medpy.metric.binary import hd as Hausdorff_Distance
from medpy.metric.binary import hd95 as Hausdorff_Distance_95
import utils_vis

HAUSDORFF_PCTILE = 95
IMG_DIR_NAME = 'volume'
LABEL_DIR_NAME = 'segmentation'
PAD_FACTOR = 16 #factor to make images divisible by.

# ===================================================
# EVALUATION FUNCTION
# ===================================================
def evaluate(model,
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
    model.eval()
    dice_score = 0
    hausd = 0
    hausd_pct = 0
    total_dice = 0
    total_hausd = 0
    sig = nn.Sigmoid()
    dices = []
    hausdorff = []
    num_island = []
    hausdorff_pct = []
    assd_list = []
    mean_bold_list = []
    assd = 0
    names = []
    voxel_to_mm = 3.

    with torch.no_grad():
        
        for batch in test:
        
            # ======================
            # run prediction
            # ======================
            logging.info("NEW BATCH")
            images = batch['img']['data']
            labels = batch['label']['data']
            fn = batch['fn']
            factor = batch['90_pct']
            low = batch['low']
            image_fn_path = batch['fn_img_path']
            img_affine = batch['affine']
            label_affine = batch['label_affine']
            subj_name = batch['subj_name']
            pad_amnt = batch['pad_amnt']

            # ======================
            # move data to gpu
            # ======================
            images = images.to(device,dtype=torch.float)
            labels = labels.to(device,dtype=torch.float)
            
            # ======================
            # make prediction
            # ======================
            outputs = model(images)
            predicted_probs = sig(outputs)

            # ======================
            # move data back to cpu
            # ======================
            images = images.to('cpu')
            predicted_probs = predicted_probs.to('cpu')
            labels = labels.to('cpu')

            # ======================
            # convert seg probabilities to binary prediction
            # ======================
            predicted = (predicted_probs > 0.5).float()

            if len(np.shape(predicted)) == 5:
                predicted = torch.squeeze(predicted, 1)
                predicted_probs = torch.squeeze(predicted_probs, 1)
                labels = torch.squeeze(labels, 1)
                images = torch.squeeze(images, 1)
            
            # ======================
            # convert to 0 / 1 labels
            # ======================
            for i in range(predicted.shape[0]):
            
                # ======================
                # clean the predicted image
                # ======================
                subject = subj_name[i]
                predicted_img = predicted[i].numpy()
                label = labels[i].numpy()
                img = images[i].numpy()
            
                # ======================
                # post process and clean
                # ======================
                if args.POST_PROCESS == 1:
                    cleaned_img = postprocess.remove_small_objects(img = predicted_img)
                    cleaned_img, _ = postprocess.remove_islands(img = cleaned_img)
                    predicted_img = cleaned_img
                total_dice += 1
                total_hausd +=1
            
                # ======================
                # print information
                # ======================
                logging.info("IMAGE: {}".format(fn[i]))
                d = dice(im1 = predicted_img,
                         im2 = label)

                # ======================
                # compute metrics
                # ======================
                if np.sum(predicted_img) > 0:
                    
                    hd = Hausdorff_Distance(result = predicted_img,
                                            reference = label,
                                            voxelspacing = voxel_to_mm)
                    
                    hd_pctile = Hausdorff_Distance_95(result = predicted_img,
                                                      reference = label,
                                                      voxelspacing = voxel_to_mm)
                    
                    assd_ind = ASSD(result = predicted_img,
                                    reference = label,
                                    voxelspacing = voxel_to_mm)
                else:
                    hd = np.nan
                    hd_pctile = np.nan
                    assd_ind = np.nan
            
                # ======================
                # compute relative mean BOLD difference
                # ======================
                bold_diff = metrics.mean_BOLD_difference(img_ref = util.unnormalize_img(img = img, low = low[i].numpy(), high = factor[i].numpy()),
                                                         label_ref = (label>0.5).astype(bool),
                                                         label_pred = predicted_img.astype(bool))

                dice_score += d
                hausd += hd
                hausd_pct += hd_pctile
                assd +=assd_ind
                logging.info("dice: {}".format(d))
                dices.append(d)
                names.append(fn[i])
                hausdorff.append(hd)
                hausdorff_pct.append(hd_pctile)
                assd_list.append(assd_ind)
                mean_bold_list.append(bold_diff)
                metrics_str = 'dice: ' + str(round(d, 3)) + \
                              ', hd: ' + str(round(hd, 3)) + \
                              ', hd95: ' + str(round(hausd_pct, 3)) + \
                              ', assd: ' + str(round(assd_ind, 3))

                # ======================
                # save the images - makes it better for later processing results.
                # ======================
                save_dir_subject = os.path.join(save_dir, subject)
            
                # ======================
                # make directories for image and segmentation
                # ======================
                if args.SAVE_NII == 1:
                    if not os.path.exists(os.path.join(save_dir_subject, 'image')):
                        os.makedirs(os.path.join(save_dir_subject, 'image'))
                    if not os.path.exists(os.path.join(save_dir_subject, 'predicted_segmentation')):
                        os.makedirs(os.path.join(save_dir_subject, 'predicted_segmentation'))
                    if not os.path.exists(os.path.join(save_dir_subject, 'true_segmentation')):
                        os.makedirs(os.path.join(save_dir_subject, 'true_segmentation'))
                
                img_pad_amnt = np.concatenate(pad_amnt[i].numpy(), axis=0)
            
                # ======================
                # unpad images
                # ======================
                img = util.unpad_img(img = util.unnormalize_img(img = img, low = low[i].numpy(), high = factor[i].numpy()), pad_amnt = img_pad_amnt)
                predicted_img = util.unpad_img(img = predicted_img, pad_amnt = img_pad_amnt)
                label = util.unpad_img(img = label, pad_amnt = img_pad_amnt)
            
                # ======================
                # save 3d prediction to subject folder
                # ======================
                if args.SAVE_NII == 1:
                    util.save_img(data = predicted_img,
                                path = os.path.join(save_dir_subject, 'predicted_segmentation'),
                                fn = 'predicted_segmentation_' + fn[i],
                                affine = img_affine[i])
                    util.save_img(data = img,
                                path = os.path.join(save_dir_subject, 'image'),
                                fn = fn[i],
                                affine = img_affine[i])
                    util.save_img(data = label,
                                path = os.path.join(save_dir_subject, 'true_segmentation'),
                                fn = 'true_segmentation_' + fn[i],
                                affine = label_affine[i])

                if args.SAVE_VIS == 1:
                    utils_vis.save_predictions_as_outlines(images = img,
                                                           preds = predicted_img,
                                                           labels = label,
                                                           title = metrics_str,
                                                           savepath = save_dir + '/vis_postprocess' + str(args.POST_PROCESS) + '_' + subject + '.png')

    logging.info('Average dice of the network on the test images: {}'.format(dice_score / total_dice))
    logging.info('Average hausdorff of the network on the test images: {}'.format(hausd / total_hausd))
    
    # ======================
    # write to file the dice scores per subject
    # ======================
    with open(os.path.join(save_dir, 'stats','stats_postprocess' + str(args.POST_PROCESS) + '.csv'), mode='w') as csv_file:
        csv_file = csv.writer(csv_file, delimiter=',')
        csv_file.writerow(['subj', 'dice', 'hausdorff', 'hausdorff_' + str(HAUSDORFF_PCTILE), 'assd', 'mean_bold_diff'])
        for j in range(len(names)):
            csv_file.writerow([names[j], dices[j], hausdorff[j], hausdorff_pct[j], assd_list[j], mean_bold_list[j]])
        csv_file.writerow(['Average', np.mean(dices), np.mean(hausdorff), np.mean(hausdorff_pct), np.mean(assd_list), np.mean(mean_bold_list)])
        csv_file.writerow(['Std', np.std(dices), np.std(hausdorff), np.std(hausdorff_pct), np.std(assd_list), np.std(mean_bold_list)])
    
# ===================================================
# MAIN FUNCTION
# ===================================================
def main(model_path,
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
    # load model and trained weights
    # ======================
    model = UNet(1)
    model = model.to(device)
    model, _, epoch, _ = util.load_checkpt(model_path, model)
    
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

    if test_only == 0:

        logging.info(' TRAINING SET SIZE: ' + str(len(train)))
        save_dir_train = os.path.join(save_dir, 'train')
        
        # ======================
        # create dir to save predictions of training data
        # ======================
        try:
            os.makedirs(os.path.join(save_dir_train, 'stats'))
        except OSError as error:
            logging.info(error)
        
        # ======================
        # evaluate training data
        # ======================
        evaluate(model = model,
                 test = train,
                 device = device,
                 save_dir = save_dir_train)             
        
        # ======================
        # create dir to save predictions of val data
        # ======================
        logging.info("START VAL")
        save_dir_val = os.path.join(save_dir, 'val')
        try:
            os.makedirs(os.path.join(save_dir_val, 'stats'))
        except OSError as error:
            logging.info(error)

        # ======================
        # evaluate val data
        # ======================
        evaluate(model = model,
                 test = val,
                 device = device,
                 save_dir = save_dir_val)
 
    # ======================
    # create dir to save predictions of test data
    # ======================
    logging.info("START TEST")
    save_dir_test = os.path.join(save_dir, 'test')
    try:
        os.makedirs(os.path.join(save_dir_test, 'stats'))
    except OSError as error:
        logging.info(error)
    
    # ======================
    # evaluate test data
    # ======================
    evaluate(model = model,
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
    
    parser.add_argument('--save_path',
                        dest = 'save_path',
                        default = '/data/scratch/nkarani/projects/qcseg/models/existing_folds_run4/', 
                        help = 'full path to location where experiment outputs will go')
    
    parser.add_argument('--model_name',
                        dest = 'model_name',
                        default = 'model_best_tmp')
    
    parser.add_argument('--data_path',
                        dest = 'data_path', 
                        default = '/data/vision/polina/projects/fetal/projects/placenta-segmentation/data/split-nifti-processed/')
    
    parser.add_argument('--eval_existing_folds',
                        type = int,
                        default = 1) # 1 / 0

    parser.add_argument('--test_only',
                        type = int,
                        default = 1) # 1 / 0

    parser.add_argument('--SAVE_NII',
                        type = int,
                        default = 0) # 1 / 0

    parser.add_argument('--SAVE_VIS',
                        type = int,
                        default = 1) # 1 / 0

    parser.add_argument('--POST_PROCESS',
                        type = int,
                        default = 1) # 1 / 0
    
    args = parser.parse_args()
    
    # ======================
    # paths
    # ======================
    model_folder = os.path.join(args.save_path, 'model') # need to get parent directory.
    model_path = os.path.join(model_folder, args.model_name + '.pt')
    save_path = os.path.join(args.save_path, 'results', args.model_name + '_output')
    img_dir = IMG_DIR_NAME
    label_dir = LABEL_DIR_NAME

    # ======================
    # define dict of subject names, 
    # depending on whether the evaluation has to be done on previously defined folds or not
    # ======================
    if args.eval_existing_folds == 1: 
        subj_folds = np.load(os.path.join(model_folder, 'model-folds.npy'), allow_pickle = 'TRUE').item()
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
    main(model_path = model_path,
         save_dir = save_path,
         data_path = args.data_path,
         img_dir = img_dir,
         label_dir = label_dir,
         subj_folds = subj_folds,
         test_only = test_only)
