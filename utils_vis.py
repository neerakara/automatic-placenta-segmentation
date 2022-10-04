# ===============================================================
# visualization functions
# ===============================================================
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import numpy as np
import logging
import skimage.segmentation

# ==========================================================
# ==========================================================
def save_single_image(image,
                      savepath,
                      nlabels=3,
                      add_pixel_each_label=False,
                      cmap='gray',
                      colorbar=False,
                      climits = [],
                      dpi = 100):
        
    plt.figure(figsize=[20,20])            
    
    if add_pixel_each_label:
        image = add_1_pixel_each_class(image, nlabels)
                
    plt.imshow(image, cmap=cmap)

    if climits != []:
        plt.clim([climits[0], climits[1]])
    plt.axis('off')
    
    if colorbar:
        plt.colorbar()
    
    plt.savefig(savepath, bbox_inches='tight', pad_inches = 0, dpi = dpi) # 
    plt.margins(0,0)
    plt.close()

# ==========================================================
# ==========================================================
def save_8_2d_image_label_pairs(images,
                                labels,
                                savepath,
                                colorbar=False,
                                climits = [],
                                dpi = 100):

    plt.figure(figsize=[80,20])            

    for idx in range(images.shape[0]):

        plt.subplot(2, 8, idx + 1)                
        plt.imshow(images[idx,:,:], cmap='gray')
        plt.colorbar()

        plt.subplot(2, 8, idx + 9)                
        plt.imshow(labels[idx,:,:], cmap='tab20')
        plt.colorbar()
    
    plt.savefig(savepath, bbox_inches='tight', pad_inches = 0, dpi = dpi) # 
    plt.margins(0,0)
    plt.close()

# ==========================================================
# ==========================================================
def save_predictions(images,
                     preds,
                     labels,
                     title,
                     savepath,
                     dpi = 100):

    # find slice with largest number of fg pixels in the ground truth
    num_pixels_fg = np.sum(labels, axis = (0,1))
    zz_largest_fg = np.argmax(num_pixels_fg)
    vis_indices = np.linspace(zz_largest_fg - 16, zz_largest_fg + 16, 8, dtype=int)

    plt.figure(figsize=[80,30])            
    
    for idx in range(images.shape[-1] // 10):

        plt.subplot(3, 8, idx + 1)                
        plt.imshow(images[:,:,vis_indices[idx]], cmap='gray')
        plt.title('image, z = ' + str(vis_indices[idx]))
        plt.colorbar()

        plt.subplot(3, 8, idx + 9)                
        plt.imshow(preds[:,:,vis_indices[idx]], cmap='tab20')
        plt.title('prediction, z = ' + str(vis_indices[idx]))
        plt.colorbar()

        plt.subplot(3, 8, idx + 17)                
        plt.imshow(labels[:,:,vis_indices[idx]], cmap='tab20')
        plt.title('ground truth, z = ' + str(vis_indices[idx]))
        plt.colorbar()
    
    plt.suptitle(title, fontsize=100)
    plt.savefig(savepath, bbox_inches='tight', pad_inches = 0, dpi = dpi) # 
    plt.margins(0,0)
    plt.close()

# ==========================================================
# ==========================================================
def save_predictions_as_outlines(images,
                                 preds,
                                 labels,
                                 title,
                                 savepath,
                                 dpi = 200):

    # visualize 5 equally spaced slices in which the placenta exists (leaving a few slices on either end)
    num_slices_vis = 5
    num_pixels_fg = np.sum(labels, axis = (0,1))
    placenta_slices = np.nonzero(num_pixels_fg)
    vis_indices = np.linspace(placenta_slices[0][0] + 4, placenta_slices[0][-1] - 4, num_slices_vis, dtype=int)

    plt.figure(figsize = [num_slices_vis * 10, 10])            
    for idx in range(num_slices_vis):

        img_slice = np.copy(images[:,:,vis_indices[idx]])
        img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))
        img_slice = 255 * img_slice

        pre_slice = preds[:,:,vis_indices[idx]]
        lbl_slice = labels[:,:,vis_indices[idx]]

        # Find external contours
        pre_contours = 1*skimage.segmentation.find_boundaries(pre_slice)
        lbl_contours = 1*skimage.segmentation.find_boundaries(lbl_slice)

        # add these contours as fixed values in the image matrix
        img_slice[pre_contours == 1] = 200
        img_slice[lbl_contours == 1] = 100

        # visualize the image
        plt.subplot(1, num_slices_vis, idx + 1)                
        plt.imshow(img_slice, cmap='gray')
        plt.title('image, z = ' + str(vis_indices[idx]))
        plt.colorbar()

    plt.suptitle(title, fontsize = 50)
    plt.savefig(savepath, bbox_inches='tight', pad_inches = 0, dpi = dpi) # 
    plt.margins(0,0)
    plt.close()

# ==========================================================
# ==========================================================
def show_images_labels_predictions(images,
                                   labels,
                                   soft_predictions):
    
    fig = plt.figure(figsize=(18, 24))
    
    tmp_images = np.copy(images.cpu().numpy())
    tmp_labels = np.copy(labels.cpu().numpy())
    tmp_predictions = np.copy(soft_predictions.detach().cpu().numpy())

    # show 4 examples per batch
    for batch_index in range(4):
        ax = fig.add_subplot(4, 3, 3*batch_index + 1, xticks=[], yticks=[])
        matplotlib_imshow(tmp_images, batch_index)
        ax = fig.add_subplot(4, 3, 3*batch_index + 2, xticks=[], yticks=[])
        matplotlib_imshow(tmp_labels, batch_index)
        ax = fig.add_subplot(4, 3, 3*batch_index + 3, xticks=[], yticks=[])
        matplotlib_imshow(tmp_predictions, batch_index)

    return fig

# ==========================================================
# ==========================================================
def matplotlib_imshow(img, b_index):
    
    img = np.squeeze(img)
    
    if len(np.shape(img)) == 4: # batch size > 1
        if np.shape(img)[0] >= b_index: # batch_size is large enough to contain this index
            img = img[b_index,:,:,img.shape[-1]//2] 
        else:
            img = img[0,:,:,img.shape[-1]//2] 
    else:
        img = img[:,:,img.shape[-1]//2] 
    
    plt.imshow(normalize_img_for_vis(img), cmap = 'gray')

    return 0

# ==========================================================
# ==========================================================
def normalize_img_for_vis(img):

    if np.percentile(img, 99) == np.percentile(img, 1):
        epsilon = 0.0001
    else:
        epsilon = 0.0
    img = (img - np.percentile(img, 1)) / (np.percentile(img, 99) - np.percentile(img, 1) + epsilon)
    img[img<0] = 0.0
    img[img>1] = 1.0

    return (img * 255).astype(np.uint8)