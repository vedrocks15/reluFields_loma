# Basic Imports....
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import sys
import compiler
import ctypes
from argparse import ArgumentParser, Namespace


# setting up directory path....
current = os.path.dirname(os.getcwd())
parent = os.path.dirname(current)
sys.path.append(parent)


# Defining constants for the entire pipeline....
# Training Hyper-params....
learning_rate = 10
num_epochs = 2000
grid_size = (10,10)


# creating directory to store intermediate results...
save_dir = os.path.join(os.getcwd(), "reluField_results")
test_img_res = os.path.join(save_dir, "test_images")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(test_img_res)

# Initalizing the feature grid....
def intialize_feature_grid(gs):
    """
    Function to intialize the feature grid to be used by the pipeline.
    Arguments:
        gs : (tuple) height & width of the feature grid.
    Returns
        feature_grid & the _dfeature_grid to store the actual 
        learnt feature values & the derivatives during training....
    """

    # setting it as pointer arrays for LOMA 
    feature_grid = (ctypes.POINTER(ctypes.c_float) * grid_size[0])()
    _dfeature_grid = (ctypes.POINTER(ctypes.c_float) * grid_size[0])()

    ## Intializing the derivative matrix & original matrix...
    for i in range(gs[0]):

        # Random normal initialization....
        row_val = [np.random.rand(1)[0] for i in range(gs[1])]
        feature_grid[i]   = (ctypes.c_float * len(row_val))(*row_val)

        # derivatives are initalized as zeros only....
        _dfeature_grid[i] = (ctypes.c_float * len(row_val))(*([0]*len(row_val)))

    
    return feature_grid, _dfeature_grid

def update_feature_grids(gs):
    """
    Helper function to zero out the derivative matrix at each epoch...
    """

    _dfeature_grid = (ctypes.POINTER(ctypes.c_float) * gs[0])()
    new_feature_grid = (ctypes.POINTER(ctypes.c_float) * gs[0])()
    
    for i in range(gs[0]):
        _dfeature_grid[i] = (ctypes.c_float * gs[1])(*([0]*gs[1]))
        new_feature_grid[i] = (ctypes.c_float * gs[1])(*([0]*gs[1]))
    
    return _dfeature_grid, new_feature_grid

def train_relu_fields(original_img, save_freq = 100):
    """
    Training function...
    Arguments :
        original_img : np array type image that will be used as the target to train ReLU Fields.
        save_freq : (int) number of epochs in which we are supposed to save intermediate results...
    """
    print("Begin Training........")
    pbar = tqdm(range(num_epochs), total = num_epochs)

    # creating an image for forward pass....
    interpolated_image_zeros = np.zeros_like(original_img)
    interpolated_image = np.copy(interpolated_image_zeros)

    # scaling factors for upsampling....
    og_shape = original_img.shape # h,w
    scale_x = grid_size[1]/og_shape[1]
    scale_y = grid_size[0]/og_shape[0]
    print("Scaling values -> along x : {} and along y : {}".format(scale_x, scale_y))

    # building feature_grids...
    feature_grid, _dfeature_grid = intialize_feature_grid(gs = grid_size)
    gradient_array = np.zeros((num_epochs//save_freq, grid_size[0]*grid_size[1]))

    # training epochs...
    for epoch in pbar:
        
        # forward_pass....
        for row_y in range(og_shape[0]):
            for row_x in range(og_shape[1]):

                # scaled values
                r_y = row_y * scale_y
                r_x = row_x * scale_x
                
                pos_val = (ctypes.c_float * 2)(*[r_y, r_x])
                grid_input = (ctypes.c_float * 2)(*[grid_size[0], grid_size[1]])

                # each run we compute a pixel value via the fractional locations
                interpolated_image[row_y, row_x] = fwd_pass(feature_grid, pos_val, grid_input)
        
        # computing loss.....
        l1_loss = l1_loss_function(interpolated_image, original_img)
        string = "Epoch : {} | loss value : {:.3f}|".format(epoch, l1_loss)
        pbar.set_description(string)

        _dfeature_grid, new_feature_grid = update_feature_grids(gs = grid_size)

        # Back Prop....
        for row_y in range(og_shape[0]):
            for row_x in range(og_shape[1]):
                
                # scaling pixels location...
                r_y = row_y * scale_y
                r_x = row_x * scale_x
                
                # upsampled locations....
                pos_val = (ctypes.c_float * len([row_x, row_y]))(*[r_y, r_x])
                grid_input = (ctypes.c_float * 2)(*[grid_size[0], grid_size[1]])

                # Applying L1 loss differential...
                if interpolated_image[row_y, row_x] > img[row_y, row_x]:
                    _dret = ctypes.c_float(-1)
                elif interpolated_image[row_y, row_x] == img[row_y, row_x]:
                    _dret = ctypes.c_float(0)
                else:
                    _dret = ctypes.c_float(1)
                
                # Computing derivatives....
                diff_code(feature_grid, 
                          _dfeature_grid, 
                          pos_val, 
                          pos_val, 
                          grid_input, 
                          grid_input, 
                          _dret)
                
        # Update step......
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                new_feature_grid[i][j] = feature_grid[i][j] + learning_rate * _dfeature_grid[i][j]
        
        
        if epoch%save_freq==0:
            # show interpolated image.....
            plt.imsave(os.path.join(test_img_res, "test_RelU_fields_interpolated_image_{}_gridSize_{}_{}.jpg".format(epoch,
                                                                                                                     grid_size[0],
                                                                                                                     grid_size[1])),
                       interpolated_image,
                       cmap='gray')
            
            
            tmp = []
            for r_i in range(grid_size[0]):
                for c_j in range(grid_size[1]):
                    tmp.append(_dfeature_grid[r_i][c_j])
            
            # saving gradient values....
            gradient_array[epoch//save_freq, : ] = tmp

            with open(os.path.join(save_dir, "gradArray.npy"), "wb") as f:
                tmp_grad_array = np.zeros((epoch//save_freq + 1, grid_size[0]*grid_size[1]))
                tmp_grad_array[:,:] = gradient_array[:epoch//save_freq+1,:]
                np.save(f, tmp_grad_array)

            
            

            # # experiment with larger resolution.....
            # larger_interpolated_image = np.zeros((1000,3000))
            # for row_y in range(1000):
            #     for row_x in range(2000):
            #         r_y = row_y * (grid_size[0]/1000)
            #         r_x = row_x * (grid_size[1]/2000)
            #         pos_val = (ctypes.c_float * len([row_x, row_y]))(*[r_y, r_x])
            #         grid_input = (ctypes.c_float * len([grid_size[0], grid_size[1]]))(*[grid_size[0], grid_size[1]])
            #         larger_interpolated_image[row_y, row_x] = fwd_pass(feature_grid, pos_val, grid_input)
            
            # plt.imsave("test_images/test_RelU_fields_larger_interpolated_image_{}.jpg".format(epoch), larger_interpolated_image, cmap='gray')

        
        # updating the final feature grid.....
        feature_grid = new_feature_grid   
        
        # saving feature grid values.....
        if epoch%save_freq==0:
            feature_np = np.zeros(grid_size)
            for r_i in range(grid_size[0]):
                for c_j in range(grid_size[1]):
                    feature_np[r_i][c_j] = feature_grid[r_i][c_j]
            
            with open(os.path.join(save_dir, "reluFeatureField_gs_{}_{}.npy").format(grid_size[0], grid_size[1]), "wb") as f:
                np.save(f, feature_np)

def inference_only_function(target_shape = (), load_file_path = ""):
    
    # loading saved feature field...
    with open(load_file_path, 'rb') as f:
        loaded_f_grid = np.load(f)
    
    
    loaded_grid_size = loaded_f_grid.shape
    # f_grid_vals to ctype
    f_grid = (ctypes.POINTER(ctypes.c_float) * loaded_grid_size[0])()
    for i in range(loaded_grid_size[0]):
        f_grid[i] =  (ctypes.c_float * loaded_grid_size[1])(*(list(loaded_f_grid[i][:])))


    loaded_scale_x = loaded_grid_size[1] / target_shape[1]
    loaded_scale_y = loaded_grid_size[0] / target_shape[0]

    interpolated_image = np.zeros(target_shape)

    # forward_pass....
    for row_y in range(target_shape[0]):
        for row_x in range(target_shape[1]):

            # scaled values
            r_y = row_y * loaded_scale_y
            r_x = row_x * loaded_scale_x
            
            pos_val = (ctypes.c_float * 2)(*[r_y, r_x])
            grid_input = (ctypes.c_float * 2)(*[loaded_grid_size[0], loaded_grid_size[1]])

            # each run we compute a pixel value via the fractional locations
            interpolated_image[row_y, row_x] = fwd_pass(f_grid, pos_val, grid_input)
    
    # show interpolated image.....
    plt.imsave(os.path.join(save_dir, "test_inference_shape_{}_{}_grid_size_{}_{}.jpg".format(target_shape[0], 
                                                                                              target_shape[1],
                                                                                              grid_size[0],
                                                                                              grid_size[1])),
                interpolated_image,
                cmap='gray')
    
if __name__ == "__main__":

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--inference", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    # lambda loss function to optimize....
    # l1 loss function gives us better results....
    l1_loss_function = lambda y,y_hat : np.mean(np.abs(y - y_hat))

    # input image for experiment....
    img = np.asarray(Image.open("./demo_train.jpg").convert('L'))

    # setting shape variables & normalizing the image...
    h,w = img.shape
    og_shape = (h,w)
    print("Size of the image loaded : ",(h,w))
    img = img/255.0

    # running reverse diff on ReluFields code...
    with open('./relu_fields_loma.py') as f:
                structs, lib = compiler.compile(f.read(),
                                                target = 'c',
                                                output_filename = './relu_fields_loma')
    
    # storing the code for forward & backward pass....
    diff_code = lib.d_rField_diff
    fwd_pass = lib.reluField2d

    # Downsample image to grid size 
    # and then upsample to original size to get interpolated image 
    # using bilinear interpolation (to compare with regular downsampling)

    downsampled_img = np.zeros((grid_size[0], grid_size[1]))
    for row_y in range(h):
        for row_x in range(w):

            # computing the downsampled indices...
            r_y = row_y * grid_size[0]/h
            r_x = row_x * grid_size[1]/w
            downsampled_img[int(r_y), int(r_x)] = img[row_y, row_x]

    # save the downsampled image...
    plt.imsave(os.path.join(save_dir,"downsampled_image.jpg"), 
               downsampled_img, 
               cmap='gray')
    

    # Why ????????????
    # d_img = (ctypes.POINTER(ctypes.c_float) * grid_size[0])()
    # for i in range(grid_size[0]):
    #     d_img[i] = (ctypes.c_float * grid_size[1])(*downsampled_img[i])

    # #Upsample downsampled image to original size using bilinear interpolation
    # upsampled_img = np.zeros_like(img)
    # for row_y in range(h):
    #     for row_x in range(w):
    #         r_y = row_y * grid_size[0]/h
    #         r_x = row_x * grid_size[1]/w
    #         if (r_x >= grid_size[1] - 1) | (r_y >= grid_size[0] - 1):
    #             continue
    #         pos_val = (ctypes.c_float * len([r_x, r_y]))(*[r_y, r_x])
    #         upsampled_img[row_y, row_x] = fwd_pass(d_img, pos_val)

    # plt.imsave("upsampled_image.jpg", upsampled_img, cmap='gray')

    # cv2_upsample = cv2.resize(downsampled_img, (w, h), interpolation=cv2.INTER_LINEAR)
    # plt.imsave("cv2_upsampled_image.jpg", cv2_upsample, cmap='gray')

    # Computing scale factors.....
    if args.inference:
        gsVal = 200
        grid_size = (gsVal,gsVal) 
        target_shape = (1000,3000)
        inference_only_function(target_shape = target_shape,
                                load_file_path = os.path.join(save_dir, "reluFeatureField_gs_{}_{}.npy").format(grid_size[0], grid_size[1]))
    else:
        for gsVal in [20,50,100,200]:
            grid_size = (gsVal,gsVal)            
            train_relu_fields(original_img = img)

        
    
    
    

    
    

    
    
    
    
 
                
