import argparse
from datasets import create_dataset
from utils import parse_configuration
import math
from models import create_model
import time
from utils.visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

"""Performs training of a specified model.

Input params:
    config_file: Either a string with the path to the JSON
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and
        model-specific settings.
    export: Whether to export the final model (default=True).
"""
def train(config_file, export=True):
    print('Reading config file...')
    configuration = parse_configuration(config_file)

    print('Initializing dataset...')
    train_dataset = create_dataset(configuration['train_dataset_params'])
    train_dataset_size = len(train_dataset)
    print('The number of training samples = {0}'.format(train_dataset_size))

    val_dataset = create_dataset(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print('The number of validation samples = {0}'.format(val_dataset_size))

    convert_dataset = create_dataset(configuration['convert_dataset_params'])
    convert_dataset_size = len(convert_dataset)
    print('The number of conversion samples = {0}'.format(convert_dataset_size))

    print('Initializing model...')
    model = create_model(configuration['model_params'])
    model.setup()

    print('Initializing visualization...')
    visualizer = Visualizer(configuration['visualization_params'])   # create a visualizer that displays images and plots

    if (type(configuration['model_params']['load_checkpoint']) == str):
        starting_epoch = configuration['model_params']['scheduler_epoch'] + 1
    else:
        starting_epoch = configuration['model_params']['load_checkpoint'] + 1
    num_epochs = configuration['model_params']['max_epochs']

    best_loss = 1000000000
    total_iters = 0

    #Loops through all epochs
    for epoch in range(starting_epoch, num_epochs):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0
        train_dataset.dataset.pre_epoch_callback(epoch)
        model.pre_epoch_callback(epoch)

        train_iterations = len(train_dataset)
        train_batch_size = configuration['train_dataset_params']['loader_params']['batch_size']
        input_size = configuration['train_dataset_params']['input_size']

        g_loss = 0
        d_loss = 0

        model.train()
        #On every epoch, loop through all data in train_dataset
        for i, data in enumerate(train_dataset):  # inner loop within one epoch
            iter_start_time = time.time()
            visualizer.reset()
            cur_data = data

            model.set_input(cur_data)         # unpack data from dataset and apply preprocessing

            # output = model.forward()
            # model.compute_loss()

            if i % configuration['model_update_freq'] == 0:
                model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

                # g_loss += model.loss_G_A2C.item() + model.loss_G_C2A.item() + model.loss_G_B2C.item() + model.loss_G_C2B.item()
                d_loss += model.loss_D_A.item() + model.loss_D_B.item() + model.loss_D_C.item()
                g_loss += model.loss_G.item()
                # d_loss += model.loss_D.item()

            if i % configuration['printout_freq'] == 0:
                t_data = iter_start_time - iter_data_time
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / train_batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                # if opt.display_id > 0:
                visualizer.plot_current_losses(epoch, float(i*train_batch_size) / train_iterations, losses)

                save_result = total_iters % 500 == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            total_iters += train_batch_size
            epoch_iter += train_batch_size

        model.eval()
        for i, data in enumerate(val_dataset):
            if (i > 0):
                break

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.test()

            real_A = model.real_A[0].permute(1, 2, 0).cpu().detach().numpy()
            C_A = model.C_A[0].permute(1, 2, 0).cpu().detach().numpy()
            fake_B = model.fake_B[0].permute(1, 2, 0).cpu().detach().numpy()
            fake_C_B = model.fake_C_B[0].permute(1, 2, 0).cpu().detach().numpy()
            recon_A = model.recon_A[0].permute(1, 2, 0).cpu().detach().numpy()
            idt_A = model.idt_A[0].permute(1, 2, 0).cpu().detach().numpy()
            recon_C_A = model.recon_C_A[0].permute(1, 2, 0).cpu().detach().numpy()

            real_B = model.real_B[0].permute(1, 2, 0).cpu().detach().numpy()
            C_B = model.C_B[0].permute(1, 2, 0).cpu().detach().numpy()
            fake_A = model.fake_A[0].permute(1, 2, 0).cpu().detach().numpy()
            fake_C_A = model.fake_C_A[0].permute(1, 2, 0).cpu().detach().numpy()
            recon_B = model.recon_B[0].permute(1, 2, 0).cpu().detach().numpy()
            idt_B = model.idt_B[0].permute(1, 2, 0).cpu().detach().numpy()
            recon_C_B = model.recon_C_B[0].permute(1, 2, 0).cpu().detach().numpy()

            fig, axs = plt.subplots(2, 7)
            axs[0, 0].imshow(real_A)
            axs[0, 0].set_title('Real A')
            axs[0, 1].imshow(C_A)
            axs[0, 1].set_title('C_A')
            axs[0, 2].imshow(fake_B)
            axs[0, 2].set_title('Fake B')
            axs[0, 3].imshow(fake_C_B)
            axs[0, 3].set_title('Fake C_B')
            axs[0, 4].imshow(recon_A)
            axs[0, 4].set_title('Recon A')
            axs[0, 5].imshow(idt_B)
            axs[0, 5].set_title('Idt A')
            axs[0, 6].imshow(recon_C_A)
            axs[0, 6].set_title('Recon C_A')
            axs[1, 0].imshow(real_B)
            axs[1, 0].set_title('Real B')
            axs[1, 1].imshow(C_B)
            axs[1, 1].set_title('C_B')
            axs[1, 2].imshow(fake_A)
            axs[1, 2].set_title('Fake A')
            axs[1, 3].imshow(fake_C_A)
            axs[1, 3].set_title('Fake C_A')
            axs[1, 4].imshow(recon_B)
            axs[1, 4].set_title('Recon B')
            axs[1, 5].imshow(idt_A)
            axs[1, 5].set_title('Idt A')
            axs[1, 6].imshow(recon_C_B)
            axs[1, 6].set_title('Recon C_B')
            plt.savefig("./plots/epoch_{}.png".format(epoch))
            plt.close()

        for i, data in enumerate(convert_dataset):
            if (i > 0):
                break

            model.set_input(data)
            model.test()

            out_trg_img_A = model.C_A[0].permute(1, 2, 0).cpu().detach().numpy()
            out_trg_img_B = model.C_B[0].permute(1, 2, 0).cpu().detach().numpy()

            #Save images
            out_trg_img_A = out_trg_img_A * 255
            out_trg_img_A = out_trg_img_A.astype(np.uint8)
            out_trg_img_B = out_trg_img_B * 255
            out_trg_img_B = out_trg_img_B.astype(np.uint8)
            output_A = Image.fromarray(out_trg_img_A)
            output_B = Image.fromarray(out_trg_img_B)
            if (not os.path.exists('./outputs/C_A')):
                os.mkdir('./outputs/C_A')
            if (not os.path.exists('./outputs/C_B')):
                os.mkdir('./outputs/C_B')
            output_A.save('./outputs/C_A/epoch_{}.png'.format(epoch))
            output_B.save('./outputs/C_B/epoch_{}.png'.format(epoch))

        # model.post_epoch_callback(epoch, visualizer)
        # train_dataset.dataset.post_epoch_callback(epoch)

        if (g_loss+d_loss < best_loss):
            best_loss = g_loss+d_loss
            print('Saving new best model at the end of epoch {0}'.format(epoch))
            model.save_networks("best")
            model.save_optimizers("best")

        print('Saving latest model at the end of epoch {0}'.format(epoch))
        model.save_networks("last")
        model.save_optimizers("last")
        # model.update_prev_losses()

        # data = OrderedDict()
        # data['G Loss'] = g_loss
        # data['D Loss'] = d_loss
        # visualizer.plot_current_epoch_loss(epoch, data)

        print('End of epoch {0} / {1} \t Time Taken: {2} sec'.format(epoch, num_epochs, time.time() - epoch_start_time))
        print('Generator Loss: {:.4f}, Discriminator Loss: {:.4f}'.format(g_loss, d_loss))

        model.update_learning_rate() # update learning rates every epoch

    if export:
        print('Exporting model')
        model.eval()
        custom_configuration = configuration['train_dataset_params']
        custom_configuration['loader_params']['batch_size'] = 1 # set batch size to 1 for tracing
        dl = train_dataset.get_custom_dataloader(custom_configuration)
        sample_input = next(iter(dl)) # sample input from the training dataset
        model.set_input(sample_input)
        model.export()

    return model.get_hyperparam_result()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Perform model training.')
    parser.add_argument('configfile', default="./config_fer.json", help='path to the configfile')

    args = parser.parse_args()
    train(args.configfile)
