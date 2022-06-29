from Model import DMFA
from tqdm import tqdm
from torch.utils.data import DataLoader
from CustomDataGenerator import DatasetLoader
from torch.autograd import Variable
from torch import nn
import torch
import cv2
# import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter

def adjust_learning_rate(optimizer, learning_rate, epoch):
# https://github.com/tianbaochou/Data-Augmentation-Pytorch/blob/master/advanced_main.py
    if (epoch + 1) % 10 == 0:
        learning_rate = learning_rate / 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    return optimizer, learning_rate
#############################################################################

# https://towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
import math

import imageio
import numpy as np
## visualize predictions and gt
def visualize_output(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        # print ('pred_edge_kk shape', pred_edge_kk.shape)
        save_path = './temp/'
        name = '{:02d}_output.png'.format(kk)
        imageio.imwrite(save_path + name, pred_edge_kk)

def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        imageio.imwrite(save_path + name, pred_edge_kk)

if __name__ == '__main__':
    base_lr = 0.0001
    weight_decay = 1e-3
    l1_criterion = nn.L1Loss().cuda()
    cel = nn.CrossEntropyLoss().cuda()
    DFModel = DMFA.DFModel().cuda()
    DFOptimizer = torch.optim.Adam(DFModel.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    # dataset_name = "FLAMES dataset (UAV)"
    # dataset_name = "Yar"
    # dataset_name = r"D:\My Research\Datasets\Fire and smoke\FLAMES dataset (UAV)"
    # dataset_name = r"Fire Segmentation Dataset"
    # dataset_path = os.path.join(r'D:\Fire Datasets\Tanveer Work', dataset_name) # Hikmat path
    dataset_path = r'D:\My Research\Datasets\Fire and smoke\FLAMES UAV Preprocessed\Classification' # Tanveer path
    batchsize = 4

    val_losses = []
    train_losses = []
    d_type = ['Train', 'Test']
    total_train_data = DatasetLoader(dataset_path, d_type[0])
    total_test_data = DatasetLoader(dataset_path, d_type[1])

    save_results_path = r"D:\Research Group\Research circle\MBZUAI\Weights/TempResults.dat"

    # train_set, test_set = torch.utils.data.random_split(total_data, [math.ceil(len(total_data) * 0.9), int(len(total_data) * 0.1)])
    train_set, valid_set = torch.utils.data.random_split(total_train_data, [math.ceil(len(total_train_data) * 0.8), int(len(total_train_data) * 0.2)])

    train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=True, num_workers=16, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batchsize, shuffle=True, num_workers=16, drop_last=True)
    save_path = "Model"
    epochs = 2
    for epoch in range(1, epochs):
        train_loss = 0.0
        train_correct = 0.0
        valid_loss = 0.0
        valid_correct = 0.0
        DFModel.train()
        train_loop = tqdm(enumerate(train_loader, start=1), total=len(train_loader), leave=False)
        for batch_id, (images, w, h, cls, lbl, index) in train_loop:

            images = Variable(images).cuda()
            lbl = Variable(lbl).cuda()
            cls = Variable(cls).cuda()
            # depths = Variable(depths).cuda()
            # y_CLASS, y_SEG = DFModel.forward(images)
            y_CLASS = DFModel.forward(images)
            
            # visualize_output(y_SEG)
            # visualize_gt(lbl)
            # y_CLASS = DFModel.forward(images)

            # seg_loss = l1_criterion(y_SEG, lbl)
            cls_loss = cel(y_CLASS, cls)
            # loss = seg_loss + cls_loss
            DFOptimizer.zero_grad()
            # cls_loss.backward()
            loss = cls_loss
            loss.backward()
            DFOptimizer.step()
            DFOptimizer, base_lr = adjust_learning_rate(DFOptimizer, base_lr, epoch)

            train_loss += cls_loss.item()
            train_correct += get_num_correct(y_CLASS, cls)

            # Update progress bar
            train_loop.set_description(f"Epoch [{epoch}/{epochs}]")
            train_loop.set_postfix(Training_loss=loss.item(), Training_accuracy=train_correct / len(train_set))
            # loop.set_postfix(accuracy=total_correct /len(valid_set))
        #
        # valid_loop = tqdm(enumerate(valid_loader, start=1), total=len(valid_loader), leave=False)
        # DFModel.eval()
        # for batch_id, (images, w, h, cls, lbl, index) in valid_loop:
        #     images = Variable(images).cuda()
        #     cls = Variable(cls).cuda()
        #     lbl = Variable(lbl).cuda()
        #
        #
        #
        #     y_CLASS, y_SEG = DFModel.forward(images)
        #
        #     cls_loss = cel(y_CLASS, cls)
        #     seg_loss = l1_criterion(y_SEG, lbl)
        #     # y_CLASS = DFModel.forward(images)
        #     loss = seg_loss + cls_loss
        #     # loss = cel(y_CLASS,cls)
        #
        #     valid_loss += loss.item()
        #     valid_correct += get_num_correct(y_CLASS, cls)
        #     # Update progress bar
        #     valid_loop.set_description("Validating")
        #     valid_loop.set_postfix(Validation_loss=loss.item(), Validation_accuracy=valid_correct / len(valid_set))

        if epoch % 10 == 0 or epoch == epochs - 1:

            with open(save_results_path, "a+") as ResultsFile:
                writing_string = "[Epoch " + str(epoch) + "/" + str(epochs) + "]" + " \nTraining_loss: " + str(
                    cls_loss.item()) + ", Training_accuracy:" + str(
                    train_correct / len(train_set)) + "\n Validation_loss: " + str(
                    loss.item()) + ", Validation_accuracy:" + str(valid_correct / len(valid_set))
                print(writing_string)
                ResultsFile.write(writing_string)
            print("Training_loss: ", cls_loss.item(), "Training_accuracy", train_correct / len(train_set))
            print("Validation_loss: ", loss.item(), "Validation_accuracy", valid_correct / len(valid_set))
            # torch.save(DFModel, save_path + dataset_name + '_%d' % epoch + 'Model.pth')
            torch.save(DFModel.state_dict(), save_path + '\\FLAMESClassification_%d' % epoch + 'Weights.pth')
            print ("Model Saved")
        #
        #
        if epoch == epochs -1:
            print("test begin!")
            # datasets = ["FLAMES dataset (UAV)"]

            # dataset_path = os.path.join(r'D:\My Research\Datasets\Fire and smoke', datasets[0])  # Tanveer path
            DFModel = DFNet.DFModel().to('cuda')
            DFModel.load_state_dict(
                torch.load(save_path + 'FLAMESClassification_%d' % epoch + 'Weights.pth'))
            # d_type = ['Train', 'Test']
            # test_dir = dataset_path + "\\" + d_type[1]
            #
            #
            # test_data = DatasetLoader(dataset_path, d_type[1])
            testloader = DataLoader(total_test_data, batch_size=1, shuffle=True, num_workers=16, drop_last=True)  # LoadtestData(test_dir)

            correct = 0
            total = 0
            train_loss = 0.0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for (images, w, h, cls, lbl, index) in testloader:
                    images = images.to('cuda')
                    cls = cls.to('cuda')
                    # calculate outputs by running images through the network
                    classification_output, segmentation_output = DFModel(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(classification_output.data, 1)
                    total += cls.size(0)
                    correct += (predicted == cls).sum().item()

                    cls_loss = cel(classification_output, cls)

                    # train_loss += cls_loss.item()

            print(f'Accuracy of the network on test images: {100 * correct // total} % and loss: {cls_loss.item()}')

            ########################################################

            ################# Class-wise accuracy ##################

            classes = ['Fire', 'No_Fire']
            # prepare to count predictions for each class
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}

            # again no gradients needed
            with torch.no_grad():
                for (images, w, h, cls, lbl, index) in testloader:
                    images = images.to('cuda')
                    cls = cls.to('cuda')
                    classification_output, segmentation_output = DFModel(images)
                    _, predictions = torch.max(classification_output, 1)
                    # collect the correct predictions for each class
                    for label, prediction in zip(cls, predictions):
                        if label == prediction:
                            correct_pred[classes[label]] += 1
                        total_pred[classes[label]] += 1

            # print accuracy for each class
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
