import os
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Graduation_Pytorch import FocalLoss
from Graduation_Pytorch.DataMapGenerator import generate_map
from Graduation_Pytorch.DataSetsLoader import MyDatasets
from Graduation_Pytorch.Draws import draw_lines
from Graduation_Pytorch.ModelConstruction import Net

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train(model, train_loader, criterion, optimizer, epoch):
    model.train()

    for idx, (img, label) in enumerate(train_loader):
        # forward
        predicted, predicted_mask = model(Variable(img).cuda())
        label = Variable(label[:, :2, :, :]).cuda()
        # backward
        optimizer.zero_grad()
        loss = criterion(predicted, label)
        loss.backward()
        if (idx + 1) % len(train_loader) == 0:
            print('TRAIN epoch: %d loss: %3.2f' % ((epoch + 1), loss))
        # update weights
        optimizer.step()

    torch.save(model.state_dict(), 'save.pth')


def test(model, test_loader, epoch):
    model.load_state_dict(torch.load("save.pth"))
    model.eval()
    recall_sum = 0
    precision_sum = 0
    accuracy_sum = 0
    metrics_sum = 0
    F1_sum = 0
    with torch.no_grad():
        for idx, (img, label) in enumerate(test_loader):
            # forward
            predicted, predicted_mask = model(Variable(img).cuda())
            label = Variable(label[:, :2, :, :]).cuda()
            ones = torch.ones_like(predicted)
            zeros = torch.zeros_like(predicted)
            tp = torch.count_nonzero(
                torch.where(torch.logical_and(torch.eq(predicted_mask, 1), torch.eq(label, 1)), ones, zeros))
            tn = torch.count_nonzero(
                torch.where(torch.logical_and(torch.eq(predicted_mask, 0), torch.eq(label, 0)), ones, zeros))
            fp = torch.count_nonzero(
                torch.where(torch.logical_and(torch.eq(predicted_mask, 1), torch.eq(label, 0)), ones, zeros))
            fn = torch.count_nonzero(
                torch.where(torch.logical_and(torch.eq(predicted_mask, 0), torch.eq(label, 1)), ones, zeros))
            recall = tp.item() / (tp.item() + fn.item())
            precision = tp.item() / (tp.item() + fp.item())
            accuracy = (tp.item() + tn.item()) / (tp.item() + tn.item() + fp.item() + fn.item())
            F1 = 2 * (recall * precision) / (recall + precision)
            recall_sum += recall
            precision_sum += precision
            accuracy_sum += accuracy
            F1_sum += F1
            metrics_sum += 1
        mean_recall = recall_sum / metrics_sum
        mean_precision = precision_sum / metrics_sum
        mean_accuracy = accuracy_sum / metrics_sum
        mean_F1 = F1_sum / metrics_sum
        print('TEST epoch: %d mean recall: %3.4f precision: %3.4f accuracy: %3.4f F1: %3.4f'
              % ((epoch + 1), mean_recall, mean_precision, mean_accuracy, mean_F1))
    return mean_recall, mean_precision, mean_accuracy, mean_F1


def main():
    mean_recalls = []
    mean_precisions = []
    mean_accuracies = []
    mean_F1s = []
    epochs = 10
    LR = 1e-4

    if not os.path.exists('imagesmap.txt'):
        generate_map('images')
        print('images map generated')
    else:
        print('train and test')
    train_ds, test_ds = torch.utils.data.random_split(MyDatasets('../Graduation_Pytorch'), [800, 200])
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
    model = Net().cuda()
    try:
        model.load_state_dict(torch.load("save.pth"))
        print('loaded')
    except:
        print('no pretrained parameter')
    criterion = FocalLoss.FocalLoss(reduction='sum', gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    for epoch in range(epochs):
        train(model, train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch)
        recall, precision, accuracy, F1 = test(model, test_loader, epoch=epoch)
        mean_recalls.append(recall)
        mean_precisions.append(precision)
        mean_accuracies.append(accuracy)
        mean_F1s.append(F1)
        scheduler.step()

    mean_recalls = [i * 100 for i in mean_recalls]
    mean_precisions = [i * 100 for i in mean_precisions]
    mean_accuracies = [i * 100 for i in mean_accuracies]
    mean_F1s = [i * 100 for i in mean_F1s]
    draw_lines(epochs, mean_recalls, mean_precisions, mean_accuracies, mean_F1s)


if __name__ == '__main__':
    s = time.time()
    main()
    print(time.time() - s)
