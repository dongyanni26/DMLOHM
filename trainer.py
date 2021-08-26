import torch
import numpy as np
import scipy.io as sio
import pandas as pd
import openpyxl

current = 'online'
dataname = 'Dioni'
# excelpath = 'G:/研究生/Test/TripletNetwork/TestResult/loss/'+dataname+'/'

def fit(train_loader, model, loss_fn, optimizer, n_epochs, log_interval, BestAcc, number,
        start_epoch=0):

    tripletmodelpath = 'E:/Postgraduates/MetricLearning/TripletNetwork/TestResult/modelparameter/net_parameter_triplet_'+current+'_'+str(number)+'data_'+dataname+'_margin1.pkl'
    # tripletloss = np.empty((200, 1), dtype = 'float32')
    for epoch in range(start_epoch, n_epochs):

        # Train stage
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, log_interval, BestAcc, tripletmodelpath)
        torch.cuda.empty_cache()
        # message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        # tripletloss[epoch, 0] = train_loss

        # val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        # val_loss /= len(val_loader)

        # message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
        #                                                                          val_loss)

    #     print(message)
    # df = pd.DataFrame(tripletloss)
    # writer = pd.ExcelWriter(excelpath + str(Number)+'4data_triplet_'+current+'_loss.xlsx')
    # df.to_excel(writer, 'triplrt_loss', float_format = '%.5f')
    # writer.save()
    # writer.close()


def train_epoch(train_loader, model, loss_fn, optimizer, log_interval, BestAcc, modelpath):

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if torch.cuda.is_available():
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            # message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     batch_idx * len(data[0]), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), np.mean(losses))
            
            # save the parameters in network
            if loss.data.cpu().numpy() < BestAcc:
                torch.save(model.state_dict(), modelpath)
                BestAcc = loss.data.cpu().numpy()

            # print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss

def classifier_fit(train_loader, model, loss_fn, optimizer, n_epochs, log_interval, BestAcc, number,
        start_epoch=0):

    classifiermodelpath = 'E:/Postgraduates/MetricLearning/TripletNetwork/TestResult/modelparameter/classifier_net_parameter_triplet_'+current+'_'+str(number)+'data_'+dataname+'_margin1.pkl'
    classifierloss = np.empty((200, 1), dtype = 'float32')
    for epoch in range(start_epoch, n_epochs):

        # Train stage
        train_loss = classifier_train_epoch(train_loader, model, loss_fn, optimizer, log_interval, BestAcc, classifiermodelpath)
        torch.cuda.empty_cache()
        # message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        classifierloss[epoch, 0] = train_loss
        # val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        # val_loss /= len(val_loader)

        # message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
        #                                                                          val_loss)

    #     print(message)
    # df = pd.DataFrame(classifierloss)
    # writer = pd.ExcelWriter('G:\研究生\Test\TripletNetwork\TestResult\loss\PaviaU' + str(number)+'dim_100data_triplet_'+current+'_cla_loss.xlsx')
    # df.to_excel(writer, 'classifier_loss', float_format = '%.5f')
    # writer.save()
    # writer.close()


def classifier_train_epoch(train_loader, model, loss_fn, optimizer, log_interval, BestAcc, modelpath):

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if torch.cuda.is_available():
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            # message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     batch_idx * len(data[0]), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), np.mean(losses))
            
            # save the parameters in network
            if loss.data.cpu().numpy() < BestAcc:
                torch.save(model.state_dict(), modelpath)
                BestAcc = loss.data.cpu().numpy()

            # print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss

# def test_epoch(val_loader, model, loss_fn, cuda):
#     with torch.no_grad():
#         model.eval()
#         val_loss = 0
#         for batch_idx, (data, target) in enumerate(val_loader):
#             target = target if len(target) > 0 else None
#             if not type(data) in (tuple, list):
#                 data = (data,)
#             if cuda:
#                 data = tuple(d.cuda() for d in data)
#                 if target is not None:
#                     target = target.cuda()

#             outputs = model(*data)

#             if type(outputs) not in (tuple, list):
#                 outputs = (outputs,)
#             loss_inputs = outputs
#             if target is not None:
#                 target = (target,)
#                 loss_inputs += target

#             loss_outputs = loss_fn(*loss_inputs)
#             loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
#             val_loss += loss.item()

#     return val_loss
