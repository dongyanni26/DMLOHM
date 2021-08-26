import os
import numpy as np
import scipy.io as sio
import spectral as sp
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import time

from networks import EmbeddingNet, Classifier, TripletNet
from loss import OnlineTripletLoss, TripletLoss
from tripletselector import RandomNegativeTripletSelector
from trainer import fit, classifier_fit
# from dataset import TripletTrainData, TripletTestData

Number = [10, 25, 40, 55, 70, 85, 100]
# Number = 85
Dim = 128
# featureD = 128
# using RNN with GRU for classification
# setting parameters
total_oa = np.zeros((10,), dtype='float32')
total_kappa = np.zeros((10,), dtype='float32')

# use different colors to represent different class
pavia_colors = np.array(
                        [[255, 255, 255],   # white
                        [216, 191, 216],    # 
                        [0, 255, 0],
                        [0, 255, 255],
                        [46, 139, 87],
                        [255, 0, 255],
                        [255, 165, 0],
                        [160, 32, 240],
                        [255, 0, 0],
                        [255, 255, 0]
                                        ])

salinas_color = np.array(
                        [[255, 255, 255],   # white
                        [255, 222, 173],    # NavajoWhite
                        [70, 130, 180],     # SteelBlue
                        [188, 143, 143],    # RosyBrown
                        [205, 92, 92],      # IndianRed
                        [160, 82, 45],      # Sienna
                        [219, 112, 147],    # PaleVioletRed
                        [147, 112, 219],    # MediumPurple
                        [193, 205, 193],    # Honeydew3
                        [105, 89, 205],     # SlateBlue3
                        [238, 106, 80],     # Coral2
                        [139, 58, 98],      # HotPink4
                        [32, 178, 170],     # LightSeaGreen
                        [95, 158, 160],     # CadetBlue
                        [0, 250, 154],      # MedSpringGreen
                        [107, 142, 35],     # OliveDrab
                        [189, 183, 107]     # DarkKhaki
                        ])

dioni_colors = np.array([[255, 255, 255],
                        [252, 3, 0],
                        [198, 53, 110],
                        [239, 232, 37],
                        [245, 125, 39],
                        # [233, 182, 73],
                        [2, 172, 50],
                        # [4, 70, 22],
                        [86, 32, 108],
                        [115, 123, 50],
                        [188, 175, 70],
                        [211, 235, 95],
                        [170, 182, 194],
                        [40, 76, 212],
                        [98, 228, 254]
                        ])

mainpath_data = 'E:/Postgraduates/MetricLearning/TripletNetwork/OriginalData/HyRANK'
mainpath_save = 'E:/Postgraduates/MetricLearning/TripletNetwork/TestResult/Dioni'
# featurepath_save = 'E:/Postgraduates/MetricLearning/TripletNetwork/TestResult/Dioni/DMLONTM/FeatureDimension/'
# writer = pd.ExcelWriter(featurepath_save+'Dioni_DMLOHM_85_Feature_Dimension_margin1.xlsx')
for dim in Number:
    # if dim == 1:
    #     dim = dim
    # else:
    #     dim = dim-1

    DataPath = mainpath_data + '/Dioni.mat'
    GtPath = mainpath_data + '/Dioni_GT.mat'
    TRPath = mainpath_data + '/Dioni_TRLabel_'+str(dim)+'.mat'
    TSPath = mainpath_data + '/Dioni_TSLabel_'+str(dim)+'.mat'

    tripletmodelpath = 'E:/Postgraduates/MetricLearning/TripletNetwork/TestResult/modelparameter/net_parameter_triplet_online_'+str(dim)+'data_Dioni_margin1.pkl'
    classifiermodelpath = 'E:/Postgraduates/MetricLearning/TripletNetwork/TestResult/modelparameter/classifier_net_parameter_triplet_online_'+str(dim)+'data_Dioni_margin1.pkl'
    print("样本数为", dim, "的结果")
    if Number == 100:
        batchsize = 129  # select from [16, 32, 64, 128]
    else:
        batchsize = 128

    for run in range(1, 11, 1):
        # load data
        Data = sio.loadmat(DataPath)
        PaviaGT = sio.loadmat(GtPath)
        TrLabel = sio.loadmat(TRPath)
        TsLabel = sio.loadmat(TSPath)

        Data = Data['dioni']
        Data = Data.astype(np.float32)
        Label = PaviaGT['dioni_gt']
        Label = np.array(Label, dtype = np.int32)
        TrLabel = TrLabel['TRLabel']
        TsLabel = TsLabel['TSLabel']

        # normalization method 2: map to zero mean and one std
        [m, n, l] = np.shape(Data)

        for i in range(l):
            mean = np.mean(Data[:, :, i])
            std = np.std(Data[:, :, i])
            Data[:, :, i] = (Data[:, :, i] - mean)/std

        # transform data to matrix
        TotalData = np.reshape(Data, [m*n, l])
        TrainDataLabel = np.reshape(TrLabel, [m*n, 1])
        Tr_index, _ = np.where(TrainDataLabel != 0)
        TrainData1 = TotalData[Tr_index, :]
        TrainDataLabel = TrainDataLabel[Tr_index, 0]
        TestDataLabel = np.reshape(TsLabel, [m*n, 1])
        Ts_index, _ = np.where(TestDataLabel != 0)
        TestData1 = TotalData[Ts_index, :]
        TestDataLabel = TestDataLabel[Ts_index, 0]

        # construct data for network
        TrainData = np.empty((len(TrainDataLabel), l, 1), dtype='float32')
        TestData = np.empty((len(TestDataLabel), l, 1), dtype='float32')

        for i in range(len(TrainDataLabel)):
            temp = TrainData1[i, :]
            temp = np.transpose(temp)
            TrainData[i, :, 0] = temp

        for i in range(len(TestDataLabel)):
            temp = TestData1[i, :]
            temp = np.transpose(temp)
            TestData[i, :, 0] = temp

        print('Training size and testing size are:', TrainData.shape, 'and', TestData.shape)

        TrainData = torch.from_numpy(TrainData)
        TrainDataLabel = torch.from_numpy(TrainDataLabel)-1
        TrainDataLabel = TrainDataLabel.long()
        dataset = torch.utils.data.TensorDataset(TrainData, TrainDataLabel)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)

        TestData = torch.from_numpy(TestData)
        TestDataLabel = torch.from_numpy(TestDataLabel)-1
        TestDataLabel = TestDataLabel.long()

        Classes = len(np.unique(TrainDataLabel))

        # train triplet network online triplet loss
        margin = 1
        embedding_net = EmbeddingNet(l, 128)
        model = embedding_net.cuda()
        loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
        lr = 1e-3
        optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 5*1e-4)
        n_epochs = 200
        log_interval = 10
        BestAcc = 100
        
        t0 = time.perf_counter()
        fit(train_loader, model, loss_fn, optimizer, n_epochs, log_interval, BestAcc, dim)
        t1 = time.perf_counter()
        # train classifier network
        classifier = Classifier(l, 128, Classes).cuda()
        optimizer2 = optim.Adam(classifier.parameters(), lr = lr, weight_decay = 1e-4)
        loss_fn2 = nn.CrossEntropyLoss()

        # load trained embeddingnet's parameters
        model_dict = classifier.state_dict()
        pretrained_dict = torch.load(tripletmodelpath)

        # delete net parameters which classifier doesn't need
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        classifier.load_state_dict(model_dict)

        BestAcc2 = 100
        t2 = time.perf_counter()
        classifier_fit(train_loader, classifier, loss_fn2, optimizer2, n_epochs ,log_interval, BestAcc2, dim)
        t3 = time.perf_counter()

        if run == 1:
            print(t3 + t1 - t0 - t2)
        # # test each class accuracy
        # # divide test set into many subsets
        # rnn.eval()
        classifier.load_state_dict(torch.load(classifiermodelpath))

        pred_y = np.empty((len(TestDataLabel)), dtype=np.int32)
        number = len(TestDataLabel)//5000
        for i in range(number):
            temp = TestData[i*5000:(i+1)*5000, :, :]

            temp = temp.view(-1, TrainData.shape[1]).cuda()
            temp2 = classifier(temp)
            temp3 = torch.max(temp2, 1)[1].squeeze()
            pred_y[i*5000:(i+1)*5000] = temp3.cpu()
            del temp, temp2, temp3

        if (i+1)*5000 < len(TestDataLabel):
            temp = TestData[(i+1)*5000:len(TestDataLabel), :, :]

            temp = temp.view(-1, TrainData.shape[1]).cuda()
            temp2 = classifier(temp)
            temp3 = torch.max(temp2, 1)[1].squeeze()
            pred_y[(i+1)*5000:len(TestDataLabel)] = temp3.cpu()
            del temp, temp2, temp3

        pred_y = torch.from_numpy(pred_y).long()
        OA = torch.sum(pred_y == TestDataLabel).type(torch.FloatTensor) / TestDataLabel.size(0)

        Classes = np.unique(TestDataLabel)
        EachAcc = np.empty(len(Classes))

        for i in range(len(Classes)):
            cla = Classes[i]
            right = 0
            sum = 0

            for j in range(len(TestDataLabel)):
                if TestDataLabel[j] == cla:
                    sum += 1
                if TestDataLabel[j] == cla and pred_y[j] == cla:
                    right += 1

            EachAcc[i] = right.__float__()/sum.__float__()

        kappa = cohen_kappa_score(TestDataLabel, pred_y)

        print("第%d次" %run, "OA值为 %f" %(OA * 100))
        print(EachAcc)
        print("第%d次" %run, "Kappa值为 %f" %kappa)


        OA = OA.numpy()
        pred_y = pred_y.cpu()
        pred_y = pred_y.numpy() + 1
        TestDataLabel = TestDataLabel.cpu()
        TestDataLabel = TestDataLabel.numpy()
        total_oa[run-1] = OA * 100
        total_kappa[run-1] = kappa

        # show the result image
        predict_all = np.reshape(TrLabel, [m*n,])
        for i in range(len(pred_y)):
            predict_all[Ts_index[i]] = pred_y[i]

        result = predict_all.reshape(m,n)
        data ={
        'mean':np.mean(total_oa),
        'std':np.std(total_oa),
        'kmean':np.mean(total_kappa),
        'kstd':np.std(total_kappa)
        }

#     df = pd.DataFrame([data])
#     df.to_excel(writer, 'Sheet'+str(dim), float_format = '%.5f')
#     writer.save()
# writer.close()
    # sio.savemat(mainpath_save + '/DMLONTM/DIoni_DMLONTM'+str(dim)+'/Dioni_'+str(dim)+'data_CNN_Triplet_online'+str(run)+'.mat', {'PredAll': result, 'OA': OA, 'TestPre': pred_y, 'TestLabel': TestDataLabel, 'Kappa':kappa})
    # sp.save_rgb(mainpath_save + '/DMLONTM/DIoni_DMLONTM'+str(dim)+'/Dioni_'+str(dim)+'data_CNN_Triplet_online'+str(run)+'.jpg', result, colors = dioni_colors)
    # f = open(mainpath_save + '/DMLONTM/DIoni_DMLONTM'+str(dim)+'/Dioni_'+str(dim)+'data_DMLONTM_mean.txt','w')
    # print("OA均值：", np.mean(total_oa), "最大值：", np.max(total_oa), "最小值：", np.min(total_oa), "方差：", np.var(total_oa), '\n',
    #         "Kappa均值：", np.mean(total_kappa), "最大值：", np.max(total_kappa), "最小值：", np.min(total_kappa),"方差：", np.var(total_kappa), file = f) 
    # f.close()
        sio.savemat(mainpath_save + '/DMLONTM/Dioni_DMLONTM'+str(dim)+'/Dioni_'+str(dim)+'data_CNN_Triplet_online'+str(run)+'.mat', {'PredAll': result, 'OA': OA * 100, 'TestPre': pred_y, 'TestLabel': TestDataLabel, 'Kappa':kappa})
        sp.save_rgb(mainpath_save + '/DMLONTM/Dioni_DMLONTM'+str(dim)+'/Dioni_'+str(dim)+'data_CNN_Triplet_online'+str(run)+'.jpg', result, colors = dioni_colors)
    f = open(mainpath_save + '/DMLONTM/Dioni_DMLONTM'+str(dim)+'/Dioni_'+str(dim)+'data_DMLONTM_margin1_mean.txt','w')
    print("OA均值：", np.mean(total_oa), "最大值：", np.max(total_oa), "最小值：", np.min(total_oa), "方差：", np.std(total_oa), '\n',
            "Kappa均值：", np.mean(total_kappa), "最大值：", np.max(total_kappa), "最小值：", np.min(total_kappa),"方差：", np.std(total_kappa), file = f) 
    f.close()