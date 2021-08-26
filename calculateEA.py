import numpy as np
import scipy.io as sio
import pandas as pd

data_name = 'Dioni'
methods = 'RNN'
# number = [25, 40, 55, 70, 85, 100]
number = [10]
savepath = 'E:/Postgraduates/MetricLearning/TripletNetwork/TestResult/'

EATemp = np.empty([12, 10])
for num in number:
    datapath = 'E:/Postgraduates/MetricLearning/TripletNetwork/TestResult/'+str(data_name)+'/'+str(methods)+'/'+str(data_name)+'_'+str(methods)+str(num)+'/'
    for run in range(1, 11, 1):
        # data = sio.loadmat(datapath+str(data_name)+'_'+str(num)+'data_CNN_Triplet_online'+str(run)+'.mat')
        # data = sio.loadmat(datapath+str(data_name)+'_'+str(num)+'data_'+str(methods)+str(run)+'.mat')
        data = sio.loadmat(datapath+'DIoni_RNN'+str(num)+'Dioni_'+str(num)+'data_'+str(methods)+str(run)+'.mat')
        pred = data['TestPre']
        pred = pred.transpose()
        TestDataLabel = data['TestLabel']
        TestDataLabel = TestDataLabel.transpose()+1
        Classes = np.unique(pred)
        EachAcc = np.empty(len(Classes))

        for i in range(0,len(Classes)):
            cla = Classes[i]
            right = 0
            sum = 0

            for j in range(len(TestDataLabel)):
                if TestDataLabel[j,0] == cla:
                    sum += 1
                if TestDataLabel[j,0] == cla and pred[j,0] == cla:
                    right += 1

            EachAcc[i] = (right.__float__()/sum.__float__())*100
        df = pd.DataFrame(EachAcc)
        writer = pd.ExcelWriter(savepath+'EachAccuracy'+str(run)+'.xlsx')
        df.to_excel(writer, 'Sheet1', float_format = '%.2f')
        writer.save()
        writer.close()     
        # print(EachAcc)
    #     EATemp[:,run-1] = EachAcc
    # EAmean_col = np.mean(EATemp, axis = 0)
    # print("样本为%d时，AA为%.4f" %(num, np.mean(EAmean_col)))