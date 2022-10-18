import torchvision.transforms as transform
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import PIL
import numpy as np
import matplotlib as mpl
import xlwt
import os


def curve_plot(ac_list, loss_list, epochs, filename):
    x1 = range(0, epochs)
    x2 = range(0, epochs)
    y1 = ac_list
    y2 = loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epochs')
    plt.ylabel('Test accuracy')
    plt.grid(color='g') # 绘制网格线
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epochs')
    plt.ylabel('Test loss')
    plt.grid(color='g')
    plt.savefig(filename)
    # plt.show()


def mat2heatmap(C2):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    f,ax=plt.subplots(figsize = (10,5))     # 设定大小，以确定index和columns作为行列标；
    sns.set(palette="muted", color_codes=True)   
    sns.set(font='SimHei', font_scale=0.8)       
    sns.set(font='YouYuan')
    sns.heatmap(data=C2, cmap='summer', center=0.5, fmt ='.2%',cbar=True,square=False,
                robust=False,annot=True, linecolor='white',linewidths=0)
    ax.tick_params(right=False,top=False,left=False, bottom=False)
    ax.set_xlabel('测试集')
    ax.set_ylabel('训练集')
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False    
    # plt.savefig('other_picture/heatmap_mj_C.png')
    plt.show() 


def confusion_matrix(label, pred):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    sns.set()
    f,ax=plt.subplots()
    C2= confusion_matrix(y_true=label, y_pred=pred, labels=[0,1,2,3,4])
    print(C2,end='|'*3)
    print(type(C2))
    # 百分比形式
    C2 = C2/(len(label)/len(range(5)))
    # sns.heatmap(C2,annot=True,ax=ax)    #画热力图   # 小数点后几位，格式设置
    sns.set(palette="muted", color_codes=True)    # seaborn样式
    sns.set(font='SimHei', font_scale=0.8)        # 解决Seaborn中文显示问题
    sns.set(font='YouYuan')
    sns.heatmap(data=C2, cmap='summer', center=0.5, fmt ='.2%',  # .2%表示显示小数点后两位的百分数
                robust=False,annot=True)
    ax.set_xlabel('Predict')
    ax.set_ylabel('Label')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.rcParams['axes.unicode_minus'] = False    # 解决无法显示"负号"的问题
    # plt.savefig('other_picture/heatmap_mj_C.png')
    plt.show()


def write_dm_to_excel(lst_name1, lst_name2, lst_name3, file_name):
    
    f = xlwt.Workbook(encoding="utf-8") #创建工作簿
    sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True) #创建sheet
    sheet1.write(0, 0, "cwt")
    for i in range(0,len(lst_name1)):
            sheet1.write(i+1,0,lst_name1[i])
    sheet1.write(0, 1, "dwwt")
    for i in range(0,len(lst_name2)):
            sheet1.write(i+1,1,lst_name2[i])
    sheet1.write(0, 2, "wt")
    for i in range(0,len(lst_name3)):
            sheet1.write(i+1,2,lst_name3[i])    
    f.save(file_name + '.xls')  #保存文件


def write_dwb_to_excel(lst_name1, lst_name2, filename):
    f = xlwt.Workbook(encoding="utf-8") #创建工作簿
    sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True) #创建sheet
    sheet1.write(0, 0, filename)
    for i in range(0,len(lst_name1)):
            sheet1.write(i+1,0,lst_name1[i])
    sheet1.write(0, 2, filename+'loss')
    for i in range(0,len(lst_name2)):
            sheet1.write(i+1,2,lst_name2[i])    
    f.save('./Excel/' + filename + '.xls')  #保存文件


def write_LP_to_excel(label, pred, filename):
    f = xlwt.Workbook(encoding='utf-8')
    sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True)
    sheet1.write(0, 0, 'label')
    for i in range(0,len(label)):
        sheet1.write(i+1, 0, label[i])
    sheet1.write(0, 2, 'pred')
    for i in range(0,len(pred)):
        sheet1.write(i+1, 2, pred[i])
    f.save('./Excel/' + filename + '.xls')


def different_method_curve(dwwt, wt, cwt, xaxix, filepath):

    # 这里导入你自己的数据
    DWWT_DRN = dwwt
    WT_CNN = wt
    CWT_CADRN = cwt
    x_axix = [i for i in range(xaxix)]
    # x_axix，train_pn_dis这些都是长度相同的list()
    # sub_axix = filter(lambda x:x%200 == 0, x_axix)
    plt.title('Result Analysis')
    # plt.plot(x_axix, DWWT_DRN, color='green', label='DWWT+DRN', marker='o')
    plt.plot(x_axix, DWWT_DRN, color='green', label='DRN', marker='o')
    plt.plot(x_axix, WT_CNN, color='red', label='CNN', marker='*')
    plt.plot(x_axix, CWT_CADRN,  color='blue', label='CADRN',marker='x')
    plt.legend() # 显示图例
    # item = './other_picture/DMac_curve'
    # if filepath.split('ac',-1)[1] == '_curve':
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Accuracy')
    # elif filepath.split('loss',-1)[1] == '_curve':
    plt.xlabel('Epoch')
    plt.ylabel('Classification Error')
    plt.savefig(filepath+'.svg')
    plt.show()
        #python 一个折线图绘制多个曲线


def write_dn_to_excel(lst_name1, lst_name2, filename,dir):
    f = xlwt.Workbook(encoding="utf-8") #创建工作簿
    sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True) #创建sheet
    sheet1.write(0, 0, filename+'accuracy')
    for i in range(0,len(lst_name1)):
            sheet1.write(i+1,0,lst_name1[i])
    sheet1.write(0, 2, filename+'loss')
    for i in range(0,len(lst_name2)):
            sheet1.write(i+1,2,lst_name2[i])    
    f.save(dir + '/' + filename + '.xls')  #保存文件


def different_network_curve(dwwt, wt, cwt, xaxix, filepath):
    DWWT_DRN = dwwt
    WT_CNN = wt
    CWT_CADRN = cwt
    x_axix = [i for i in range(xaxix)]
    sub_axix = filter(lambda x:x % 200 == 0, x_axix)
    plt.title('Result Analysis')
    # plt.plot(x_axix, DWWT_DRN, color='green', label='DWWT+DRN', marker='o')
    plt.plot(x_axix, DWWT_DRN, color='green', label='DRN', marker='o')
    plt.plot(x_axix, WT_CNN, color='red', label='CNN', marker='*')
    plt.plot(x_axix, CWT_CADRN,  color='blue', label='CADRN',marker='x')
    plt.legend() 
    # item = './other_picture/DMac_curve'
    # if filepath.split('ac',-1)[1] == '_curve':
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Accuracy')
    # elif filepath.split('loss',-1)[1] == '_curve':
    plt.xlabel('Epoch')
    plt.ylabel('Classification Error')
    plt.savefig(filepath+'.svg')
    plt.show()


if __name__ == "__main__":
    dwwt=[i for i in range(10,15)]
    wt=[i for i in range(5,10)]
    cwt=[i for i in range(0,5)]
    DWWT_DRN = dwwt
    WT_CNN = wt
    CWT_CADRN = cwt
    filepaths = './other_picture/DMac_curve'
    different_method_curve(DWWT_DRN,WT_CNN,CWT_CADRN,xaxix=5,filepath=filepaths)
    