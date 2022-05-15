# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:07:18 2020

@author: G. I. Choudary
"""
import numpy as np

def confusionMatrix(total_classes, gt, predicted):
    total = total_classes-len(gt)
    TP = len(list(set(predicted).intersection(set(gt))))
    FP = len(predicted)-TP
    FN = len(gt)-TP
    TN = total_classes-(TP+FP+FN)
    Accuracy = (TP+TN)/total
    MissclassificationRate = (FP+FN)/total
    TPR = TP/(FN+TP) #Sensitivity, recall
    FPR = FP/(TN+FP) 
    TNR= TN/(TN+FP) #Specificity, selectivity
    FNR= FN/(FN+TP) #miss rate
    Precision=0
    Fscore =0
    LikelihoodRatio=0
    
    if (FP+ TP) !=0:
        Precision = TP/(FP+ TP)
    
    AUC = (TP+(0.5*FP))/len(predicted)
    
    if (Precision+TPR) !=0:
        Fscore = 2*((Precision*TPR)/(Precision+TPR))
    
    if (1-TPR) !=0:
        LikelihoodRatio = Precision/(1-TPR)
    
    return TP, FP, FN, TN, Accuracy, MissclassificationRate, TPR, FPR, FNR, TNR, Precision, AUC, Fscore, LikelihoodRatio


def getCM(predicted, t_classes):
    cm =[]
    for list in predicted:
        if len(list[1])>0:
            cm.append(confusionMatrix(t_classes, list[0], list[1]))
    
    return cm

def printStat(cm):
    confusion_map=[list(i) for i in zip(*cm)]
    onfusion_mean =[np.mean(k) for k in confusion_map]
    onfusion_std =[np.std(k) for k in confusion_map]
    print("TP: "+str(onfusion_mean[0]))
    print("FP: "+str(onfusion_mean[1]))
    print("FN: "+str(onfusion_mean[2]))
    print("TN: "+str(onfusion_mean[3]))
    print("Accuracy: "+str(onfusion_mean[4]))
    print("MissclassificationRate: "+str(onfusion_mean[5]))
    print("TPR: "+str(onfusion_mean[6]))
    print("FPR: "+str(onfusion_mean[7]))
    print("FNR: "+str(onfusion_mean[8]))
    print("TNR: "+str(onfusion_mean[9]))
    print("Precision: "+str(onfusion_mean[10]))
    print("AUC: "+str(onfusion_mean[11]))
    print("AUC STD: "+str(onfusion_std[11]))
    print("Fscore: "+str(onfusion_mean[12]))
    print("LikelihoodRatio: "+str(onfusion_mean[13]))
    
    