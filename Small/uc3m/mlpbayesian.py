#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 12:38:54 2018

@author: mlazaro
"""

#import math
import random

#------------------------------------------------------------------------------
# definición de métodos útiles
#------------------------------------------------------------------------------
import numpy as np
from scipy import special
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin

#from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
)


def calc_MAE_AMAE_CM(CM):
    nClases = CM.shape[0]
    AMAEu = np.zeros(nClases)
    MAEu = np.zeros(nClases)
    for kclase in range(nClases):
        for kotra in range(nClases):
            MAEu[kclase] += CM[kclase,kotra]*np.abs(kclase-kotra)#/y_test.shape[0]
            AMAEu[kclase] += CM[kclase,kotra]*np.abs(kclase-kotra)/np.sum(CM, axis=1)[kclase]

    MAE = np.sum(MAEu)/np.sum(CM)
    AMAE = np.mean(AMAEu)

    return MAE, AMAE

def calc_MAE_AMAE(y,ye):
    #labels, ocurrences = np.unique(y,return_counts=True)
    labels = np.unique(y)

    MAE = np.mean(np.abs(y-ye))
    MAEu = np.zeros((labels.shape[0]))
    for kclase in range(labels.shape[0]):
        vc = np.nonzero(y==labels[kclase])
        MAEu[kclase] = np.mean(np.abs(y[vc]-ye[vc]))

    AMAE = np.mean(MAEu)

    return MAE, AMAE

def compute_metrics(y,ye,metrics):

    metricas=np.zeros(len(metrics))
    for k in range(len(metrics)):
        name_metric = metrics[k]
        if name_metric.lower() == 'mae':
            #MAE, AMAE = calc_MAE_AMAE_CM(confusion_matrix(y, ye))
            MAE, AMAE = calc_MAE_AMAE(y, ye)
            metricas[k] = MAE

        elif name_metric.lower() == 'amae':
            #MAE, AMAE = calc_MAE_AMAE(confusion_matrix(y, ye))
            MAE, AMAE = calc_MAE_AMAE(y, ye)
            metricas[k] = AMAE

        elif name_metric.lower() == 'accuracy':
            metricas[k] = accuracy_score(y, ye)

        elif name_metric.lower() == 'balanced_accuracy':
            metricas[k] = balanced_accuracy_score(y, ye)

        elif name_metric.lower() == 'f1':
            metricas[k] = f1_score(y, ye, average='macro')

        elif name_metric.lower() == 'cohen_kappa':
            metricas[k] = cohen_kappa_score(y, ye)

        elif name_metric.lower() == 'cohen_kappa_linear':
            metricas[k] = cohen_kappa_score(y, ye, weights='linear')

        elif name_metric.lower() == 'cohen_kappa_quadratic':
            metricas[k] = cohen_kappa_score(y, ye, weights='quadratic')

        elif name_metric.lower() == 'matthews':
            metricas[k] = matthews_corrcoef(y, ye)

    return metricas

#------------------------------------------------------------------------------
# Dictionary to map textual names of Parzen windows with numbers
#------------------------------------------------------------------------------
dict_parzen = {
    'gauss': 0,
    'uniform': 1,
    'linear': 2,
    'linear_inv': -2,
    'quadratic': 3,
    'quadratic_inv': -3,
    'cubic': 4,
    'cubic_inv': -4,
    'fourth': 5,
    'fourth_inv': -5,
    'triangle': 10,
    'abs': 11,
    'exponential': 13
}

#------------------------------------------------------------------------------
"""
% y = fdpGeneralN(x,tFDP);
%
% Función que calcula la función densidad de probabilidad especificada
% y parámetro deltaS
%
%  tFDP : parametros FDP base para el estimador de Parzen
%
%     tFDP(1) indica el tipo de FDP
%      >  0 : FDP Gausiana (99% de probabilidad en centroS +/- deltaS)
%      >  1 : FDP Uniforme en centroS +/- deltaS  (norma L1 en +/-1)
%      >  2 : FDP Lineal Creciente en centroS +/- deltaS (norma L2 en +/-1)
%      > -2 : FDP Lineal Decreciente en centroS +/- deltaS
%      >  3 : FDP cuadrática Creciente en centroS +/- deltaS (norma L3 en +/-1)
%      > -3 : FDP cuadrática Decreciente en centroS +/- deltaS (complementp norma L3 en +/-1)
%      >  4 : FDP cúbica Creciente en centroS +/- deltaS (norma L4 en +/-1)
%      > -4 : FDP cúbica Decreciente en centroS +/- deltaS (complementp norma L4 en +/-1)
%      >  5 : FDP cuarta Creciente en centroS +/- deltaS (norma L5 en +/-1)
%      > -5 : FDP cuarta Decreciente en centroS +/- deltaS (complementp norma L5 en +/-1)
%
%      > 10 : FDP Triangular simétrica en centroS +/- deltaS
%      > 11 : FDP Valor Absoluto en centroS +/- deltaS
%      > 12 : FDP Integral de activación tanh
%           (con 99% de probabilidad en centroS +/- deltaS)
%      > 13 : FDP exponencial decreciente, inicio en centroS, con factor de caída deltaS
%
%      > 102, -102, 110, 111: generalizaciones de 2,-2,10,11 pero con
%           transiciones sinusoidales en lugar de lineales
%
%      > 202, -202, 210, 211: generalizaciones de 2,-2,10,11 pero con
%           transiciones exponenciales (x^2) en lugar de lineales
%
%      > 302, -302, 310, 311: generalizaciones de 2,-2,10,11 pero con
%           transiciones exponenciales inversas (1 - x^2)
%
%
%     tFDP(2) indica el parámetro de ancho deltaS (soporte centroS +/- deltaS)
%     tFDP(3) indica el parámetro de media (centroS)
%
%--------------------------------------------------------------------------
%         Autor: Marcelino Lázaro
%      Creación: diciembre 2014
% Actualización: abril 2017
%--------------------------------------------------------------------------
"""
def fdpGeneralN(x, tipoFDP=[0, 1, 0]):
    tFDP = tipoFDP[0]
    if isinstance(tFDP, str):
        tFDP = dict_parzen[tFDP]

    deltaS = float(tipoFDP[1])
    centroS = tipoFDP[2]
    diff = x - centroS
    abs_diff = np.abs(diff)

    y = np.zeros_like(x)
    in_range = abs_diff <= deltaS

    if tFDP == 0:
        #adapta = 3.09022 # 99.8%
        #adapta = 2.32635 # 98%
        adapta = 2.5758  # 99%
        y[in_range] = np.exp(-np.power(diff[in_range], 2) / (2 * np.power(deltaS / adapta, 2))) / (np.sqrt(2 * np.pi) * deltaS / adapta)

    if tFDP == 1:
        y[in_range] = 1.0 / (2.0 * deltaS)
    elif tFDP == 2:
        y[in_range] = (diff[in_range] + deltaS) / (2.0 * deltaS * deltaS)
    elif tFDP == -2:
        y[in_range] = (deltaS - diff[in_range]) / (2.0 * deltaS * deltaS)
    elif tFDP == 3:
        y[in_range] = 3.0 / 8.0 * np.power(1 + diff[in_range] / deltaS, 2) / deltaS
    elif tFDP == -3:
        y[in_range] = 3.0 / 8.0 * np.power(1 - diff[in_range] / deltaS, 2) / deltaS
    elif tFDP == 4:
        y[in_range] = 1.0 / 4.0 * np.power(1 + diff[in_range] / deltaS, 3) / deltaS
    elif tFDP == -4:
        y[in_range] = 1.0 / 4.0 * np.power(1 - diff[in_range] / deltaS, 3) / deltaS
    elif tFDP == 5:
        y[in_range] = 5.0 / 32.0 * np.power(1 + diff[in_range] / deltaS, 4) / deltaS
    elif tFDP == -5:
        y[in_range] = 5.0 / 32.0 * np.power(1 - diff[in_range] / deltaS, 4) / deltaS
    elif tFDP == 10:
        y[in_range] = (1 - abs_diff[in_range] / deltaS) / deltaS
    elif tFDP == 11:
        y[in_range] = abs_diff[in_range] / (deltaS * deltaS)
    elif tFDP == 13:
        y[in_range] = np.exp(-(diff[in_range]+deltaS)*2.5)

    return y


# def fdpGeneralN(x,tipoFDP=[0,1,0]):
#     tFDP=tipoFDP[0]
#     if type(tFDP)==str:
#         tFDP = dict_parzen[tFDP]

#     deltaS=tipoFDP[1]
#     centroS=tipoFDP[2]

#     if tFDP == 0:
#         #adapta = 3.09022 # 99.8%
#         #adapta = 2.32635 # 98%
#         adapta = 2.5758 # 99%
#         y=np.exp(-np.power((x-centroS),2)/(2*np.power(float(deltaS)/adapta,2)))/(np.sqrt(2*3.1415926535)*float(deltaS)/adapta)
#     elif tFDP == 1:
#         y = np.where(np.abs(x-centroS) <= deltaS, 1.0/(2.0*float(deltaS)), 0)

#     elif tFDP == 2:
#         y = np.where(np.abs(x-centroS) <= deltaS, (x-centroS+deltaS)/float(deltaS)/float(deltaS)/2.0, 0)

#     elif tFDP == -2:
#         y = np.where(np.abs(x-centroS) <= deltaS, (-x+centroS+deltaS)/float(deltaS)/float(deltaS)/2.0, 0)

#     elif tFDP == 3:
#         y = np.where(np.abs(x-centroS) <= deltaS, 3.0/8.0/float(deltaS)*np.power(1+(x-centroS)/float(deltaS),2), 0)

#     elif tFDP == -3:
#         y = np.where(np.abs(x-centroS) <= deltaS, 3.0/8.0/float(deltaS)*np.power(1-(x-centroS)/float(deltaS),2), 0)

#     elif tFDP == 4:
#         y = np.where(np.abs(x-centroS) <= deltaS, 1.0/4.0/float(deltaS)*np.power(1+(x-centroS)/float(deltaS),3), 0)

#     elif tFDP == -4:
#         y = np.where(np.abs(x-centroS) <= deltaS, 1.0/4.0/float(deltaS)*np.power(1-(x-centroS)/float(deltaS),3), 0)

#     elif tFDP == 5:
#         y = np.where(np.abs(x-centroS) <= deltaS, 5.0/32.0/float(deltaS)*np.power(1+(x-centroS)/float(deltaS),4), 0)

#     elif tFDP == -5:
#         y = np.where(np.abs(x-centroS) <= deltaS, 5.0/32.0/float(deltaS)*np.power(1-(x-centroS)/float(deltaS),4), 0)

#     elif tFDP == 10:
#         y = np.where(np.abs(x-centroS) <= deltaS, (1-np.abs((x-centroS)/float(deltaS)))/float(deltaS), 0)

#     elif tFDP == 11:
#         y = np.where(np.abs(x-centroS) <= deltaS, np.abs(x-centroS)/float(deltaS)/float(deltaS), 0)

#     elif tFDP == 13:
#         y = np.where(x >= centroS, deltaS*np.exp(-deltaS*(x-centroS)), 0)


#     return y
#------------------------------------------------------------------------------
"""
% y = intGeneralN(x,tFDP);
%
% Función que calcula la integral de la distribución especificada
% desde menos infinito a x
%
%  tFDP : tipo de FDP base para el estimador de Parzen
%
%  tFDP : parametros FDP base para el estimador de Parzen
%
%     tFDP(1) indica el tipo de FDP
%      >  0 : FDP Gausiana (99% de probabilidad en centroS +/- deltaS)
%      >  1 : FDP Uniforme en centroS +/- deltaS  (norma L1 en +/-1)
%      >  2 : FDP Lineal Creciente en centroS +/- deltaS (norma L2 en +/-1)
%      > -2 : FDP Lineal Decreciente en centroS +/- deltaS
%      >  3 : FDP cuadrática Creciente en centroS +/- deltaS (norma L3 en +/-1)
%      > -3 : FDP cuadrática Decreciente en centroS +/- deltaS (complementp norma L3 en +/-1)
%      >  4 : FDP cúbica Creciente en centroS +/- deltaS (norma L4 en +/-1)
%      > -4 : FDP cúbica Decreciente en centroS +/- deltaS (complementp norma L4 en +/-1)
%      >  5 : FDP cuarta Creciente en centroS +/- deltaS (norma L5 en +/-1)
%      > -5 : FDP cuarta Decreciente en centroS +/- deltaS (complementp norma L5 en +/-1)
%
%      > 10 : FDP Triangular simétrica en centroS +/- deltaS
%      > 11 : FDP Valor Absoluto en centroS +/- deltaS
%      > 12 : FDP Integral de activación tanh
%           (con 99% de probabilidad en centroS +/- deltaS)
%      > 13 : FDP exponencial decreciente, inicio en centroS, con factor de caída deltaS
%
%      > 102, -102, 110, 111: generalizaciones de 2,-2,10,11 pero con
%           transiciones sinusoidales en lugar de lineales
%
%      > 202, -202, 210, 211: generalizaciones de 2,-2,10,11 pero con
%           transiciones exponenciales (x^2) en lugar de lineales
%
%      > 302, -302, 310, 311: generalizaciones de 2,-2,10,11 pero con
%           transiciones exponenciales inversas (1 - x^2)
%
%
%     tFDP(2) indica el parámetro de ancho deltaS (soporte centroS +/- deltaS)
%     tFDP(3) indica el parámetro de media (centroS)
%
%--------------------------------------------------------------------------
%         Autor: Marcelino Lázaro
%      Creación: diciembre 2014
% Actualización: abril 2017
%--------------------------------------------------------------------------
%
"""
def intGeneralN(x,tipoFDP=[0,1,0]):
    tFDP=tipoFDP[0]
    if type(tFDP)==str:
        tFDP = dict_parzen[tFDP]

    deltaS=float(tipoFDP[1])
    centroS=float(tipoFDP[2])

    diff = x - centroS
    abs_diff = np.abs(diff)
    in_range = abs_diff <= deltaS

    y = np.ones_like(x)

    if tFDP == 0:
        #adapta = 3.09022 # 99.8%
        #adapta = 2.32635 # 98%
        adapta = 2.5758 # 99%

        y[in_range]=special.erfc(-0.7071*diff[in_range]/(deltaS/adapta))/2.0

    elif tFDP == 1:
        y[in_range] = (diff[in_range]+deltaS)/(2*deltaS)

    elif tFDP == 2:
        y[in_range] = np.power(diff[in_range]+deltaS,2)/(4.0*deltaS*deltaS)

    elif tFDP == -2:
        y[in_range] = 1-np.power(deltaS-diff[in_range],2)/(4.0*deltaS*deltaS)

    elif tFDP == 3:
        y[in_range] = 0.125*np.power(1+diff[in_range]/deltaS,3)

    elif tFDP == -3:
        y[in_range] = 1-0.125*np.power(1-diff[in_range]/deltaS,3)

    elif tFDP == 4:
        y[in_range] = 0.0625*np.power(1+diff[in_range]/deltaS,4)

    elif tFDP == -4:
        y[in_range] = 1-0.0625*np.power(1-diff[in_range]/deltaS,4)

    elif tFDP == 5:
        y[in_range] = 0.03125*np.power(1+diff[in_range]/deltaS,5)

    elif tFDP == -5:
        y[in_range] = 1-0.03125*np.power(1-diff[in_range]/deltaS,5)

    elif tFDP == 10:
        y[in_range] = (diff[in_range]+deltaS)/deltaS - 0.5-np.multiply(diff[in_range],abs_diff[in_range])/(2*deltaS*deltaS)

    elif tFDP == 11:
        y[in_range] = 0.5+np.multiply(diff[in_range],abs_diff[in_range])/(2*deltaS*deltaS)

    elif tFDP ==13:
        y[in_range] = 1-np.exp(-(diff[in_range]+deltaS)*2.5)

    y[x < (centroS - deltaS)] = 0.0

    return y


# def intGeneralN(x,tipoFDP=[0,1,0]):
#     tFDP=tipoFDP[0]
#     if type(tFDP)==str:
#         tFDP = dict_parzen[tFDP]

#     deltaS=tipoFDP[1]
#     centroS=tipoFDP[2]

#     if tFDP == 0:
#         #adapta = 3.09022 # 99.8%
#         #adapta = 2.32635 # 98%
#         adapta = 2.5758 # 99%

#         y=special.erfc(0.7071*(centroS-x)/(deltaS/adapta))/2.0

#     elif tFDP == 1:
#         y = np.where(np.abs(x-centroS) <= deltaS, (x-centroS+float(deltaS))/(2*deltaS), 1.0)
#         y[np.nonzero(x<(centroS-deltaS))] = 0.0

#     elif tFDP == 2:
#         y = np.where(np.abs(x-centroS) <= deltaS, np.power(x-centroS+float(deltaS),2)/(4.0*deltaS*deltaS), 1.0)
#         y[np.nonzero(x<(centroS-deltaS))] = 0.0


#     elif tFDP == -2:
#         y = np.where(np.abs(x-centroS) <= deltaS, 1-np.power(centroS+float(deltaS)-x,2)/(4.0*deltaS*deltaS), 1.0)
#         y[np.nonzero(x<(centroS-deltaS))] = 0.0


#     elif tFDP == 3:
#         y = np.where(np.abs(x-centroS) <= deltaS, 0.125*np.power(1+(x-centroS)/deltaS,3), 1.0)
#         y[np.nonzero(x<(centroS-deltaS))] = 0.0


#     elif tFDP == -3:
#         y = np.where(np.abs(x-centroS) <= deltaS, 1-0.125*np.power(1-(x-centroS)/deltaS,3), 1.0)
#         y[np.nonzero(x<(centroS-deltaS))] = 0.0

#     elif tFDP == 4:
#         y = np.where(np.abs(x-centroS) <= deltaS, 0.0625*np.power(1+(x-centroS)/deltaS,4), 1.0)
#         y[np.nonzero(x<(centroS-deltaS))] = 0.0

#     elif tFDP == -4:
#         y = np.where(np.abs(x-centroS) <= deltaS, 1-0.0625*np.power(1-(x-centroS)/deltaS,4), 1.0)
#         y[np.nonzero(x<(centroS-deltaS))] = 0.0

#     elif tFDP == 5:
#         y = np.where(np.abs(x-centroS) <= deltaS, 0.03125*np.power(1+(x-centroS)/deltaS,5), 1.0)
#         y[np.nonzero(x<(centroS-deltaS))] = 0.0

#     elif tFDP == -5:
#         y = np.where(np.abs(x-centroS) <= deltaS, 1-0.03125*np.power(1-(x-centroS)/deltaS,5), 1.0)
#         y[np.nonzero(x<(centroS-deltaS))] = 0.0

#     elif tFDP == 10:
#         y = np.where(np.abs(x-centroS) <= deltaS, (x-centroS+deltaS)/deltaS - 0.5-np.multiply((x-centroS),np.abs(x-centroS))/(2*deltaS*deltaS), 1.0)
#         y[np.nonzero(x<(centroS-deltaS))] = 0.0


#     elif tFDP == 11:
#         y = np.where(np.abs(x-centroS) <= deltaS, 0.5+np.multiply((x-centroS),np.abs(x-centroS))/(2*deltaS*deltaS), 1.0)
#         y[np.nonzero(x<(centroS-deltaS))] = 0.0

#     elif tFDP ==13:
#         y = np.where(x >= centroS, 1-np.exp(-deltaS*(x-centroS)), 0.0)
#         #y[np.nonzero(x<(centroS-deltaS))] = 0.0



#     return y



#------------------------------------------------------------------------------
"""
batches = generate_batches(y)
"""
#------------------------------------------------------------------------------
def generate_batches(nBatch, param, mode='random'):

    if mode == 'random':
        # param: number of samples in the training set
        if nBatch == param:
            list_samples_batch = []
            list_samples_batch.append(list(range(param)))
            num_batches = 1

        else:

            indices = list(range(param))
            random.shuffle(indices)

            l = len(indices)
            #for ndx in range(0, l, nBatch):
            #    yield indices[ndx:min(ndx + nBatch, l)]

            num_batches = int(np.ceil(param/nBatch))
            list_samples_batch = []
            for ndx in range(0, l, nBatch):
                list_aux = indices[ndx:min(ndx + nBatch, l)]
                list_aux.sort()
                list_samples_batch.append(list_aux)


    elif mode == 'class_equitative':
        # All classes have the same number of samples in each batch (repetition for minority)
        #param: class labels for the train set
        if nBatch == param.shape[0]:
            list_samples_batch = []
            list_samples_batch.append(list(range(param.shape[0])))
            num_batches = 1

        else:
            class_labels, samples_class = np.unique(param, return_counts=True)
            samples_max = np.max(samples_class)
            samples_min = np.min(samples_class)
            num_classes = class_labels.shape[0]

            batch_samples_class = np.ceil(nBatch/num_classes).astype(int)
            num_batches = np.ceil(samples_max/batch_samples_class).astype(int)

            list_samples_class = []
            for k in range(num_classes):
                mod = int((batch_samples_class*num_batches) // samples_class[k])
                rem = int((batch_samples_class*num_batches) % samples_class[k])
                ind_class = np.nonzero(param==class_labels[k])[0]
                list_aux = list(ind_class)*mod + list(np.random.choice(ind_class, rem))
                random.shuffle(list_aux)
                list_samples_class.append(list_aux)

            list_samples_batch = []
            for kbatch in range(num_batches):
                list_aux = []
                for kclass in range(num_classes):
                    list_aux += list_samples_class[kclass][batch_samples_class*kbatch:batch_samples_class*(kbatch+1)]

                list_aux.sort()
                list_samples_batch.append(list_aux)

    elif mode == 'representative':
        # All classes have at least 1 sample in each batch (repetition for minority)
        #param: class labels for the train set
        if nBatch == param.shape[0]:
            list_samples_batch = []
            list_samples_batch.append(list(range(param.shape[0])))
            num_batches = 1

        else:
            class_labels, samples_class = np.unique(param, return_counts=True)
            samples_max = np.max(samples_class)
            samples_min = np.min(samples_class)
            num_classes = class_labels.shape[0]
            num_samples = param.shape[0]

            #batch_samples_class = np.ceil(nBatch/num_classes).astype(int)
            num_batches = np.ceil(num_samples/nBatch).astype(int)
            batch_samples_class = []
            list_samples_class = []
            for k in range(num_classes):
                batch_samples_class.append(np.ceil(nBatch*samples_class[k]/num_samples).astype(int))

                mod = int((batch_samples_class[k]*num_batches) // samples_class[k])
                rem = int((batch_samples_class[k]*num_batches) % samples_class[k])
                ind_class = np.nonzero(param==class_labels[k])[0]
                list_aux = list(ind_class)*mod + list(np.random.choice(ind_class, rem))
                random.shuffle(list_aux)
                list_samples_class.append(list_aux)

            list_samples_batch = []
            for kbatch in range(num_batches):
                list_aux = []
                for kclass in range(num_classes):
                    list_aux += list_samples_class[kclass][batch_samples_class[kclass]*kbatch:batch_samples_class[kclass]*(kbatch+1)]

                list_aux.sort()
                list_samples_batch.append(list_aux)


    for kbatch in range(num_batches):
        yield list_samples_batch[kbatch]



#------------------------------------------------------------------------------
"""
  MLPnn  CLASS

  General architecture of a MLP neural network

  Parameters
  ----------

  * layers_size : number of neurons of each hidden layer [list or tuple] nHidden
  * activations : type of activation of each layer [list] nHidden + 1
      Includes the activation of the output layer
      Possible types:
         - tanh
         - relu
         - logistic
         - identity
         - softmax
"""
#------------------------------------------------------------------------------
class MLPnn(BaseEstimator, ClassifierMixin):

    def __init__(self, layers_size=(100,), activations=['relu', 'linear']): #, update='momentum', update_params=[1e-4, 0.95]):
        self.layers_size = layers_size
        self.activations = activations


    def _init_coefs(self, dim_in, dim_out, type_init):
        # Use the initialization method recommended by
        # Glorot et al.
        if type_init == 'normal':
            sd = np.sqrt(2.0 / (dim_in + dim_out))
            coefs = np.float32(np.random.normal(0.0, sd, (dim_out,dim_in+1)))

        elif type_init == 'uniform':
            sd = np.sqrt(6.0 / (dim_in + dim_out))
            coefs = np.float32(np.random.uniform(-sd, sd, (dim_out,dim_in+1)))

        return coefs

    def _initialize(self, dim_in, dim_out, type_init='uniform'):

        self.coefs_ = []
        self.dW = []
        self.dWm = []
        self.num_hidden = len(self.layers_size)
        self.learning_rate = self.learning_rate_init

        nDims = [dim_in] + list(self.layers_size) + [dim_out]
        for k in range(self.num_hidden+1):
            self.coefs_.append(self._init_coefs(nDims[k], nDims[k+1], type_init))
            self.dW.append(self.coefs_[k]*0.0)
            self.dWm.append(self.coefs_[k]*0.0)

    def _forward_pass(self, x):
        nCapas=len(self.coefs_)
        No=nCapas-1

        Np=x.shape[1]

        Os=list()

        xcapa = x.copy()
        for ko in range(nCapas):

            o=np.matmul(self.coefs_[ko],np.concatenate((xcapa,np.ones((1,Np)))));

            if self.activations[ko] == 'tanh':
                o = np.tanh(o)
            elif self.activations[ko] == 'logistic':
                o = 1.0/(1+np.exp(-1*o))
            elif self.activations[ko] == 'relu':
                o[o<0]=0.0
            elif self.activations[ko] == 'softmax':
                #o = np.exp(o)
                # Option to avoid numeric issues (same result)
                o = np.exp(o-np.max(o,axis=0))
                o = o/np.sum(o,axis=0)

            Os.append(o)
            xcapa=o.copy()

        return (Os[-1],Os[0:No])

    def _forward_pass_W(self, x, W):
        nCapas=len(W)
        No=nCapas-1

        Np=x.shape[1]

        Os=list()

        xcapa = x.copy()
        for ko in range(nCapas):

            o=np.matmul(W[ko],np.concatenate((xcapa,np.ones((1,Np)))));

            if self.activations[ko] == 'tanh':
                o = np.tanh(o)
            elif self.activations[ko] == 'logistic':
                o = 1.0/(1+np.exp(-1*o))
            elif self.activations[ko] == 'relu':
                o[o<0]=0.0
            elif self.activations[ko] == 'softmax':
                #o = np.exp(o)
                # Option to avoid numeric issues (same result)
                o = np.exp(o-np.max(o,axis=0))
                o = o/np.sum(o,axis=0)

            Os.append(o)
            xcapa=o.copy()

        return (Os[-1],Os[0:No])

    def _forward_pass_fast(self, x):
        nCapas=len(self.coefs_)

        Np=x.shape[1]

        xcapa = x.copy()
        for ko in range(nCapas):

            o=np.matmul(self.coefs_[ko],np.concatenate((xcapa,np.ones((1,Np)))));

            if self.activations[ko] == 'tanh':
                o = np.tanh(o)
            elif self.activations[ko] == 'logistic':
                o = 1.0/(1+np.exp(-1*o))
            elif self.activations[ko] == 'relu':
                o[o<0]=0.0
            elif self.activations[ko] == 'softmax':
                #o = np.exp(o)
                # Option to avoid numeric issues (same result)
                o = np.exp(o-np.max(o,axis=0))
                o = o/np.sum(o,axis=0)

            xcapa=o.copy()

        return o

    def _cost(self, y, ye, param=[]):

        # Se evalúa el coste inicial
        coste=np.mean(np.square(y-ye))

        return coste


    def _backpropagation(self, x, Os, d):
        # x: input, Os: network outputs, d: derivative of loss function (with activation)

        nCapas = self.num_hidden + 1
        dim_in, nBatchSize = x.shape

        dW=[[] for k in range(nCapas)]

        dW[nCapas-1]=np.matmul(d,np.transpose(np.concatenate((Os[-1],np.ones((1,nBatchSize))))))

        o=Os[-1]

        for ko in range(nCapas-2,-1,-1):
            wp=self.coefs_[ko+1]
            d=np.matmul(np.transpose(wp[:,0:-1]),d)

            if self.activations[ko]=='relu':
                d = np.where(o<=0,0,d)

            elif self.activations[ko]=='tanh':
                d=np.multiply(d,1-np.square(o))

            elif self.activations[ko]=='logistic':
                d=np.multiply(d,o-np.square(o))

            #elif self.activations[ko]=='linear':
            #    d=np.multiply(d,np.ones(np.shape(ye)))

            if ko == 0:
                o=x
            else:
                o=Os[ko-1]

            dW[ko]=np.matmul(d,np.transpose(np.concatenate((o,np.ones((1,nBatchSize))))))

        return dW


    def _update_W(self, dW,mX,mW):

        nCapas = self.num_hidden + 1
        if self.update =='momentum':
            for ko in range(nCapas):
                self.dWm[ko]=self.dWm[ko]*self.momentum+self.learning_rate*np.multiply(dW[ko],mW[ko])
                self.coefs_[ko]=self.coefs_[ko]-np.multiply(self.dWm[ko],mW[ko])

        elif self.update =='gradient':
            for ko in range(nCapas):
                self.coefs_[ko]=self.coefs_[ko]-self.learning_rate*np.multiply(dW[ko],mW[ko])

    def _update_W_X(self, dW,mX,mW,x,y,param,coste):

        nCapas = self.num_hidden + 1
        if self.update == 'gradient-adapt':
            Wdon=[[] for k in range(nCapas)]

            for ko in range(nCapas):
                Wdon[ko]=np.multiply(self.coefs_[ko],mW[ko])-self.learning_rate*np.multiply(dW[ko],mW[ko])

            #----------------------------------------------------------------------
            # Estima de los costes y ajuste del parámetro de paso (GRADIENTE)
            #----------------------------------------------------------------------
            ye, Os = self._forward_pass_W(np.matmul(mX,x),Wdon)
            costen = self._cost(y, ye)
            #------------------------------------------------------------------
            # Ajuste del parámetro de paso
            #------------------------------------------------------------------
            if costen >= coste:
                aumenta=True

                while aumenta:
                    self.learning_rate = self.learning_rate/self.learning_rate_dec

                    for ko in range(nCapas):
                        #Wn[kb]=W[kb]-mu*dW[kb]
                        #Wdon[ko]=Wdo[ko]-mu*np.multiply(dW[ko],mW[ko])
                        Wdon[ko]=np.multiply(self.coefs_[ko],mW[ko])-self.learning_rate*np.multiply(dW[ko],mW[ko])

                    #(ye,Os)=mlp(np.matmul(mX,x[:,indBatch]),Wdon,tAct)
                    (ye,Os)=self._forward_pass_W(np.matmul(mX,x),Wdon)
                    costen = self._cost(y, ye, param)

                    if (costen < coste):
                        aumenta = False
                    elif  (self.learning_rate<1e-20):
                        aumenta = False

                        return True


            for ko in range(nCapas):
                self.coefs_[ko]=self.coefs_[ko]-self.learning_rate*np.multiply(dW[ko],mW[ko])

            self.learning_rate=self.learning_rate*self.learning_rate_inc

            return False

    def get_params(self, deep=True):
        return {'layers_size': self.layers_size,
                'activations': self.activations#,
                #'coefs_': self.coefs_
                }


    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


    # def set_params(self, params):
    #     self.coefs_ = params
    #     self.num_hidden = len(self.coefs_)-1
    #     self.dW = []
    #     self.dWm = []
    #     for k in range(self.num_hidden+1):
    #         self.dW.append(self.coefs_[k]*0.0)
    #         self.dWm.append(self.coefs_[k]*0.0)

    #     self.learning_rate = self.learning_rate_init


#------------------------------------------------------------------------------
"""
  MLPRegressor CLASS

  Classifier using a MLP neural network

  Parameters
  ----------

  * layers_size : number of neurons of each hidden layer (list nHidden)
  * activations : type of activation of each layer (list: nHidden + 1)
      Includes the activation of the output layer
      Possible types:
         - tanh
         - relu
         - logistic
         - identity
         -softmax
  * loss_func: loss function used to train ('mmse', 'entropy')
  * update: type of gradient update
      Possible types:
          - momentum
          - gradient
          - gradient-adapt
  * update_params: params for the type of update
      Possible values:
          - [mu, momentum] (momentum)
          - [mu] (gradient)
          - [mu, muCrec, muDec] (gradient-adapt)
  * weights: weight for each sample (TODO)
  * drop_out: probability of drop-out for each layer, including input (list: nHidden + 1)
  * epoch_params: list [number of epochs, batch_size]
  * W : list with the weights of each layer (nHidden + 1)
  * type_init: type of initizalization for network weights ('uniform', 'gaussian')
  * type_bin: type of output for binary classification ('single', 'multi')
  * flag_evo: flag to compute the loss value after each epoch


"""
#------------------------------------------------------------------------------

class MLPregressor(MLPnn):

    def __init__(self, layers_size, activations,
                 loss_func = 'mmse',
                 update='momentum',
                 update_params=[1e-4,0.95],
                 drop_out = [],
                 n_epochs = 1000,
                 n_batch = 128,
                 W=[],
                 type_init = 'uniform',
                 flag_evo=False
                 ):

        super().__init__(layers_size, activations, W = W)


        self.loss_func = loss_func
        self.update = update
        self.update_params = update_params
        self.drop_out = drop_out
        self.flag_evo = flag_evo
        self.type_init = type_init
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.dWm = []

    def _preprocess_data(self, x, y, sample_weight):

        if len(y.shape)==1:
            yout = y.reshape(1,y.shape[0])
        else:
            yout = y.T

        xout = x.T

        return xout, yout

    def _cost(self, y, ye, sample_weight=[]):
        # param (pesos for wmmse, wentropy)

        # Se evalúa el coste inicial
        if self.loss_func =='mmse':
            coste=np.mean(np.square(y-ye))
        elif self.loss_func =='wmmse':
            coste=np.mean(np.multiply(sample_weight,np.square(y-ye)))

        return coste


    def _derivatives_cost(self, y, ye, params):

        dim_out, nBatchSize = y.shape


        if self.loss_func == 'mmse':
            d=2.0*(ye-y)/nBatchSize

        elif self.loss_func == 'wmmse':
            d=2.0*np.multiply((ye-y),params)/nBatchSize

        if self.activations[-1] == 'tanh':
            d=np.multiply(d,1-np.square(ye))

        elif self.activations[-1] == 'logistic':
            d=np.multiply(d,ye-np.square(ye))

        elif self.activations[-1] == 'relu':
            d = np.where(ye<=0,0,d)

        elif self.activations[-1] == 'softmax':
            auxP = ye-np.square(ye)
            for kA in range(dim_out):
                for kB in range(dim_out):
                    if kB != kA:
                        auxP[kA,:] += np.multiply(d[kA,:],d[kB,:])

            d=np.multiply(d,auxP)
            #d=np.multiply(d,ye-np.square(ye))

        return d

    def gradiente_Numerico(self, y, ye, params):

        pasoGrad = 1e-4

        #ye = self._forward_pass_fast(x)
        costeRef = self._cost(y, ye, params)
        grad_d = np.zeros(ye.shape)
        for kA in range(ye.shape[0]):
            for kB in range(ye.shape[1]):
                yeG = 1.0*ye
                yeG[kA,kB] = yeG[kA,kB] + pasoGrad

                costeGrad = self._cost(y,yeG, params)
                grad_d[kA,kB] =  (costeGrad - costeRef)/pasoGrad

        return grad_d

    def predict(self, x):

        ye = self._forward_pass_fast(x.T)

        if ye.shape[0] == 1:
            ye = ye[0,:]
        else:
            ye = ye.T

        return ye



    def fit(self, x, y, class_weight=None, warm_start=False):

        self.warm_start = warm_start


        x, y = self._preprocess_data(x,y)

        dim_in, num_pat = x.shape
        dim_out = y.shape[0]

        if (len(self.coefs_) == 0) or (warm_start==False):
            self._initialize(dim_in, dim_out, self.type_init)

        nEpochs = self.epoch_params[0]
        nBatch = self.epoch_params[1]
        nCapas = self.num_hidden + 1

        if class_weight == 'balanced':
            weights = np.ones(y.shape)
            if y.shape[0] == 1:
                u_val, u_counts = np.unique(y, return_counts=True)
                for kU in range(1,u_val.shape[0]):
                    weights[np.nonzero(y==u_val[kU])]=u_counts[0]/u_counts[kU]
            else:
                u_counts = np.sum(y, axis=1)
                for kU in range(1,len(self.classes_)):
                    weights[:,np.nonzero(y[kU,:]==1)]=u_counts[0]/u_counts[kU]

        elif type(class_weight) ==  dict:
            weights = np.ones(y.shape)
            if y.shape[0] == 1:
                for key in class_weight.keys():
                    weights[np.nonzero(y==key)] = class_weight[key]
            else:
                for key in class_weight.keys():
                    weights[:,np.nonzero(y[key,:])] = class_weight[key]


        else:
            weights=[]

        if (nBatch == 0) or (nBatch > num_pat):
            nBatch = num_pat

        if self.flag_evo:
            evoCosteEpoch=np.zeros(nEpochs+1)
            ye = self._forward_pass_fast(x)
            evoCosteEpoch[0]=self._cost(y,ye,weights)

        #dW=[[] for k in range(nCapas)]
        Wdo=[[] for k in range(nCapas)]
        Wdon=[[] for k in range(nCapas)]
        mX=np.diag(np.ones(dim_in)).astype(float)
        mW=[[] for k in range(nCapas)]

        if len(self.drop_out)==0:
            self.drop_out=[0 for k in range(nCapas)]

        if len(self.dWm) == 0:

            for ko in range(nCapas):
                self.dWm.append(np.zeros((np.shape(self.coefs_[ko]))))

        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]/(1-self.drop_out[ko])
            mW[ko]=np.ones(np.shape(self.coefs_[ko]))

        #--------------------------------------------------------------------------
        # Se inicia el procedimiento de entrenamiento
        #--------------------------------------------------------------------------
        for kEpoch in range(nEpochs):

            for indBatch in generate_batches(nBatch, num_pat):
                #nBatchSize = len(indBatch)
                if self.loss_func in ['wmmse']:
                    weightsBatch = weights[:,indBatch]
                else:
                    weightsBatch = []

                # Generation of the Drop-Out Masks
                if np.sum(self.drop_out)>0:
                    mX=np.diag(np.random.uniform(0,1,dim_in)>self.drop_out[0]).astype(float)
                    for ko in range(nCapas-1):
                        (Nb,Na)=np.shape(self.coefs_[ko])
                        mW[ko]=np.matmul(np.diag((np.random.uniform(0,1,Nb)>self.DO[ko+1])).astype(float), np.ones((Nb,Na)))
                        Wdo[ko]=np.multiply(self.coefs_[ko],mW[ko])

                    Wdo[nCapas-1]=self.coefs_[nCapas-1]
                else:
                    Wdo=self.coefs_.copy()

                #(ye,Os)=mlp(np.matmul(mX,x[:,indBatch]),Wdo,tAct)
                (ye,Os) = self._forward_pass_W(x[:,indBatch],Wdo)
                #----------------------------------------------------------------------
                # Cálculo de gradientes
                #----------------------------------------------------------------------
                d = self._derivatives_cost(y[:,indBatch], ye, weightsBatch)
                dW = self._backpropagation(x[:,indBatch], Os, d)
                #----------------------------------------------------------------------
                # Nuevos pesos iteración
                #----------------------------------------------------------------------
                if self.update == 'gradient-adapt':
                    coste = self._cost(y[:,indBatch],ye,weightsBatch)
                    salida = self._update_W_X(dW,mX,mW,x[:,indBatch],y[:,indBatch],weightsBatch,coste)

                    if salida:
                        for ko in range(nCapas):
                            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])


                        evoCosteEpoch[kEpoch+1:nEpochs+1]=coste
                        print("Step size in the limit (zero) ...")

                        return evoCosteEpoch

                else:
                    self._update_W(dW,mX,mW)


            #----------------------------------------------------------------------
            # Actualización del coste (global)
            #----------------------------------------------------------------------
            if self.flag_evo:
                for ko in range(nCapas):
                    Wdon[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

                (ye,Os)=self._forward_pass_W(x,Wdon)
                evoCosteEpoch[kEpoch+1] = self._cost(y,ye,weights)


        # Training is over

        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

        if self.flag_evo == False:
            ye=self._forward_pass_fast(x)
            evoCosteEpoch = self._cost(y,ye)

        return evoCosteEpoch


#------------------------------------------------------------------------------
"""
  MLPRegressor CLASS

  Classifier using a MLP neural network

  Parameters
  ----------

  * layers_size : number of neurons of each hidden layer (list nHidden)
  * activations : type of activation of each layer (list: nHidden + 1)
      Includes the activation of the output layer
      Possible types:
         - tanh
         - relu
         - logistic
         - identity
         -softmax
  * loss_func: loss function used to train ('mmse', 'entropy')
  * update: type of gradient update
      Possible types:
          - momentum
          - gradient
          - gradient-adapt
  * update_params: params for the type of update
      Possible values:
          - [mu, momentum] (momentum)
          - [mu] (gradient)
          - [mu, muCrec, muDec] (gradient-adapt)
  * weights: weight for each sample (TODO)
  * drop_out: probability of drop-out for each layer, including input (list: nHidden + 1)
  * epoch_params: list [number of epochs, batch_size]
  * W : list with the weights of each layer (nHidden + 1)
  * type_init: type of initizalization for network weights ('uniform', 'gaussian')
  * type_bin: type of output for binary classification ('single', 'multi')
  * flag_evo: flag to compute the loss value after each epoch


"""
#------------------------------------------------------------------------------

class MLPRegressor(MLPnn):

    def __init__(self,layers_size=[100],
                 activations=['relu','identity'],
                 update='momentum',
                 learning_rate_init=1e-4,
                 learning_rate_inc=1.05,
                 learning_rate_dec=2,
                 momentum = 0.9,
                 drop_out = [],
                 n_epochs = 1000,
                 n_batch = 128,
                 type_init = 'uniform',
                 flag_evo=False
                 ):

        super().__init__(layers_size, activations)

        self.update = update
        self.learning_rate_init = learning_rate_init
        self.learning_rate_inc = learning_rate_inc
        self.learning_rate_dec = learning_rate_dec
        self.momentum = momentum
        self.drop_out = drop_out
        self.flag_evo = flag_evo
        self.type_init = type_init
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.dWm = []

    def _preprocess_data(self, x, y, sample_weight):

        if len(y.shape)==1:
            yout = y.reshape(1,y.shape[0])
        else:
            yout = y.T

        xout = x.T

        if type(sample_weight) == np.ndarray:
            self.loss_func = 'wmmse'
            weights = np.ones(y.shape)
            weights = sample_weight.T
        elif type(sample_weight) ==  dict:
            self.loss_func = 'wmmse'
            weights = np.ones(yout.shape)
            for key in sample_weight.keys():
                weights[key,:] = sample_weight[key]
        else:
            weights = None
            self.loss_func = 'mmse'

        return xout, yout, weights

    def _cost(self, y, ye, sample_weight=None):

        # Cost evaluation
        if self.loss_func =='mmse':
            coste=np.mean(np.square(y-ye))
        elif self.loss_func =='wmmse':
            coste=np.mean(np.multiply(sample_weight,np.square(y-ye)))

        return coste


    def _derivatives_cost(self, y, ye, sample_weight):

        dim_out, nBatchSize = y.shape


        if self.loss_func == 'mmse':
            d=2.0*(ye-y)/nBatchSize

        elif self.loss_func == 'wmmse':
            d=2.0*np.multiply((ye-y),sample_weight)/nBatchSize

        if self.activations[-1] == 'tanh':
            d=np.multiply(d,1-np.square(ye))

        elif self.activations[-1] == 'logistic':
            d=np.multiply(d,ye-np.square(ye))

        elif self.activations[-1] == 'relu':
            d = np.where(ye<=0,0,d)

        elif self.activations[-1] == 'softmax':
            auxP = ye-np.square(ye)
            for kA in range(dim_out):
                for kB in range(dim_out):
                    if kB != kA:
                        auxP[kA,:] += np.multiply(d[kA,:],d[kB,:])

            d=np.multiply(d,auxP)
            #d=np.multiply(d,ye-np.square(ye))

        return d

    def gradiente_Numerico(self, y, ye, params):

        pasoGrad = 1e-4

        #ye = self._forward_pass_fast(x)
        costeRef = self._cost(y, ye, params)
        grad_d = np.zeros(ye.shape)
        for kA in range(ye.shape[0]):
            for kB in range(ye.shape[1]):
                yeG = 1.0*ye
                yeG[kA,kB] = yeG[kA,kB] + pasoGrad

                costeGrad = self._cost(y,yeG, params)
                grad_d[kA,kB] =  (costeGrad - costeRef)/pasoGrad

        return grad_d

    def predict(self, x):

        ye = self._forward_pass_fast(x.T)

        if ye.shape[0] == 1:
            ye = ye[0,:]
        else:
            ye = ye.T

        return ye

    def predict_hidden(self,x):
        out_layers = self._forward_pass(x.T)[1]

        out_ret=[]
        for k in range(len(out_layers)):
            out_ret.append(out_layers[k].T)

        return out_ret


    def _fit(self, x, y, sample_weight=None, warm_start=False, incremental=False):

        self.warm_start = warm_start

        x, y, weights = self._preprocess_data(x,y,sample_weight)

        dim_in, num_pat = x.shape
        dim_out = y.shape[0]

        first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
        )

        if first_pass:#(len(self.coefs_) == 0) or (self.warm_start==False):
            self._initialize(dim_in, dim_out, self.type_init)

        # if (len(self.coefs_) == 0) or (warm_start==False):
        #     self._initialize(dim_in, dim_out, self.type_init)

        nCapas = self.num_hidden + 1


        if (self.n_batch == 0) or (self.n_batch > num_pat):
            self.n_batch = num_pat

        if self.flag_evo:
            evoCosteEpoch=np.zeros(self.n_epochs+1)
            ye = self._forward_pass_fast(x)
            evoCosteEpoch[0]=self._cost(y,ye,weights)

        #dW=[[] for k in range(nCapas)]
        Wdo=[[] for k in range(nCapas)]
        Wdon=[[] for k in range(nCapas)]
        mX=np.diag(np.ones(dim_in)).astype(float)
        mW=[[] for k in range(nCapas)]

        if len(self.drop_out)==0:
            self.drop_out=[0 for k in range(nCapas)]

        if len(self.dWm) == 0:

            for ko in range(nCapas):
                self.dWm.append(np.zeros((np.shape(self.coefs_[ko]))))

        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]/(1-self.drop_out[ko])
            mW[ko]=np.ones(np.shape(self.coefs_[ko]))

        #--------------------------------------------------------------------------
        # Se inicia el procedimiento de entrenamiento
        #--------------------------------------------------------------------------
        for kEpoch in range(self.n_epochs):

            for indBatch in generate_batches(self.n_batch, num_pat):
                #nBatchSize = len(indBatch)
                if self.loss_func in ['wmmse']:
                    weightsBatch = weights[:,indBatch]
                else:
                    weightsBatch = []

                # Generation of the Drop-Out Masks
                if np.sum(self.drop_out)>0:
                    mX=np.diag(np.random.uniform(0,1,dim_in)>self.drop_out[0]).astype(float)
                    for ko in range(nCapas-1):
                        (Nb,Na)=np.shape(self.coefs_[ko])
                        mW[ko]=np.matmul(np.diag((np.random.uniform(0,1,Nb)>self.DO[ko+1])).astype(float), np.ones((Nb,Na)))
                        Wdo[ko]=np.multiply(self.coefs_[ko],mW[ko])

                    Wdo[nCapas-1]=self.coefs_[nCapas-1]
                else:
                    Wdo=self.coefs_.copy()

                #(ye,Os)=mlp(np.matmul(mX,x[:,indBatch]),Wdo,tAct)
                (ye,Os) = self._forward_pass_W(x[:,indBatch],Wdo)
                #----------------------------------------------------------------------
                # Cálculo de gradientes
                #----------------------------------------------------------------------
                d = self._derivatives_cost(y[:,indBatch], ye, weightsBatch)
                dW = self._backpropagation(x[:,indBatch], Os, d)
                #----------------------------------------------------------------------
                # Nuevos pesos iteración
                #----------------------------------------------------------------------
                if self.update == 'gradient-adapt':
                    coste = self._cost(y[:,indBatch],ye,weightsBatch)
                    salida = self._update_W_X(dW,mX,mW,x[:,indBatch],y[:,indBatch],weightsBatch,coste)

                    if salida:
                        for ko in range(nCapas):
                            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])


                        evoCosteEpoch[kEpoch+1:self.n_epochs+1]=coste
                        print("Step size in the limit (zero) ...")

                        return evoCosteEpoch

                else:
                    self._update_W(dW,mX,mW)


            #----------------------------------------------------------------------
            # Actualización del coste (global)
            #----------------------------------------------------------------------
            if self.flag_evo:
                for ko in range(nCapas):
                    Wdon[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

                (ye,Os)=self._forward_pass_W(x,Wdon)
                evoCosteEpoch[kEpoch+1] = self._cost(y,ye,weights)


        # Training is over

        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

        if self.flag_evo == False:
            ye=self._forward_pass_fast(x)
            evoCosteEpoch = self._cost(y,ye)

        return evoCosteEpoch

    def fit(self, X, y, sample_weight=None):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        Returns
        -------
        self : object
            Returns a trained MLP model.
        """
        #self._validate_params()
        self.n_features_in_ = X.shape[0]
        if X.shape[0] != y.shape[0]:
            raise ValueError('Sizes of X and y are not compatible..')

        return self._fit(X, y, sample_weight, incremental=False)

#------------------------------------------------------------------------------
"""
  MLPAutoencoder CLASS

  Classifier using a MLP neural network

  Parameters
  ----------

  * layers_size : number of neurons of each hidden layer (list nHidden)
  * activations : type of activation of each layer (list: nHidden + 1)
      Includes the activation of the output layer
      Possible types:
         - tanh
         - relu
         - logistic
         - identity
         -softmax
  * loss_func: loss function used to train ('mmse', 'entropy')
  * update: type of gradient update
      Possible types:
          - momentum
          - gradient
          - gradient-adapt
  * update_params: params for the type of update
      Possible values:
          - [mu, momentum] (momentum)
          - [mu] (gradient)
          - [mu, muCrec, muDec] (gradient-adapt)
  * weights: weight for each sample (TODO)
  * drop_out: probability of drop-out for each layer, including input (list: nHidden + 1)
  * epoch_params: list [number of epochs, batch_size]
  * W : list with the weights of each layer (nHidden + 1)
  * type_init: type of initizalization for network weights ('uniform', 'gaussian')
  * type_bin: type of output for binary classification ('single', 'multi')
  * flag_evo: flag to compute the loss value after each epoch


"""
#------------------------------------------------------------------------------

class MLPAutoencoder(MLPnn):

    def __init__(self,layers_size=(100,),
                 activations=['tanh','identity'],
                 update='momentum',
                 learning_rate_init=1e-4,
                 learning_rate_inc=1.05,
                 learning_rate_dec=2,
                 momentum = 0.9,
                 drop_out = [],
                 n_epoch = 1000,
                 n_batch = 128,
                 type_init = 'uniform',
                 noise_type = 'gaussian',
                 noise_param = 0.1,
                 noise_gen = 'fixed',
                 warm_start = False,
                 flag_evo=False
                 ):

        super().__init__(layers_size, activations)

        self.update = update
        self.learning_rate_init = learning_rate_init
        self.learning_rate_inc = learning_rate_inc
        self.learning_rate_dec = learning_rate_dec
        self.momentum = momentum
        self.drop_out = drop_out
        self.flag_evo = flag_evo
        self.type_init = type_init
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.noise_type = noise_type
        self.noise_param = noise_param
        self.noise_gen = noise_gen
        self.warm_start = warm_start

        self.dW = []
        self.dWm = []

    def _preprocess_data(self, x, y, sample_weight):

        if len(y.shape)==1:
            yout = y.reshape(1,y.shape[0])
        else:
            yout = y.T

        if self.noise_gen == 'fixed':
            if self.noise_type == 'gaussian':
                noise = np.random.normal(0,self.noise_param,np.shape(x.T))
            xout = x.T + noise
        else:
            xout = x.T

        if type(sample_weight) == np.ndarray:
            self.loss_func = 'wmmse'
            weights = sample_weight.T
        elif type(sample_weight) ==  dict:
            self.loss_func = 'wmmse'
            weights = np.ones(yout.shape)
            for key in sample_weight.keys():
                weights[key,:] = sample_weight[key]
        elif type(sample_weight) == tuple:
            self.loss_func = 'wmmse'
            if len(sample_weight[1].shape) ==  1:
                weights = np.reshape(np.repeat(sample_weight[1],yout.shape[0]), (yout.shape[1],yout.shape[0])).T
            else:
                weights = sample_weight[1].T

            for k, key in enumerate(sample_weight[0].keys()):
                    weights[key,:] = sample_weight[0][key]*weights[key,:]

        else:
            weights = None
            self.loss_func = 'mmse'

        return xout, yout, weights

    def _cost(self, y, ye, sample_weight=None):

        # Cost evaluation
        if self.loss_func =='mmse':
            coste=np.mean(np.square(y-ye))
        elif self.loss_func =='wmmse':
            coste=np.mean(np.multiply(sample_weight,np.square(y-ye)))

        return coste


    def _derivatives_cost(self, y, ye, sample_weight):

        dim_out, nBatchSize = y.shape


        if self.loss_func == 'mmse':
            d=2.0*(ye-y)/nBatchSize

        elif self.loss_func == 'wmmse':
            d=2.0*np.multiply((ye-y),sample_weight)/nBatchSize

        if self.activations[-1] == 'tanh':
            d=np.multiply(d,1-np.square(ye))

        elif self.activations[-1] == 'logistic':
            d=np.multiply(d,ye-np.square(ye))

        elif self.activations[-1] == 'relu':
            d = np.where(ye<=0,0,d)

        elif self.activations[-1] == 'softmax':
            auxP = ye-np.square(ye)
            for kA in range(dim_out):
                for kB in range(dim_out):
                    if kB != kA:
                        auxP[kA,:] += np.multiply(d[kA,:],d[kB,:])

            d=np.multiply(d,auxP)
            #d=np.multiply(d,ye-np.square(ye))

        return d

    def gradiente_Numerico(self, y, ye, params):

        pasoGrad = 1e-4

        #ye = self._forward_pass_fast(x)
        costeRef = self._cost(y, ye, params)
        grad_d = np.zeros(ye.shape)
        for kA in range(ye.shape[0]):
            for kB in range(ye.shape[1]):
                yeG = 1.0*ye
                yeG[kA,kB] = yeG[kA,kB] + pasoGrad

                costeGrad = self._cost(y,yeG, params)
                grad_d[kA,kB] =  (costeGrad - costeRef)/pasoGrad

        return grad_d

    def predict(self, x):

        ye = self._forward_pass_fast(x.T)

        if ye.shape[0] == 1:
            ye = ye[0,:]
        else:
            ye = ye.T

        return ye

    def predict_hidden(self,x):
        out_layers = self._forward_pass(x.T)[1]

        out_ret=[]
        for k in range(len(out_layers)):
            out_ret.append(out_layers[k].T)

        return out_ret


    def _fit(self, x, y, sample_weight=None, incremental=False):

        #self.warm_start = warm_start

        x, y, weights = self._preprocess_data(x,y,sample_weight)

        dim_in, num_pat = x.shape
        dim_out = y.shape[0]

        first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
        )

        if first_pass:
            self._initialize(dim_in, dim_out, self.type_init)
        else:
            if not hasattr(self, "num_hidden"):
                self.num_hidden = len(self.coefs_)-1

            if len(self.dW) == 0:
                self.dW = []
                self.dWm = []
                for k in range(self.num_hidden+1):
                    self.dW.append(self.coefs_[k]*0.0)
                    self.dWm.append(self.coefs_[k]*0.0)

                self.learning_rate = self.learning_rate_init

        nCapas = self.num_hidden + 1

        if (self.n_batch == 0) or (self.n_batch > num_pat):
            self.n_batch = num_pat

        if self.flag_evo:
            evoCosteEpoch=np.zeros(self.n_epoch+1)
            ye = self._forward_pass_fast(x)
            evoCosteEpoch[0]=self._cost(y,ye,weights)

        #dW=[[] for k in range(nCapas)]
        Wdo=[[] for k in range(nCapas)]
        Wdon=[[] for k in range(nCapas)]
        mX=np.diag(np.ones(dim_in)).astype(float)
        mW=[[] for k in range(nCapas)]

        if len(self.drop_out)==0:
            self.drop_out=[0 for k in range(nCapas)]

        # if len(self.dWm) == 0:
        #     for ko in range(nCapas):
        #         self.dWm.append(np.zeros((np.shape(self.coefs_[ko]))))

        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]/(1-self.drop_out[ko])
            mW[ko]=np.ones(np.shape(self.coefs_[ko]))

        #--------------------------------------------------------------------------
        # Se inicia el procedimiento de entrenamiento
        #--------------------------------------------------------------------------
        for kEpoch in range(self.n_epoch):
            if self.noise_gen == 'epoch':
                if self.noise_type == 'gaussian':
                    noise = np.random.normal(0,self.noise_param,x.shape)

            for indBatch in generate_batches(self.n_batch, num_pat):
                #nBatchSize = len(indBatch)
                if self.loss_func in ['wmmse']:
                    weights_batch = weights[:,indBatch]
                else:
                    weights_batch = []

                # Generation of the Drop-Out Masks
                if np.sum(self.drop_out)>0:
                    mX=np.diag(np.random.uniform(0,1,dim_in)>self.drop_out[0]).astype(float)
                    for ko in range(nCapas-1):
                        (Nb,Na)=np.shape(self.coefs_[ko])
                        mW[ko]=np.matmul(np.diag((np.random.uniform(0,1,Nb)>self.drop_out[ko+1])).astype(float), np.ones((Nb,Na)))
                        Wdo[ko]=np.multiply(self.coefs_[ko],mW[ko])

                    Wdo[nCapas-1]=self.coefs_[nCapas-1].copy()
                else:
                    Wdo=self.coefs_.copy()

                #(ye,Os)=mlp(np.matmul(mX,x[:,indBatch]),Wdo,tAct)
                if self.noise_gen == 'epoch':
                    (ye,Os) = self._forward_pass_W(x[:,indBatch]+noise[:,indBatch],Wdo)
                else:
                    (ye,Os) = self._forward_pass_W(x[:,indBatch],Wdo)
                #----------------------------------------------------------------------
                # Cálculo de gradientes
                #----------------------------------------------------------------------
                d = self._derivatives_cost(y[:,indBatch], ye, weights_batch)
                if self.noise_gen == 'epoch':
                    dW = self._backpropagation(x[:,indBatch]+noise[:,indBatch], Os, d)
                else:
                    dW = self._backpropagation(x[:,indBatch], Os, d)
                #----------------------------------------------------------------------
                # Nuevos pesos iteración
                #----------------------------------------------------------------------
                if self.update == 'gradient-adapt':
                    coste = self._cost(y[:,indBatch],ye,weights_batch)
                    if self.noise_gen == 'epoch':
                        salida = self._update_W_X(dW,mX,mW,x[:,indBatch]+noise[:,indBatch],y[:,indBatch],weights_batch,coste)
                    else:
                        salida = self._update_W_X(dW,mX,mW,x[:,indBatch],y[:,indBatch],weights_batch,coste)

                    if salida:
                        for ko in range(nCapas):
                            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])


                        evoCosteEpoch[kEpoch+1:self.n_epochs+1]=coste
                        print("Step size in the limit (zero) ...")

                        return evoCosteEpoch

                else:
                    self._update_W(dW,mX,mW)


            #----------------------------------------------------------------------
            # Actualización del coste (global)
            #----------------------------------------------------------------------
            if self.flag_evo:
                for ko in range(nCapas):
                    Wdon[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

                (ye,Os)=self._forward_pass_W(x,Wdon)
                evoCosteEpoch[kEpoch+1] = self._cost(y,ye,weights)


        # Training is over

        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

        if self.flag_evo == False:
            ye=self._forward_pass_fast(x)
            evoCosteEpoch = self._cost(y,ye)

        self.loss_ = evoCosteEpoch

        return self

    def fit(self, X, y, sample_weight=None):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        Returns
        -------
        self : object
            Returns a trained MLP model.
        """
        #self._validate_params()
        self.n_features_in_ = X.shape[0]
        if X.shape[0] != y.shape[0]:
            raise ValueError('Sizes of X and y are not compatible..')

        self._fit(X, y, sample_weight, incremental=False)

        return self

    def get_params(self, deep=True):
        return {
            'layers_size': self.layers_size,
            'activations': self.activations,
            'update': self.update,
            'learning_rate_init': self.learning_rate_init,
            'learning_rate_inc': self.learning_rate_inc,
            'learning_rate_dec': self.learning_rate_dec,
            'momentum': self.momentum,
            'drop_out': self.drop_out,
            'n_epoch': self.n_epoch,
            'n_batch': self.n_batch,
            'type_init': self.type_init,
            'noise_type': self.noise_type,
            'noise_param': self.noise_param,
            'noise_gen': self.noise_gen,
            'warm_start': self.warm_start,
            'flag_evo': self.flag_evo
        }

    # def set_params(self, params):
    #     self.coefs_ = params
    #     self.num_hidden = len(self.coefs_)-1
    #     self.dW = []
    #     self.dWm = []
    #     for k in range(self.num_hidden+1):
    #         self.dW.append(self.coefs_[k]*0.0)
    #         self.dWm.append(self.coefs_[k]*0.0)

    #     self.learning_rate = self.learning_rate_init

#------------------------------------------------------------------------------
"""
  MLPBayesBin  CLASS

  Classifier for binary problems using a single output MLP neural network
  The decision rule is threshold based (threshold at 0)
  The loss function is an estimate of the Bayes risk based on the Parzen
  Windows estimator

  Parameters
  ----------

  * layers_size : number of neurons of each hidden layer [list or tuple] nHidden
  * activations : type of activation of each layer [list] nHidden + 1
      Includes the activation of the output layer (this must be identity or tanh)
      Possible types:
         - tanh
         - relu
         - logistic
         - identity
         - softmax
  * parzen_params: params of the Parzen Window [list]
        - params_parzen[0] : type of window
             + 'gauss'
             + 'uniform'
             + 'linear'
             + 'linear_inv'
             + 'quadratic'
             + 'quadratic_inv'
        - params_parzen[1] : support of the window around center
        - params_parzen[2] : center of the window

  * class_prob : probabilities of the classes  [list] nClasses
  * class_costs : costs of misclassification for each class [list] nClases
  * update: type of gradient update
      Possible types:
          - momentum
          - gradient
          - gradient-adapt
  * learning_rate_init: initial value of the learning rate [real]
  * learning_rate_inc: increment factor for learning rate, if 'gradient-adapt' [real]
  * learning_rate_dec: decrease factor for learning rate, if 'gradient-adapt' [real]
  * momentum: momentum value, if 'momentum' [real]
  * drop_out: probability of drop-out for each layer, including input [list] nHidden + 1
  * n_epoch: number of epochs [int]
  * n_batch: size of the mini-batch (0 if batch-mode is used) [int]
  * type_init: type of initizalization for network weights ('uniform', 'gaussian')
  * warm_start: flag to allow a warm start from previous solution [bool]
  * flag_evo: flag to compute the loss value after each epoch [bool]


"""
#------------------------------------------------------------------------------

class MLPBayesBin(MLPnn):

    def __init__(self, layers_size=[100],
                 activations=['relu','identity'],
                 parzen_params=['gauss',1,0],
                 class_prob=[0.5, 0.5],
                 class_cost=[1,1],
                 update='momentum',
                 learning_rate_init=1e-4,
                 learning_rate_inc=1.05,
                 learning_rate_dec=2,
                 momentum = 0.9,
                 drop_out = [],
                 n_epoch = 1000,
                 n_batch = 250,
                 type_batch = 'representative',
                 type_init = 'uniform',
                 warm_start=False,
                 flag_evo=False
                 ):

        super().__init__(layers_size, activations)

        #self.name = 'MLPBayesBin'
        self.layers_size = layers_size
        self.activations = activations
        self.parzen_params = parzen_params
        self.class_prob = class_prob
        self.class_cost = class_cost
        self.update = update
        self.learning_rate_init = learning_rate_init
        self.learning_rate_inc = learning_rate_inc
        self.learning_rate_dec = learning_rate_dec
        self.momentum = momentum
        self.drop_out = drop_out
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.type_batch = type_batch
        self.type_init = type_init
        self.warm_start = warm_start
        self.flag_evo = flag_evo
        #


    def _preprocess_data(self, x, y):
        nClases = 2

        # Labels converted to -1, +1
        if 0 not in list(np.unique(y)):
            y = y - 1

        self.classes_ = list(np.unique(y))


        if len(self.class_prob) == 0:
            self.class_prob = np.zeros(nClases)
            for kclase in range(nClases):
                v = np.nonzero(y==kclase)
                self.class_prob[kclase] = len(v[0])/y.shape[0]

        if len(self.class_cost) == 0:
            self.class_cost = np.ones(nClases)

        yout = np.where(y==0, -1,1)
        xout = x.T

        return xout, yout

    def _cost(self, y, ye, param=[]):

        v0 = np.nonzero(y==-1)
        v1 = np.nonzero(y==1)

        coste = self.class_cost[0] * self.class_prob[0] * np.mean(intGeneralN(ye[0,v0],self.parzen_params))
        coste += self.class_cost[1] * self.class_prob[0] * np.mean(intGeneralN(-ye[0,v1],self.parzen_params))

        return coste


    def _derivatives_cost(self, y, ye):

        nBatchSize = y.shape[0]

        d=np.zeros((1,nBatchSize))

        v0 = np.nonzero(y==-1)
        v1 = np.nonzero(y==1)

        d[0,v0] = self.class_cost[0] * self.class_prob[0] * fdpGeneralN(ye[0,v0],self.parzen_params) / len(v0[0])
        d[0,v1] = -self.class_cost[1] * self.class_prob[1] * fdpGeneralN(-ye[0,v1],self.parzen_params) / len(v1[0])

        # Propagation through the nonlinear function of the output laye
        if self.activations[-1] == 'tanh':
            d=np.multiply(d,1-np.square(ye))

        elif self.activations[-1] == 'logistic':
            d=np.multiply(d,ye-np.square(ye))

        elif self.activations[-1] == 'relu':
            d = np.where(ye<=0,0,d)

        return d

    def gradiente_Numerico(self, y,ye):

        pasoGrad = 1e-4

        #ye = self._forward_pass_fast(x)
        costeRef = self._cost(y,ye)
        grad_d = np.zeros(ye.shape)
        for kA in range(ye.shape[0]):
            for kB in range(ye.shape[1]):
                yeG = 1.0*ye
                yeG[kA,kB] = yeG[kA,kB] + pasoGrad

                costeGrad = self._cost(y,yeG)
                grad_d[kA,kB] =  (costeGrad - costeRef)/pasoGrad

        return grad_d

    def predict(self, x):

        ys = self._forward_pass_fast(x.T)
        #ye = np.argmax(ys, axis=0)
        ye = np.where(ys[0,:]>0,1,0)

        return ye


    def soft_output(self, x):

        ys = self._forward_pass_fast(x.T)
        ye = ys[0,:]

        return ye


    def _fit(self, x, y, x_val=np.zeros(0), y_val=np.zeros(0), val_stop=[0,0], incremental=False):

        if type(x_val) == float:

            nFolds = np.ceil(1/x_val).astype(int)
            nBatchCV = np.ceil(x.shape[0]/nFolds).astype(int)

            indices_CV =[]
            for val in generate_batches(nBatchCV,y, mode='representative'):
                indices_CV.append(val)

            ind_val = indices_CV[-1]
            #ind_train = [v for v in range(x_train.shape[0]) if v not in ind_val]
            ind_train=[]
            for kTrain in range(nFolds-1):
                ind_train += indices_CV[kTrain]
            # Getting the unique values
            ind_train = list(set(ind_train))

            x_val = x[ind_val,:]
            y_val = y[ind_val]
            x = x[ind_train,:]
            y = y[ind_train]

        x, y = self._preprocess_data(x,y)
        if x_val.shape[0] > 0:
            x_val, y_val = self._preprocess_data(x_val,y_val)


        dim_in, num_pat = x.shape
        dim_out = 1

        first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
        )

        if first_pass:#(len(self.coefs_) == 0) or (self.warm_start==False):
            self._initialize(dim_in, dim_out, self.type_init)

        nEpochs = self.n_epoch
        nBatch = self.n_batch
        nCapas = self.num_hidden + 1

        if (nBatch == 0) or (nBatch > num_pat):
            nBatch = num_pat

        if (self.flag_evo) or (val_stop[0] > 0):
            evoCosteEpoch=np.zeros(nEpochs+1)
            if x_val.shape[0] > 0:
                ye = self._forward_pass_fast(x_val)
                coste = self._cost(y_val,ye)
            else:
                ye = self._forward_pass_fast(x)
                coste = self._cost(y,ye)


            evoCosteEpoch[0]=coste

            if val_stop[0] > 0:
                opt_cost = coste
                opt_epoch = 0
                opt_W = self.coefs_.copy()


        #dW=[[] for k in range(nCapas)]
        Wdo=[[] for k in range(nCapas)]
        Wdon=[[] for k in range(nCapas)]
        mX=np.diag(np.ones(dim_in)).astype(float)
        mW=[[] for k in range(nCapas)]

        if len(self.drop_out)==0:
            self.drop_out=[0 for k in range(nCapas)]

        if len(self.dWm) == 0:
            for ko in range(nCapas):
                self.dWm.append(np.zeros((np.shape(self.coefs_[ko]))))

        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]/(1-self.drop_out[ko])
            mW[ko]=np.ones(np.shape(self.coefs_[ko]))

        #--------------------------------------------------------------------------
        # Se inicia el procedimiento de entrenamiento
        #--------------------------------------------------------------------------
        if self.type_batch in ['class_equitative', 'representative']:
            paramsBatch = y

        for kEpoch in range(nEpochs):

            for indBatch in generate_batches(nBatch, paramsBatch, mode=self.type_batch):
                # Generation of the Drop-Out Masks
                if np.sum(self.drop_out)>0:
                    mX=np.diag(np.random.uniform(0,1,dim_in)>self.drop_out[0]).astype(float)
                    for ko in range(nCapas-1):
                        (Nb,Na)=np.shape(self.coefs_[ko])
                        mW[ko]=np.matmul(np.diag((np.random.uniform(0,1,Nb)>self.drop_out[ko+1])).astype(float), np.ones((Nb,Na)))
                        Wdo[ko]=np.multiply(self.coefs_[ko],mW[ko])

                    Wdo[nCapas-1]=self.coefs_[nCapas-1]
                else:
                    Wdo=self.coefs_.copy()

                (ye,Os) = self._forward_pass_W(x[:,indBatch],Wdo)
                #----------------------------------------------------------------------
                # Cálculo de gradientes
                #----------------------------------------------------------------------
                d = self._derivatives_cost(y[indBatch], ye)
                #d = self.gradiente_Numerico(y[indBatch], ye)
                dW = self._backpropagation(x[:,indBatch], Os, d)
                #----------------------------------------------------------------------
                # Nuevos pesos iteración
                #----------------------------------------------------------------------
                if self.update == 'gradient-adapt':
                    coste = self._cost(y[indBatch],ye)
                    salida = self._update_W_X(dW,mX,mW,x[:,indBatch],y[indBatch],paramsBatch,coste)

                    if salida:
                        for ko in range(nCapas):
                            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

                        evoCosteEpoch[kEpoch+1:nEpochs+1]=coste
                        print("Step size in the limit (zero) ...")

                        if val_stop[0] == 0:
                            opt_epoch = kEpoch
                        else:
                            self.coefs_ = opt_W

                        return opt_epoch, evoCosteEpoch

                else:
                    self._update_W(dW,mX,mW)

            #----------------------------------------------------------------------
            # Actualización del coste (global) y validación
            #----------------------------------------------------------------------
            if (self.flag_evo) or (val_stop[0] > 0):
                for ko in range(nCapas):
                    Wdon[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

                if x_val.shape[0] > 0:
                    (ye,Os)=self._forward_pass_W(x_val,Wdon)
                    coste = self._cost(y_val,ye)
                else:
                    (ye,Os)=self._forward_pass_W(x,Wdon)
                    coste = self._cost(y,ye)


                evoCosteEpoch[kEpoch+1] = coste

                if val_stop[0] > 0:
                    if coste < opt_cost:
                        opt_cost = coste
                        opt_epoch = kEpoch+1
                        opt_W = self.coefs_.copy()

                    if kEpoch > val_stop[1]:
                        rel_dec = (np.min(evoCosteEpoch[kEpoch-val_stop[1]:kEpoch])-evoCosteEpoch[kEpoch])/np.min(evoCosteEpoch[kEpoch-val_stop[1]:kEpoch])

                        if rel_dec < val_stop[0]:
                            self.coefs_ = opt_W

                            break

        # Training is over
        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

        if (self.flag_evo == False) and (val_stop[0]==0):
            ye=self._forward_pass_fast(x)
            evoCosteEpoch = self._cost(y,ye)
            opt_epoch = nEpochs

        elif val_stop[0] == 0:
            opt_epoch = nEpochs

        self.loss_ = evoCosteEpoch
        self.opt_epoch = opt_epoch


        return self

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns a trained MLP model.
        """
        #self._validate_params()
        self.n_features_in_ = X.shape[0]
        if X.shape[0] != y.shape[0]:
            raise ValueError('Sizes of X and y are not compatible..')

        return self._fit(X, y, incremental=False)

    def fit_cv_stop(self, x, y, val_stop, metrics=['MAE', 'AMAE'], nFolds=5, verbose=False):
        #metrics = ['MAE', 'AMAE', 'accuracy', 'balanced_accuracy', 'f1', 'cohen_kappa', 'cohen_kappa_linear', 'cohen_kappa_quadratic', 'matthews']

        nBatchCV = np.ceil(x.shape[0]/nFolds).astype(int)

        indices_CV =[]
        for val in generate_batches(nBatchCV,y, mode='representative'):
            indices_CV.append(val)

        epoch_params = self.epoch_params.copy()
        update_params = self.update_params

        self.epoch_params[0] = 0

        final_ep, evoCosteEpoch = self.fit(x,y)

        initial_W = self.coefs_.copy()
        self.epoch_params = epoch_params

        number_epoch_CV = np.zeros(nFolds).astype(int)

        cv_metrics = np.zeros((len(metrics),nFolds))


        for kCV in range(nFolds):
            self.coefs_ = initial_W.copy()
            self.update_params = update_params

            ind_val = indices_CV[kCV]
            #ind_train = [v for v in range(x_train.shape[0]) if v not in ind_val]
            ind_train=[]
            for kTrain in range(nFolds):
                if kTrain != kCV:
                    ind_train += indices_CV[kTrain]
            # Getting the unique values
            ind_train = list(set(ind_train))

            #final_ep, evoCosteEpoch = model.fit(x_train,y_train, val_stop=[1e-3,5])
            final_ep, evoCosteEpoch = self.fit(x[ind_train,:],y[ind_train], x_val=x[ind_val,:], y_val=y[ind_val], val_stop=val_stop)
            if verbose:
                print('Fold %d : (Initial cost = %2.3f) Final epoch = %d'%(kCV+1, evoCosteEpoch[0], final_ep))

            ye_val = self.predict(x[ind_val,:])
            cv_metrics[:,kCV] = compute_metrics(y[ind_val], ye_val, metrics)

            number_epoch_CV[kCV] = final_ep


        # Re-train the network with the whole training set
        # ------------------------------------------------
        self.coefs_ = initial_W.copy()
        self.update_params = update_params
        self.epoch_params = [int(number_epoch_CV.max()*1.1), epoch_params[1]]

        final_ep, evoCosteEpoch = self.fit(x,y, val_stop=val_stop)

        return final_ep, evoCosteEpoch, cv_metrics


#------------------------------------------------------------------------------
"""
  MLPBayesBinW  CLASS

  Classifier for binary problems using a single output MLP neural network
  The decision rule is threshold based (threshold at 0)
  The loss function is an estimate of the Bayes risk based on the Parzen
  Windows estimator

  Parameters
  ----------

  * layers_size : number of neurons of each hidden layer [list or tuple] nHidden
  * activations : type of activation of each layer [list] nHidden + 1
      Includes the activation of the output layer (this must be identity or tanh)
      Possible types:
         - tanh
         - relu
         - logistic
         - identity
         - softmax
  * parzen_params: params of the Parzen Window [list]
        - params_parzen[0] : type of window
             + 'gauss'
             + 'uniform'
             + 'linear'
             + 'linear_inv'
             + 'quadratic'
             + 'quadratic_inv'
        - params_parzen[1] : support of the window around center
        - params_parzen[2] : center of the window

  * class_prob : probabilities of the classes  [list] nClasses
  * class_costs : costs of misclassification for each class [list] nClases
  * update: type of gradient update
      Possible types:
          - momentum
          - gradient
          - gradient-adapt
  * learning_rate_init: initial value of the learning rate [real]
  * learning_rate_inc: increment factor for learning rate, if 'gradient-adapt' [real]
  * learning_rate_dec: decrease factor for learning rate, if 'gradient-adapt' [real]
  * momentum: momentum value, if 'momentum' [real]
  * drop_out: probability of drop-out for each layer, including input [list] nHidden + 1
  * n_epoch: number of epochs [int]
  * n_batch: size of the mini-batch (0 if batch-mode is used) [int]
  * type_init: type of initizalization for network weights ('uniform', 'gaussian')
  * warm_start: flag to allow a warm start from previous solution [bool]
  * flag_evo: flag to compute the loss value after each epoch [bool]


"""
#------------------------------------------------------------------------------

class MLPBayesBinW(MLPnn):

    def __init__(self, layers_size=(30,),
                 activations=['relu','identity'],
                 parzen_params=['gauss',1,0],
                 class_prob=[0.5, 0.5],
                 class_cost=[1,1],
                 update='momentum',
                 learning_rate_init=1e-4,
                 learning_rate_inc=1.05,
                 learning_rate_dec=2,
                 momentum = 0.9,
                 drop_out = [],
                 n_epoch = 1000,
                 n_batch = 250,
                 type_batch = 'representative',
                 type_init = 'uniform',
                 warm_start=False,
                 flag_evo=False
                 ):

        super().__init__(layers_size, activations)

        #self.name = 'MLPBayesBin'
        self.layers_size = layers_size
        self.activations = activations
        self.parzen_params = parzen_params
        self.class_prob = class_prob
        self.class_cost = class_cost
        self.update = update
        self.learning_rate_init = learning_rate_init
        self.learning_rate_inc = learning_rate_inc
        self.learning_rate_dec = learning_rate_dec
        self.momentum = momentum
        self.drop_out = drop_out
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.type_batch = type_batch
        self.type_init = type_init
        self.warm_start = warm_start
        self.flag_evo = flag_evo
        #
        # Initializing attributes that will be set in fit
        self.coefs_ = None


    def _preprocess_data(self, x, y, sample_weight):
        nClases = 2

        # Labels converted to -1, +1
        if 0 not in list(np.unique(y)):
            y = y - 1

        self.classes_ = list(np.unique(y))


        if len(self.class_prob) == 0:
            self.class_prob = np.zeros(nClases)
            for kclase in range(nClases):
                v = np.nonzero(y==kclase)
                self.class_prob[kclase] = len(v[0])/y.shape[0]

        if len(self.class_cost) == 0:
            self.class_cost = np.ones(nClases)

        if type(sample_weight) == type(None):
            sample_weight = np.ones((y.shape[0],))


        yout = np.where(y==0, -1,1)
        xout = x.T

        return xout, yout, sample_weight

    def _cost(self, y, ye, sample_weight):

        v0 = np.nonzero(y!=1)
        v1 = np.nonzero(y==1)

        #coste = self.class_cost[0] * self.class_prob[0] * np.mean(intGeneralN(ye[0,v0],self.parzen_params))
        #coste += self.class_cost[1] * self.class_prob[0] * np.mean(intGeneralN(-ye[0,v1],self.parzen_params))

        coste = self.class_cost[0] * self.class_prob[0] * np.matmul(intGeneralN(ye[0,v0],self.parzen_params), sample_weight[v0])
        coste += self.class_cost[1] * self.class_prob[0] * np.matmul(intGeneralN(-ye[0,v1],self.parzen_params), sample_weight[v1])

        return coste


    def _derivatives_cost(self, y, ye, sample_weight):

        nBatchSize = y.shape[0]

        d=np.zeros((1,nBatchSize))

        v0 = np.nonzero(y==-1)
        v1 = np.nonzero(y==1)

        d[0,v0] = self.class_cost[0] * self.class_prob[0] * np.multiply(fdpGeneralN(ye[0,v0],self.parzen_params),sample_weight[v0]) / len(v0[0])
        d[0,v1] = -self.class_cost[1] * self.class_prob[1] * np.multiply(fdpGeneralN(-ye[0,v1],self.parzen_params), sample_weight[v1]) / len(v1[0])

        # Propagation through the nonlinear function of the output laye
        if self.activations[-1] == 'tanh':
            d=np.multiply(d,1-np.square(ye))

        elif self.activations[-1] == 'logistic':
            d=np.multiply(d,ye-np.square(ye))

        elif self.activations[-1] == 'relu':
            d = np.where(ye<=0,0,d)

        return d

    def gradiente_Numerico(self, y,ye):

        pasoGrad = 1e-4

        #ye = self._forward_pass_fast(x)
        costeRef = self._cost(y,ye)
        grad_d = np.zeros(ye.shape)
        for kA in range(ye.shape[0]):
            for kB in range(ye.shape[1]):
                yeG = 1.0*ye
                yeG[kA,kB] = yeG[kA,kB] + pasoGrad

                costeGrad = self._cost(y,yeG)
                grad_d[kA,kB] =  (costeGrad - costeRef)/pasoGrad

        return grad_d


    def predict(self, x):

        ys = self._forward_pass_fast(x.T)
        #ye = np.argmax(ys, axis=0)
        ye = np.where(ys[0,:]>0,1,0)

        return ye

    def score(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(y.shape)

        # Scoring logic here, taking sample_weight into account
        y_pred = self.predict(X)
        score_metric = balanced_accuracy_score(y, y_pred)
        #score_metric = self._cost(y, y_pred, sample_weight)

        return score_metric

    def soft_output(self, x):

        ys = self._forward_pass_fast(x.T)
        ye = ys[0,:]

        return ye


    def _fit(self, x, y, sample_weight, x_val=np.zeros(0), y_val=np.zeros(0), val_stop=[0,0], incremental=False):

        if type(x_val) == float:

            nFolds = np.ceil(1/x_val).astype(int)
            nBatchCV = np.ceil(x.shape[0]/nFolds).astype(int)

            indices_CV =[]
            for val in generate_batches(nBatchCV,y, mode='representative'):
                indices_CV.append(val)

            ind_val = indices_CV[-1]
            #ind_train = [v for v in range(x_train.shape[0]) if v not in ind_val]
            ind_train=[]
            for kTrain in range(nFolds-1):
                ind_train += indices_CV[kTrain]
            # Getting the unique values
            ind_train = list(set(ind_train))

            x_val = x[ind_val,:]
            y_val = y[ind_val]
            x = x[ind_train,:]
            y = y[ind_train]

        x, y, sample_weight = self._preprocess_data(x,y,sample_weight)
        if x_val.shape[0] > 0:
            x_val, y_val = self._preprocess_data(x_val,y_val)


        dim_in, num_pat = x.shape
        dim_out = 1

        first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
        )

        if first_pass:#(len(self.coefs_) == 0) or (self.warm_start==False):
            self._initialize(dim_in, dim_out, self.type_init)

        nEpochs = self.n_epoch
        nBatch = self.n_batch
        nCapas = self.num_hidden + 1

        if (nBatch == 0) or (nBatch > num_pat):
            nBatch = num_pat

        if (self.flag_evo) or (val_stop[0] > 0):
            evoCosteEpoch=np.zeros(nEpochs+1)
            if x_val.shape[0] > 0:
                ye = self._forward_pass_fast(x_val)
                coste = self._cost(y_val,ye)
            else:
                ye = self._forward_pass_fast(x)
                coste = self._cost(y,ye)


            evoCosteEpoch[0]=coste

            if val_stop[0] > 0:
                opt_cost = coste
                opt_epoch = 0
                opt_W = self.coefs_.copy()


        #dW=[[] for k in range(nCapas)]
        Wdo=[[] for k in range(nCapas)]
        Wdon=[[] for k in range(nCapas)]
        mX=np.diag(np.ones(dim_in)).astype(float)
        mW=[[] for k in range(nCapas)]

        if len(self.drop_out)==0:
            self.drop_out=[0 for k in range(nCapas)]

        if len(self.dWm) == 0:
            for ko in range(nCapas):
                self.dWm.append(np.zeros((np.shape(self.coefs_[ko]))))

        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]/(1-self.drop_out[ko])
            mW[ko]=np.ones(np.shape(self.coefs_[ko]))

        #--------------------------------------------------------------------------
        # Se inicia el procedimiento de entrenamiento
        #--------------------------------------------------------------------------
        if self.type_batch in ['class_equitative', 'representative']:
            paramsBatch = y

        for kEpoch in range(nEpochs):

            for indBatch in generate_batches(nBatch, paramsBatch, mode=self.type_batch):
                # Generation of the Drop-Out Masks
                if np.sum(self.drop_out)>0:
                    mX=np.diag(np.random.uniform(0,1,dim_in)>self.drop_out[0]).astype(float)
                    for ko in range(nCapas-1):
                        (Nb,Na)=np.shape(self.coefs_[ko])
                        mW[ko]=np.matmul(np.diag((np.random.uniform(0,1,Nb)>self.drop_out[ko+1])).astype(float), np.ones((Nb,Na)))
                        Wdo[ko]=np.multiply(self.coefs_[ko],mW[ko])

                    Wdo[nCapas-1]=self.coefs_[nCapas-1]
                else:
                    Wdo=self.coefs_.copy()

                (ye,Os) = self._forward_pass_W(x[:,indBatch],Wdo)
                #----------------------------------------------------------------------
                # Cálculo de gradientes
                #----------------------------------------------------------------------
                d = self._derivatives_cost(y[indBatch], ye, sample_weight[indBatch])
                #d = self.gradiente_Numerico(y[indBatch], ye)
                dW = self._backpropagation(x[:,indBatch], Os, d)
                #----------------------------------------------------------------------
                # Nuevos pesos iteración
                #----------------------------------------------------------------------
                if self.update == 'gradient-adapt':
                    coste = self._cost(y[indBatch],ye, self.cost_weights[0,indBatch])
                    salida = self._update_W_X(dW,mX,mW,x[:,indBatch],y[indBatch],paramsBatch,coste)

                    if salida:
                        for ko in range(nCapas):
                            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

                        evoCosteEpoch[kEpoch+1:nEpochs+1]=coste
                        print("Step size in the limit (zero) ...")

                        if val_stop[0] == 0:
                            opt_epoch = kEpoch
                        else:
                            self.coefs_ = opt_W

                        return opt_epoch, evoCosteEpoch

                else:
                    self._update_W(dW,mX,mW)

            #----------------------------------------------------------------------
            # Actualización del coste (global) y validación
            #----------------------------------------------------------------------
            if (self.flag_evo) or (val_stop[0] > 0):
                for ko in range(nCapas):
                    Wdon[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

                if x_val.shape[0] > 0:
                    (ye,Os)=self._forward_pass_W(x_val,Wdon)
                    coste = self._cost(y_val,ye)
                else:
                    (ye,Os)=self._forward_pass_W(x,Wdon)
                    coste = self._cost(y,ye)


                evoCosteEpoch[kEpoch+1] = coste

                if val_stop[0] > 0:
                    if coste < opt_cost:
                        opt_cost = coste
                        opt_epoch = kEpoch+1
                        opt_W = self.coefs_.copy()

                    if kEpoch > val_stop[1]:
                        rel_dec = (np.min(evoCosteEpoch[kEpoch-val_stop[1]:kEpoch])-evoCosteEpoch[kEpoch])/np.min(evoCosteEpoch[kEpoch-val_stop[1]:kEpoch])

                        if rel_dec < val_stop[0]:
                            self.coefs_ = opt_W

                            break

        # Training is over
        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

        if (self.flag_evo == False) and (val_stop[0]==0):
            ye=self._forward_pass_fast(x)
            evoCosteEpoch = self._cost(y,ye, sample_weight)
            opt_epoch = nEpochs

        elif val_stop[0] == 0:
            opt_epoch = nEpochs

        self.loss_ = evoCosteEpoch
        self.opt_epoch = opt_epoch


        return self

    def fit(self, X, y, sample_weight=None):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        Returns
        -------
        self : object
            Returns a trained MLP model.
        """
        #self._validate_params()
        self.n_features_in_ = X.shape[0]
        if X.shape[0] != y.shape[0]:
            raise ValueError('Sizes of X and y are not compatible..')

        self._fit(X, y, sample_weight, incremental=False)

        return self

    def fit_cv_stop(self, x, y, val_stop, metrics=['MAE', 'AMAE'], nFolds=5, verbose=False):
        #metrics = ['MAE', 'AMAE', 'accuracy', 'balanced_accuracy', 'f1', 'cohen_kappa', 'cohen_kappa_linear', 'cohen_kappa_quadratic', 'matthews']

        nBatchCV = np.ceil(x.shape[0]/nFolds).astype(int)

        indices_CV =[]
        for val in generate_batches(nBatchCV,y, mode='representative'):
            indices_CV.append(val)

        epoch_params = self.epoch_params.copy()
        update_params = self.update_params

        self.epoch_params[0] = 0

        final_ep, evoCosteEpoch = self.fit(x,y)

        initial_W = self.coefs_.copy()
        self.epoch_params = epoch_params

        number_epoch_CV = np.zeros(nFolds).astype(int)

        cv_metrics = np.zeros((len(metrics),nFolds))

        for kCV in range(nFolds):
            self.coefs_ = initial_W.copy()
            self.update_params = update_params

            ind_val = indices_CV[kCV]
            #ind_train = [v for v in range(x_train.shape[0]) if v not in ind_val]
            ind_train=[]
            for kTrain in range(nFolds):
                if kTrain != kCV:
                    ind_train += indices_CV[kTrain]
            # Getting the unique values
            ind_train = list(set(ind_train))

            #final_ep, evoCosteEpoch = model.fit(x_train,y_train, val_stop=[1e-3,5])
            final_ep, evoCosteEpoch = self.fit(x[ind_train,:],y[ind_train], x_val=x[ind_val,:], y_val=y[ind_val], val_stop=val_stop)
            if verbose:
                print('Fold %d : (Initial cost = %2.3f) Final epoch = %d'%(kCV+1, evoCosteEpoch[0], final_ep))

            ye_val = self.predict(x[ind_val,:])
            cv_metrics[:,kCV] = compute_metrics(y[ind_val], ye_val, metrics)

            number_epoch_CV[kCV] = final_ep


        # Re-train the network with the whole training set
        # ------------------------------------------------
        self.coefs_ = initial_W.copy()
        self.update_params = update_params
        self.epoch_params = [int(number_epoch_CV.max()*1.1), epoch_params[1]]

        final_ep, evoCosteEpoch = self.fit(x,y, val_stop=val_stop)


        return final_ep, evoCosteEpoch, cv_metrics

    def get_params(self, deep=True):
        return {
            'layers_size': self.layers_size,
            'activations': self.activations,
            'parzen_params': self.parzen_params,
            'class_prob': self.class_prob,
            'class_cost': self.class_cost,
            'update': self.update,
            'learning_rate_init': self.learning_rate_init,
            'learning_rate_inc': self.learning_rate_inc,
            'learning_rate_dec': self.learning_rate_dec,
            'momentum': self.momentum,
            'drop_out': self.drop_out,
            'n_epoch': self.n_epoch,
            'n_batch': self.n_batch,
            'type_batch': self.type_batch,
            'type_init': self.type_init,
            'warm_start': self.warm_start,
            'flag_evo': self.flag_evo
        }


    # def set_params(self, **params):
    #     for key, value in params.items():
    #         setattr(self, key, value)
    #     return self


class MLPBayesBinMT(MLPnn):

    def __init__(self, layers_size=(100,),
                 activations=['relu','identity'],
                 parzen_params=['gauss',1,0],
                 class_prob=None,
                 class_cost=None,
                 update='momentum',
                 learning_rate_init=1e-4,
                 learning_rate_inc=1.05,
                 learning_rate_dec=2,
                 momentum = 0.9,
                 drop_out = [],
                 n_epoch = 100,
                 n_batch = 128,
                 type_batch = 'representative',
                 type_init = 'uniform',
                 warm_start=False,
                 flag_evo=False
                 ):

        super().__init__(layers_size, activations)

        #self.name = 'MLPBayesBin'
        self.layers_size = layers_size
        self.activations = activations
        self.parzen_params = parzen_params
        self.class_prob = class_prob
        self.class_cost = class_cost
        self.update = update
        self.learning_rate_init = learning_rate_init
        self.learning_rate_inc = learning_rate_inc
        self.learning_rate_dec = learning_rate_dec
        self.momentum = momentum
        self.drop_out = drop_out
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.type_batch = type_batch
        self.type_init = type_init
        self.warm_start = warm_start
        self.flag_evo = flag_evo
        #
        self.dW = []
        self.dWm = []



    def _preprocess_data(self, x, y, sample_weight):
        nClases = 2

        # Ensure y is a 2D array
        if y.ndim == 1:
            y = y.reshape(-1,1)

        self.num_task = y.shape[1]

        # Convert labels to 0 and 1 if they are not already in that format
        if 0 not in list(np.unique(y)):
            y = y - 1

        self.classes_ = np.unique(y.astype(int)).tolist()

        # Calculate class probabilities if not already provided
        if not self.class_prob:
            self.class_prob = {}
            for ktask in range(self.num_task):
                unique, counts = np.unique(y[:, ktask], return_counts=True)
                aux = np.zeros(nClases)
                #for label, count in zip(unique, counts):
                for label, count in enumerate(counts):
                    aux[label] = count / y.shape[0]
                self.class_prob[ktask] = aux

        # Set cost classes to ones if not provided
        if not self.class_cost:
            self.class_cost = {ktask: np.ones(nClases) for ktask in range(self.num_task)}

        # Initialize sample_weight to ones if not provided
        if type(sample_weight) == np.ndarray:
            if sample_weight.ndim == 1:
                sample_weight = np.tile(sample_weight, (self.num_task, 1))
            else:
                sample_weight = sample_weight.T
        elif type(sample_weight) ==  dict:
            self.loss_func = 'wmmse'
            weights = np.ones_like(y.T)
            for key in sample_weight.keys():
                weights[key,:] = sample_weight[key]
        elif type(sample_weight) == tuple:
            if sample_weight[1].ndim ==  1:
                weights = np.tile(sample_weight, (self.num_task, 1))
            else:
                weights = sample_weight[1].T

            for k, key in enumerate(sample_weight[0].keys()):
                    weights[key,:] = sample_weight[0][key]*weights[key,:]
        else:
            sample_weight = np.ones_like(y.T)


        # Convert y labels to -1 and 1
        yout = np.where(y.T == 0, -1, 1)
        xout = x.T

        # Costs and probabilities: Numpy array (for efficiency)
        self.cbayes = np.zeros((self.num_task,nClases))
        for ktask in range(self.num_task):
            self.cbayes[ktask,0] = self.class_cost[ktask][0] * self.class_prob[ktask][0]
            self.cbayes[ktask,1] = self.class_cost[ktask][1] * self.class_prob[ktask][1]


        return xout, yout, sample_weight

    def _cost(self, y, ye, sample_weight):
        cost = 0
        for ktask in range(self.num_task):
            v0 = np.nonzero(y[ktask, :]==-1)
            v1 = np.nonzero(y[ktask, :]==1)
            #cost += self.class_cost[ktask][0] * self.class_prob[ktask][0] * np.matmul(intGeneralN(ye[ktask, v0]), sample_weight[ktask, v0].T) / len(v0[0])
            #cost += self.class_cost[ktask][1] * self.class_prob[ktask][1] * np.matmul(intGeneralN(-ye[ktask, v1]), sample_weight[ktask, v1].T) / len(v1[0])
            cost += self.cbayes[ktask, 0] * np.mean(intGeneralN(ye[ktask, v0]) * sample_weight[ktask, v0])
            cost += self.cbayes[ktask, 1] * np.mean(intGeneralN(-ye[ktask, v1]) * sample_weight[ktask, v1])

        return cost


    def _derivatives_cost(self, y, ye, sample_weight):

        d=np.zeros((y.shape))
        for ktask in range(self.num_task):
            v0 = np.nonzero(y[ktask, :]==-1)
            v1 = np.nonzero(y[ktask, :]==1)
            #d[ktask, v0] = self.class_cost[ktask][0] * self.class_prob[ktask][0] * np.multiply(fdpGeneralN(ye[ktask, v0],self.parzen_params),sample_weight[ktask, v0]) / len(v0[0])
            #d[ktask, v1] = -self.class_cost[ktask][1] * self.class_prob[ktask][1] * np.multiply(fdpGeneralN(-ye[ktask, v1],self.parzen_params), sample_weight[ktask, v1]) / len(v1[0])
            d[ktask, v0] = self.cbayes[ktask, 0] * np.multiply(fdpGeneralN(ye[ktask, v0],self.parzen_params),sample_weight[ktask, v0]) / len(v0[0])
            d[ktask, v1] = -self.cbayes[ktask, 1] * np.multiply(fdpGeneralN(-ye[ktask, v1],self.parzen_params), sample_weight[ktask, v1]) / len(v1[0])

        # Propagation through the nonlinear function of the output laye
        if self.activations[-1] == 'tanh':
            d=np.multiply(d,1-np.square(ye))

        elif self.activations[-1] == 'logistic':
            d=np.multiply(d,ye-np.square(ye))

        elif self.activations[-1] == 'relu':
            d = np.where(ye<=0,0,d)

        return d

    def gradiente_Numerico(self, y, ye, sample_weight):

        pasoGrad = 1e-4

        #ye = self._forward_pass_fast(x)
        costeRef = self._cost(y, ye, sample_weight)
        grad_d = np.zeros(ye.shape)
        for kA in range(ye.shape[0]):
            for kB in range(ye.shape[1]):
                yeG = 1.0*ye
                yeG[kA,kB] = yeG[kA,kB] + pasoGrad

                costeGrad = self._cost(y, yeG, sample_weight)
                grad_d[kA,kB] =  (costeGrad - costeRef)/pasoGrad

        return grad_d

    # def get_cost_weights(self, cost_weights):

    #     nPat = cost_weights.shape[0]
    #     self.cost_weights = np.ones((1,nPat))
    #     self.cost_weights[0,:] = cost_weights


    def predict(self, x):
        ys = self._forward_pass_fast(x.T)
        #ye = np.argmax(ys, axis=0)
        ye = np.where(ys>0,1,0)

        return ye.T


    def soft_output(self, x):
        ys = self._forward_pass_fast(x.T)

        return ys.T


    def _fit(self, x, y, sample_weight, x_val=np.zeros(0), y_val=np.zeros(0), val_stop=[0,0], incremental=False):

        if type(x_val) == float:
            #print('    Generating validation set...')

            nFolds = np.ceil(1/x_val).astype(int)
            nBatchCV = np.ceil(x.shape[0]/nFolds).astype(int)

            indices_CV =[]
            for val in generate_batches(nBatchCV,y, mode='representative'):
                indices_CV.append(val)

            ind_val = indices_CV[-1]
            #ind_train = [v for v in range(x_train.shape[0]) if v not in ind_val]
            ind_train=[]
            for kTrain in range(nFolds-1):
                ind_train += indices_CV[kTrain]
            # Getting the unique values
            ind_train = list(set(ind_train))

            x_val = x[ind_val,:]
            y_val = y[ind_val]
            x = x[ind_train,:]
            y = y[ind_train]

            #print('    Validation set generated')


        x, y, sample_weight = self._preprocess_data(x,y,sample_weight)
        if x_val.shape[0] > 0:
            x_val, y_val = self._preprocess_data(x_val,y_val)


        dim_in, num_pat = x.shape
        dim_out = y.shape[0]

        first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
        )

        if first_pass:
            self._initialize(dim_in, dim_out, self.type_init)
        else:
            if not hasattr(self, "num_hidden"):
                self.num_hidden = len(self.coefs_)-1

            if len(self.dW) == 0:
                self.dW = []
                self.dWm = []
                for k in range(self.num_hidden+1):
                    self.dW.append(self.coefs_[k]*0.0)
                    self.dWm.append(self.coefs_[k]*0.0)

                self.learning_rate = self.learning_rate_init

        nEpochs = self.n_epoch
        nBatch = self.n_batch
        nCapas = self.num_hidden + 1

        if (nBatch == 0) or (nBatch > num_pat):
            nBatch = num_pat

        if (self.flag_evo) or (val_stop[0] > 0):
            evoCosteEpoch=np.zeros(nEpochs+1)
            if x_val.shape[0] > 0:
                ye = self._forward_pass_fast(x_val)
                coste = self._cost(y_val,ye,sample_weight)
            else:
                ye = self._forward_pass_fast(x)
                coste = self._cost(y,ye,sample_weight)


            evoCosteEpoch[0]=coste

            if val_stop[0] > 0:
                opt_cost = coste
                opt_epoch = 0
                opt_W = self.coefs_.copy()


        Wdo=[[] for k in range(nCapas)]
        Wdon=[[] for k in range(nCapas)]
        mX=np.diag(np.ones(dim_in)).astype(float)
        mW=[[] for k in range(nCapas)]

        if len(self.drop_out)==0:
            self.drop_out=[0 for k in range(nCapas)]

        # if len(self.dWm) == 0:
        #     for ko in range(nCapas):
        #         self.dWm.append(np.zeros((np.shape(self.coefs_[ko]))))

        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]/(1-self.drop_out[ko])
            mW[ko]=np.ones(np.shape(self.coefs_[ko]))

        #--------------------------------------------------------------------------
        # Se inicia el procedimiento de entrenamiento
        #--------------------------------------------------------------------------
        if self.type_batch in ['class_equitative', 'representative']:
            paramsBatch = y[0,:]

        for kEpoch in range(nEpochs):

            for indBatch in generate_batches(nBatch, paramsBatch, mode=self.type_batch):
                # Generation of the Drop-Out Masks
                if np.sum(self.drop_out)>0:
                    mX=np.diag(np.random.uniform(0,1,dim_in)>self.drop_out[0]).astype(float)
                    for ko in range(nCapas-1):
                        (Nb,Na)=np.shape(self.coefs_[ko])
                        mW[ko]=np.matmul(np.diag((np.random.uniform(0,1,Nb)>self.drop_out[ko+1])).astype(float), np.ones((Nb,Na)))
                        Wdo[ko]=np.multiply(self.coefs_[ko],mW[ko])

                    Wdo[nCapas-1]=self.coefs_[nCapas-1]
                else:
                    Wdo=self.coefs_.copy()

                (ye,Os) = self._forward_pass_W(x[:,indBatch],Wdo)
                #----------------------------------------------------------------------
                # Cálculo de gradientes
                #----------------------------------------------------------------------
                d = self._derivatives_cost(y[:,indBatch], ye, sample_weight[:,indBatch])
                #d = self.gradiente_Numerico(y[:,indBatch], ye, sample_weight[:,indBatch])
                dW = self._backpropagation(x[:,indBatch], Os, d)
                #----------------------------------------------------------------------
                # Nuevos pesos iteración
                #----------------------------------------------------------------------
                if self.update == 'gradient-adapt':
                    coste = self._cost(y[indBatch],ye, self.cost_weights[0,indBatch])
                    salida = self._update_W_X(dW,mX,mW,x[:,indBatch],y[indBatch],paramsBatch,coste)

                    if salida:
                        for ko in range(nCapas):
                            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

                        evoCosteEpoch[kEpoch+1:nEpochs+1]=coste
                        print("Step size in the limit (zero) ...")

                        if val_stop[0] == 0:
                            opt_epoch = kEpoch
                        else:
                            self.coefs_ = opt_W

                        return opt_epoch, evoCosteEpoch

                else:
                    self._update_W(dW,mX,mW)

            #----------------------------------------------------------------------
            # Actualización del coste (global) y validación
            #----------------------------------------------------------------------
            if (self.flag_evo) or (val_stop[0] > 0):
                for ko in range(nCapas):
                    Wdon[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

                if x_val.shape[0] > 0:
                    (ye,Os)=self._forward_pass_W(x_val,Wdon)
                    coste = self._cost(y_val,ye)
                else:
                    (ye,Os)=self._forward_pass_W(x,Wdon)
                    coste = self._cost(y, ye, sample_weight)


                evoCosteEpoch[kEpoch+1] = coste

                if val_stop[0] > 0:
                    if coste < opt_cost:
                        opt_cost = coste
                        opt_epoch = kEpoch+1
                        opt_W = self.coefs_.copy()

                    if kEpoch > val_stop[1]:
                        rel_dec = (np.min(evoCosteEpoch[kEpoch-val_stop[1]:kEpoch])-evoCosteEpoch[kEpoch])/np.min(evoCosteEpoch[kEpoch-val_stop[1]:kEpoch])

                        if rel_dec < val_stop[0]:
                            self.coefs_ = opt_W

                            break

        # Training is over
        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

        if (self.flag_evo == False) and (val_stop[0]==0):
            ye=self._forward_pass_fast(x)
            evoCosteEpoch = self._cost(y,ye, sample_weight)
            opt_epoch = nEpochs

        elif val_stop[0] == 0:
            opt_epoch = nEpochs

        self.loss_ = evoCosteEpoch
        self.opt_epoch = opt_epoch


        return self

    def fit(self, X, y, sample_weight=None):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
        Returns
        -------
        self : object
            Returns a trained MLP model.
        """
        #self._validate_params()
        self.n_features_in_ = X.shape[0]
        if X.shape[0] != y.shape[0]:
            raise ValueError('Sizes of X and y are not compatible..')

        return self._fit(X, y, sample_weight, incremental=False)

    def fit_cv_stop(self, x, y, val_stop, metrics=['MAE', 'AMAE'], nFolds=5, verbose=False):
        #metrics = ['MAE', 'AMAE', 'accuracy', 'balanced_accuracy', 'f1', 'cohen_kappa', 'cohen_kappa_linear', 'cohen_kappa_quadratic', 'matthews']

        nBatchCV = np.ceil(x.shape[0]/nFolds).astype(int)

        indices_CV =[]
        for val in generate_batches(nBatchCV,y, mode='representative'):
            indices_CV.append(val)

        epoch_params = self.epoch_params.copy()
        update_params = self.update_params

        self.epoch_params[0] = 0

        final_ep, evoCosteEpoch = self.fit(x,y)

        initial_W = self.coefs_.copy()
        self.epoch_params = epoch_params

        number_epoch_CV = np.zeros(nFolds).astype(int)

        #cost_value_CV = np.zeros(nFolds)
        #AMAE_CV_val = np.zeros(nFolds)
        #MAE_CV_val = np.zeros(nFolds)

        cv_metrics = np.zeros((len(metrics),nFolds))


        for kCV in range(nFolds):
            self.coefs_ = initial_W.copy()
            self.update_params = update_params

            ind_val = indices_CV[kCV]
            #ind_train = [v for v in range(x_train.shape[0]) if v not in ind_val]
            ind_train=[]
            for kTrain in range(nFolds):
                if kTrain != kCV:
                    ind_train += indices_CV[kTrain]
            # Getting the unique values
            ind_train = list(set(ind_train))

            #final_ep, evoCosteEpoch = model.fit(x_train,y_train, val_stop=[1e-3,5])
            final_ep, evoCosteEpoch = self.fit(x[ind_train,:],y[ind_train], x_val=x[ind_val,:], y_val=y[ind_val], val_stop=val_stop)
            if verbose:
                print('Fold %d : (Initial cost = %2.3f) Final epoch = %d'%(kCV+1, evoCosteEpoch[0], final_ep))

            ye_val = self.predict(x[ind_val,:])
            cv_metrics[:,kCV] = compute_metrics(y[ind_val], ye_val, metrics)

            number_epoch_CV[kCV] = final_ep


            #cost_value_CV[kCV] = evoCosteEpoch[final_ep]

            # CM_val = confusion_matrix(y[ind_val], ye_val)
            # MAE_val, AMAE_val = calc_MAE_AMAE(CM_val)

            # MAE_CV_val[kCV] = MAE_val
            # AMAE_CV_val[kCV] = AMAE_val

            # print('   MAE (val) = %2.3f' %(MAE_val))
            # print('  AMAE (val) = %2.3f' %(AMAE_val))

        # Re-train the network with the whole training set
        # ------------------------------------------------
        self.coefs_ = initial_W.copy()
        self.update_params = update_params
        self.epoch_params = [int(number_epoch_CV.max()*1.1), epoch_params[1]]

        final_ep, evoCosteEpoch = self.fit(x,y, val_stop=val_stop)
        #print(' ')
        #print('Final training : (Initial cost = %2.3f) Final epoch = %d'%(evoCosteEpoch[0], final_ep))

        #ye_val = self.predict(x)
        #CM_val = confusion_matrix(y, ye_val)
        #MAE_val, AMAE_val = calc_MAE_AMAE(CM_val)
        #print('  AMAE (train) = %2.3f' %(AMAE_val))
        #print('   MAE (train) = %2.3f' %(MAE_val))

        return final_ep, evoCosteEpoch, cv_metrics


#------------------------------------------------------------------------------
"""
  MLPBayes  CLASS

  Classifier for multiclass problems using multiple output MLP neural network
  The decision rule is based on arg-max
  The loss function is an estimate of the Bayesian cost based on the Parzen
  Windows estimator

  Parameters
  ----------

  * layers_size : number of neurons of each hidden layer [list or tuple] nHidden
  * activations : type of activation of each layer [list] nHidden + 1
      Includes the activation of the output layer (this must be identity or tanh)
      Possible types:
         - tanh
         - relu
         - logistic
         - identity
         - softmax
  * parzen_params: params of the Parzen Window [list]
        - params_parzen[0] : type of window
             + 'gauss'
             + 'uniform'
             + 'linear'
             + 'linear_inv'
             + 'quadratic'
             + 'quadratic_inv'
        - params_parzen[1] : support of the window around center
        - params_parzen[2] : center of the window
  * class_prob : probabilities of the classes  [list] nClasses
  * cost_matriz : decission costs among classes  [array] nClasses x nClasses
  * update: type of gradient update
      Possible types:
          - momentum
          - gradient
          - gradient-adapt
  * learning_rate_init: initial value of the learning rate [real]
  * learning_rate_inc: increment factor for learning rate, if 'gradient-adapt' [real]
  * learning_rate_dec: decrease factor for learning rate, if 'gradient-adapt' [real]
  * momentum: momentum value, if 'momentum' [real]
  * drop_out: probability of drop-out for each layer, including input [list] nHidden + 1
  * n_epoch: number of epochs [int]
  * n_batch: size of the mini-batch (0 if batch-mode is used) [int]
  * type_init: type of initizalization for network weights ('uniform', 'gaussian')
  * warm_start: flag to allow a warm start from previous solution [bool]
  * flag_evo: flag to compute the loss value after each epoch [bool]


"""
#------------------------------------------------------------------------------

class MLPBayes(MLPnn):

    def __init__(self, layers_size=[100],
                 activations=['relu', 'identity'],
                 loss_func='BayesMSpair',
                 parzen_params=['gauss',1,0],
                 class_prob=[0.5, 0.5],
                 cost_matrix=np.array([[0,1],[1,0]]),
                 update='momentum',
                 learning_rate_init=1e-4,
                 learning_rate_inc=1.05,
                 learning_rate_dec=2,
                 momentum = 0.9,
                 drop_out = [],
                 n_epoch = 1000,
                 n_batch = 250,
                 type_batch = 'representative',
                 type_init = 'uniform',
                 warm_start = False,
                 flag_evo=False
                 ):

        super().__init__(layers_size, activations)

        self.loss_func = loss_func
        self.parzen_params = parzen_params
        self.class_prob = class_prob
        self.cost_matrix = cost_matrix
        self.update = update
        self.learning_rate_init = learning_rate_init
        self.learning_rate_inc = learning_rate_inc
        self.learning_rate_dec = learning_rate_dec
        self.momentum = momentum
        self.drop_out = drop_out
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.type_batch = type_batch
        self.type_init = type_init
        self.warm_start = warm_start
        self.flag_evo = flag_evo
        #

    def _preprocess_data(self, x, y):

        if 0 not in list(np.unique(y)):
            y = y - 1

        if not hasattr(self, "class_labels"):
            self.classes_ = list(np.unique(y))

        nClases = len(self.classes_)


        if len(self.class_prob) == 0:
            self.class_prob = np.zeros(nClases)
            for kclase in range(nClases):
                v = np.nonzero(y==kclase)
                self.class_prob[kclase] = len(v[0])/y.shape[0]

        if len(self.cost_matrix) == 0:
            self.cost_matrix = np.zeros((nClases, nClases))
            for kclase in range(nClases):
                for kotra in range(nClases):
                    self.cost_matrix[kotra,kclase] = np.abs(kclase-kotra)

        yout = np.where(y==0, 1,0)
        for k in range(1,nClases):
            yout = np.vstack((yout,np.where(y==k, 1,0)))

        xout = x.T

        return xout, yout

    def _cost(self, y, ye, param=[]):

        nClases = len(self.classes_)
        coste=0

        if self.loss_func=='BayesMSpair':

            for ktrue in range(nClases):
                #vc=find(y[ktrue,:],1)
                vc = np.nonzero(y[ktrue,:]==1)
                for kdec in range(nClases):
                    #costeDec = intGeneralN(ye[kdec,vc]-ye[ktrue,vc],self.parzen_params)
                    costeDec = 1.0 - intGeneralN(ye[ktrue,vc]-ye[kdec,vc],self.parzen_params)
                    coste += self.cost_matrix[kdec,ktrue]*self.class_prob[ktrue]*np.mean(costeDec)

        elif self.loss_func=='BayesMSsum':

            for ktrue in range(nClases):
                #vc=find(y[ktrue,:],1)
                vc = np.nonzero(y[ktrue,:]==1)
                for kdec in range(nClases):
                    #otras=setdiff(range(nClases),[kdec])
                    otras = [x for x in range(nClases) if x not in [kdec]]
                    costeDec = 0.0
                    for kotra in otras:
                        costeDec += 1.0 - intGeneralN(ye[kotra,vc]-ye[kdec,vc],self.parzen_params)

                    coste += self.cost_matrix[kdec,ktrue]*self.class_prob[ktrue]*np.mean(costeDec)


        elif self.loss_func=='BayesMSmax':

            for ktrue in range(nClases):
                #vc=find(y[ktrue,:],1)
                vc = np.nonzero(y[ktrue,:]==1)
                for kdec in range(nClases):
                    #otras=setdiff(range(nClases),[kdec])
                    otras = [x for x in range(nClases) if x not in [kdec]]
                    aux = np.max(ye[otras,:],axis=0)
                    costeDec = 1.0 - intGeneralN(aux[vc]-ye[kdec,vc],self.parzen_params)
                    coste += self.cost_matrix[kdec,ktrue]*self.class_prob[ktrue]*np.mean(costeDec)

        elif self.loss_func=='BayesMStot':

            for ktrue in range(nClases):
                #vc=find(y[ktrue,:],1)
                vc = np.nonzero(y[ktrue,:]==1)
                for kdec in range(nClases):
                    #otras=setdiff(range(nClases),[kdec])
                    otras = [x for x in range(nClases) if x not in [kdec]]
                    costeDec = 1.0
                    for kotra in otras:
                        costeDec = costeDec*(1.0 - intGeneralN(ye[kotra,vc]-ye[kdec,vc],self.parzen_params))

                    coste += self.cost_matrix[kdec,ktrue]*self.class_prob[ktrue]*np.mean(costeDec)

        # for kclase in range(nClases):
        #     coste += self.class_prob[kclase]*self.cost_matrix[nClases-1,kclase]
        #     vc=np.nonzero(y[0,:]==kclase)
        #     for kotra in range(kclase):
        #         costeOtra=intGeneralN(self.thresholds[kotra]-ye[0,vc],self.parzen_params)
        #         coste += (self.cost_matrix[kotra,kclase]-self.cost_matrix[kotra+1,kclase])*self.class_prob[kclase]*np.mean(costeOtra)
        #     for kotra in range(kclase,nClases-1):
        #         costeOtra=1-intGeneralN(ye[0,vc]-self.thresholds[kotra],self.parzen_params)
        #         coste += (self.cost_matrix[kotra,kclase]-self.cost_matrix[kotra+1,kclase])*self.class_prob[kclase]*np.mean(costeOtra)

        return coste

    def _update(self, dthresholds,dW,mX,mW):

        nCapas = self.num_hidden + 1
        if self.update =='momentum':
            for ko in range(nCapas):
                self.dWm[ko]=self.dWm[ko]*self.update_params[1]+self.update_params[0]*np.multiply(dW[ko],mW[ko])
                self.coefs_[ko]=self.coefs_[ko]-np.multiply(self.dWm[ko],mW[ko])

                self.thresholdsMom = self.thresholdsMom*self.update_params[1] + self.update_params[0]*dthresholds
                self.thresholds = self.thresholds-self.thresholdsMom

        elif self.update =='gradient':
            for ko in range(nCapas):
                self.coefs_[ko]=self.coefs_[ko]-self.update_params[0]*np.multiply(dW[ko],mW[ko])
                self.thresholds = self.thresholds - self.update_params[0] * dthresholds

    def _update_X(self, dthresholds,dW,mX,mW,x,y,coste):
        #TODO: implement threshold update

        valueTHR = self.thresholds

        nCapas = self.num_hidden + 1
        if self.update == 'gradient-adapt':
            Wdon=[[] for k in range(nCapas)]

            self.thresholds = valueTHR - self.update_params[0] * dthresholds

            for ko in range(nCapas):
                Wdon[ko]=np.multiply(self.coefs_[ko],mW[ko])-self.update_params[0]*np.multiply(dW[ko],mW[ko])

            #----------------------------------------------------------------------
            # Estima de los costes y ajuste del parámetro de paso (GRADIENTE)
            #----------------------------------------------------------------------
            ye, Os = self._forward_pass_W(np.matmul(mX,x),Wdon)
            costen = self._cost(y, ye)
            #------------------------------------------------------------------
            # Ajuste del parámetro de paso
            #------------------------------------------------------------------
            if costen >= coste:
                aumenta=True

                while aumenta:
                    self.update_params[0] = self.update_params[0]/self.update_params[2]

                    self.thresholds = valueTHR - self.update_params[0] * dthresholds

                    for ko in range(nCapas):
                        Wdon[ko]=np.multiply(self.coefs_[ko],mW[ko])-self.update_params[0]*np.multiply(dW[ko],mW[ko])

                    #(ye,Os)=mlp(np.matmul(mX,x[:,indBatch]),Wdon,tAct)
                    (ye,Os)=self._forward_pass_W(np.matmul(mX,x),Wdon)
                    costen = self._cost(y, ye)

                    if (costen < coste):
                        aumenta = False
                    elif  (self.update_params[0]<1e-20):
                        aumenta = False

                        return True

            #self.thresholds = valueTHR - self.update_params[0] * dthresholds
            for ko in range(nCapas):
                self.coefs_[ko]=self.coefs_[ko]-self.update_params[0]*np.multiply(dW[ko],mW[ko])

            self.update_params[0]=self.update_params[0]*self.update_params[1]

            return False

    def _derivatives_cost(self, y, ye, params):

        nClases = self.cost_matrix.shape[0]
        dim_out, nBatchSize = y.shape

        #d=np.zeros((1,nBatchSize))
        d=np.zeros((dim_out,nBatchSize))

        nPclase=np.zeros(nClases)
        for kclase in range(nClases):
            #vc=find(y[kclase,:],1)
            vc = np.nonzero(y[kclase,:]==1)
            nPclase[kclase]=len(vc[0])

        if self.loss_func=='BayesMSpair':

           for ktrue in range(nClases):
               #vc=find(y[ktrue,indBatch],1)
               vc = np.nonzero(y[ktrue,:]==1)
               ye_true = ye[:,vc]

               #otras=setdiff(range(nClases),[ktrue])
               otras = [x for x in range(nClases) if x not in [ktrue]]
               for kotra in otras:
                   adaptOtra=fdpGeneralN(ye[ktrue,vc]-ye[kotra,vc],self.parzen_params)
                   d[ktrue,vc] -= adaptOtra*self.cost_matrix[kotra,ktrue]*self.class_prob[ktrue]/nPclase[ktrue]
                   d[kotra,vc] = adaptOtra*self.cost_matrix[kotra,ktrue]*self.class_prob[ktrue]/nPclase[ktrue]

        elif self.loss_func=='BayesMSsum':
           for ktrue in range(nClases):
               #vc=find(y[ktrue,indBatch],1)
               vc = np.nonzero(y[ktrue,:]==1)

               for kdec in range(nClases):
                   #otras=setdiff(range(nClases),[kdec])
                   otras = [x for x in range(nClases) if x not in [kdec]]
                   for kotra in otras:
                       adaptOtra=fdpGeneralN(ye[kotra,vc]-ye[kdec,vc],self.parzen_params)
                       d[kdec,vc] += adaptOtra*self.cost_matrix[kdec,ktrue]*self.class_prob[ktrue]/nPclase[ktrue]
                       d[kotra,vc] -= adaptOtra*self.cost_matrix[kdec,ktrue]*self.class_prob[ktrue]/nPclase[ktrue]

        elif self.loss_func=='BayesMStot':
           for ktrue in range(nClases):
               #vc=find(y[ktrue,indBatch],1)
               vc = np.nonzero(y[ktrue,:]==1)

               for kdec in range(nClases):
                   #otras=setdiff(range(nClases),[kdec])
                   otras = [x for x in range(nClases) if x not in [kdec]]
                   for kotra in otras:
                       adaptOtra=fdpGeneralN(ye[kotra,vc]-ye[kdec,vc],self.parzen_params)

                       #dimprod = setdiff(range(nClases),[kdec,kotra])
                       dimprod = [x for x in range(nClases) if x not in [kdec,kotra]]
                       adaptProd = 1.0
                       for kprod in dimprod:
                           adaptProd = adaptProd * (1.0 - intGeneralN(ye[kprod,vc]-ye[kdec,vc],self.parzen_params))

                       d[kdec,vc] += np.multiply(adaptOtra,adaptProd)*self.cost_matrix[kdec,ktrue]*self.class_prob[ktrue]/nPclase[ktrue]
                       d[kotra,vc] -= np.multiply(adaptOtra,adaptProd)*self.cost_matrix[kdec,ktrue]*self.class_prob[ktrue]/nPclase[ktrue]

        elif self.loss_func=='BayesMSmax':
           for ktrue in range(nClases):
               #vc=find(y[ktrue,indBatch],1)
               vc = np.nonzero(y[ktrue,:]==1)[0]
               ye_true = ye[:,vc]

               for kdec in range(nClases):
                   #otras=setdiff(range(nClases),[kdec])
                   otras = [x for x in range(nClases) if x not in [kdec]]
                   ye_max = np.max(ye_true[otras,:],axis=0)
                   ve_max = np.argmax(ye_true[otras,:],axis=0)
                   adaptOtra=fdpGeneralN(ye_max - ye_true[kdec,:],self.parzen_params)
                   d[kdec,vc] += adaptOtra*self.cost_matrix[kdec,ktrue]*self.class_prob[ktrue]/nPclase[ktrue]
                   for kotra in range(len(otras)):
                       #vo = find(ve_max,kotra)
                       vo = np.nonzero(ve_max==kotra)
                       if len(vo)>0:
                           ind_otra = otras[kotra]
                           d[ind_otra,vc[vo]] -= (fdpGeneralN(ye_max[vo] - ye_true[kdec,vo],self.parzen_params)*self.cost_matrix[kdec,ktrue]*self.class_prob[ktrue]/nPclase[ktrue])[0,:]

        # Propagation through the nonlinear function of the output laye
        if self.activations[-1] == 'tanh':
            d=np.multiply(d,1-np.square(ye))

        elif self.activations[-1] == 'logistic':
            d=np.multiply(d,ye-np.square(ye))

        elif self.activations[-1] == 'relu':
            d = np.where(ye<=0,0,d)

        return d

    def gradiente_Numerico(self, y,ye):

        pasoGrad = 1e-4

        #ye = self._forward_pass_fast(x)
        costeRef = self._cost(y,ye)
        grad_d = np.zeros(ye.shape)
        for kA in range(ye.shape[0]):
            for kB in range(ye.shape[1]):
                yeG = 1.0*ye
                yeG[kA,kB] = yeG[kA,kB] + pasoGrad

                costeGrad = self._cost(y,yeG)
                grad_d[kA,kB] =  (costeGrad - costeRef)/pasoGrad

        return grad_d

    def predict(self, x):

        ys = self._forward_pass_fast(x.T)
        ye = np.argmax(ys, axis=0)

        return ye


    def soft_output(self, x):

        ys = self._forward_pass_fast(x.T)
        ye = ys.T

        return ye


    def _fit(self, x, y, x_val=np.zeros(0), y_val=np.zeros(0), val_stop=[0,0], incremental=False):

        if type(x_val) == float:
            #print('    Generating validation set...')

            nFolds = np.ceil(1/x_val).astype(int)
            nBatchCV = np.ceil(x.shape[0]/nFolds).astype(int)

            indices_CV =[]
            for val in generate_batches(nBatchCV,y, mode='representative'):
                indices_CV.append(val)

            ind_val = indices_CV[-1]
            #ind_train = [v for v in range(x_train.shape[0]) if v not in ind_val]
            ind_train=[]
            for kTrain in range(nFolds-1):
                ind_train += indices_CV[kTrain]
            # Getting the unique values
            ind_train = list(set(ind_train))

            x_val = x[ind_val,:]
            y_val = y[ind_val]
            x = x[ind_train,:]
            y = y[ind_train]

            #print('    Validation set generated')



        #nClases = self.cost_matrix.shape[0]

        x, y = self._preprocess_data(x,y)
        if x_val.shape[0] > 0:
            x_val, y_val = self._preprocess_data(x_val,y_val)


        dim_in, num_pat = x.shape
        dim_out = y.shape[0]

        first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
        )

        if first_pass:#(len(self.coefs_) == 0) or (self.warm_start==False):
            self._initialize(dim_in, dim_out, self.type_init)

        nEpochs = self.n_epoch
        nBatch = self.n_batch
        nCapas = self.num_hidden + 1

        if (nBatch == 0) or (nBatch > num_pat):
            nBatch = num_pat

        if (self.flag_evo) or (val_stop[0] > 0):
            evoCosteEpoch=np.zeros(nEpochs+1)
            if x_val.shape[0] > 0:
                ye = self._forward_pass_fast(x_val)
                coste = self._cost(y_val,ye)
            else:
                ye = self._forward_pass_fast(x)
                coste = self._cost(y,ye)


            evoCosteEpoch[0]=coste

            if val_stop[0] > 0:
                opt_cost = coste
                opt_epoch = 0
                opt_W = self.coefs_.copy()


        #dW=[[] for k in range(nCapas)]
        Wdo=[[] for k in range(nCapas)]
        Wdon=[[] for k in range(nCapas)]
        mX=np.diag(np.ones(dim_in)).astype(float)
        mW=[[] for k in range(nCapas)]

        if len(self.drop_out)==0:
            self.drop_out=[0 for k in range(nCapas)]

        if len(self.dWm) == 0:
            for ko in range(nCapas):
                self.dWm.append(np.zeros((np.shape(self.coefs_[ko]))))

        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]/(1-self.drop_out[ko])
            mW[ko]=np.ones(np.shape(self.coefs_[ko]))

        #--------------------------------------------------------------------------
        # Se inicia el procedimiento de entrenamiento
        #--------------------------------------------------------------------------
        if self.type_batch in ['class_equitative', 'representative']:
            paramsBatch = y[0,:]

        for kEpoch in range(nEpochs):

            for indBatch in generate_batches(nBatch, paramsBatch, mode=self.type_batch):
                # Generation of the Drop-Out Masks
                if np.sum(self.drop_out)>0:
                    mX=np.diag(np.random.uniform(0,1,dim_in)>self.drop_out[0]).astype(float)
                    for ko in range(nCapas-1):
                        (Nb,Na)=np.shape(self.coefs_[ko])
                        mW[ko]=np.matmul(np.diag((np.random.uniform(0,1,Nb)>self.drop_out[ko+1])).astype(float), np.ones((Nb,Na)))
                        Wdo[ko]=np.multiply(self.coefs_[ko],mW[ko])

                    Wdo[nCapas-1]=self.coefs_[nCapas-1]
                else:
                    Wdo=self.coefs_.copy()

                (ye,Os) = self._forward_pass_W(x[:,indBatch],Wdo)
                #----------------------------------------------------------------------
                # Cálculo de gradientes
                #----------------------------------------------------------------------
                d = self._derivatives_cost(y[:,indBatch], ye, paramsBatch)
                dW = self._backpropagation(x[:,indBatch], Os, d)
                #----------------------------------------------------------------------
                # Nuevos pesos iteración
                #----------------------------------------------------------------------
                if self.update == 'gradient-adapt':
                    coste = self._cost(y[:,indBatch],ye)
                    salida = self._update_W_X(dW,mX,mW,x[:,indBatch],y[:,indBatch],paramsBatch,coste)

                    if salida:
                        for ko in range(nCapas):
                            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

                        evoCosteEpoch[kEpoch+1:nEpochs+1]=coste
                        print("Step size in the limit (zero) ...")

                        if val_stop[0] == 0:
                            opt_epoch = kEpoch
                        else:
                            self.coefs_ = opt_W

                        return opt_epoch, evoCosteEpoch

                else:
                    self._update_W(dW,mX,mW)

            #----------------------------------------------------------------------
            # Actualización del coste (global) y validación
            #----------------------------------------------------------------------
            if (self.flag_evo) or (val_stop[0] > 0):
                for ko in range(nCapas):
                    Wdon[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

                if x_val.shape[0] > 0:
                    (ye,Os)=self._forward_pass_W(x_val,Wdon)
                    coste = self._cost(y_val,ye)
                else:
                    (ye,Os)=self._forward_pass_W(x,Wdon)
                    coste = self._cost(y,ye)


                evoCosteEpoch[kEpoch+1] = coste

                if val_stop[0] > 0:
                    if coste < opt_cost:
                        opt_cost = coste
                        opt_epoch = kEpoch+1
                        opt_W = self.coefs_.copy()

                    if kEpoch > val_stop[1]:
                        rel_dec = (np.min(evoCosteEpoch[kEpoch-val_stop[1]:kEpoch])-evoCosteEpoch[kEpoch])/np.min(evoCosteEpoch[kEpoch-val_stop[1]:kEpoch])

                        if rel_dec < val_stop[0]:
                            self.coefs_ = opt_W

                            break

        # Training is over
        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

        if (self.flag_evo == False) and (val_stop[0]==0):
            ye=self._forward_pass_fast(x)
            evoCosteEpoch = self._cost(y,ye)
            opt_epoch = nEpochs
        elif val_stop[0] == 0:
            opt_epoch = nEpochs

        self.opt_epoch = opt_epoch
        self.loss_ = evoCosteEpoch

        return self

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns a trained MLP model.
        """
        #self._validate_params()
        self.n_features_in_ = X.shape[0]
        if X.shape[0] != y.shape[0]:
            raise ValueError('Sizes of X and y are not compatible..')

        return self._fit(X, y, incremental=False)

    def fit_cv_stop(self, x, y, val_stop, metrics=['MAE', 'AMAE'], nFolds=5, verbose=False):
        #metrics = ['MAE', 'AMAE', 'accuracy', 'balanced_accuracy', 'f1', 'cohen_kappa', 'cohen_kappa_linear', 'cohen_kappa_quadratic', 'matthews']

        nBatchCV = np.ceil(x.shape[0]/nFolds).astype(int)

        indices_CV =[]
        for val in generate_batches(nBatchCV,y, mode='representative'):
            indices_CV.append(val)

        epoch_params = self.epoch_params.copy()
        update_params = self.update_params

        self.epoch_params[0] = 0

        final_ep, evoCosteEpoch = self.fit(x,y)

        initial_W = self.coefs_.copy()
        self.epoch_params = epoch_params

        number_epoch_CV = np.zeros(nFolds).astype(int)

        #cost_value_CV = np.zeros(nFolds)
        #AMAE_CV_val = np.zeros(nFolds)
        #MAE_CV_val = np.zeros(nFolds)

        cv_metrics = np.zeros((len(metrics),nFolds))


        for kCV in range(nFolds):
            self.coefs_ = initial_W.copy()
            self.update_params = update_params

            ind_val = indices_CV[kCV]
            #ind_train = [v for v in range(x_train.shape[0]) if v not in ind_val]
            ind_train=[]
            for kTrain in range(nFolds):
                if kTrain != kCV:
                    ind_train += indices_CV[kTrain]
            # Getting the unique values
            ind_train = list(set(ind_train))

            #final_ep, evoCosteEpoch = model.fit(x_train,y_train, val_stop=[1e-3,5])
            final_ep, evoCosteEpoch = self.fit(x[ind_train,:],y[ind_train], x_val=x[ind_val,:], y_val=y[ind_val], val_stop=val_stop)
            if verbose:
                print('Fold %d : (Initial cost = %2.3f) Final epoch = %d'%(kCV+1, evoCosteEpoch[0], final_ep))

            ye_val = self.predict(x[ind_val,:])
            cv_metrics[:,kCV] = compute_metrics(y[ind_val], ye_val, metrics)

            number_epoch_CV[kCV] = final_ep


            #cost_value_CV[kCV] = evoCosteEpoch[final_ep]

            # CM_val = confusion_matrix(y[ind_val], ye_val)
            # MAE_val, AMAE_val = calc_MAE_AMAE(CM_val)

            # MAE_CV_val[kCV] = MAE_val
            # AMAE_CV_val[kCV] = AMAE_val

            # print('   MAE (val) = %2.3f' %(MAE_val))
            # print('  AMAE (val) = %2.3f' %(AMAE_val))

        # Re-train the network with the whole training set
        # ------------------------------------------------
        self.coefs_ = initial_W.copy()
        self.update_params = update_params
        self.epoch_params = [int(number_epoch_CV.max()*1.1), epoch_params[1]]

        final_ep, evoCosteEpoch = self.fit(x,y, val_stop=val_stop)
        #print(' ')
        #print('Final training : (Initial cost = %2.3f) Final epoch = %d'%(evoCosteEpoch[0], final_ep))

        #ye_val = self.predict(x)
        #CM_val = confusion_matrix(y, ye_val)
        #MAE_val, AMAE_val = calc_MAE_AMAE(CM_val)
        #print('  AMAE (train) = %2.3f' %(AMAE_val))
        #print('   MAE (train) = %2.3f' %(MAE_val))

        return final_ep, evoCosteEpoch, cv_metrics


#------------------------------------------------------------------------------
"""
  MLPOrdinal  CLASS

  Classifier for Ordinal problems using single output MLP neural network
  The decision rule is based on thresholds
  The loss function is an estimate of the Bayesian risk based on the Parzen
  Windows estimator

  Parameters
  ----------

  * layers_size : number of neurons of each hidden layer [list or tuple] nHidden
  * activations : type of activation of each layer [list] nHidden + 1
      Includes the activation of the output layer (this must be identity or tanh)
      Possible types:
         - tanh
         - relu
         - logistic
         - identity
         - softmax
  * parzen_params: params of the Parzen Window [list]
        - params_parzen[0] : type of window
             + 'gauss'
             + 'uniform'
             + 'linear'
             + 'linear_inv'
             + 'quadratic'
             + 'quadratic_inv'
        - params_parzen[1] : support of the window around center
        - params_parzen[2] : center of the window
  * class_prob : probabilities of the classes  [list] nClasses
  * cost_matriz : decission costs among classes  [array] nClasses x nClasses
  * update: type of gradient update
      Possible types:
          - momentum
          - gradient
          - gradient-adapt
  * learning_rate_init: initial value of the learning rate [real]
  * learning_rate_inc: increment factor for learning rate, if 'gradient-adapt' [real]
  * learning_rate_dec: decrease factor for learning rate, if 'gradient-adapt' [real]
  * momentum: momentum value, if 'momentum' [real]
  * drop_out: probability of drop-out for each layer, including input [list] nHidden + 1
  * n_epoch: number of epochs [int]
  * n_batch: size of the mini-batch (0 if batch-mode is used) [int]
  * type_init: type of initizalization for network weights ('uniform', 'gaussian')
  * warm_start: flag to allow a warm start from previous solution [bool]
  * flag_evo: flag to compute the loss value after each epoch [bool]

"""
#------------------------------------------------------------------------------

class MLPOrdinal(MLPnn):

    def __init__(self, layers_size=[10],
                 activations = ['relu', 'identity'],
                 parzen_params=['gauss',1,0],
                 class_prob=[],
                 cost_matrix=[],
                 update='momentum',
                 learning_rate_init=1e-4,
                 learning_rate_inc=1.05,
                 learning_rate_dec=2,
                 momentum = 0.9,
                 n_epoch = 1000,
                 n_batch = 250,
                 drop_out = [],
                 thresholds=[],
                 type_batch = 'representative',
                 type_init = 'uniform',
                 warm_start = False,
                 flag_evo=False
                 ):

        super().__init__(layers_size, activations)

        self.parzen_params = parzen_params
        self.class_prob = class_prob
        self.cost_matrix = cost_matrix
        self.update = update
        self.learning_rate_init = learning_rate_init
        self.learning_rate_inc = learning_rate_inc
        self.learning_rate_dec = learning_rate_dec
        self.momentum = momentum
        self.drop_out = drop_out
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.thresholds = thresholds
        self.type_batch = type_batch
        self.type_init = type_init
        self.warm_start = warm_start
        self.flag_evo = flag_evo


    def _preprocess_data(self, x, y):

        if 0 not in list(np.unique(y)):
            y = y - 1

        if not hasattr(self, "class_labels"):
            self.classes_ = list(np.unique(y))

        nClases = len(self.classes_)


        if len(self.thresholds) == 0:
            self.thresholds = 0.2*np.linspace(-1,1,nClases+1)[1:-1]

        if len(self.class_prob) == 0:
            self.class_prob = np.zeros(nClases)
            for kclase in range(nClases):
                v = np.nonzero(y==kclase)
                self.class_prob[kclase] = len(v[0])/y.shape[0]

        if len(self.cost_matrix) == 0:
            self.cost_matrix = np.zeros((nClases, nClases))
            for kclase in range(nClases):
                for kotra in range(nClases):
                    self.cost_matrix[kotra,kclase] = np.abs(kclase-kotra)

        yout = y.reshape(1,y.shape[0])

        xout = x.T

        return xout, yout

    def _cost(self, y, ye):

        nClases = len(self.classes_)
        coste=0

        for kclase in range(nClases):
            coste += self.class_prob[kclase]*self.cost_matrix[nClases-1,kclase]
            vc=np.nonzero(y[0,:]==kclase)
            for kotra in range(kclase):
                costeOtra=intGeneralN(self.thresholds[kotra]-ye[0,vc],self.parzen_params)
                coste += (self.cost_matrix[kotra,kclase]-self.cost_matrix[kotra+1,kclase])*self.class_prob[kclase]*np.mean(costeOtra)
            for kotra in range(kclase,nClases-1):
                costeOtra=1-intGeneralN(ye[0,vc]-self.thresholds[kotra],self.parzen_params)
                coste += (self.cost_matrix[kotra,kclase]-self.cost_matrix[kotra+1,kclase])*self.class_prob[kclase]*np.mean(costeOtra)

        return coste

    def _update(self, dthresholds,dW,mX,mW):

        nCapas = self.num_hidden + 1
        if self.update =='momentum':
            for ko in range(nCapas):
                self.dWm[ko]=self.dWm[ko]*self.momentum+self.learning_rate*np.multiply(dW[ko],mW[ko])
                self.coefs_[ko]=self.coefs_[ko]-np.multiply(self.dWm[ko],mW[ko])

                self.thresholdsMom = self.thresholdsMom*self.momentum + self.learning_rate*dthresholds
                self.thresholds = self.thresholds-self.thresholdsMom

        elif self.update =='gradient':
            for ko in range(nCapas):
                self.coefs_[ko]=self.coefs_[ko]-self.learning_rate*np.multiply(dW[ko],mW[ko])
                self.thresholds = self.thresholds - self.learning_rate * dthresholds

    def _update_X(self, dthresholds,dW,mX,mW,x,y,coste):

        valueTHR = self.thresholds

        nCapas = self.num_hidden + 1
        if self.update == 'gradient-adapt':
            Wdon=[[] for k in range(nCapas)]

            self.thresholds = valueTHR - self.learning_rate * dthresholds

            for ko in range(nCapas):
                Wdon[ko]=np.multiply(self.coefs_[ko],mW[ko])-self.learning_rate*np.multiply(dW[ko],mW[ko])

            #----------------------------------------------------------------------
            # Estima de los costes y ajuste del parámetro de paso (GRADIENTE)
            #----------------------------------------------------------------------
            ye, Os = self._forward_pass_W(np.matmul(mX,x),Wdon)
            costen = self._cost(y, ye)
            #------------------------------------------------------------------
            # Ajuste del parámetro de paso
            #------------------------------------------------------------------
            if costen >= coste:
                aumenta=True

                while aumenta:
                    self.learning_rate = self.learning_rate/self.learning_rate_dec

                    self.thresholds = valueTHR - self.learning_rate * dthresholds

                    for ko in range(nCapas):
                        Wdon[ko]=np.multiply(self.coefs_[ko],mW[ko])-self.learning_rate*np.multiply(dW[ko],mW[ko])

                    #(ye,Os)=mlp(np.matmul(mX,x[:,indBatch]),Wdon,tAct)
                    (ye,Os)=self._forward_pass_W(np.matmul(mX,x),Wdon)
                    costen = self._cost(y, ye)

                    if (costen < coste):
                        aumenta = False
                    elif  (self.learning_rate<1e-20):
                        aumenta = False

                        return True

            #self.thresholds = valueTHR - self.update_params[0] * dthresholds
            for ko in range(nCapas):
                self.coefs_[ko]=self.coefs_[ko]-self.learning_rate*np.multiply(dW[ko],mW[ko])

            self.learning_rate=self.learning_rate*self.learning_rate_inc

            return False

    def _derivatives_cost(self, y, ye, params):

        nClases = self.cost_matrix.shape[0]
        dim_out, nBatchSize = y.shape

        d=np.zeros((1,nBatchSize))
        dthresholds=0.0*self.thresholds

        nPclase=np.zeros(nClases)
        for kclase in range(nClases):
            #vc=find(y[kclase,:],1)
            vc = np.nonzero(y[0,:]==kclase)
            nPclase[kclase]=len(vc[0])

        for kclase in range(nClases):
            #vc=find(y[0,indBatch],kclase+1)
            vc = np.nonzero(y[0,:] == kclase)
            for kumbral in range(nClases-1):
                dthresholds[kumbral]+=np.sum(fdpGeneralN(self.thresholds[kumbral]-ye[0,vc],self.parzen_params))*(self.cost_matrix[kumbral,kclase]-self.cost_matrix[kumbral+1,kclase])*self.class_prob[kclase]/nPclase[kclase]

            for kotra in range(kclase):
                adaptOtra=fdpGeneralN(self.thresholds[kotra]-ye[0,vc],self.parzen_params)
                d[0,vc] -= adaptOtra*(self.cost_matrix[kotra,kclase]-self.cost_matrix[kotra+1,kclase])*self.class_prob[kclase]/nPclase[kclase]

            for kotra in range(kclase,nClases-1):
                adaptOtra=fdpGeneralN(ye[0,vc]-self.thresholds[kotra],self.parzen_params)
                d[0,vc] -= adaptOtra*(self.cost_matrix[kotra,kclase]-self.cost_matrix[kotra+1,kclase])*self.class_prob[kclase]/nPclase[kclase]


        if self.activations[-1] == 'tanh':
            d=np.multiply(d,1-np.square(ye))

        elif self.activations[-1] == 'logistic':
            d=np.multiply(d,ye-np.square(ye))

        elif self.activations[-1] == 'relu':
            d = np.where(ye<=0,0,d)

        return d, dthresholds

    def gradiente_Numerico(self, y,ye):

        pasoGrad = 1e-4

        #ye = self._forward_pass_fast(x)
        costeRef = self._cost(y,ye)
        grad_d = np.zeros(ye.shape)
        for kA in range(ye.shape[0]):
            for kB in range(ye.shape[1]):
                yeG = 1.0*ye
                yeG[kA,kB] = yeG[kA,kB] + pasoGrad

                costeGrad = self._cost(y,yeG)
                grad_d[kA,kB] =  (costeGrad - costeRef)/pasoGrad

        grad_u = np.zeros(self.thresholds.shape)

        valTH = 1.0 * self.thresholds

        for kA  in range(len(self.thresholds)):
            self.thresholds = 1.0*valTH
            self.thresholds[kA] += pasoGrad
            costeGrad = self._cost(y,ye)
            grad_u[kA] =  (costeGrad - costeRef)/pasoGrad

        self.thresholds = 1.0*valTH

        return grad_d, grad_u

    def predict(self, x):

        ys = self._forward_pass_fast(x.T)

        nClases = self.cost_matrix.shape[0]

        ye = np.zeros(x.shape[0])

        for k in range(nClases-1):
            #vu=findG(ye[0,:],umbrales[0,k])
            vu = np.nonzero(ys[0,:]>self.thresholds[k])
            if len(vu)>0:
                ye[vu] += 1

        return ye

    def soft_output(self, x):

        y = self._forward_pass_fast(x.T)

        if y.shape[0] == 1:
            ye = y[0,:]
        else:
            ye = y.T

        return ye


    #def fit(self, x, y, warm_start='False'):
    def _fit(self, x, y, x_val=np.zeros(0), y_val=np.zeros(0), val_stop=[0,0], incremental=False):

        if type(x_val) == float:
            #print('    Generating validation set...')

            nFolds = np.ceil(1/x_val).astype(int)
            nBatchCV = np.ceil(x.shape[0]/nFolds).astype(int)

            indices_CV =[]
            for val in generate_batches(nBatchCV,y, mode='representative'):
                indices_CV.append(val)

            ind_val = indices_CV[-1]
            #ind_train = [v for v in range(x_train.shape[0]) if v not in ind_val]
            ind_train=[]
            for kTrain in range(nFolds-1):
                ind_train += indices_CV[kTrain]
            # Getting the unique values
            ind_train = list(set(ind_train))

            x_val = x[ind_val,:]
            y_val = y[ind_val]
            x = x[ind_train,:]
            y = y[ind_train]


        x, y = self._preprocess_data(x,y)
        if x_val.shape[0] > 0:
            x_val, y_val = self._preprocess_data(x_val,y_val)

        dim_in, num_pat = x.shape
        dim_out = y.shape[0]

        first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
        )

        if first_pass:
            self._initialize(dim_in, dim_out, self.type_init)
            self.thresholdsMom = 0.0 * self.thresholds

        nEpochs = self.n_epoch
        nBatch = self.n_batch
        nCapas = self.num_hidden + 1

        if (nBatch == 0) or (nBatch > num_pat):
            nBatch = num_pat

        if (self.flag_evo) or (val_stop[0] > 0):
            evoCosteEpoch=np.zeros(nEpochs+1)
            if x_val.shape[0] > 0:
                ye = self._forward_pass_fast(x_val)
                coste = self._cost(y_val,ye)
            else:
                ye = self._forward_pass_fast(x)
                coste = self._cost(y,ye)


            evoCosteEpoch[0]=coste

            if val_stop[0] > 0:
                opt_cost = coste
                opt_epoch = 0
                opt_W = self.coefs_.copy()


        Wdo=[[] for k in range(nCapas)]
        Wdon=[[] for k in range(nCapas)]
        mX=np.diag(np.ones(dim_in)).astype(float)
        mW=[[] for k in range(nCapas)]

        if len(self.drop_out)==0:
            self.drop_out=[0 for k in range(nCapas)]

        if len(self.dWm) == 0:
            self.thresholdsMom = 0.0*self.thresholds
            for ko in range(nCapas):
                self.dWm.append(np.zeros((np.shape(self.coefs_[ko]))))

        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]/(1-self.drop_out[ko])
            mW[ko]=np.ones(np.shape(self.coefs_[ko]))

        #--------------------------------------------------------------------------
        # Se inicia el procedimiento de entrenamiento
        #--------------------------------------------------------------------------
        if self.type_batch in ['class_equitative', 'representative']:
            paramsBatch = y[0,:]

        for kEpoch in range(nEpochs):

            for indBatch in generate_batches(nBatch, paramsBatch, mode=self.type_batch):
            #for indBatch in generate_batches(nBatch, y[0,:], 'class_equitative'):
                #nBatchSize = len(indBatch)

                # Generation of the Drop-Out Masks
                if np.sum(self.drop_out)>0:
                    mX=np.diag(np.random.uniform(0,1,dim_in)>self.drop_out[0]).astype(float)
                    for ko in range(nCapas-1):
                        (Nb,Na)=np.shape(self.coefs_[ko])
                        mW[ko]=np.matmul(np.diag((np.random.uniform(0,1,Nb)>self.drop_out[ko+1])).astype(float), np.ones((Nb,Na)))
                        Wdo[ko]=np.multiply(self.coefs_[ko],mW[ko])

                    Wdo[nCapas-1]=self.coefs_[nCapas-1]
                else:
                    Wdo=self.coefs_.copy()

                (ye,Os) = self._forward_pass_W(x[:,indBatch],Wdo)
                #----------------------------------------------------------------------
                # Cálculo de gradientes
                #----------------------------------------------------------------------
                d, dthresholds = self._derivatives_cost(y[:,indBatch], ye, paramsBatch)
                dW = self._backpropagation(x[:,indBatch], Os, d)
                #----------------------------------------------------------------------
                # Nuevos pesos iteración
                #----------------------------------------------------------------------
                if self.update == 'gradient-adapt':
                    coste = self._cost(y[:,indBatch],ye)
                    salida = self._update_X(dthresholds,dW,mX,mW,x[:,indBatch],y[:,indBatch],coste)

                    if salida:
                        for ko in range(nCapas):
                            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

                        evoCosteEpoch[kEpoch+1:nEpochs+1]=coste
                        print("Step size in the limit (zero) ...")

                        #return evoCosteEpoch

                        if val_stop[0] == 0:
                            opt_epoch = kEpoch
                        else:
                            self.coefs_ = opt_W

                        return opt_epoch, evoCosteEpoch

                else:
                    self._update(dthresholds,dW,mX,mW)

            #----------------------------------------------------------------------
            # Actualización del coste (global)
            #----------------------------------------------------------------------
            if self.flag_evo:
                for ko in range(nCapas):
                    Wdon[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

                #(ye,Os)=self._forward_pass_W(x,Wdon)
                #evoCosteEpoch[kEpoch+1] = self._cost(y,ye)

                if x_val.shape[0] > 0:
                    (ye,Os)=self._forward_pass_W(x_val,Wdon)
                    coste = self._cost(y_val,ye)
                else:
                    (ye,Os)=self._forward_pass_W(x,Wdon)
                    coste = self._cost(y,ye)


                evoCosteEpoch[kEpoch+1] = coste

                if val_stop[0] > 0:
                    if coste < opt_cost:
                        opt_cost = coste
                        opt_epoch = kEpoch+1
                        opt_W = self.coefs_.copy()

                    if kEpoch > val_stop[1]:
                        rel_dec = (np.min(evoCosteEpoch[kEpoch-val_stop[1]:kEpoch])-evoCosteEpoch[kEpoch])/np.min(evoCosteEpoch[kEpoch-val_stop[1]:kEpoch])

                        if rel_dec < val_stop[0]:
                            self.coefs_ = opt_W

                            break


        # Training is over
        # for ko in range(nCapas):
        #     self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

        # if self.flag_evo == False:
        #     ye=self._forward_pass_fast(x)
        #     evoCosteEpoch = self._cost(y,ye)

        # Training is over
        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])

        if (self.flag_evo == False) and (val_stop[0]==0):
            ye=self._forward_pass_fast(x)
            evoCosteEpoch = self._cost(y,ye)
            opt_epoch = nEpochs
        elif val_stop[0] == 0:
            opt_epoch = nEpochs

        self.loss_ = evoCosteEpoch
        self.opt_epoch = opt_epoch

        return self

    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.
        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns a trained MLP model.
        """
        #self._validate_params()
        self.n_features_in_ = X.shape[0]
        if X.shape[0] != y.shape[0]:
            raise ValueError('Sizes of X and y are not compatible..')

        return self._fit(X, y, incremental=False)


    def fit_cv_stop(self, x, y, val_stop, metrics=['MAE', 'AMAE'], nFolds=5, verbose=False):
        #metrics = ['MAE', 'AMAE', 'accuracy', 'balanced_accuracy', 'f1', 'cohen_kappa', 'cohen_kappa_linear', 'cohen_kappa_quadratic', 'matthews']

        nBatchCV = np.ceil(x.shape[0]/nFolds).astype(int)

        indices_CV =[]
        for val in generate_batches(nBatchCV,y, mode='representative'):
            indices_CV.append(val)

        epoch_params = self.epoch_params.copy()
        update_params = self.update_params

        self.epoch_params[0] = 0

        final_ep, evoCosteEpoch = self.fit(x,y)

        initial_W = self.coefs_.copy()
        self.epoch_params = epoch_params

        number_epoch_CV = np.zeros(nFolds).astype(int)

        #cost_value_CV = np.zeros(nFolds)
        #AMAE_CV_val = np.zeros(nFolds)
        #MAE_CV_val = np.zeros(nFolds)

        cv_metrics = np.zeros((len(metrics),nFolds))


        for kCV in range(nFolds):
            self.coefs_ = initial_W.copy()
            self.update_params = update_params

            ind_val = indices_CV[kCV]
            #ind_train = [v for v in range(x_train.shape[0]) if v not in ind_val]
            ind_train=[]
            for kTrain in range(nFolds):
                if kTrain != kCV:
                    ind_train += indices_CV[kTrain]
            # Getting the unique values
            ind_train = list(set(ind_train))

            #final_ep, evoCosteEpoch = model.fit(x_train,y_train, val_stop=[1e-3,5])
            final_ep, evoCosteEpoch = self.fit(x[ind_train,:],y[ind_train], x_val=x[ind_val,:], y_val=y[ind_val], val_stop=val_stop)
            if verbose:
                print('Fold %d : (Initial cost = %2.3f) Final epoch = %d'%(kCV+1, evoCosteEpoch[0], final_ep))

            ye_val = self.predict(x[ind_val,:])
            cv_metrics[:,kCV] = compute_metrics(y[ind_val], ye_val, metrics)

            number_epoch_CV[kCV] = final_ep


            #cost_value_CV[kCV] = evoCosteEpoch[final_ep]

            # CM_val = confusion_matrix(y[ind_val], ye_val)
            # MAE_val, AMAE_val = calc_MAE_AMAE(CM_val)

            # MAE_CV_val[kCV] = MAE_val
            # AMAE_CV_val[kCV] = AMAE_val

            # print('   MAE (val) = %2.3f' %(MAE_val))
            # print('  AMAE (val) = %2.3f' %(AMAE_val))

        # Re-train the network with the whole training set
        # ------------------------------------------------
        self.coefs_ = initial_W.copy()
        self.update_params = update_params
        self.epoch_params = [int(number_epoch_CV.max()*1.1), epoch_params[1]]

        final_ep, evoCosteEpoch = self.fit(x,y, val_stop=val_stop)
        #print(' ')
        #print('Final training : (Initial cost = %2.3f) Final epoch = %d'%(evoCosteEpoch[0], final_ep))

        #ye_val = self.predict(x)
        #CM_val = confusion_matrix(y, ye_val)
        #MAE_val, AMAE_val = calc_MAE_AMAE(CM_val)
        #print('  AMAE (train) = %2.3f' %(AMAE_val))
        #print('   MAE (train) = %2.3f' %(MAE_val))

        return final_ep, evoCosteEpoch, cv_metrics


#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------


