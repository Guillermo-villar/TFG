#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 12:38:54 2018

@author: mlazaro
"""

#------------------------------------------------------------------------------
# definición de métodos útiles
#------------------------------------------------------------------------------
import numpy as np
from scipy import special
#import math
import random

# Set random seed
np.random.seed(42)  # For NumPy
random.seed(42)     # For Python's random module

#from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.base import BaseEstimator


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
def fdpGeneralN(x,tipoFDP=[0,1,0]):
    tFDP=tipoFDP[0]
    if type(tFDP)==str:
        tFDP = dict_parzen[tFDP]
        
    deltaS=tipoFDP[1]
    centroS=tipoFDP[2]
        
    if tFDP == 0:
        #adapta = 3.09022 # 99.8%
        #adapta = 2.32635 # 98%
        adapta = 2.5758 # 99%
        y=np.exp(-np.power((x-centroS),2)/(2*np.power(float(deltaS)/adapta,2)))/(np.sqrt(2*3.1415926535)*float(deltaS)/adapta)
    elif tFDP == 1:
        y = np.where(np.abs(x-centroS) <= deltaS, 1.0/(2.0*float(deltaS)), 0)
      
    elif tFDP == 2:
        y = np.where(np.abs(x-centroS) <= deltaS, (x-centroS+deltaS)/float(deltaS)/float(deltaS)/2.0, 0)
        
    elif tFDP == -2:
        y = np.where(np.abs(x-centroS) <= deltaS, (-x+centroS+deltaS)/float(deltaS)/float(deltaS)/2.0, 0)
        
    elif tFDP == 3:
        y = np.where(np.abs(x-centroS) <= deltaS, 3.0/8.0/float(deltaS)*np.power(1+(x-centroS)/float(deltaS),2), 0)
        
    elif tFDP == -3:
        y = np.where(np.abs(x-centroS) <= deltaS, 3.0/8.0/float(deltaS)*np.power(1-(x-centroS)/float(deltaS),2), 0)
                
    elif tFDP == 4:
        y = np.where(np.abs(x-centroS) <= deltaS, 1.0/4.0/float(deltaS)*np.power(1+(x-centroS)/float(deltaS),3), 0)
        
    elif tFDP == -4:
        y = np.where(np.abs(x-centroS) <= deltaS, 1.0/4.0/float(deltaS)*np.power(1-(x-centroS)/float(deltaS),3), 0)
        
    elif tFDP == 5:
        y = np.where(np.abs(x-centroS) <= deltaS, 5.0/32.0/float(deltaS)*np.power(1+(x-centroS)/float(deltaS),4), 0)
        
    elif tFDP == -5:
        y = np.where(np.abs(x-centroS) <= deltaS, 5.0/32.0/float(deltaS)*np.power(1-(x-centroS)/float(deltaS),4), 0)
        
    elif tFDP == 10:
        y = np.where(np.abs(x-centroS) <= deltaS, (1-np.abs((x-centroS)/float(deltaS)))/float(deltaS), 0)
        
    elif tFDP == 11:
        y = np.where(np.abs(x-centroS) <= deltaS, np.abs(x-centroS)/float(deltaS)/float(deltaS), 0)
        
    elif tFDP == 13:
        y = np.where(x >= centroS, deltaS*np.exp(-deltaS*(x-centroS)), 0)
        
            
    return y    
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
        
    deltaS=tipoFDP[1]
    centroS=tipoFDP[2]
    
    if tFDP == 0:
        #adapta = 3.09022 # 99.8%
        #adapta = 2.32635 # 98%
        adapta = 2.5758 # 99%
        
        y=special.erfc(0.7071*(centroS-x)/(deltaS/adapta))/2.0
        
    elif tFDP == 1:
        y = np.where(np.abs(x-centroS) <= deltaS, (x-centroS+float(deltaS))/(2*deltaS), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0
        
    elif tFDP == 2:
        y = np.where(np.abs(x-centroS) <= deltaS, np.power(x-centroS+float(deltaS),2)/(4.0*deltaS*deltaS), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0
        
            
    elif tFDP == -2:
        y = np.where(np.abs(x-centroS) <= deltaS, 1-np.power(centroS+float(deltaS)-x,2)/(4.0*deltaS*deltaS), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     
        

    elif tFDP == 3:
        y = np.where(np.abs(x-centroS) <= deltaS, 0.125*np.power(1+(x-centroS)/deltaS,3), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     


    elif tFDP == -3:
        y = np.where(np.abs(x-centroS) <= deltaS, 1-0.125*np.power(1-(x-centroS)/deltaS,3), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     

    elif tFDP == 4:
        y = np.where(np.abs(x-centroS) <= deltaS, 0.0625*np.power(1+(x-centroS)/deltaS,4), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     

    elif tFDP == -4:
        y = np.where(np.abs(x-centroS) <= deltaS, 1-0.0625*np.power(1-(x-centroS)/deltaS,4), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     

    elif tFDP == 5:
        y = np.where(np.abs(x-centroS) <= deltaS, 0.03125*np.power(1+(x-centroS)/deltaS,5), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     

    elif tFDP == -5:
        y = np.where(np.abs(x-centroS) <= deltaS, 1-0.03125*np.power(1-(x-centroS)/deltaS,5), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     

    elif tFDP == 10:
        y = np.where(np.abs(x-centroS) <= deltaS, (x-centroS+deltaS)/deltaS - 0.5-np.multiply((x-centroS),np.abs(x-centroS))/(2*deltaS*deltaS), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     

        
    elif tFDP == 11:
        y = np.where(np.abs(x-centroS) <= deltaS, 0.5+np.multiply((x-centroS),np.abs(x-centroS))/(2*deltaS*deltaS), 1.0)
        y[np.nonzero(x<(centroS-deltaS))] = 0.0                     

    elif tFDP ==13:
        y = np.where(x >= centroS, 1-np.exp(-deltaS*(x-centroS)), 0.0)
        #y[np.nonzero(x<(centroS-deltaS))] = 0.0                     

        
        
    return y



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
class MLPnn(BaseEstimator):
    
    def __init__(self, layers_size=[100], activations=['relu', 'linear']): #, update='momentum', update_params=[1e-4, 0.95]):
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
  * cost_classess : costs of misclassification for each class [list] nClases             
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
  * flag_Evo: flag to compute the loss value after each epoch [bool]

  
"""
#------------------------------------------------------------------------------ 
            
class MLPBayesBin(MLPnn):
    
    def __init__(self, layers_size=[100], 
                 activations=['relu','identity'],
                 parzen_params=['gauss',1,0],      
                 class_prob=[0.5, 0.5],
                 cost_classes=[1,1],
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
                 flag_Evo=False
                 ):
        
        super().__init__(layers_size, activations)
        
        #self.name = 'MLPBayesBin'
        self.layers_size = layers_size
        self.activations = activations
        self.parzen_params = parzen_params        
        self.class_prob = class_prob
        self.cost_classes = cost_classes 
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
        self.flag_Evo = flag_Evo
        #        
        
    
    def _preprocess_data(self, x, y):
        nClases = 2
        
        # Labels converted to -1, +1
        if 0 not in list(np.unique(y)):
            y = y - 1
                
        self.class_labels = list(np.unique(y))                        
                        
        
        if len(self.class_prob) == 0:
            self.class_prob = np.zeros(nClases)
            for kclase in range(nClases):
                v = np.nonzero(y==kclase)
                self.class_prob[kclase] = len(v[0])/y.shape[0]
                
        if len(self.cost_classes) == 0:
            self.cost_classes = np.ones(nClases)
                                    
        yout = np.where(y==0, -1,1)                    
        xout = x.T                          
        
        return xout, yout
    
    def _cost(self, y, ye, param=[]):                
        
        v0 = np.nonzero(y==-1)
        v1 = np.nonzero(y==1)
        
        coste = self.cost_classes[0] * self.class_prob[0] * np.mean(intGeneralN(ye[0,v0],self.parzen_params))
        coste += self.cost_classes[1] * self.class_prob[1] * np.mean(intGeneralN(-ye[0,v1],self.parzen_params))
                    
        return coste
    
        
    def _derivatives_cost(self, y, ye):
        
        nBatchSize = y.shape[0]                 
            
        d=np.zeros((1,nBatchSize))
        
        v0 = np.nonzero(y==-1)
        v1 = np.nonzero(y==1)
        
        d[0,v0] = self.cost_classes[0] * self.class_prob[0] * fdpGeneralN(ye[0,v0],self.parzen_params) / len(v0[0])
        d[0,v1] = -self.cost_classes[1] * self.class_prob[1] * fdpGeneralN(-ye[0,v1],self.parzen_params) / len(v1[0])
       
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
                      
        if (self.flag_Evo) or (val_stop[0] > 0):
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
            if (self.flag_Evo) or (val_stop[0] > 0):
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
            
        if (self.flag_Evo == False) and (val_stop[0]==0):
            ye=self._forward_pass_fast(x)
            evoCosteEpoch = self._cost(y,ye)
            
            """
            # Convert predictions to class labels (-1 or 1)
            yepred = np.where(ye < 0, -1, 1)[0]
            
            # Print cost after training
            print(f"Cost after training: {evoCosteEpoch:.4f}")
            
            # Print balanced accuracy score
            balanced_acc = balanced_accuracy_score(y, yepred)
            print(f"Balanced Accuracy = {balanced_acc:.4f}")
            """
            
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
    
                 # layers_size=[100], 
                 # activations=['relu','identity'],
                 # W=[],
                 # parzen_params=['gauss',1,0],
                 # #class_labels = [],                 
                 # class_prob=[0.5, 0.5],
                 # cost_classes=[1,1],
                 # update='momentum', 
                 # update_params=[1e-4,0.95],
                 # #weigths = None, 
                 # drop_out = [],
                 # epoch_params = [1000,250],                  
                 # #thresholds=[],
                 # type_batch = 'representative',
                 # type_init = 'uniform',
                 # #type_bin = 'single',
                 # flag_Evo=False
                 # ):
    
    # def get_params(self, deep=True):
    #     return {"layers_size": self.layer_size, 
    #             "activations": self.activations,
    #             "W": self.coefs_,
    #             "parzen_params": self.parzen_params, 
    #             "class_prob" : self.class_prob,
    #             "cost_classes" : self.cost_classes,
    #             "update" : self.update,
    #             "update_params" : self.update_params,
    #             "drop_out" : self.drop_out,
    #             "epoch_params" : self.epoch_params,
    #             "type_batch" : self.type_batch,
    #             "type_init" : self.type_init,
    #             "flag_Evo" : self.flag_Evo
    #             }

    # def set_params(self, **parameters):
    #     for parameter, value in parameters.items():
    #         setattr(self, parameter, value)
            
    #     return self


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
  * flag_Evo: flag to compute the loss value after each epoch [bool]
  
  
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
                 flag_Evo=False
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
        self.flag_Evo = flag_Evo
        #
            
    def _preprocess_data(self, x, y):
        
        if 0 not in list(np.unique(y)):
            y = y - 1
        
        if not hasattr(self, "class_labels"):
            self.class_labels = list(np.unique(y))                        
            
        nClases = len(self.class_labels)
                
        
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
        
        nClases = len(self.class_labels)
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
                      
        if (self.flag_Evo) or (val_stop[0] > 0):
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
            if (self.flag_Evo) or (val_stop[0] > 0):
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
            
        if (self.flag_Evo == False) and (val_stop[0]==0):
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
  * flag_Evo: flag to compute the loss value after each epoch [bool]
  
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
                 flag_Evo=False
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
        self.flag_Evo = flag_Evo

    
    def _preprocess_data(self, x, y):
        
        if 0 not in list(np.unique(y)):
            y = y - 1
        
        if not hasattr(self, "class_labels"):
            self.class_labels = list(np.unique(y))                        
            
        nClases = len(self.class_labels)
                
        
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
        
        nClases = len(self.class_labels)
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
            
        if (self.flag_Evo) or (val_stop[0] > 0):
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
            if self.flag_Evo:
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
            
        # if self.flag_Evo == False:
        #     ye=self._forward_pass_fast(x)
        #     evoCosteEpoch = self._cost(y,ye)
            
        # Training is over            
        for ko in range(nCapas):
            self.coefs_[ko]=self.coefs_[ko]*(1-self.drop_out[ko])
            
        if (self.flag_Evo == False) and (val_stop[0]==0):
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


