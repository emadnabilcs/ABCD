"""
Created on 10/10/2021
@author: Abdullah M. Alshanqiti (a.m.alshanqiti@gmail.com)
"""
##
##
##

from _dsPreProcessing import Hadiths
from _mLayers import mLayers
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from datetime import datetime
import numpy as np
import os
##
##
##
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
##
##
##

class RnnLstm():

    def __init__(self, Xt, transformLayer=False, isForNarrators=False):
        
        #
        #
        self.TransformerLayers = transformLayer
        self.rnn = mLayers(Xt, isForNarrators).doBuild(self.TransformerLayers)        
        self.rnn.summary()        


    def get_layer (self, iIndex):
        return self.rnn.get_layer(index=iIndex)


    def doTrain (self, trainDS, Epochs):
        self.rnn.fit(
            trainDS, 
            epochs=Epochs)
        return None
    
    def doTrainXY (self, Xt, Yt, Epochs):
        self.rnn.fit(
            Xt, 
            Yt, 
            epochs=Epochs)
        return None


    def getPrediction (self, strIns):
        #
        preList = []
        for x_val in strIns:
            pred = self.rnn(x_val)
            preList.append(pred.numpy()[0])
        #
        return np.array(preList) 

        
    #
    #
    
    def doSave (self, rModelName, Version):
        self.rnn.save_weights("../models/{}_Transformer_{}_.v{}".format(
            rModelName,
            str(self.TransformerLayers), 
            Version)) 

    #
    #
    
    def doLoad (self, rModelName, VersionFile):
        self.rnn.load_weights("../models/{}_Transformer_{}_.v{}".format(
            rModelName,
            str(self.TransformerLayers), 
            VersionFile))    

##
##
##
##
##
##
##
##
############################################################
############################################################

vers = "5"
Epochs = 100
TrainingInsteadOfLoading = True

#
#
#

def saveResults(modelNameV, res, doOverwrite='a'):
    with open("../evaluation/resultsTrace.txt", doOverwrite) as resTxt:
        resTxt.write("{}\t{} \n".format(modelNameV, res))
        print (modelNameV, "\t", res, "\n")
#
#
#

def dRound(vR, d=5):
    return round(vR, d)

#
#
#

def getRMSE (y_true, y_pred):
    
    rmse = dRound(mean_squared_error(y_true, y_pred, squared=True))
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))    
    return "(RMSE= {} \t cm= {} )\n".format(rmse, str(cm))
#
#
#

def printRMSEForAllModels(y_val, predNarrators, predMatn, att=""):
    # print and trace RMSE restuls ...
    saveResults("RMSE {}- Narrators:".format(att), getRMSE(y_val, predNarrators))
    saveResults("RMSE {}- Matn  :".format(att), getRMSE(y_val, predMatn))
    saveResults("RMSE {}- Narrators and Matn:".format(att), getRMSE(y_val, ((predNarrators + predMatn)*.5)))

#
#
#

def TrainingOrLoading (rModel, ds, rModelName):
    if TrainingInsteadOfLoading:
        rModel.doTrain(ds, Epochs)
        rModel.doSave(rModelName, vers)
        #
        #
        loss = round(rModel.rnn.history.history['loss'][-1], 5)
        accuracy = round(rModel.rnn.history.history['accuracy'][-1], 5)
        saveResults("\t Training [{}]: loss= {}, accuracy={}".format(rModelName, loss, accuracy),"")
        #
    else:
        rModel.doLoad(rModelName, vers)        
    return rModel
#
#    
#
 
def main():   
    #
    #
    saveResults(datetime.now(), "", "w")
    #
    #
    #
    # Loading datasets: X and Y 
    #   [iFeatures=0]  for Narrators | [iFeatures=1]  for Matn | [iFeatures=2]  for merging Matn + Narrators  
    iFeatures = 0
    hadiths = Hadiths()
    trainNarrators, xNarrators, x_valNarrators, y_val = hadiths.loadDataset(iFeatures)
    #
    #    
    iFeatures = 1
    trainMatn, xMatn, x_valMatn, _ = hadiths.loadDataset(iFeatures)
    hadiths = None # for freeing the mem
    #
    #
    # Experiment 1
    # *******************************************************************
    # Create the first two rnn models that don't include transformer layers based on LSTM
    transformerLayers = False
    isForNarrators=True
    rnnNarrators = RnnLstm(xNarrators, transformerLayers, isForNarrators)    
    rnnMatn = RnnLstm(xMatn, transformerLayers)
    #
    #
    # Training | Loading
    rnnNarrators = TrainingOrLoading(rnnNarrators, trainNarrators, "Narrators")
    rnnMatn = TrainingOrLoading(rnnMatn, trainMatn, "Matn")
    #
    #    
    # Prediction and RMSE restuls...    
    printRMSEForAllModels (y_val, 
                           rnnNarrators.getPrediction(x_valNarrators), 
                           rnnMatn.getPrediction(x_valMatn))    
    #    
    #
    #
    #
    # Experiment 2
    # *******************************************************************
    # Create the second two rnn models that include transformer layers
    transformerLayers = True  
    isForNarrators=True
    rnnNarrators = RnnLstm(xNarrators, transformerLayers, isForNarrators)
    xNarrators = None   # for freeing the mem
    #
    rnnMatn = RnnLstm(xMatn, transformerLayers)
    xMatn = None        # for freeing the mem
    #
    #
    # Training | Loading
    rnnNarrators = TrainingOrLoading(rnnNarrators, trainNarrators, "Narrators")
    rnnMatn = TrainingOrLoading(rnnMatn, trainMatn, "Matn")
    #
    #
    # Prediction and RMSE restuls...    
    printRMSEForAllModels (y_val, 
                           rnnNarrators.getPrediction(x_valNarrators), 
                           rnnMatn.getPrediction(x_valMatn), "Transformer-Layers")         
    #
    #
    # 
    print ("\n\nProcess completed.")


if __name__ == "__main__":
    main()
##
##
##
##
##
##
##
## End
############################################################

