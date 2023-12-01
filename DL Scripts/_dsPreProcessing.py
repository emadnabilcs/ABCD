"""
Created on 10/10/2021
@author: Abdullah M. Alshanqiti (a.m.alshanqiti@gmail.com)
"""
import pandas as pd
import tensorflow as tf

##
##
##
##
##
##
##
##
############################################################

class Hadith():

     def __init__(self, HID, MATN, NArrators, VErdict):
         self.hID=HID
         self.Matn = MATN
         self.Narrators = NArrators
         self.Verdict = VErdict

##
##
##
############################################################

class Hadiths():

    def __init__(self):
        self.listHadiths = []
        self.X=None
        self.Y=None
     
        
    def addHadith(self, hID, Matn, Narrators, Verdict):
         
        # TODO: Clean textual Matn
                 
        # TODO: Narrators
        Narrators = [].append(Narrators)
        # TODO: Verdict lable
            
        self.listHadiths.append(Hadith(hID, Matn, Narrators, Verdict))
        
        return

    
    def VerdictOneHotEncoding(self, iVerd):          
         
        #
        # Marfue=0 | Mawquf=1 | Maqtue=3
        if iVerd==0:
            return "1,0,0"
        elif iVerd==1:
            return "0,1,0"
        elif iVerd==2:
            return "0,0,1"
        #
        return None
    
    

    def saveDS(self, strDefPath = "../resources/{}.csv"):
        
        # saving Matn ...
        MatnFile = open(strDefPath.format("Matn"), "w", encoding="utf-8")
        
        # saving Narrators ...
        NarratorsFile = open(strDefPath.format("Narrators"), "w", encoding="utf-8")
                
        # saving Matn and Narrators ...
        MatnAndNarratorsFile = open(strDefPath.format("MatnAndNarrators"), "w", encoding="utf-8")        
        
        # saving Verdict ...
        VerdictFile = open(strDefPath.format("Verdict"), "w", encoding="utf-8")
        
        for h in self.listHadiths:
            MatnFile.write("{}\n".format(h.Matn))
            NarratorsFile.write("{}\n".format(h.Narrators))
            MatnAndNarratorsFile.write("{},{}\n".format(h.Matn,h.Narrators))
            VerdictFile.write("{}\n".format(h.Verdict))     
        #
        #
        # closing files ...
        MatnFile.close()
        NarratorsFile.close()
        MatnAndNarratorsFile.close()
        VerdictFile.close()
        

        
    # loading Dataset : training, validation & prediction parts
    ###################################################################
    def loadDataset(self, iFeatures=0, iSplit=-1, batch_size=100):
             
        #  iFeatures=0 -> Narrators | iFeatures<>1 -> Matn
        #
        strDefPath =""
        if iFeatures==0:
            strDefPath ="Narrators"
        else:
            strDefPath="Matn"
        #
        #
        self.X = pd.DataFrame(pd.read_csv("../resources/{}.csv".format(strDefPath), header=None)).values
        #
        #
        #
        # We fetch the labels only once
        if self.Y is None:        
            # One-Hot Encoding for verdict types
            # Marfue=0 (1,0,0) | Mawquf=1 (0,1,0) | Maqtue=3 (0,0,1) 
            self.Y = pd.DataFrame(pd.read_csv("../resources/Verdict.csv", header=None)).values
        #
        #
        #
        #
        # Reserve (size=iSplit) samples for validation.
        if iSplit<0 or iSplit>=len(self.X):
            iSplit = round(.3*len(self.X))            
            #iSplit=10
        #x_tra = self.X[:-iSplit]   print ("x_tra", x_tra)
        #y_tra = self.Y[:-iSplit]   print ("y_tra", y_tra)   
        x_val = self.X[-iSplit:]   
        y_val = self.Y[-iSplit:]   
        #
        #
        # Prepare the training dataset.
        trainDS = tf.data.Dataset.from_tensor_slices((self.X[:-iSplit], self.Y[:-iSplit]))
        trainDS = trainDS.shuffle(buffer_size=1024).batch(batch_size)        
        #
        #    
        # Prepare the validation dataset.
        ##valDS = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        ##valDS = valDS.batch(batch_size)
        #
        #        
        #for element in trainDS.as_numpy_iterator():
        #    print(element)
        #xTrain = list(trainDS.as_numpy_iterator())[0]
        
        #print (trainDS)
        
        #print ("valDS\n", valDS)
        #return trainDS, valDS, self.X, self.Y, x_val, y_val
        return trainDS, self.X, x_val, y_val

##
##
##
##
##
##
##
##
############################################################


    
def main():
    
    # Testing ...
    # Loading all Hadiths DS
    hadiths = Hadiths()
    
       
    trainNarrators, xNarrators, x_valNarrators, y_val = hadiths.loadDataset(0)
    #print ("trainNarrators", trainNarrators)
    #print ("xNarrators", xNarrators)
    #print ("x_valNarrators", x_valNarrators)
    print ("y_val", y_val)
    print ("y_val", y_val.shape)

    
    #trainMatn, xMatn, x_valMatn, _ = hadiths.loadDataset(1)    
    #print ("trainMatn", trainMatn)
    #print ("xMatn", xMatn)
    #print ("x_valMatn", x_valMatn.shape)
    
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



