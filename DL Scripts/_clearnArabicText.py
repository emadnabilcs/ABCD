"""
Created on 10/10/2021
@author: Abdullah M. Alshanqiti (a.m.alshanqiti@gmail.com)
"""
import re as regEx
import pandas as pd
from nltk.tokenize import word_tokenize
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_alef_maksura_ar
from camel_tools.utils.normalize import normalize_alef_ar
from camel_tools.utils.normalize import normalize_teh_marbuta_ar
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

#
#
#
arStemmer = Analyzer(MorphologyDB.builtin_db())
#
#
#

def removeNonArabicChar(strText):    
    # 
    # remove english and non-arabic (including special) characters 
    strText = regEx.compile('([^\n\u060C-\u064A\.:؟?])').sub(' ', strText)
    #
    # remove extra spaces 
    return regEx.sub(" +", " ", strText)


def getStemedWToken(wToken):
    #
    try:            
        stemObject = arStemmer.analyze(wToken)
        
        # Remove Tashkeel and Normailse
        strText = dediac_ar(stemObject[0]['stem'])     
        return strText
    except:
        return wToken
       
def getlemmatized (strMatn):
    #
    getTokens = word_tokenize(strMatn)        
    strMatn = ""
    
    for strToken in getTokens:
        strToken = strToken.strip()
        if len(strToken)<=1:
            continue        
        strMatn += getStemedWToken(
            strToken) + ' ' 
        
    return strMatn.strip()
    
def add_if_key_not_exist(dict_obj, key, value):
    if key not in dict_obj:
        dict_obj.update({key: value})
            
def txtNormalize(strMatn):

    #
    #
    # Normailse 
    strMatn = normalize_teh_marbuta_ar(strMatn)   # for Alha
    strMatn = normalize_alef_ar(strMatn)          # for Alhamza
    strMatn = normalize_alef_maksura_ar(strMatn) 
    #
    # Remove Tashkeel
    strMatn = dediac_ar(strMatn)
    #
    #
    # Remove newline
    strMatn = strMatn.replace('\n', ' ')
    strMatn = removeNonArabicChar(strMatn)
    #
    #
    strMatn = getlemmatized (strMatn)
    #
    #
    # remove extra spaces 
    strMatn = regEx.sub(" +", " ", strMatn)
    
    return strMatn


def readWriteNarratorsToCSVs():
    # 
    #
    AllNarratorsTokens = {}
    sequence_length = -1
    #
    #
    wFile = open("../resources/Narrators.csv", "w", encoding="utf-8")
    data = pd.read_excel ("../resources/original/HadithDatabase2.xlsx") 
    df_Narrators = pd.DataFrame(data, columns= ['Chains'])
    #
    # 
    for i in range(len(df_Narrators)):   
        #
        strNarrators = df_Narrators.iat[i, 0] 
        strNarrators = reFormateNarrotors (strNarrators)
        #
        wTokens = word_tokenize(strNarrators)
        for strToken in wTokens:
            add_if_key_not_exist(AllNarratorsTokens, strToken, None)  
        
        
        lenStrNarrators = len(wTokens)
        if lenStrNarrators>sequence_length:
            sequence_length = lenStrNarrators
        #
        #
        wFile.write("{}\n".format(strNarrators))        
    #
    #
    wFile.close()
    #    
    #
    print ("\n\t Max_Tokens: {} \n\t Max-sequence-length: {}\n\n".format(len(AllNarratorsTokens), sequence_length))
    
    return None
    

def reFormateNarrotors(strNarrators):
    #
    listIndices = [pos for pos, char in enumerate(strNarrators) if char == ')']  
    strCleanedNarrators = ""
    for element in range(0, len(strNarrators)):
        if element in listIndices or element+1 in listIndices:
            continue
        ch = strNarrators[element]
        if ch =='\n':
            ch =' '
        strCleanedNarrators += ch
    #
    #
    return strCleanedNarrators.strip()

    
def readWriteMatnToCSV():
    # 
    #
    AllMatnTokens = {}
    sequence_length = -1
    #
    #
    wFile = open("../resources/Matn.csv", "w", encoding="utf-8")
    data = pd.read_excel ("../resources/original/HadithDatabase.V3.xlsx") 
    df_Matn = pd.DataFrame(data, columns= ['Matn'])
    #
    # 
    for i in range(len(df_Matn)):   
        #
        strMatn = df_Matn.iat[i, 0] 

        #
        wTokens = word_tokenize(strMatn)
        for strToken in wTokens:
            add_if_key_not_exist(AllMatnTokens, strToken, None)  
        
        
        lenStrMatn = len(wTokens)
        if lenStrMatn>sequence_length:
            sequence_length = lenStrMatn
        #
        #
        wFile.write("{}\n".format(txtNormalize(strMatn)))        
    #
    #
    wFile.close()
    #    
    #
    print ("\n\t Max_Tokens: {} \n\t Max-sequence-length: {}\n\n".format(len(AllMatnTokens), sequence_length))
    
    return None




def hadithDSXLSX():

    data = pd.read_excel ("../resources/original/HadithDatabase2.xlsx") 
    df_Narrotors = pd.DataFrame(data, columns= ['Chains'])
    
    strNarrotors = df_Narrotors.iat[0, 0] 
    print ("Before", strNarrotors)
    strNarrotors = reFormateNarrotors(strNarrotors)
    print ("\nAfter", strNarrotors)
    
    
    return None



if __name__ == "__main__":
    readWriteMatnToCSV()
    #hadithDSXLSX()
    #readWriteNarratorsToCSVs()    
    print ("\n\nProcess completed.")    



