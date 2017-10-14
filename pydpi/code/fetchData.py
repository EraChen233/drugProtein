# -*- coding: UTF-8 -*-  
#prama warning(disable:8888)
# Date 2017.7.11
# Author chenjiyun

import pydrug, pypro, pydpi
import pandas as pd
import csv
import multiprocessing
import featrue2

originalData = pd.read_csv('./code/originalData/all.csv')

proteinCsv = pd.read_csv('./code/originalData/protein.csv')
drugCsv = pd.read_csv('./code/originalData/drug.csv');
human = originalData[originalData['Species']=="Human"];
proteinObject=pypro.PyPro()
drugObject=pydrug.PyDrug()
proteinCharacter=['proteinId','AAComp','CTD','DPComp','MoranAuto','MoreauBrotoAuto','APAAC','QSO','Triad','SOCN','PAAC'];
drugCharacter = ['drugId','Constitution','Topology','Connectivity','Estate','Kappa','MOE','Geary','Moran','MoreauBroto','Charge','MolProperty','AllDescriptor']


length = human.shape[0]
step = 10
info = ""

# def func():
#   print "hello timer!"

def fetchProtein(index,proteinId,startIndex,endIndex) :
    try:
        mess = "Downloading protein from "+str(startIndex)+" to " + str(endIndex) + ": "+str(index)+"_"+proteinId
        print("\033[1;37;41m%s\033[0m" % mess)
        protein = [];
        ps=proteinObject.GetProteinSequenceFromID(proteinId)
        proteinObject.ReadProteinSequence(ps)

        protein.append(proteinId)
        protein.append( str(proteinObject.GetAAComp()) )
        protein.append( str(proteinObject.GetCTD()) )
        protein.append( str(proteinObject.GetDPComp()) )
        protein.append( str(proteinObject.GetMoranAuto()) )
        protein.append( str(proteinObject.GetMoreauBrotoAuto()) )
        protein.append( str(proteinObject.GetAPAAC()) )
        protein.append( str(proteinObject.GetQSO()) )
        protein.append( str(proteinObject.GetTriad()) )
        protein.append( str(proteinObject.GetSOCN()) )
        protein.append( str(proteinObject.GetPAAC()) )
        # 三肽组成暂时不要，字符串长度超过 2 * (2^15) 表格放不下。
        # 转换为对象形式，差不多是六七千个属性
        # protein.append( str(proteinObject.GetTPComp()))
        return protein
    except Exception,e:
        return TimeOutOrError(index,"protein",proteinId,e)

def fetchDrug(index,drugId,startIndex,endIndex) : 
    try:
        mess = "Downloading Drug "+str(startIndex)+" to " + str(endIndex) + ": "+str(index)+"_"+drugId
        print("\033[1;37;41m%s\033[0m" % mess)
        drug = []
        drug.append(drugId)
        smi = pydrug.getmol.GetMolFromDrugbank(drugId)
        drugObject.ReadMolFromSmile(smi)

        drug.append( str(drugObject.GetConstitution()) )
        drug.append( str(drugObject.GetTopology()) )
        drug.append( str(drugObject.GetConnectivity()) )
        drug.append( str(drugObject.GetEstate()) )
        drug.append( str(drugObject.GetKappa()) )
        drug.append( str(drugObject.GetMOE()) )
        drug.append( str(drugObject.GetGeary()) )
        drug.append( str(drugObject.GetMoran()) )
        drug.append( str(drugObject.GetMoreauBroto()) )
        drug.append( str(drugObject.GetCharge()) )
        drug.append( str(drugObject.GetMolProperty()) )
        drug.append( str(drugObject.GetAllDescriptor()).replace(',',";") )
        return drug
    except Exception,e:
        return TimeOutOrError(index,"drug",drugId,e)

def fetchDPI1(index,proteinId,drugId,startIndex,endIndex,type) :
    mess = "Downloading DPI 1 "+type+" from "+str(startIndex)+" to "+str(endIndex)+"。Current index is "+str(index)+" : "+proteinId + "_"+drugId
    print("\033[1;37;41m%s\033[0m" % mess)
    dpi=pydpi.PyDPI()
    smi = pydrug.getmol.GetMolFromDrugbank(drugId)
    dpi.ReadMolFromSmile(smi)
    ddict=dpi.GetConnectivity()
    ddict.update(dpi.GetKappa())
    ps=dpi.GetProteinSequenceFromID('P48039')
    dpi.ReadProteinSequence(ps)
    pdict=dpi.GetAAComp()
    pdict.update(dpi.GetAPAAC())
    return dpi.GetDPIFeature1(ddict,pdict)

def fetchDPI2(index,proteinId,drugId,startIndex,endIndex,type) :
    mess = "Downloading DPI 2 "+type+" from "+str(startIndex)+" to "+str(endIndex)+"。Current index is "+str(index)+": "+proteinId+'_'+drugId
    print("\033[1;37;41m%s\033[0m" % mess)    
    dpi=pydpi.PyDPI()
    smi = pydrug.getmol.GetMolFromDrugbank(drugId)
    dpi.ReadMolFromSmile(smi)
    ddict=dpi.GetConnectivity()
    ddict.update(dpi.GetKappa())
    ps=dpi.GetProteinSequenceFromID('P48039')
    dpi.ReadProteinSequence(ps)
    pdict=dpi.GetAAComp()
    pdict.update(dpi.GetAPAAC())
    return dpi.GetDPIFeature2(ddict,pdict)

def TimeOutOrError(index,arg,id,e) : 
    with open("code/logs/{0}/{1}_{2}.txt".format(arg,index,id),"w") as logfile: 
        logfile.write(e.message)
        logfile.write("\n")
        print("\033[1;37;41m%s\033[0m" % e.message)
    logfile.close()

#arg : GetDPIFeature1 or GetDPIFeature2
#type : 正样本 or fu样本
def fetchDPIs(arg,type,data,startIndex,endIndex) :
    stepDpi = step
    sampleLen = len(data)
    #proteinLen = proteinCsv.shape[0]
    featrue1Head = ['protein_drug','dpi'+str(arg)]
    if endIndex <= 0 :
        endIndex = sampleLen
    for i in range(startIndex/stepDpi,endIndex/stepDpi+1):
        with open('code/dataset/dpi{0}_{1}/{2}.csv'.format(arg,type,i), "w") as csvfile :
            writer = csv.writer(csvfile)
            if arg == 1 :
                writer.writerow(featrue1Head)
            else:
                featrue2Head = featrue2.head()
                writer.writerow(['protein_drug']+featrue2.head())
            for j in range(0,stepDpi):
                index = i * step + j;
                if index < sampleLen :
                    # if index in errorList :
                        # continue;
                    arr = [data[index]]
                    proteinId = data[index].split('_')[0]
                    drugId = data[index].split('_')[1]
                    try:
                        if arg == 1:
                            arr.append(fetchDPI1(index,proteinId,drugId,startIndex,endIndex,type))
                        else :
                            currHead = []
                            drugEntery = fetchDPI2(index,proteinId,drugId,startIndex,endIndex,type)
                            #for item in drugEntery : 
                            #    currHead.append(item)
                            #if sorted(currHead) != featrue2Head:
                            #    raise Exception("表格头部不对")#异常被抛出，print函数无法执行
                            for item in featrue2Head :
                                arr.append(drugEntery[item])
                            
                        writer.writerow(arr)
                    except Exception,e:
                        FetchDpiError(arg,type,index,proteinId,drugId,e)

def FetchDpiError(arg,type,index,proteinId,drugId,e) : 
    if e.message == "表格头部不对":
        fileName = "code/logs/dpi{0}_{1}/{2}_{3}_{4}表格头部不对.txt".format(arg,type,index,proteinId,drugId)
    else :
        fileName = "code/logs/dpi{0}_{1}/{2}_{3}_{4}.txt".format(arg,type,index,proteinId,drugId)
    with open(fileName,"w") as logfile: 
        logfile.write(e.message)
        logfile.write("\n")
        print("\033[1;37;41m%s\033[0m" % e.message)
    logfile.close()
      
    
def fetchDatas(data,arg,tableHead,fetchFun,startIndex,endIndex=0,errorList=[]) :
    if endIndex > 0 :
      length = endIndex
    else :
      length = data.shape[0]
    for i in range(startIndex/step,length/step+1) :
        with open("code/dataset/{0}/{0}_{1}.csv".format(arg,i),"w") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow(tableHead)
            for j in range(0,step):
                index = i * step +j
                if index<data.shape[0] :
                    if (index in errorList) and (arg =="drug"):
                        continue;
                    try:
                        id = data.iat[index,0].strip()
                        # d = fetchFun(index,id)
                        writer.writerow(fetchFun(index,id,startIndex,endIndex))
                    except Exception,e:
                        TimeOutOrError(index,arg,id,e)
        csvfile.close()

def fetch(arg,startIndex = 0,endIndex=0,drump=[]):
  if arg == 'drug':
    fetchDatas(drugCsv,'drug',drugCharacter,fetchDrug,startIndex,endIndex,drump)
  if arg == "protein":
    fetchDatas(proteinCsv,'protein',proteinCharacter,fetchProtein,startIndex,endIndex)
  
  return;
