# -*- coding: UTF-8 -*-  
#prama warning(disable:8888)
# Date 2017.7.11
# Author chenjiyun

import pydrug, pypro, pydpi
import pandas as pd
import csv
import multiprocessing

import fetchData

originalData = pd.read_csv('./code/originalData/all_target_ids_all.csv');
proteinCsv = pd.read_csv('./code/originalData/protein.csv');
drugCsv = pd.read_csv('./code/originalData/drug.csv');
originalData = pd.read_csv('./code/originalData/all_target_ids_all.csv');
human = originalData[originalData['Species']=="Human"];
proteinObject=pypro.PyPro()
drugObject=pydrug.PyDrug()
proteinCharacter=['proteinId','AAComp','CTD','DPComp','MoranAuto','MoreauBrotoAuto','APAAC','QSO','Triad','SOCN','PAAC','TPComp'];
drugCharacter = ['drugId','Constitution','Topology','Connectivity','Estate','Kappa','MOE','Geary','Moran','MoreauBroto','Charge','MolProperty','AllDescriptor']

length = human.shape[0]
step = 10
info = ""

# def func():
#   print "hello timer!"

def fetchProtein(index,proteinId) :
    try:
        mess = "Downloading protein "+str(index)+" "+proteinId
        print("\033[1;37;41m%s\033[0m" % mess)
        protein = [];
        ps=proteinObject.GetProteinSequenceFromID(proteinId)
        proteinObject.ReadProteinSequence(ps)

        protein.append(proteinId)
        protein.append( proteinObject.GetAAComp() )
        protein.append( proteinObject.GetCTD() )
        protein.append( proteinObject.GetDPComp() )
        protein.append( proteinObject.GetMoranAuto() )
        protein.append( proteinObject.GetMoreauBrotoAuto() )
        protein.append( proteinObject.GetAPAAC() )
        protein.append( proteinObject.GetQSO() )
        protein.append( proteinObject.GetTriad() )
        protein.append( proteinObject.GetSOCN() )
        protein.append( proteinObject.GetPAAC() )
        protein.append( proteinObject.GetTPComp() )
        return protein
    except Exception,e:
        return TimeOutOrError(index,"protein",proteinId,e)

def fetchDrug(index,drugId) : 
    try:
        mess = "Downloading Drug "+str(index)+" "+drugId
        print("\033[1;37;41m%s\033[0m" % mess)
        drug = []
        drug.append(drugId)
        smi = pydrug.getmol.GetMolFromDrugbank(drugId)
        drugObject.ReadMolFromSmile(smi)

        drug.append( drugObject.GetConstitution() )
        drug.append( drugObject.GetTopology() )
        drug.append( drugObject.GetConnectivity() )
        drug.append( drugObject.GetEstate() )
        drug.append( drugObject.GetKappa() )
        drug.append( drugObject.GetMOE() )
        drug.append( drugObject.GetGeary() )
        drug.append( drugObject.GetMoran() )
        drug.append( drugObject.GetMoreauBroto() )
        drug.append( drugObject.GetCharge() )
        drug.append( drugObject.GetMolProperty() )
        drug.append( drugObject.GetAllDescriptor() )
        return drug
    except Exception,e:
        return TimeOutOrError(index,"drug",drugId,e)


def TimeOutOrError(index,arg,id,e) : 
    with open("code/logs/{0}/log_{1}_{2}.txt".format(arg,index,id),"w") as logfile: 
        logfile.write(e.message)
        logfile.write("\n")
        print("\033[1;37;41m%s\033[0m" % e.message)
    logfile.close()


def fetchDatas(data,arg,tableHead,fetchFun) :
    length = data.shape[0]
    for i in range(0,length/step+1) :
        with open("code/dataset/{0}/{0}_{1}.csv".format(arg,i),"w") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow(tableHead)
            for j in range(0,step-1):
                index = i * step +j
                try:
                    id = data.iat[index,0]
                    # d = fetchFun(index,id)
                    writer.writerow(fetchFun(index,id))
                except Exception,e:
                    TimeOutOrError(index,arg,id,e)
        csvfile.close()




fetchDatas(proteinCsv,'protein',proteinCharacter,fetchProtein)
fetchDatas(drugCsv,'drug',drugCharacter,fetchDrug)

















def start_process():
    mess = 'Starting '+ multiprocessing.current_process().name
    print("\033[1;37;41m%s\033[0m" % mess)

# pool_size=multiprocessing.cpu_count()*2
# pool=multiprocessing.Pool(processes=pool_size, initializer=start_process)
def fetchData() : 
    try:
        index = 0
        proteinId = 0
        drugId = 0
        info = ""
        e = None
        for i in range(1,(len/step+1)):
            with open("dataset/dataset_{0}.csv".format(i),"w") as csvfile: 
                writer = csv.writer(csvfile)
                #先写入columns_name
                writer.writerow(drugCharacter + proteinCharacter)
                for y in range(0,step-1):
                    index = i * step + y
                    if index <len :
                        proteinId = human.iat[index,0]
                        tempProtein = fetchProtein(proteinId)
                        drugIds = human.iat[index,12].split(';')
                        for drugId in drugIds:
                            try:
                                def callback(tempDrug):
                                    info = " Info#  "+ str(index)+" : "+proteinId
                                    print("\033[1;37;41m%s\033[0m" % info)
                                    writer.writerow(tempDrug + tempProtein)
                                    
                                tempDrug = pool.apply_async(fetchDrug,drugId,callback = callback)
                                # info = " Info#  "+ str(index)+" : "+proteinId + " "+drugId
                                
                                # tempDrug = fetchDrug(drugId)
                                #print tempDrug
                                #print tempProtein
                            except Exception,e:
                                TimeOutOrError(index,proteinId,"drugId",info,e)
            csvfile.close()
                        

    except Exception,e:
        TimeOutOrError(index,proteinId,drugId,info,e)


# fetchData()
# pool.close()
# pool.join()


# def fetchData() : 
#     try:
#         index = 0
#         proteinId = 0
#         drugId = 0
#         info = ""
#         e = None
#         for i in range(1,(len/step+1)):
#             with open("dataset/dataset_{0}.csv".format(i),"w") as csvfile: 
#                 writer = csv.writer(csvfile)
#                 #先写入columns_name
#                 writer.writerow(drugCharacter + proteinCharacter)
#                 for y in range(0,step-1):
#                     index = i * step + y
#                     if index <len :
#                         proteinId = human.iat[index,0]
#                         tempProtein = fetchProtein(proteinId)
#                         drugIds = human.iat[index,12].split(';')
#                         for drugId in drugIds:
#                             try:
#                                 pool_size=multiprocessing.cpu_count()*2
#                                 pool=multiprocessing.Pool(processes=pool_size, initializer=start_process,)
#                                 pool_outputs=pool.map(do_calculation,inputs)
#                                 pool.close()
#                                 pool.join()
                                
#                                 info = " Info#  "+ str(index)+" : "+proteinId + " "+drugId
#                                 print("\033[1;37;41m%s\033[0m" % info)
#                                 tempDrug = fetchDrug(drugId)
#                                 #print tempDrug
#                                 #print tempProtein
#                                 writer.writerow(tempDrug + tempProtein)
#                             except Exception,e:
#                                 TimeOutOrError(index,proteinId,drugId,info,e)

#     except Exception,e:
#         TimeOutOrError(index,proteinId,drugId,info,e)





# for x

#写入多行用writerows
# writer.writerows([[0,1,3],[1,2,3],[2,3,4]])

# table = pd.DataFrame(result,index=None,columns=['AAComp','CTD','DPComp','MoranAuto','MoreauBrotoAuto','APAAC','QSO','Triad','SOCN','TPComp','PAAC'])
# table.to_csv('./dataset.csv',index=None)
