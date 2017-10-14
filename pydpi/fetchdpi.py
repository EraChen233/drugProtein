# -*- coding: utf-8 -*-
from code import fetchData
import pandas as pd
import random

# drug :4950
# protein : 2313
# positive 12320
# negative 15000

# fetchDPIs
# arg : GetDPIFeature1 or GetDPIFeature2
# type : positive or negetive 

#正样本
# positiveCsv = pd.read_csv('./code/originalData/positive.csv');
# positiveData = positiveCsv.iloc[:,0]
# fetchData.fetchDPIs(1,'positive',positiveData,9030,9039) 



# 负样本
negetiveCsv = pd.read_csv('./code/originalData/negetive.csv');
negetiveData = negetiveCsv.iloc[:,0]


fetchData.fetchDPIs(1,'negative',negetiveData,1444,1559)
# fetchData.fetchDPIs(1,'negative',negetiveData,13360,13369)
#fetchData.fetchDPIs(1,'negative',negetiveData,11752,11999)
# fetchData.fetchDPIs(1,'negative',negetiveData,12760,12999)
#fetchData.fetchDPIs(1,'negative',negetiveData,13991,13999)
# fetchData.fetchDPIs(1,'negative',negetiveData,14902,15001)  #done

# fetchData.fetchDPIs(1,'negative',negetiveData,12100,12999)

# ----------------- feature2 positive ---------------------

#fetchData.fetchDPIs(2,'positive',positiveData,888,999) done
#fetchData.fetchDPIs(2,'positive',positiveData,1790,1999) done
#fetchData.fetchDPIs(2,'positive',positiveData,2588,2999) done
#fetchData.fetchDPIs(2,'positive',positiveData,3722,3999) done
#fetchData.fetchDPIs(2,'positive',positiveData,4446,4999) done

#fetchData.fetchDPIs(2,'positive',positiveData,5540,5999)  done
# fetchData.fetchDPIs(2,'positive',positiveData,6664,6999) done
#fetchData.fetchDPIs(2,'positive',positiveData,7024,7999)  done
#fetchData.fetchDPIs(2,'positive',positiveData,8424,8999)  done
#fetchData.fetchDPIs(2,'positive',positiveData,9287,9999)  done

# fetchData.fetchDPIs(2,'positive',positiveData,10797,10999)  done
# fetchData.fetchDPIs(2,'positive',positiveData,11296,11999) done
# fetchData.fetchDPIs(2,'positive',positiveData,12297,12399) done



# ------------------- feature2 negative -------------
# fetchData.fetchDPIs(2,'negative',negetiveData,3210,3219)    #done
# fetchData.fetchDPIs(2,'negative',negetiveData,1259,1999)  done
# fetchData.fetchDPIs(2,'negative',negetiveData,2974,2999)  done
# fetchData.fetchDPIs(2,'negative',negetiveData,3357,3999)  done
# fetchData.fetchDPIs(2,'negative',negetiveData,4697,4999)  done

# fetchData.fetchDPIs(2,'negative',negetiveData,5080,5249)  #done
# fetchData.fetchDPIs(2,'negative',negetiveData,6132,6999)  done
# fetchData.fetchDPIs(2,'negative',negetiveData,7079,7999) done
# fetchData.fetchDPIs(2,'negative',negetiveData,8414,8999) done
#fetchData.fetchDPIs(2,'negative',negetiveData,9896,9999) done

# fetchData.fetchDPIs(2,'negative',negetiveData,10000,10999)  done
# fetchData.fetchDPIs(2,'negative',negetiveData,11355,11999)  done
# fetchData.fetchDPIs(2,'negative',negetiveData,12371,12999)  done
# fetchData.fetchDPIs(2,'negative',negetiveData,13000,13999)  done
# fetchData.fetchDPIs(2,'negative',negetiveData,14367,15100)  done


