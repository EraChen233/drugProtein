# -*- coding: UTF-8 -*-
import fun

# fun.splitData()
# fun.getPositiveSample()
# fun.getNegetiveData(37000)  # 随机生成五倍负样本
# fun.dp1split()   # 拆分dp1的数据

# dpi1 数据拆分
# fun.dp1split()

# 蛋白质数据拆分
# fun.proteinsplit('AAComp',1,'init')
# fun.proteinsplit('CTD',3)
# fun.proteinsplit('DPComp',1)
# fun.proteinsplit('MoranAuto',0)
# fun.proteinsplit('MoreauBrotoAuto',0)
# fun.proteinsplit('APAAC',1)
# fun.proteinsplit('QSO',2)
# fun.proteinsplit('Triad',1)
# fun.proteinsplit('SOCN',1)
# fun.proteinsplit('PAAC',2)


# drug 数据拆分
# head = ['Constitution', 'Topology', 'Connectivity', 'Estate', 'Kappa', 'MOE', 'Geary', 'Moran', 'MoreauBroto', 'Charge', 'MolProperty', 'AllDescriptor'] # 不需要AllDescriptor
# head = ['Constitution', 'Topology', 'Connectivity', 'Estate', 'Kappa', 'MOE', 'Geary', 'Moran', 'MoreauBroto', 'Charge', 'MolProperty']
# for i in range(0,len(head)):
#     if i == 0:
#         fun.drugsplit(head[i],'init')
#     elif i == len(head)-1:
#         fun.drugsplit(head[i],';')
#     else:
#         fun.drugsplit(head[i])

# protein drug 数据合并
fun.joinProteinOrDrug('negetive*3')

# 拆分dpi 2 的protein_drug
# fun.splitdpi2()

# 检查数据缺失值
# fun.drawEmpty('drug_protein*2')

# 检查数据连续还是离散
# fun.continuousOrDiscrete('drug_protein*2_1')
