from code import gbdt
from code import gbdt_lr

logFile = './train2.log'
filename = 'drug_protein_original'
boost_round = 100
keepId = False

# gbdt.start(filename, boost_round, keepId, logFile)
gbdt_lr.start(filename, boost_round, keepId, logFile)
