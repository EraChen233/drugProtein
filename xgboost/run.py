from code import gbdt

logFile = './train1.log'
filename = 'drug_protein'
boost_round = 2
keepId = False

gbdt.start(filename, boost_round, keepId, logFile)
