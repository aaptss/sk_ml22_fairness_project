f = 5
s = 7

filename = 'metrics/metrics_census_adaWeights.txt'

accuracy = [0.9363, 0.9405, 0.9313, 0.9401, 0.9365]
bal_acc = [0.6056, 0.5893, 0.6059, 0.6115, 0.6229]
eq_odds = [0.0845, 0.0996, 0.0799, 0.0850, 0.0945]
TPR_prot = [0.0048, 0.0015, 0.0077, 0.0061, 0.0077]
TPR_non_prot = [0.0231, 0.0219, 0.0228, 0.0235, 0.0272]
TNR_prot = [0.9580, 0.9702, 0.9505, 0.9608, 0.9588]
TNR_non_prot = [0.8918, 0.8910, 0.8857, 0.8932, 0.8838]

metrics = [accuracy, bal_acc, eq_odds, TPR_prot, TPR_non_prot, TNR_prot, TNR_non_prot]
metrics = [[metrics[i][j] for i in range(s)] for j in range(f)]
m = []
for i in range(f):
    m += metrics[i]


a = {'name': f*['Accuracy', 'Bal. Acc.', 'Eq.Odds', 'TPR Prot', 'TPR Non-Prot', 'TNR Prot', 'TNR Non-Prot'],
     'value': m,
     'model': s*f*['AdaWeights']}

print(a)

fil = open(filename, 'w+')
fil.write(str(a))
