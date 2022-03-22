function [accuracy, bal_acc, eq_odds, TPR_prot, TPR_non_prot, TNR_prot, TNR_non_prot] = getMetrics(classifier, x, y, sensitive)
    decisionThreshold = 0.5;
    nonSensitive = ~sensitive;
    
    %% obtain classification scores
    scores = classifier.predict(x);
    
    positiveClassification = (scores>decisionThreshold);
    positive = (y>decisionThreshold);
    
    %Accuracy
    correctClassifcation = 1-xor(positiveClassification,positive); 
    accuracy = sum(correctClassifcation)/length(y);
    
    %TPR
    TPR = sum(and(positiveClassification,positive))/sum(positive);
    
    %TNR
    TNR = sum(~or(positiveClassification,positive))/sum(positive==0);
    
    %Balanced accuracy
    bal_acc = (TPR+TNR)/2;
    
    %TPR protected
    TPR_prot = sum(and(and(positiveClassification,positive),sensitive))/sum(sensitive);
    
    %TPR non-protected
    TPR_non_prot = sum(and(and(positiveClassification,positive),nonSensitive))/sum(nonSensitive);
    
    %TNR protected
    TNR_prot = sum(and(~or(positiveClassification,positive),sensitive))/sum(sensitive);
    
    %TNR non-protected
    TNR_non_prot = sum(and(~or(positiveClassification,positive),nonSensitive))/sum(nonSensitive);
    
    %Equalized Odds
    FPR_prot = 1 - TNR_prot;
    FPR_non_prot = 1 - TNR_non_prot;
    dFPR = FPR_prot - FPR_non_prot;
    FNR_prot = 1 - TPR_prot;
    FNR_non_prot = 1 - TPR_non_prot;
    dFNR = FNR_prot - FNR_non_prot;
    eq_odds = abs(dFPR) + abs(dFNR);

end