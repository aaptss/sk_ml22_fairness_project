function [objective, accuracy, pRule, DFPR, DFNR] = obtainMetrics(classifier, x, y, sensitive, objectiveWeights)
    decisionThreshold = 0.5;
    nonSensitive = ~sensitive;
    
    if nargin<4
        objectiveWeights = zeros(5,1);
    end
    %% obtain classification scores
    scores = classifier.predict(x);
    
    positiveClassification = (scores>decisionThreshold);
    positive = (y>decisionThreshold);
    
    correctClassifcation = 1-xor(positiveClassification,positive); 
    accuracy = sum(correctClassifcation)/length(y);
    
    %FPR parity
    DFPR = sum(correctClassifcation(sensitive)==0 & positive(sensitive)==0)/sum(positive(sensitive)==0) ...
             - sum(correctClassifcation(nonSensitive)==0 & positive(nonSensitive)==0)/sum(positive(nonSensitive)==0);
    %FNR parity
    DFNR = sum(positiveClassification(sensitive)==0 & positive(sensitive)==1)/sum(positive(sensitive)==1) ...
             - sum(positiveClassification(nonSensitive)==0 & positive(nonSensitive)==1)/sum(positive(nonSensitive)==1);
         
    %pRule
    pRule = min(sum(positiveClassification(sensitive))/sum(positiveClassification(nonSensitive)) * sum(nonSensitive)/sum(sensitive), ...
                sum(positiveClassification(nonSensitive))/sum(positiveClassification(sensitive)) * sum(sensitive)/sum(nonSensitive));

    
    objective = objectiveWeights(1)*accuracy + objectiveWeights(2)*pRule + objectiveWeights(3)*abs(DFPR) + objectiveWeights(4)*abs(DFNR);
   
end