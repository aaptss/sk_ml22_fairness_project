function [accuracy, bal_acc, eq_odds, TPR_prot, TPR_non_prot, TNR_prot, TNR_non_prot] = main(data)
    % [x, y, sensitive, training, test] = dataImport.importAdultData();
    [x, y, sensitive, training, test] = data();
    folds = 1;

    %x = [x, sensitive]; % introduces disparate treatment (is redundant for Compass data)

    classifier = classifiers.AdaptiveWeights(classifiers.SimpleLogisticClassifier(0.0001));

    accs = 0;
    pRules = 0;
    DFPRs = 0;
    DFNRs = 0;

    accuracy = zeros(1,folds);
    bal_acc = zeros(1,folds);
    eq_odds = zeros(1,folds);
    TPR_prot = zeros(1,folds);
    TPR_non_prot = zeros(1,folds);
    TNR_prot = zeros(1,folds);
    TNR_non_prot = zeros(1,folds);

    %validationFunction = @(c,x,y,s)obtainMetrics(c,x,y,s,[2, 0, 0, -1, -1]);
    % validationFunction = @(c,x,y,s)obtainMetrics(c,x,y,s,[1, 0, 1, 0, 0]);
    validationFunction = @(c,x,y,s)obtainMetrics(c,x,y,s,[1, 1, 1, 0, 0]);

    for fold=1:folds
        if(folds~=1)
            training = randsample(1:length(y), length(training));
            test = setdiff(1:length(y), training);
        end
        classifier.train(x(training,:),y(training),sensitive(training),validationFunction);
        [~, acc, pRule, DFPR, DFNR] = validationFunction(classifier,x(test,:),y(test),sensitive(test));
        accs = accs+acc/folds;
        pRules = pRules+pRule/folds;
        DFPRs = DFPRs+DFPR/folds;
        DFNRs = DFNRs+DFNR/folds;
        fprintf('\nCurrent evaluation on fold %d: acc = %f , pRule = %f , DFPR = %f , DFNR = %f \n\n\n', fold, accs*folds/fold, pRules*folds/fold, DFPRs*folds/fold, DFNRs*folds/fold); 

        [accuracy(fold), bal_acc(fold), eq_odds(fold), TPR_prot(fold), TPR_non_prot(fold), TNR_prot(fold), TNR_non_prot(fold)] = getMetrics(classifier,x(test,:),y(test),sensitive(test));
    end

    mean_accuracy = mean(accuracy);
    mean_bal_acc = mean(bal_acc);
    mean_eq_odds = mean(eq_odds);
    mean_TPR_prot = mean(TPR_prot);
    mean_TPR_non_prot = mean(TPR_non_prot);
    mean_TNR_prot = mean(TNR_prot);
    mean_TNR_non_prot = mean(TNR_non_prot);

    std_accuracy = std(accuracy);
    std_bal_acc = std(bal_acc);
    std_eq_odds = std(eq_odds);
    std_TPR_prot = std(TPR_prot);
    std_TPR_non_prot = std(TPR_non_prot);
    std_TNR_prot = std(TNR_prot);
    std_TNR_non_prot = std(TNR_non_prot);

    fprintf('\nAccuracy mean: %f , std = %f \n\n', mean_accuracy, std_accuracy);
    fprintf('\nBalanced accuracy mean: %f , std = %f \n\n', mean_bal_acc, std_bal_acc);
    fprintf('\nEqualized Odds: %f , std = %f \n\n', mean_eq_odds, std_eq_odds);
    fprintf('\nTPR prot.: %f , std = %f \n\n', mean_TPR_prot, std_TPR_prot);
    fprintf('\nTPR non-prot: %f , std = %f \n\n', mean_TPR_non_prot, std_TPR_non_prot);
    fprintf('\nTNR prot.: %f , std = %f \n\n', mean_TNR_prot, std_TNR_prot);
    fprintf('\nTNR non-prot: %f , std = %f \n\n', mean_TNR_non_prot, std_TNR_non_prot);

end