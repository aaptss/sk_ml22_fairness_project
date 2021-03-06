function [x, y, sensitive, training, test] = importCensusData()
    ftoreadData = '+dataImport/census-income-train.csv';
    ftoreadDataTest = '+dataImport/census-income-test.csv';
    fidData = fopen(ftoreadData);
    fidDataTest = fopen(ftoreadDataTest);
    incomeData = textscan(fidData, '%f%f%s%f%f%s%f%s%s%s%s%s%s%s%f%f%f%s%s%s%f%s%s%s%s%s%f%s%s%s%s%f%f%f%f%s', 'Delimiter',',');
    incomeDataTest = textscan(fidDataTest, '%f%f%s%f%f%s%f%s%s%s%s%s%s%s%f%f%f%s%s%s%f%s%s%s%s%s%f%s%s%s%s%f%f%f%f%s', 'Delimiter',',');
    a = incomeData(:,1);
    a = vertcat(a{:});
    testSplitPoint = size(a,1);
    incomeData = cat(1,incomeData,incomeDataTest);
    
    incomeData_f =[dataImport.convertToDouble(incomeData(:,2)), ...
        dataImport.convertToValues(incomeData(:,3)), ...
        dataImport.convertToDouble(incomeData(:,4)), ...
        dataImport.convertToDouble(incomeData(:,5)), ...
        dataImport.convertToValues(incomeData(:,6)), ...
        dataImport.convertToDouble(incomeData(:,7)), ...
        dataImport.convertToValues(incomeData(:,8)), ...
        dataImport.convertToValues(incomeData(:,9)), ...
        dataImport.convertToValues(incomeData(:,10)), ...
        dataImport.convertToValues(incomeData(:,11)), ...
        dataImport.convertToValues(incomeData(:,12)), ...
        dataImport.convertToValues(incomeData(:,13)), ...
        dataImport.convertToValues(incomeData(:,14)), ...
        dataImport.convertToDouble(incomeData(:,15)), ...
        dataImport.convertToDouble(incomeData(:,16)), ...
        dataImport.convertToDouble(incomeData(:,17)), ...
        dataImport.convertToValues(incomeData(:,18)), ...
        dataImport.convertToValues(incomeData(:,19)), ...
        dataImport.convertToValues(incomeData(:,20)), ...
        dataImport.convertToDouble(incomeData(:,21)), ...
        dataImport.convertToValues(incomeData(:,22)), ...
        dataImport.convertToValues(incomeData(:,23)), ...
        dataImport.convertToValues(incomeData(:,24)), ...
        dataImport.convertToValues(incomeData(:,25)), ...
        dataImport.convertToValues(incomeData(:,26)), ...
        dataImport.convertToDouble(incomeData(:,27)), ...
        dataImport.convertToValues(incomeData(:,28)), ...
        dataImport.convertToValues(incomeData(:,29)), ...
        dataImport.convertToValues(incomeData(:,30)), ...
        dataImport.convertToValues(incomeData(:,31)), ...
        dataImport.convertToDouble(incomeData(:,32)), ...
        dataImport.convertToDouble(incomeData(:,33)), ...
        dataImport.convertToDouble(incomeData(:,34)), ...
        dataImport.convertToDouble(incomeData(:,35))];
    
    a = incomeData(:,36);
    a = vertcat(a{:});
    
    y = ones(size(incomeData_f,1),1);
    for i=1:length(y)
        if(strcmp(cellstr(a(i)),'- 50000.')==1)
            y(i) = 0;
        end
    end
    
    sensitive = false(size(incomeData_f,1), 1);
    a = incomeData(:,13);
%     a = incomeData(:,11);
    a = vertcat(a{:});
    for j=1:size(incomeData_f,1)
        sensitive(j) = strcmp(cellstr(a(j)),'Female')==1;%females are sensitive
        %sensitive(j) = strcmp(cellstr(a(j)),'White')==0;%non-whites are sensitive
    end;
    
    x = incomeData_f;
    
    %% generate training and test data
    training = 1:testSplitPoint;
    test = (testSplitPoint+1):length(y);
end
