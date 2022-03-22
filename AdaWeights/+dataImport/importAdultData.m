function [x, y, sensitive, training, test] = importAdultData()
    ftoreadData = '+dataImport/adult.data';
    ftoreadDataTest = '+dataImport/adult.test';
    fidData = fopen(ftoreadData);
    fidDataTest = fopen(ftoreadDataTest);
    incomeData = textscan(fidData, '%f%s%f%s%f%s%s%s%s%s%f%f%f%s%s', 'Delimiter',',');
    incomeDataTest = textscan(fidDataTest, '%f%s%f%s%f%s%s%s%s%s%f%f%f%s%s', 'Delimiter',',');
    a = incomeData(:,1);
    a = vertcat(a{:});
    testSplitPoint = size(a,1);
    incomeData = cat(1,incomeData,incomeDataTest);

    x =[dataImport.convertToDouble(incomeData(:,1)), ...
        dataImport.convertToValues(incomeData(:,2)), ...
        dataImport.convertToDouble(incomeData(:,3)), ...
        dataImport.convertToValues(incomeData(:,4)), ...
        dataImport.convertToDouble(incomeData(:,5)), ...
        dataImport.convertToValues(incomeData(:,6)), ...
        dataImport.convertToValues(incomeData(:,7)), ...
        dataImport.convertToValues(incomeData(:,8)), ...
        dataImport.convertToValues(incomeData(:,9)), ...
        dataImport.convertToValues(incomeData(:,10)), ...
        dataImport.convertToDouble(incomeData(:,11)), ...
        dataImport.convertToDouble(incomeData(:,12)), ...
        dataImport.convertToDouble(incomeData(:,13)), ...
        dataImport.convertToValues(incomeData(:,14))];
    
    a = incomeData(:,15);
    a = vertcat(a{:});
    
    y = ones(size(x,1),1);
    for i=1:length(y)
        if(strcmp(cellstr(a(i)),'<=50K')==1)
            y(i) = 0;
        elseif(strcmp(cellstr(a(i)),'<=50K.')==1)
            y(i) = 0;
        end
    end
    
    
    sensitive = false(size(x,1), 1);
    a = incomeData(:,10);
%     a = incomeData(:,9);
    a = vertcat(a{:});
    for j=1:size(x,1)
        sensitive(j) = strcmp(cellstr(a(j)),'Female')==1;%females are sensitive
        %sensitive(j) = strcmp(cellstr(a(j)),'White')==0;%non-whites are sensitive
    end;
    
    %% generate training and test data
    training = 1:testSplitPoint;
    test = (testSplitPoint+1):length(y);
end
