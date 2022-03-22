function [x, y, sensitive, training, test] = importCompassData()
    ftoreadData = '+dataImport/compas-scores-two-years.csv';
    fidData = fopen(ftoreadData);
    fgetl(fidData);
    data = textscan(fidData, '%f%q%q%q%q%q%q%f%q%q%f%f%f%f%f%f%q%q%q%q%q%f%q%q%f%q%q%f%q%q%q%q%f%f%q%q%q%q%q%f%q%q%q%f%q%q%q%q%f%f%f%f%f', 'Delimiter',',');
    
    a = data(10);
    a = vertcat(a{:});
    b = data(23);
    b = vertcat(b{:});

    mask = strcmp(cellstr(a),'Caucasian')|strcmp(cellstr(a),'African-American')&strcmp(cellstr(b),'O')==0;
    
    tmp = data(15);
    tmp = vertcat(tmp{:});
    
    x =[dataImport.convertToValues(data(:,10)), ...
        dataImport.convertToValues(data(:,6)), ...
        dataImport.convertToValues(data(:,9)), ...
        tmp/mean(tmp), ...
        dataImport.convertToValues(data(:,23))];
    x = x(mask);
    
    y = data(53);
    y = vertcat(y{:});
    y = y(mask);
    
    a = data(10);
    a = vertcat(a{:});
    a = a(mask);
    sensitive = strcmp(a,'Caucasian')==0;
    
    training = 1:floor(size(x,1)*0.5);
    test = (length(training)+1):length(y);
end