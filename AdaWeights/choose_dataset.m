function [func] = choose_dataset(s)
    if strcmp(s,'adult')
        func = '@()dataImport.importAdultData()';
    elseif strcmp(s,'bank')
        func = '@()dataImport.importBankData()';
    elseif strcmp(s,'census')
        func = '@()dataImport.importCensusData()';
    elseif strcmp(s,'compass')
        func = '@()dataImport.importCompassData()';
    else
        fprintf('No such dataset!');
        return
    end
end
