function x = convertToValues(data)
    %% convert to integer values
    data = vertcat(data{:});
    classes = unique(data);
    x = zeros(size(data,1), size(classes,1));
    for i=1:size(classes,1)
        for j=1:size(data,1)
            x(j, i) = strcmp(cellstr(data(j)),classes(i));
        end;
    end;
end
