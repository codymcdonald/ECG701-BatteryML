% Clear workspace
clear; clc;

% Get the directory where this script is located
scriptDir = fileparts(mfilename('fullpath'));

% Find all .mat files in the script directory
matFiles = dir(fullfile(scriptDir, '*.mat'));

% Check if any .mat files were found
if isempty(matFiles)
    fprintf('❌ No .mat files found in directory: %s\n', scriptDir);
    return;
end

fprintf('Found %d .mat files to process:\n', length(matFiles));
for i = 1:length(matFiles)
    fprintf('%d. %s\n', i, matFiles(i).name);
end

% Process each .mat file
for fileIdx = 1:length(matFiles)
    matFile = matFiles(fileIdx).name;
    fprintf('\nProcessing file %d of %d: %s\n', fileIdx, length(matFiles), matFile);
    
    % Load .mat data
    load(fullfile(scriptDir, matFile));
    
    % Extract base name from matFile (remove .mat extension)
    [~, baseName, ~] = fileparts(matFile);
    
    % Output Excel file with same name as input .mat file
    excelFile = fullfile(scriptDir, sprintf('%s.xlsx', baseName));
    
    % Delete existing file if needed
    if isfile(excelFile)
        delete(excelFile);
    end
    
    % Discharge cycle counter
    discharge_counter = 0;
    
    % Get the variable name from the mat file
    varName = who('-file', fullfile(scriptDir, matFile));
    dataStruct = eval(varName{1});
    
    % Loop through all cycles
    for i = 1:length(dataStruct.cycle)
        cycle = dataStruct.cycle(i);
        
        if strcmp(cycle.type, 'discharge') && isfield(cycle, 'data')
            data = cycle.data;
            
            % Extract arrays (convert to column vectors)
            Time = data.Time(:);
            Voltage_measured = data.Voltage_measured(:);
            Current_measured = data.Current_measured(:);
            Temperature_measured = data.Temperature_measured(:);
            Current_load = data.Current_load(:);
            Voltage_load = data.Voltage_load(:);
            Capacity = repmat(data.Capacity, length(Time), 1);
            Ambient_temperature = repmat(cycle.ambient_temperature, length(Time), 1);
            
            % Truncate to the shortest vector length
            len = min([ ...
                length(Time), length(Voltage_measured), length(Current_measured), ...
                length(Temperature_measured), length(Current_load), ...
                length(Voltage_load), length(Capacity), length(Ambient_temperature)]);
            
            % Discharge sequence counter
            discharge_counter = discharge_counter + 1;
            
            % Build table
            T = table( ...
                repmat(discharge_counter, len, 1), ...
                Time(1:len), ...
                Voltage_measured(1:len), ...
                Current_measured(1:len), ...
                Temperature_measured(1:len), ...
                Current_load(1:len), ...
                Voltage_load(1:len), ...
                Capacity(1:len), ...
                Ambient_temperature(1:len), ...
                'VariableNames', {'Cycle', 'Time', 'Voltage_measured', ...
                                'Current_measured', 'Temperature_measured', ...
                                'Current_load', 'Voltage_load', ...
                                'Capacity', 'Ambient_temperature'} ...
            );
            
            % Append to Excel
            writetable(T, excelFile, 'WriteMode', 'append');
            
            fprintf('✅ Cycle %d (original index %d) written with %d rows.\n', ...
                discharge_counter, i, len);
        end
    end
    
    fprintf('📄 Done! Discharge cycles exported to: %s\n', excelFile);
end

fprintf('\nAll files processed successfully!\n');
