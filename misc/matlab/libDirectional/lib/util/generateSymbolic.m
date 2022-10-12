function generateSymbolic(D)
    % Requires the symbolic toolbox, if you need to generate further files.
    % Files are already generated for d = 2:6.
    
    name = 'cBinghamNorm';
    filename = mFilePath(D, name);
    if exist(filename, 'file') == 0
        [c, X] = cBinghamNormSymbolic(D);
        mFileExport(c, X, name);
    end
        
    name = 'cBinghamGradLogNorm';
    filename = mFilePath(D, name);
    if exist(filename, 'file') == 0
        grad_log_c = cBinghamGradLogNormSymbolic(c, X);
        mFileExport(grad_log_c, X, name);
    end
        
    name = 'cBinghamGradNormDividedByNorm';
    filename = mFilePath(D, name);
    if exist(filename, 'file') == 0
        grad_c_divided_by_c = cBinghamGradNormDividedByNormSymbolic(c, X);
        mFileExport(grad_c_divided_by_c, X, name);
    end
end

function filename = mFilePath(D, name)
    thisFilePath = mfilename('fullpath');
    filename = sprintf('%s%d.m', name, D);
    filename = fullfile(fileparts(thisFilePath), ['autogenerated/' filename]);
end

function mFileExport(expression, variables, name)
    D = numel(variables);
    filename = mFilePath(D, name);
    matlabFunction(expression, 'file', filename, 'vars', {variables});
end