# matlab function for calling Python programs

function pred = matpy(data1, data2)
    net = py.importlib.import_module('model');
    py.importlib.reload(net);
    model = py.importlib.import_module('modelpred');
    py.importlib.reload(model);
    pred = model.prediction(pyargs('data1', data1, 'data2' ,data2));
end
