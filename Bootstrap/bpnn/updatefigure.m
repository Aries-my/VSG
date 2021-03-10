function figHandle = updatefigure(net, i, figHandle, option, trainPerf, valPerf, testPerf, gradient, fail)
%UPDATEFIGURE Update errors, gradients, validation checks.

%   Date: August 31, 2016
%   Author: kalvin chern (E-mail:zhongsheng.chen@outlook.com)


if i == 1
    return;
end

interval = fix(log2(i));
if nargin == 7
    
    % create legend for curves.
    legendStr = {'Training'};
    if option.validation
        legendStr = [legendStr, {'Validation'}];
    end
    if option.testing
        legendStr = [legendStr,{'Testing'}];
    end
    
    % create plot data.
    ind = 1 : fix(log2(i)): i;
    xplot = ind';
    eplot = trainPerf(ind)';
    if option.validation
        xplot = [xplot, ind'];
        eplot = [eplot, valPerf(ind)'];
    end
    if option.testing
        xplot = [xplot, ind'];
        eplot = [eplot, testPerf(ind)'];
    end
    
    % Plot errors.
    figure(figHandle);
    object = plot(xplot, eplot);
    objectSize = size(object, 1);
    figHandle.Name = 'Neural Network Training Performance';
    object(1).Color = 'b'; object(1).LineStyle = '-'; object(1).Marker = 'none';      % Training error curve
    if objectSize == 2
        if option.validation
            object(objectSize).Color = 'g'; object(objectSize).LineStyle = '--'; object(objectSize).Marker = 'none'; % validation error curve
        end
        if option.testing
            object(objectSize).Color = 'r'; object(objectSize).LineStyle = '-.'; object(objectSize).Marker = 'none'; % Testing error curve
        end
    end
    if objectSize == 3
        if option.validation
            object(objectSize - 1).Color = 'g'; object(objectSize - 1).LineStyle = '--'; object(objectSize - 1).Marker = 'none'; % validation error curve
        end
        if option.testing
            object(objectSize).Color = 'r'; object(objectSize).LineStyle = '-.'; object(objectSize).Marker = 'none'; % Testing error curve
        end
    end
    ax = gca;
    ax.XLim = [0, i + 10];
    ax.XLabel.String = 'Epochs';
    ax.YLabel.String = (['Errors (', lower(net.performFcn), ')']);
    legend(ax, legendStr, 'Location', 'NE');
    drawnow
end

if nargin == 9
    
    ind = 1 : interval: i;
    xplot = ind';
    if option.validation
        subPlotSize = 3;
    else
        subPlotSize = 2;
    end
    figure(figHandle)
    figHandle.Name = 'Neural Network Training State';
    % Plot training errors.
    eplot = trainPerf(ind)';
    train = subplot(subPlotSize, 1, 1);
    plot(train, xplot, eplot, 'b-');
    train.Title.String = sprintf('Training errors = %3.6f, at %d epoch', trainPerf(i), i);
    train.XLabel.String = 'Epochs';
    train.YLabel.String = sprintf('Erros (%s)',lower(net.performFcn));
    
    % Plot gradients.
    eplot = gradient(ind)';
    grad = subplot(subPlotSize, 1, 2);
    plot(grad, xplot, eplot, 'k:');
    grad.Title.String = sprintf('Gradient = %3.6f, at %d epoch', gradient(i), i);
    grad.XLabel.String = 'Epochs';
    grad.YLabel.String = sprintf('Gradient');
    
    % Plot number of succesive iteration of validaton performance fails to decrease.
    if option.validation
        eplot = fail(ind)';
        failchk = subplot(subPlotSize, 1, 3);
        scatter(failchk, xplot, eplot, 'MarkerFaceColor', 'r');
        failchk.Title.String = sprintf('Validation Checks = %d, at %d epoch', fail(i), i);
        failchk.XLabel.String = 'Epochs';
        failchk.YLabel.String = sprintf('Fails');
    end
    
    drawnow
end
