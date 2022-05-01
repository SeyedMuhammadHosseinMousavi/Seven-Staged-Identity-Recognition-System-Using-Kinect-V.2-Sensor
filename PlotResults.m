function PlotResults(Targets,Outputs,Name)

    Errors=Targets-Outputs;

    figure;

    subplot(2,2,[1 2]);
    plot(Targets,'r:');
    hold on;
    plot(Outputs,'b');
    legend('Targets','Outputs');
    title([Name ' - Targets and Outputs']);
    ylabel('Targets and Outputs');
    
    ErrorMean=mean(Errors);
    ErrorStD=std(Errors);
    MSE=mean(Errors.^2);
    RMSE=sqrt(MSE);
    
    subplot(2,2,3);
    plot(Errors);
    ylabel('Errors');
    title(['MSE = ' num2str(MSE) ', RMSE = ' num2str(RMSE)]);
    
    nBin=max(round(numel(Errors)/20),10);
    
    subplot(2,2,4);
    histfit(Errors,nBin);
    xlabel('Errors');
    title(['Mean = ' num2str(ErrorMean) ', StD = ' num2str(ErrorStD)]);
    
end