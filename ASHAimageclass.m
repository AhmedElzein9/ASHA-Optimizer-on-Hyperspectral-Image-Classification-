function [Result]=ASHAimageclass(train_input,train_target,test_input,test_target,locTrain,locTest,img,gt,maxobj)
% L={discr,ensemble,kernel,knn,linear,nb,net,svm,tree};
class=max(max(gt));
[r,c,~]=size(img);
B=ones(r,c);
options = struct('Optimizer','bayesopt','UseParallel',true,'ShowPlots',false,'MaxObjectiveEvaluations',maxobj,'Kfold',2);
for i=1:1
    tic;
[Result.Mdl{i},Result.OptimizationRESULTS_pavia{i}] = fitcauto(train_input,train_target,'HyperparameterOptimizationOptions',options,'OptimizeHyperparameters','all','Learners','svm');
Result.tra_out{i}=predict(Result.Mdl{i},train_input);
[Result.tra_error{i}]=confusion.getMatrix(train_target,Result.tra_out{i});
Result.test_out{i}=predict(Result.Mdl{i},test_input);
[Result.test_error{i}]=confusion.getMatrix(test_target,Result.test_out{i});
classified_image = gt;
classified_image(locTrain)=Result.tra_out{i};
classified_image(locTest)=Result.test_out{i};
[Result.All_image_error{i}]=confusion.getMatrix(gt,classified_image);
classified_image=reshape(classified_image,r,c);
Result.classified_image{i}=classified_image;
[Result.Errors{i}]=Image_quality(Result.classified_image{i},gt);
gt=double(gt);gt=gt+B;gt=uint8(gt);
Result.classified_image{i}=double(Result.classified_image{i});Result.classified_image{i}=Result.classified_image{i}+B;Result.classified_image{i}=uint8(Result.classified_image{i});
[Result.kappa{i}] = kappaindex(gt,Result.classified_image{i},class);
gt=double(gt);gt=gt-B;gt=uint8(gt);
Result.classified_image{i}=double(Result.classified_image{i});Result.classified_image{i}=Result.classified_image{i}-B;Result.classified_image{i}=uint8(Result.classified_image{i});
Result.time{i}=toc;
figure(i)
 subplot(1,2,1)
 imshow(classified_image,[]), colormap jet, title('Classified Image');
 subplot(1,2,2)
 imshow(gt,[]), colormap jet, title('Ground Reality Image');
 saveFolder = 'G:\Other computers\My Laptop\PhD Turkey\Papers\CA\Result1';
 filename = fullfile(saveFolder, ['figure_1' num2str(i) '.fig']);
 filename1 = fullfile(saveFolder, ['figure_1' num2str(i) '.emf']);
 saveas(gcf, filename);
 saveas(gcf, filename1);
close all
% Specify the MAT-file name (e.g., 'my_results.mat')
matFileName = 'Result.mat';
% Combine the directory and file name to create the full file path
fullFilePath = fullfile(saveFolder, matFileName);
% Save the specific variables to the specified path
save(fullFilePath, 'Result');
end