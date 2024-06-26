function [Errors]=Image_quality(img,baseimg)
% class=max(max(baseimg));
[r,c,~]=size(img);
Errors.mse_value = immse(img,baseimg);
Errors.psnr_value = psnr(img,baseimg);
Errors.ssimval = ssim(img,baseimg);
cp=classperf(baseimg,img);
Errors.GeneralAccuracy=cp.CorrectRate;
ass=baseimg-img;
mn=tabulate(ass(:));
Errors.False_Alarm=mn(1,2);
Errors.Missed_Alarm=mn(3,2);
Errors.Total_Error=Errors.False_Alarm+Errors.Missed_Alarm;
Errors.Total_Error_Rate=Errors.Total_Error./(r*c);