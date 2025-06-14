% 图像处理第一次作业 
clear; 
clc;
close all;

main_dir = 'images/'; % 源图片放置在 images 文件夹中
file_type = '.jpg';   % 图片文件类型

%% 读取文件夹中文件，且做尺寸调整(保持长宽比) 统一图片到同一大小
% （空白区域用黑色）。例如，所有图片均为 600 x 1000(max_Y) x 3
image_files = dir([main_dir, '*', file_type]);
len = length(image_files);
% 先找出缩放高=600后，对应的最大宽度 maxy
max_Y = 0;
for i = 1 : len
    tmp = imread(fullfile(main_dir, image_files(i).name));
    [h, w, ~] = size(tmp);
    scale   = 600 / h; 
    new_w   = round(w * scale);
    if new_w > max_Y
        max_Y = new_w;
    end
end

% (2) 正式读取、缩放、居中贴图到 (600 x max_Y)
image_coll = cell(1, len);
image_name    = cell(1, len);

for i = 1 : len
    image_name{i} = image_files(i).name;
    
    % 读取并缩放
    ori_img = imread(fullfile(main_dir, image_files(i).name));
    img_d   = im2double(ori_img);
    [h, w, ~] = size(img_d);
    scale    = 600 / h;
    resiz  = imresize(img_d, scale, 'bicubic');
    
    % 在黑色背景上居中贴图
    back_bg  = zeros(600, max_Y, 3);% 黑色背景，max_Y 要大于最大Y尺度，以确保数组不越界
    [rh, rw, ~] = size(resiz);
    start_col = floor((max_Y - rw)/2) + 1;  % 让图片水平居中
    back_bg(1:rh, start_col:start_col+rw-1, :) = resiz;
    
    image_coll{i} = back_bg;
end

%% 建立新文件夹（之后方便将保存的视频帧存到该文件夹中）
new_main_dir = 'video_images/';%建立video_images文件夹
if ~exist(new_main_dir, 'dir')
     % 使用mkdir函数建立文件夹
    %%%%%%%%%%%%%%%%%%%%%%%%
    mkdir(new_main_dir);
    %%%%%%%%%%%%%%%%%%%%%%%%
end

%% 将处理的图片集合 image_collect 进行“动画特效”处理， 至少满足作业要求, 动画顺序可任意调整
%%% 处理后的每帧图通过imwrite存到 new_main_dir 中 %%%
%%% 可以将处理中的图片通过imshow展示，若想通过figure展示大致视频效果，可以在每次imshow后加入pause(0.001)函数 %%%
figure(1);
set(gcf, 'unit','centimeters','position',[5 5 20 20]);
set(gca, 'Position',[.2, .2, .7, .65]);
noofflip = 60; % 每段动画输出 60 帧
saveIDX = 1; % 保存视频帧的序号
%% 从黑色背景到第一张图片 （淡入淡出，灰度变换类）（示例）
blakbg = zeros(600, max_Y, 3);
imb = image_coll{1};
for f = 0 : noofflip - 1
    alpha = f / (noofflip - 1);
    frame = (1 - alpha)*blakbg + alpha*imb;
    imshow(frame); pause(0.001);
    imwrite(frame, [new_main_dir, num2str(saveIDX,'%05d'), file_type]);
    saveIDX = saveIDX + 1;
end
for iPair = 1 : 19
    ima = image_coll{iPair};
    imb = image_coll{iPair+1};
    [H, W, ~] = size(ima);

    switch iPair
%% 普通淡入淡出
        case 1

            for f = 0 : noofflip-1
                alpha = f / (noofflip-1);
                outF  = (1 - alpha)*ima + alpha*imb;
                imshow(outF); pause(0.001);
                imwrite(outF, [new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX = saveIDX + 1;
            end
%% 棋盘格淡入淡出 （灰度变换类）
        case 2
            block_size = 40;
            chbox = checkerboard(block_size, ceil(H/block_size), ceil(W/block_size));
            chbox = chbox(1:H,1:W);
            mask1 = (chbox>0.5);
            mask2 = ~mask1;
            m1_3  = repmat(mask1,[1,1,3]);
            m2_3  = repmat(mask2,[1,1,3]);
            
            outF = ima;
            half_seg = noofflip/2; % 前30帧白格，后30帧黑格
            for f = 0 : half_seg-1
                alpha = f/(half_seg-1);
                outF(m1_3) = (1-alpha)*ima(m1_3) + alpha*imb(m1_3);
                imshow(outF); pause(0.001);
                imwrite(outF, [new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX = saveIDX + 1;
            end
            for f = 0 : half_seg-1
                alpha = f/(half_seg-1);
                outF(m2_3) = (1-alpha)*ima(m2_3) + alpha*imb(m2_3);
                imshow(outF); pause(0.001);
                imwrite(outF, [new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX = saveIDX + 1;
            end
%% 从左到右淡入
        case 3
            frame = ima;
            for f = 0 : noofflip - 1
                frac = f / (noofflip-1);
                col_lim = round(frac * W);
                col_lim = max(col_lim,1);
                frame(:,1:col_lim,:) = imb(:,1:col_lim,:);
                imshow(frame); pause(0.001);
                imwrite(frame, [new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX = saveIDX + 1;
            end
%% 从上到下淡入
        case 4
            frame = ima;
            for f = 0 : noofflip - 1
                frac = f / (noofflip-1);
                row_lim = round(frac * H);
                row_lim = max(row_lim,1);
                frame(1:row_lim,:,:) = imb(1:row_lim,:,:);
                imshow(frame); pause(0.001);
                imwrite(frame, [new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX = saveIDX + 1;
            end
%% 随机块淡入
        case 5
            frame = ima;
            block_s = 50;
            row_blk = ceil(H/block_s);
            col_blk = ceil(W/block_s);
            tot_blk = row_blk * col_blk;
            idx_map = reshape(1:tot_blk,[row_blk,col_blk]);
            rand_seq= randperm(tot_blk);

            blocks_per_frame = ceil(tot_blk/noofflip);
            cur_idx = 1;
            for ff = 1 : noofflip
                end_idx = cur_idx + blocks_per_frame - 1;
                if end_idx>tot_blk, end_idx=tot_blk; end
                subset = rand_seq(cur_idx:end_idx);
                for bID = subset
                    [rB,cB] = find(idx_map==bID);
                    rS = (rB-1)*block_s +1; rE = min(rB*block_s,H);
                    cS = (cB-1)*block_s +1; cE = min(cB*block_s,W);
                    frame(rS:rE, cS:cE,:) = imb(rS:rE, cS:cE,:);
                end
                imshow(frame); pause(0.001);
                imwrite(frame, [new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX = saveIDX + 1;

                cur_idx = end_idx+1;
                if cur_idx>tot_blk
                    for leftf = ff+1 : noofflip
                        imshow(frame); pause(0.001);
                        imwrite(frame, [new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                        saveIDX=saveIDX+1;
                    end
                    break;
                end
            end
%% 垂直百叶窗
        case 6
            frame = ima;
            num_strips = 10;
            strip_w = ceil(W/num_strips);
            seg_per_strip = noofflip/num_strips; 
            used_fr = 0;
            for sID = 1:num_strips
                cS = (sID-1)*strip_w +1;
                cE = min(sID*strip_w,W);
                for ff = 1:seg_per_strip
                    alpha = ff/seg_per_strip;
                    frame(:, cS:cE, :) = (1-alpha)*ima(:, cS:cE,:) + alpha*imb(:, cS:cE,:);
                    imshow(frame); pause(0.001);
                    imwrite(frame, [new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                    saveIDX = saveIDX + 1;
                    used_fr = used_fr+1;
                end
            end
            % 如果有剩余, 补齐(整除问题)
            while used_fr<noofflip
                imshow(frame); pause(0.001);
                imwrite(frame, [new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX = saveIDX+1;
                used_fr=used_fr+1;
            end
%% 圆形中心向外渐显
        case 7
            frame = ima;
            cenr = round(H/2);
            cenc = round(W/2);
            maxR = sqrt((H-cenr)^2 + (W-cenc)^2);
            [xx,yy] = meshgrid(1:W,1:H);
            distMap = sqrt((yy-cenr).^2 + (xx-cenc).^2);
            for f = 0:noofflip-1
                alpha = f/(noofflip-1);
                curr  = alpha*maxR;
                mask  = (distMap<=curr);
                mask3 = repmat(mask,[1,1,3]);
                frame(mask3)= imb(mask3);
                imshow(frame);pause(0.001);
                imwrite(frame, [new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX=saveIDX+1;
            end
%% 对角线擦除
        case 8
            frame = ima;
            diagmax= H+W;
            for f = 0:noofflip-1
                alpha = f/(noofflip-1);
                curD  = round(alpha*diagmax);
                mask  = bsxfun(@le, ((1:H)' + (1:W)), curD);
                mask3 = repmat(mask,[1,1,3]);
                frame(mask3)= imb(mask3);
                imshow(frame); pause(0.001);
                imwrite(frame, [new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX=saveIDX+1;
            end
%% 水平波浪
        case 9
            amp=20; T=40;
            for f = 0:noofflip-1
                alpha = f/(noofflip-1);
                temp  = (1-alpha)*ima + alpha*imb;
                wave_img = zeros(size(temp));
                for r=1:H
                    shiftX = round(amp*sin(2*pi*r/T + alpha*2*pi));
                    cStart = 1+shiftX; cEnd=W+shiftX;
                    cut_s=0; cut_e=0;
                    if cStart<1
                        cut_s=1-cStart; cStart=1;
                    end
                    if cEnd>W
                        cut_e=cEnd-W; cEnd=W;
                    end
                    if cStart<=cEnd
                        wave_img(r,cStart:cEnd,:) = temp(r,(1+cut_s):(W-cut_e),:);
                    end
                end
                imshow(wave_img); pause(0.001);
                imwrite(wave_img, [new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX=saveIDX+1;
            end
%% 像素化渐变
        case 10
            for f = 0:noofflip-1
                alpha = f/(noofflip-1);
                big_s = 40; 
                cur_s= 1 + round((big_s-1)*(1-alpha));
                % 缩小后再放大 => 像素块
                smallB= imresize(imb,1/cur_s,'nearest');
                pixB  = imresize(smallB,[H,W],'nearest');
                out   = (1-alpha)*ima + alpha*pixB;
                imshow(out); pause(0.001);
                imwrite(out, [new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX=saveIDX+1;
            end
%% 水平百叶窗(分成水平条带)
        case 11
            frame = ima;
            row_strips=10;
            band_h= ceil(H/row_strips);
            seg_per_strip= noofflip/row_strips;
            used_f=0;
            for bID=1:row_strips
                rS= (bID-1)*band_h+1;
                rE= min(bID*band_h,H);
                for ff=1:seg_per_strip
                    alpha= ff/seg_per_strip;
                    frame(rS:rE,:,:)= (1-alpha)*ima(rS:rE,:,:) + alpha*imb(rS:rE,:,:);
                    imshow(frame);pause(0.001);
                    imwrite(frame, [new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                    saveIDX=saveIDX+1;
                    used_f=used_f+1;
                end
            end
            while used_f<noofflip
                imshow(frame);pause(0.001);
                imwrite(frame, [new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX=saveIDX+1;
                used_f=used_f+1;
            end
 %% 十字擦除(从中心扩展)
        case 12
            frame = ima;
            midR = round(H/2); 
            midC = round(W/2);
            for f=0:noofflip-1
                alpha= f/(noofflip-1);
                cross_w= round(alpha* min(midR,midC));
                
                r_t= midR-cross_w; 
                r_b= midR+cross_w;
                c_l= midC-cross_w; 
                c_r= midC+cross_w;
                r_t   = max(1, r_t);
                r_b   = min(H, r_b);
                c_l  = max(1, c_l);
                c_r = min(W, c_r);
                if r_t <= r_b
                    frame(r_t:r_b,:,:)= imb(r_t:r_b,:,:);
                end
                if c_l <= c_r
                    frame(:,c_l:c_r,:)= imb(:,c_l:c_r,:);
                end
                imshow(frame);pause(0.001);
                imwrite(frame,[new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX=saveIDX+1;
            end
%% 双对角线擦除
        case 13
            frame= ima;
            diagmax= H+W;
            for f=0:noofflip-1
                alpha= f/(noofflip-1);
                curD= round(alpha*diagmax);
                mask1= bsxfun(@le, ((1:H)' + (1:W)), curD);
                mask2= bsxfun(@le, ((1:H)' + (W-(1:W))), curD);
                mask= mask1 | mask2;
                mask3= repmat(mask,[1,1,3]);
                frame(mask3)= imb(mask3);
                imshow(frame);pause(0.001);
                imwrite(frame,[new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX=saveIDX+1;
            end
%% 中心放大矩形
        case 14
            frame=ima; halfH= round(H/2); halfW= round(W/2);
            for f=0:noofflip-1
                alpha= f/(noofflip-1);
                dR= round(alpha*halfH);
                dC= round(alpha*halfW);
                rS= max(1, halfH-dR); rE= min(H, halfH+dR);
                cS= max(1, halfW-dC); cE= min(W, halfW+dC);
                frame(rS:rE,cS:cE,:)= imb(rS:rE,cS:cE,:);
                imshow(frame);pause(0.001);
                imwrite(frame,[new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX=saveIDX+1;
            end
%% 钻石(菱形)     
        case 16
            frame=ima;
            cR= round(H/2); cC= round(W/2);
            for f=0:noofflip-1
                alpha= f/(noofflip-1);
                maxd= alpha*max(cR,cC);
                [xx,yy]= meshgrid(1:W,1:H);
                dist= abs(yy-cR)+ abs(xx-cC);
                mask= (dist<= maxd);
                mask3= repmat(mask,[1,1,3]);
                frame(mask3)= imb(mask3);
                imshow(frame);pause(0.001);
                imwrite(frame,[new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX=saveIDX+1;
            end

%% 渐变叠加(从上往下替换)
        case 17
            frame=ima;
            for f=0:noofflip-1
                frac= f/(noofflip-1);
                cut= round(frac*H);
                if cut<1, cut=1; end
                frame(1:cut,:,:)= imb(1:cut,:,:);
                imshow(frame);pause(0.001);
                imwrite(frame,[new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX=saveIDX+1;
            end

%% 颜色通道交替        
        case 18
            frame=ima;
            halfseg= floor(noofflip/3);
            remain= noofflip-3*halfseg;
            % R通道
            for ff=1:halfseg
                alpha= ff/halfseg;
                frame(:,:,1)= (1-alpha)*ima(:,:,1)+ alpha*imb(:,:,1);
                imshow(frame);pause(0.001);
                imwrite(frame,[new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX=saveIDX+1;
            end
            % G通道
            for ff=1:halfseg
                alpha= ff/halfseg;
                frame(:,:,2)= (1-alpha)*ima(:,:,2)+ alpha*imb(:,:,2);
                imshow(frame);pause(0.001);
                imwrite(frame,[new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX=saveIDX+1;
            end
            % B通道
            for ff=1:halfseg
                alpha= ff/halfseg;
                frame(:,:,3)= (1-alpha)*ima(:,:,3)+ alpha*imb(:,:,3);
                imshow(frame);pause(0.001);
                imwrite(frame,[new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX=saveIDX+1;
            end
            % 若还有剩余remai帧,保持frame
            for rr=1:remain
                imshow(frame);pause(0.001);
                imwrite(frame,[new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX=saveIDX+1;
            end
%% 随机行逐渐替换
        case 19
            frame=ima;
            row_order= randperm(H);
            rows_pf= ceil(H/noofflip);
            idx_now=1;
            for ff=1:noofflip
                ed= idx_now+rows_pf-1;
                if ed>H, ed=H; end
                subset= row_order(idx_now:ed);
                frame(subset,:,:)= imb(subset,:,:);
                imshow(frame);pause(0.001);
                imwrite(frame,[new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                saveIDX=saveIDX+1;
                
                idx_now= ed+1;
                if idx_now>H
                    for rrr=ff+1:noofflip
                        imshow(frame);pause(0.001);
                        imwrite(frame,[new_main_dir, num2str(saveIDX,'%05d'), file_type]);
                        saveIDX=saveIDX+1;
                    end
                    break;
                end
            end
    end
end
%% 缩放
img = image_coll{1};
for f=0:noofflip-1
    frac= f/(noofflip-1);
    scale_factor= 1 + 0.5*frac; 
    tmp = imresize(img, scale_factor);
    [hh, ww, ~]= size(tmp);
    bg= zeros(600, max_Y, 3);
    rr= min(hh,600); cc= min(ww,max_Y);
    bg(1:rr,1:cc,:) = tmp(1:rr,1:cc,:);
    imshow(bg); pause(0.001);
    imwrite(bg,[new_main_dir, num2str(saveIDX,'%05d'), file_type]);
    saveIDX=saveIDX+1;
end

%% 平移
img = image_coll{2};
for f=0:noofflip-1
    frac= f/(noofflip-1);
    shiftX= 300*frac;
    transition_image= imtranslate(img,[shiftX,0],'FillValues',0);
    imshow(transition_image);pause(0.001);
    imwrite(transition_image,[new_main_dir, num2str(saveIDX,'%05d'), file_type]);
    saveIDX=saveIDX+1;
end

%% 旋转
img = image_coll{3};
for f=0:noofflip-1
    frac= f/(noofflip-1);
    ang= 360*frac;
    rot_img= imrotate(img,ang,'bicubic','crop');
    imshow(rot_img); pause(0.001);
    imwrite(rot_img,[new_main_dir, num2str(saveIDX,'%05d'), file_type]);
    saveIDX=saveIDX+1;
end

%% 单页翻转
img = image_coll{4};
[HH, WW, ~]= size(img);
for f=0:noofflip-1
    frac= f/(noofflip-1);
    angle= 180*frac;
    sf= abs(cosd(angle));
    newW= round(WW*sf); if newW<1,newW=1; end
    flip_part= imresize(img,[HH,newW]);
    bg= zeros(600, max_Y, 3);
    cc= min(newW, max_Y);
    bg(1:HH,1:cc,:)= flip_part(1:HH,1:cc,:);
    imshow(bg);pause(0.001);
    imwrite(bg,[new_main_dir, num2str(saveIDX,'%05d'), file_type]);
    saveIDX= saveIDX+1;
end

%% 把 new_main_dir 中存好的图片制作成视频
animation = VideoWriter('photo_album','MPEG-4');
% 使用VideoWriter建立视频对象 animation，并设置相关参数（例如帧率等）
animation.FrameRate = 24;  % 每秒24真
open(animation);
all_frames = dir([new_main_dir,'*',file_type]);
n_frames   = length(all_frames);
for i=1:n_frames
    fpath = fullfile(new_main_dir, all_frames(i).name);
    frame_data= imread(fpath);
    writeVideo(animation, frame_data);
end
close(animation);

