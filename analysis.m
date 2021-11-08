%% Main Analysis

FLAG_Performance = 0;
FLAG_Psychometric = 0;
FLAG_Traj_Dist = 1;

if FLAG_Psychometric
    figure
    X = [50, 100, 150, 250, 300, 350];
end

%% PERFORMANCE AND PSYCHOMETRIC DATA

% models_folder = 'models/other/MULTI_90*';
% models_folder = 'models/MULTI_90*STD*';
models_folder = 'models/MULTI_90*retrain*';
% models_folder = 'models/MULTI_90*';
Folders = dir(models_folder);
clear rule
for model_dir=1:length(Folders)
    folder = struct2cell(Folders(model_dir));
    %     folder_name = 'models/other/' + string(folder(1)) + '/';
    folder_name = 'models/' + string(folder(1)) + '/';
    Files = dir(folder_name + '*.mat');
    
    for targetfile=1:length(Files)
        file = struct2cell(Files(targetfile));
        file = folder_name + string(file(1));
        load(file)
        disp(file)
        
        %         if contains(file,'500ms_delay_test_250ms')
        %             rule(model_dir, targetfile)=250;
        %         elseif contains(file,'500ms_delay_test_500ms')
        %             rule(model_dir, targetfile)=500;
        %         elseif contains(file,'500ms_delay_test_750ms')
        %             rule(model_dir, targetfile)=750;
        %         elseif contains(file,'500ms_delay_test_1000ms')
        %             rule(model_dir, targetfile)=1000;
        %         elseif contains(file,'500ms_delay_test_1250ms')
        %             rule(model_dir, targetfile)=1250;
        %         elseif contains(file,'retrain_1000ms_0_test_250')
        %             rule(model_dir, targetfile)=250.1;
        %         elseif contains(file,'retrain_1000ms_0_test_500')
        %             rule(model_dir, targetfile)=500.1;
        %         elseif contains(file,'retrain_1000ms_0_test_750')
        %             rule(model_dir, targetfile)=750.1;
        %         elseif contains(file,'retrain_1000ms_0_test_1000')
        %             rule(model_dir, targetfile)=1000.1;
        %         elseif contains(file,'retrain_1000ms_0_test_1250')
        %             rule(model_dir, targetfile)=1250.1;
        %         else
        %             disp('error: file type not found')
        %         end
        
        if contains(file,'test_250ms')
            rule(model_dir, targetfile)=250;
        elseif contains(file,'test_500ms')
            rule(model_dir, targetfile)=500;
        elseif contains(file,'test_750ms')
            rule(model_dir, targetfile)=750;
        elseif contains(file,'test_1000ms')
            rule(model_dir, targetfile)=1000;
        elseif contains(file,'test_1250ms')
            rule(model_dir, targetfile)=1250;
        end
        
        [numTrials, tmax, numIn] = size(x);
        [~, ~, numRec] = size(h);
        [~, ~, numOut] = size(y);
        %         taxis = (dt:dt:tmax*dt)/1000;
        taxis = (1:1:tmax);
        
        if FLAG_Performance
            std_first_ind = find(stim_order==0);
            std_first_ind = find(stim_order==1);
            performance(model_dir, targetfile, 1) = Int_Discrim_Perf(y, y_hat, respond); %total
            performance(model_dir, targetfile, 2) = Int_Discrim_Perf(y(std_first_ind,:), y_hat(std_first_ind,:), respond(std_first_ind)); %std first
            performance(model_dir, targetfile, 3) = Int_Discrim_Perf(y(std_first_ind,:), y_hat(std_first_ind,:), respond(std_first_ind)); %std second
        end
        
        if FLAG_Psychometric && rule(model_dir, targetfile)==1000
            clear input output
            Y = Get_Y_PsychoMetric(X, x, y_hat, stim_order);
            [beta, THR, input, output, R, J] = PsychoMetricFitSigmoid(X,Y);
            scatter(X,Y)
            plot(input,output)
%             legend('Seed 905','Seed 906', 'Seed 907', 'Seed 908', 'Seed 909')
            hold on
        end
    end
end


%% PERFORMANCE BOXPLOTS
if FLAG_Performance
    perf_ind = 3;
    [r,c] = find(rule == 250);
    Gr250 = [r,c];
    [r,c] = find(rule == 500);
    Gr500 = [r,c];
    [r,c] = find(rule == 750);
    Gr750 = [r,c];
    [r,c] = find(rule == 1000);
    Gr1000 = [r,c];
    [r,c] = find(rule == 1250);
    Gr1250 = [r,c];
    
    [r,c] = find(rule == 250.1);
    Gr250retrain = [r,c];
    [r,c] = find(rule == 500.1);
    Gr500retrain = [r,c];
    [r,c] = find(rule == 750.1);
    Gr750retrain = [r,c];
    [r,c] = find(rule == 1000.1);
    Gr1000retrain = [r,c];
    [r,c] = find(rule == 1250.1);
    Gr1250retrain = [r,c];
    
    figure
    set(gcf,'position',[150 150 600 300],'color','w')
    colors = ['b','r', 'g', 'y', 'm'];
    
    %     hbp = boxplot([performance(Gr250(:,1),Gr250(1,2),perf_ind);performance(Gr250retrain(:,1),Gr250retrain(1,2),perf_ind); ...
    %         performance(Gr500(:,1),Gr500(1,2),perf_ind);performance(Gr500retrain(:,1),Gr500retrain(1,2),perf_ind);...
    %         performance(Gr750(:,1),Gr750(1,2),perf_ind);performance(Gr750retrain(:,1),Gr750retrain(1,2),perf_ind); ...
    %         performance(Gr1000(:,1),Gr1000(1,2),perf_ind);performance(Gr1000retrain(:,1),Gr1000retrain(1,2),perf_ind); ...
    %         performance(Gr1250(:,1),Gr1250(1,2),perf_ind);performance(Gr1250retrain(:,1),Gr1250retrain(1,2),perf_ind)], ...
    %         [ones(length(Gr250),1);2*ones(length(Gr500),1);3*ones(length(Gr750),1);4*ones(length(Gr1000),1);5*ones(length(Gr1250),1); ...
    %         6*ones(length(Gr250),1);7*ones(length(Gr500),1);8*ones(length(Gr750),1); 9*ones(length(Gr1000),1);10*ones(length(Gr1250),1)], ...
    %         'PlotStyle','compact','position',[0.95 0.975 1.025 1.05 1.10 1.125 1.175 1.20 1.25 1.275],'color',colors);
    
    hbp = boxplot([performance(Gr250(:,1),Gr250(1,2),perf_ind); ...
        performance(Gr500(:,1),Gr500(1,2),perf_ind);...
        performance(Gr750(:,1),Gr750(1,2),perf_ind); ...
        performance(Gr1000(:,1),Gr1000(1,2),perf_ind); ...
        performance(Gr1250(:,1),Gr1250(1,2),perf_ind);], ...
        [ones(length(Gr250),1);2*ones(length(Gr500),1);3*ones(length(Gr750),1);4*ones(length(Gr1000),1);5*ones(length(Gr1250),1)], ...
        'PlotStyle','compact','Labels',{'250' '500' '750' '1000' '1250'},'color',colors);
    
    hand = findobj(hbp,'Tag','Box');
    set(hand,'linewidth',14)
    hand = findobj(hbp,'Tag','Whisker');
    set(hand,'linewidth',2)
    hand = findobj(hbp,'Tag','MedianOuter');
    set(hand,'MarkerSize',8)
    box off
    ylim([0 1])
    set(gca,'linewidth',1,'fontweight','bold','fontsize',8)
    ylabel('Performance','fontsize',12,'fontweight','bold')
    xlabel('Delay (ms)','fontsize',12,'fontweight','bold')
end
%% SEQ and DIM
% delay_start = 87;
% delay_end = 188;
% ActThr = 0.001;
% numEntropyBins = 60;
% datadelay = squeeze(mean(h(long(:,1),delay_start:delay_end, :), 1))';
% [val] = max(datadelay,[],2);
% actdatadelay = datadelay(val>=ActThr, :);
% [~,~,~,~,explained,~] = pca(actdatadelay','centered','off');
% dim = find(cumsum(explained)>95,1)
% disp(size(actdatadelay, 1))
% sqi = SeqIndexDB(actdatadelay,numEntropyBins)

%% EUCLIDEAN DISTANCE ANALYSIS

% Just look at one condition: Standard first, Long (350 ms)
% Separate short and long trials
% Neural trajectories in each batch of trials
% Calculate Euclidean distance at each pair of time steps

load('/Users/rehanchinoy/Library/CloudStorage/Box-Box/TensorFlowProjects/Timing_WorkingMemory_19/Rehan_19/IntervalDiscrimination/models/MULTI_100_Interval_Discrim_bs_50_retrain/MULTI_100_Interval_Discrim_bs_50_retrain_random_int1_onset.mat')
std_first_ind = find(std_order==0);
Act_Thr = 0.001;
for i=1:length(std_first_ind)
    std_first_ind(2,i) = (int2_offs(i)-int2_ons(i)) - (int1_offs(i)-int1_ons(i));
end
std_first_ind = std_first_ind';
std_first_ind = sortrows(std_first_ind,2);

S_Sh = true; %CMP = 120

if S_Sh == true
    val = find(std_first_ind(:,2) == -8);
else
    val = find(std_first_ind(:,2) == -1);
end
std_first_ind = std_first_ind(val,1);

[~,short_ind] = find(delay(std_first_ind) == 500);
[~, long_ind] = find(delay(std_first_ind) == 1000);

short_traj = squeeze(mean(h(short_ind,:,:), 1))';
long_traj = squeeze(mean(h(long_ind,:,:), 1))';

for j = 1:size(y_hat,2)
    for k=1:size(y_hat,2)
        distance(j,k) =  norm(short_traj(:,j) - long_traj(:,k));
    end
end

imagesc(distance)
colorbar

first_secondary = diag(distance(120:132, int1_ons(std_first_ind(1)):int1_offs(std_first_ind(1)))); %TODO:store onsets/offsets as variables
second_secondary = diag(distance(int1_ons(std_first_ind(1)):int1_offs(std_first_ind(1)), 170:182));

figure
plot(first_secondary)
hold on
plot(second_secondary)
title('Trajectory Distances')
legend('Second Interval of 500 ms delay trials - First STD Interval ', 'Second Interval of 1000 ms delay trials - First STD Interval')

%% NEURAL TRAJECTORY PLOTS
% Remove inactive units
% Just look at one condition: Standard first, Long (350 ms)
% Separate S-SH,..., S-L trials
% Neural trajectories in each batch of trials, sorted carefully

load('/Users/rehanchinoy/Library/CloudStorage/Box-Box/TensorFlowProjects/Timing_WorkingMemory_19/Rehan_19/IntervalDiscrimination/models/MULTI_100_Interval_Discrim_bs_50/MULTI_100_Interval_Discrim_bs_50double_noise.mat')
Act_Thr = 0.1;
a = max(h, [], 1);
a = squeeze(a);
a = max(a, [], 1);
[~, active_units] = find(a > Act_Thr);

std_first_ind = find(std_order==0);
for i=1:length(std_first_ind)
    inputs = find(x(std_first_ind(1,i),:,1)>0.4);
    std_first_ind(2,i) = (inputs(7)-inputs(5)) - (inputs(3)-inputs(1));
end
std_first_ind = std_first_ind';
std_first_ind = sortrows(std_first_ind,2);

val = find(std_first_ind(:,2) == -8); 
STD_120_trials = std_first_ind(val,1);
STD_120_inputs = find(x(STD_120_trials(1),:,1)>0.4);
% 
% val = find(trial_ind(:,2) == -10); 
% STD_100_trials = trial_ind(val,1);
% STD_100_inputs = find(x(STD_100_trials(1),:,1)>0.4);
% 
% val = find(trial_ind(:,2) == -5);
% STD_150_trials = trial_ind(val,1);
% STD_150_inputs = find(x(STD_150_trials(1),:,1)>0.4);
% 
val = find(std_first_ind(:,2) == 8); 
STD_280_trials = std_first_ind(val,1);
STD_280_inputs = find(x(STD_280_trials(1),:,1)>0.4);
% 
% val = find(trial_ind(:,2) == 10); 
% STD_300_trials = trial_ind(val,1);
% STD_300_inputs = find(x(STD_300_trials(1),:,1)>0.4);
% 
% val = find(trial_ind(:,2) == 15); 
% STD_350_trials = trial_ind(val,1);
% STD_350_inputs = find(x(STD_350_trials(1),:,1)>0.4);


std_first_ind = find(std_order==1);
for i=1:length(std_first_ind)
    inputs = find(x(std_first_ind(1,i),:,1)>0.4);
    std_first_ind(2,i) = (inputs(7)-inputs(5)) - (inputs(3)-inputs(1));
end
std_first_ind = std_first_ind';
std_first_ind = sortrows(std_first_ind,2);

val = find(std_first_ind(:,2) == 8);
CMP120_STD_trials = std_first_ind(val,1);
CMP120_STD_inputs = find(x(CMP120_STD_trials(1),:,1)>0.4);

val = find(std_first_ind(:,2) == 4);
CMP160_STD_trials = std_first_ind(val,1);
CMP160_STD_inputs = find(x(CMP160_STD_trials(1),:,1)>0.4);

val = find(std_first_ind(:,2) == 2);
CMP180_STD_trials = std_first_ind(val,1);
CMP180_STD_inputs = find(x(CMP180_STD_trials(1),:,1)>0.4);

val = find(std_first_ind(:,2) == 1); 
CMP190_STD_trials = std_first_ind(val,1);
CMP190_STD_inputs = find(x(CMP190_STD_trials(1),:,1)>0.4);

val = find(std_first_ind(:,2) == -8); 
CMP280_STD_trials = std_first_ind(val,1);
CMP280_STD_inputs = find(x(CMP280_STD_trials(1),:,1)>0.4);

val = find(std_first_ind(:,2) == -4); 
CMP240_STD_trials = std_first_ind(val,1);
CMP240_STD_inputs = find(x(CMP240_STD_trials(1),:,1)>0.4);

val = find(std_first_ind(:,2) == -2); 
CMP220_STD_trials = std_first_ind(val,1);
CMP220_STD_inputs = find(x(CMP220_STD_trials(1),:,1)>0.4);

val = find(std_first_ind(:,2) == -1); 
CMP210_STD_trials = std_first_ind(val,1);
CMP210_STD_inputs = find(x(CMP210_STD_trials(1),:,1)>0.4);

Event = squeeze(mean(h(CMP120_STD_trials,CMP120_STD_inputs(4):CMP120_STD_inputs(5),active_units), 1))'; %SORT
[Traj, sortind, peaktimes, cellorder, P] = SortEvent(Event);

figure
% imagesc(squeeze(mean(h(CMP120_STD_trials,CMP120_STD_inputs(4):352,active_units(sortind)), 1))'./max(squeeze(mean(h(CMP120_STD_trials,CMP120_STD_inputs(4):352,active_units(sortind)), 1)))')
imagesc(squeeze(mean(h(CMP120_STD_trials,:,active_units(sortind)), 1))')
title('120-STD, Sorted by 120-STD')
line(50*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(62*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(112*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(132*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
colorbar

figure
% imagesc(squeeze(mean(h(CMP120_STD_trials,CMP120_STD_inputs(4):352,active_units(sortind)), 1))'./max(squeeze(mean(h(CMP120_STD_trials,CMP120_STD_inputs(4):352,active_units(sortind)), 1)))')
imagesc(squeeze(mean(h(CMP160_STD_trials,:,active_units(sortind)), 1))')
title('160-STD, Sorted by 120-STD')
line(50*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(66*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(116*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(136*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
colorbar

% figure
% imagesc(squeeze(mean(h(CMP100_STD_trials,CMP100_STD_inputs(4):CMP100_STD_inputs(5)-1,sortind), 1))')
% title('100-STD, Sorted by STD-50')
% colorbar
% 
% figure
% imagesc(squeeze(mean(h(CMP150_STD_trials,CMP150_STD_inputs(4):CMP150_STD_inputs(5)-1,sortind), 1))')
% title('150-STD, Sorted by STD-50')
% colorbar

figure
% imagesc(squeeze(mean(h(STD_120_trials,STD_120_inputs(4):352,active_units(sortind)), 1))'./max(squeeze(mean(h(STD_120_trials,STD_120_inputs(4):352,active_units(sortind)), 1)))')
imagesc(squeeze(mean(h(STD_280_trials,:,active_units(sortind)), 1))')
title('STD-280, Sorted by 120-STD')
line(50*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(70*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(120*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(148*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
colorbar


figure
% imagesc(squeeze(mean(h(STD_120_trials,STD_120_inputs(4):352,active_units(sortind)), 1))'./max(squeeze(mean(h(STD_120_trials,STD_120_inputs(4):352,active_units(sortind)), 1)))')
imagesc(squeeze(mean(h(STD_120_trials,:,active_units(sortind)), 1))')
title('STD-120, Sorted by 120-STD')
line(50*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(70*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(120*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(132*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
colorbar

% figure
% imagesc(squeeze(mean(h(CMP250_STD_trials,CMP250_STD_inputs(4):CMP250_STD_inputs(5)-1,sortind), 1))')
% title('250-STD, Sorted by STD-50')
% colorbar
% 
% figure
% imagesc(squeeze(mean(h(CMP300_STD_trials,CMP300_STD_inputs(4):CMP300_STD_inputs(5)-1,sortind), 1))')
% title('300-STD, Sorted by STD-50')
% colorbar
figure
% imagesc(squeeze(mean(h(CMP280_STD_trials,CMP280_STD_inputs(4):352,active_units(sortind)), 1))'./max(squeeze(mean(h(CMP280_STD_trials,CMP280_STD_inputs(4):352,active_units(sortind)), 1)))')
imagesc(squeeze(mean(h(CMP240_STD_trials,:,active_units(sortind)), 1))')
line(50*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(74*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(124*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(144*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
title('240-STD, Sorted by 120-STD')
colorbar



figure
% imagesc(squeeze(mean(h(CMP280_STD_trials,CMP280_STD_inputs(4):352,active_units(sortind)), 1))'./max(squeeze(mean(h(CMP280_STD_trials,CMP280_STD_inputs(4):352,active_units(sortind)), 1)))')
imagesc(squeeze(mean(h(CMP280_STD_trials,:,active_units(sortind)), 1))')
line(50*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(78*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(128*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
line(148*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
title('280-STD, Sorted by 120-STD')
colorbar

% figure
% imagesc(squeeze(mean(h(STD_100_trials,STD_100_inputs(4):STD_100_inputs(5)-1,sortind), 1))')
% title('STD-100, Sorted by STD-50')
% 
% figure
% imagesc(squeeze(mean(h(STD_150_trials,STD_150_inputs(4):STD_150_inputs(5)-1,sortind), 1))')
% title('STD-150, Sorted by STD-50')
% 
% figure
% imagesc(squeeze(mean(h(STD_250_trials,STD_250_inputs(4):STD_250_inputs(5)-1,sortind), 1))')
% title('STD-250, Sorted by STD-50')
% 
% figure
% imagesc(squeeze(mean(h(STD_300_trials,STD_300_inputs(4):STD_300_inputs(5)-1,sortind), 1))')
% title('STD-300, Sorted by STD-50')
% 
% figure
% imagesc(squeeze(mean(h(STD_350_trials,STD_350_inputs(4):STD_350_inputs(5)-1,sortind), 1))')
% title('STD-350, Sorted by STD-50')

% imagesc(squeeze(mean(h(trial_ind,inputs(4):inputs(5)-1,active_units(sortind)), 1))'./max(squeeze(mean(h(trial_ind,inputs(4):inputs(5),active_units(sortind)), 1)))')


%% DELAY ANALYSIS
load('/Users/rehanchinoy/Library/CloudStorage/Box-Box/TensorFlowProjects/Timing_WorkingMemory_19/Rehan_19/IntervalDiscrimination/models/MULTI_100_Interval_Discrim_bs_50_retrain/MULTI_100_Interval_Discrim_bs_50_retrain_fixed_int1_onset.mat')
std_second_ind = find(std_order==1);
Act_Thr = 0.001;
for i=1:length(std_second_ind)
    std_second_ind(2,i) = (int2_offs(i)-int2_ons(i)) - (int1_offs(i)-int1_ons(i));
end
std_second_ind = std_second_ind';
std_second_ind = sortrows(std_second_ind,2);

[val1, ~] = find(std_second_ind(:,2) == 8);
[val2, ~] = find(std_second_ind(:,2) == -8);

CMP120_std_second_ind = std_second_ind(val1,1);
CMP280_std_second_ind = std_second_ind(val2,1);

[~,short_ind] = find(delay(CMP120_std_second_ind) == 500);
[~, long_ind] = find(delay(CMP120_std_second_ind) == 1000);

short_traj = squeeze(mean(h(short_ind,:,:), 1))';
long_traj = squeeze(mean(h(long_ind,:,:), 1))';

Event = squeeze(short_traj);
[Traj, sortind, peaktimes, cellorder, P] = SortEvent(Event); %SORT

figure
% imagesc(squeeze(mean(h(CMP120_std_second_ind(short_ind),:,sortind), 1))'./max(squeeze(mean(h(CMP120_std_second_ind(short_ind),:,sortind), 1)))')
imagesc(squeeze(mean(h(CMP120_std_second_ind(short_ind),:,sortind), 1))')
title('120-STD Short, Sorted by 120-STD Short')
colorbar

figure
% imagesc(squeeze(mean(h(CMP120_std_second_ind(long_ind),:,sortind), 1))'./max(squeeze(mean(h(CMP120_std_second_ind(long_ind),:,sortind), 1)))')
imagesc(squeeze(mean(h(CMP120_std_second_ind(long_ind),:,sortind), 1))')
title('120-STD Long, Sorted by 120-STD Short')
colorbar

[~,short_ind] = find(delay(CMP280_std_second_ind) == 500);
[~, long_ind] = find(delay(CMP280_std_second_ind) == 1000);

figure
% imagesc(squeeze(mean(h(CMP280_std_second_ind(short_ind),:,sortind), 1))'./max(squeeze(mean(h(CMP280_std_second_ind(short_ind),:,sortind), 1)))')
imagesc(squeeze(mean(h(CMP280_std_second_ind(short_ind),:,sortind), 1))')
title('280-STD Short, Sorted by 120-STD Short')
colorbar

figure
% imagesc(squeeze(mean(h(CMP280_std_second_ind(long_ind),:,sortind), 1))'./max(squeeze(mean(h(CMP280_std_second_ind(long_ind),:,sortind), 1)))')
imagesc(squeeze(mean(h(CMP280_std_second_ind(long_ind),:,sortind), 1))')
title('280-STD Long, Sorted by 120-STD Short')
colorbar



%% TUNING ACROSS INTERVALS

% Select active unit for one trial type
% Note if Ex or Inh
% Image recurrent activity across all trial types
% Image recurrent activity across different delays (need struct with
% multiple files)

load('/Users/rehanchinoy/Library/CloudStorage/Box-Box/TensorFlowProjects/Timing_WorkingMemory_19/Rehan_19/IntervalDiscrimination/models/MULTI_100_Interval_Discrim_bs_50/MULTI_100_Interval_Discrim_bs_50.mat')
% Act_Thr = 0.001;
std_first_ind = find(std_order==1);
for i=1:length(std_first_ind)
    inputs = find(x(std_first_ind(1,i),:,1)>0.4);
    std_first_ind(2,i) = (inputs(7)-inputs(5)) - (inputs(3)-inputs(1));
end
std_first_ind = std_first_ind';
std_first_ind = sortrows(std_first_ind,2);

% val = find(std_first_ind(:,2) == 4);
% CMP120STD_trials = std_first_ind(val,1);
% CMP120STD_inputs = find(x(CMP120STD_trials(1),:,1)>0.4);
val = find(std_first_ind(:,2) == 8);
CMP120STD_trials = std_first_ind(val,1);
CMP120STD_inputs = find(x(CMP120STD_trials(1),:,1)>0.4);

% Event = squeeze(mean(h(trials,70:120,1))');
Event = squeeze(mean(h(CMP120STD_trials,62:112,:), 1))';
max_activity = max(Event,[],2);
% active_units = find(max_activity>Act_Thr);
[~,indices] = sort(max_activity);
for i=1:100
    index = indices(end-i);

    if EI_matrix(index,index) == 1
        type = 'Excitatory';
    else
        type = 'Inhibitory';
    end

    figure
%     imagesc(squeeze(h(std_first_ind(:,1),CMP120STD_inputs(4):CMP120STD_inputs(5),index))./max(squeeze(h(std_first_ind(:,1),CMP120STD_inputs(4):CMP120STD_inputs(5),index))))
    imagesc(squeeze(h(std_first_ind(:,1),:,index)))
    hold on
    line(int1_ons(std_first_ind(:,1)), 1:size(std_first_ind, 1), 'color','r','linestyle',':','linewidth',2)
    line(int1_offs(std_first_ind(:,1)), 1:size(std_first_ind, 1), 'color','r','linestyle',':','linewidth',2)
    line(int2_ons(std_first_ind(:,1)), 1:size(std_first_ind, 1), 'color','r','linestyle',':','linewidth',2)
    line(int2_offs(std_first_ind(:,1)), 1:size(std_first_ind, 1), 'color','r','linestyle',':','linewidth',2)
    title(['Unit ', num2str(index), ': ' type])
    colorbar
    
    waitforbuttonpress
end

%% TUNING ACROSS DELAYS
% Select active unit for one delay type
% Image activity during short and long trials

load('/Users/rehanchinoy/Library/CloudStorage/Box-Box/TensorFlowProjects/Timing_WorkingMemory_19/Rehan_19/IntervalDiscrimination/models/MULTI_100_Interval_Discrim_bs_50_retrain/MULTI_100_Interval_Discrim_bs_50_retrain_fixed_int1_onset.mat')
std_second_ind = find(std_order==1);
Act_Thr = 0.001;
for i=1:length(std_second_ind)
    std_second_ind(2,i) = (int2_offs(i)-int2_ons(i)) - (int1_offs(i)-int1_ons(i));
end
std_second_ind = std_second_ind';
std_second_ind = sortrows(std_second_ind,2);

[val1, ~] = find(std_second_ind(:,2) == 8);

CMP120_std_second_ind = std_second_ind(val1,1);

[~,short_ind] = find(delay(CMP120_std_second_ind) == 500);
[~, long_ind] = find(delay(CMP120_std_second_ind) == 1000);

short_traj = squeeze(mean(h(CMP120_std_second_ind(short_ind),:,:), 1))';
long_traj = squeeze(mean(h(CMP120_std_second_ind(long_ind),:,:), 1))';

Event = squeeze(short_traj(:, 70:120)); %JUST LOOK AT DELAY
[Traj, sortind, peaktimes, cellorder, P] = SortEvent(Event); %SORT
max_activity = max(Event,[],2);
[value,indices] = sort(max_activity);
for i=1:10
    index = indices(end-i);

    if EI_matrix(index,index) == 1
        type = 'Excitatory';
    else
        type = 'Inhibitory';
    end

    figure
%     imagesc(short_traj(sortind, :)./max(short_traj(sortind,:)))
    imagesc(squeeze(h(std_second_ind(short_ind),:,index))./max(squeeze(h(std_second_ind(:,1),:,index))))
    hold on
    line(50*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
    line(62*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
    line(112*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
    line(132*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
    title(['Unit ', num2str(index), ': ' type, 'Short'])
    colorbar
    
    figure
%     imagesc(long_traj(sortind, :)./max(short_traj(sortind,:)))
    imagesc(squeeze(h(std_second_ind(long_ind),:,index))./max(squeeze(h(std_second_ind(:,1),:,index))))

    hold on
    line(50*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
    line(62*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
    line(162*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
    line(182*ones(256,1), 1:256, 'color','r','linestyle',':','linewidth',2)
    title(['Unit ', num2str(index), ': ' type, 'Long'])
    colorbar
end


