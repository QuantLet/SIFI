fpath = '~\SIFI\'

reordering = [1,2,3,4,5,6,7,8, 9,10,11,12,16,17,18,19,20,21,22,23,24,25, 13,14,15,26,27,28];  % reordering for ordered SIFIs
data=xlsread('SIFI_Daily_Return28.xls','DataInput','E:AF');
data=data(:,reordering);
year = xlsread('SIFI_Daily_Return28.xls','DataInput','B:B');
DailyIV=xlsread('DailyIV28.xls','Sheet1','B:AC');
DailyIV = DailyIV(:,reordering);

[T,N]=size(data);
Return=100*data;    % Panel returns series across N SIFI
h = 90;

% Return = Return(:, [1:8 9:12 16:25 13:15 26:28])

ReturnToSave = (Return((h+1):(T),:));
save('ReturnToSave.mat','ReturnToSave')

X = xlsread('Market-wide-covariates.xls','DataInput','E:I');   % read market data
asset = xlsread('Node-specificdata28Currency.xls','nodespecificA', 'B:AC');
debt = xlsread('Node-specificdata28Currency.xls','nodespecificD', 'B:AC');
asset = asset(:,reordering);
debt = debt(:, reordering);
nodespec_asset= zeros(T,N);
nodespec_debt= zeros(T,N);

for k=1:T
    for i=2007:2015
        if (year(k)==i) 
            nodespec_asset(k,:) = asset(i-2006,:);
            nodespec_debt(k,:) = debt(i-2006,:);
        end
    end
end

Rpath = 'C:\Program Files\R\R-3.3.1\bin';
RscriptFileName = '~\SIFI\rq_check.r';


%which.one = 1; % 'SIM';
which.one = 2; % 'NG';
%which.one = 3; % 'equal';
Fisher = 1;
nsplits = 3; 


[Tx,Nx] = size(X);
sheetname = {'S_2008','S_2009','S_2010','S_2011','S_2012','S_2013','S_2014','S_2015'};
varnames = {'const','network', 'lagged Y', 'TEDrate','VIX','S&P500 return'};
firmnames = {'JP.MORGAN.CHASE','BANK.OF.AMERICA',	'BANK.OF.NEW.YORK.MELLON','CITIGROUP','GOLDMAN.SACHS.GP.','MORGAN.STANLEY','STATE.STREET','WELLS.FARGO','ROYAL.BANK.OF.SCTL.GP.','BARCLAYS','HSBC.HDG.','STANDARD.CHARTERED','BANK.OF.CHINA.','INDUSTRIAL.COML.BK.OF.CHINA.','CHINA.CON.BANK.','BNP.PARIBAS','CREDIT.AGRICOLE','SOCIETE.GENERALE','DEUTSCHE.BANK','UNICREDIT','ING.GROEP','BANCO.SANTANDER','NORDEA.BANK','CREDIT.SUISSE.GROUP.N','UBS.GROUP','MITSUBISHI.UFJ.FINL.GP.','MIZUHO.FINL.GP.','SUMITOMO.MITSUI.FINL.GP'};
geogr_loc = [1,1,1,1,1,1,1,1, 2,2,2,2, 3,3,3, 2,2,2,2,2,2,2,2,2,2, 3,3,3];


firmnames = firmnames(reordering);
geogr_loc = geogr_loc(reordering);


whichstats = {'beta', 'covb', 'r', 'rsquare','tstat'};
% z.alpha = norminv(0.975,0,1);
VaR=zeros(T-h,N);  %  Value-at-Risk at p%
ES=zeros(T-h,N);   %  Expected shortfall at p%
MeanIV=zeros(T-h,N);
p=0.05;          % 5% significance level

for k=1:(T-h)
    tempData=Return(k:(k+(h-1)),:);
    
    PartialReturn=zeros(h,N);
    for j=1:N
       Xpartial=tempData; Xpartial(:,j)=[];
       mdl = regstats(tempData(:,j),Xpartial,'linear', whichstats);
       PartialReturn(:,j) = mdl.r; 
    end 
    
    VaR(k,:)=quantile(PartialReturn,p,1);   % Calculate the p quantile for each column of X (dim = 1).
    for j=1:N
        ES(k,j)=mean(PartialReturn(find(PartialReturn(:,j) < VaR(k,j)),j));
    end
    MeanIV(k,:)=mean(DailyIV(k:(k+(h-1)),:));
   
end

% 
    whichstats = {'beta', 'covb', 'r', 'rsquare','tstat'};
    residVaR = zeros(T-h,N);
    residES = zeros(T-h,N);
    residIV = zeros(T-h,N);
    residR2 = zeros(N,3);
    for j=1:N

        XIV = MeanIV; XIV(:,j) = [];
        mdl = regstats(MeanIV(:,j),XIV,'linear', whichstats);
        residIV(:,j) = mdl.r;
        residR2(j,3) = mdl.rsquare;
    end

% VaR=residVaR; ES = residES; 
MeanIV = residIV;



% calculate rolling change of risk variables

delta_VaR=zeros(T-h,N);
delta_ES=zeros(T-h,N);
delta_IV=zeros(T-h,N);

    for i=1:N
        delta_VaR(:,i)=zscore(VaR(:,i));
        delta_ES(:,i)=zscore(ES(:,i));
        delta_IV(:,i)=zscore(MeanIV(:,i));
    end


% 
% %%% Measure cosine similarity in the context of Euclidean space   %%%%
% 
V=zeros(N,3);
Centered_V=zeros(N,3);
S=zeros(N,N,(T-h));
Connect=zeros((T-h),1);

for t=1:(T-h)
    for i=1:N
        
        V(i,:) = [delta_VaR(t,i) delta_ES(t,i) delta_IV(t,i)];
        Centered_V(i,:)= V(i,:)-mean(V(i,:));   % centered vector (deMean)
        
        for j=(i+1):N
            V(j,:) = [delta_VaR(t,j) delta_ES(t,j) delta_IV(t,j)];
            Centered_V(j,:)= V(j,:)-mean(V(j,:));  % centered vector (deMean)
            S(i,j,t) = (dot(Centered_V(i,:), Centered_V(j,:))/(norm(Centered_V(i,:))*norm(Centered_V(j,:))));   % centered cosine similarity
            S(j,i,t) = S(i,j,t);
            Connect(t,1)=sum(sum( S(:,:,t)));    %% Total connectedness of Similarity Matrix
           
        end
    end
%     xlswrite('CenteredSimilarityMatrixRolling',S(:,:,t),sheetname{t},'B2');
end


    years = unique(year((1+h):T));
    ind=zeros(length(years),1);   %  Expected shortfall at p%
    for i=1:length(years)
        ind(i) = find(year((1+h):T) == years(i),1,'first');
    end;
    years = years(2:end);
    ind=ind(2:end);


for iyear = 1:length(years)
    sub3 = figure
    figure(sub3)
    clims = [-1 1]
    %imagesc(S(:,:,ind(iyear)-1),clims)
    imagesc(S(:,:,1),clims)
    colormap(summer)
    colorbar
    title(year(ind(iyear)+h-1), 'FontSize', 12);
    axis image
    set(sub3,'PaperPosition',[0 0, 7 7]) 
    set(sub3, 'PaperSize', [7 7]);
    saveas(sub3,fullfile(fpath,strcat('heatmapS',num2str(year(ind(iyear)+h-1)))),'pdf') ; 
end




% %%% Ng approach   %%%%
% 
BinAdj=zeros(N,N,(T-h));
breaksNg = zeros(T-h, 2 + 7+7);
quantNg = zeros(T-h,  9);
eig_A=zeros((T-h),1);
Centralnode=zeros(N,(T-h));
 
for t=1:(T-h)    
%for t=1:2
    disp(t);
    if (Fisher==0)
        sigma=abs(sqrt(h)*S(:,:,t));   % take absolute value for correlation matrix
    end;
    if (Fisher==1)
        sigma = 0.5* log((1+S(:,:,t))./(1-S(:,:,t)));
    end; 
    rm1=[];m1=[];m1true=[];
    
    %  read upper triangle of corr matrix
    for i=1:N;
        m1=[m1;sigma(i,i+1:N)'];
        m1true = [m1true;S(i,i+1:N,t)'];
    end;
    quantNg(t,:) = quantile(m1true, [0.1,0.2,0.3, 0.4, 0.5,0.6,0.7, 0.8,0.9]);
    
    %  sort sample correlation as ordered correlation
    
    [dum,id_m]=sort(m1,1);
    m1=m1(id_m);
    [dumtrue,id_mtrue]=sort(m1true,1);
    m1true_sorted=m1true(id_mtrue);
    
    % % Probability integral transformation of the ordered correlations
    NN=N*(N-1)/2;
    q=2;    % q-order spacings
    %ym=normcdf(sqrt(h-3)*m1); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ym=normcdf(m1); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    dm=[ym(2:NN)-ym(1:NN-1)];
    
    if  (nsplits==2)
        N1=splitreg(dm,ones(NN-1,1),0.1);   % sumsample  S (1:N1)    N1+1 is breaking point
        N2=N1;                         % sumsample  L (N1+1:NN);
    end; 
    if  (nsplits==3)
        [N1 N2]=splitreg3(dm,ones(NN-1,1),0.1);
    end;
    
    breaksNg(t,1:2) = [N1 N2];
    if (Fisher==0)
        breaksNg(t,3:16) = [m1((N1-3):(N1+3))'/sqrt(h), m1((N2-3):(N2+3))'/sqrt(h)];
    end;
    if (Fisher==1)
        breaksNg(t,3:16) = [(exp(2*m1((N1-3):(N1+3))')-1)./(exp(2*m1((N1-3):(N1+3))')+1),  (exp(2*m1((N2-3):(N2+3))')-1)./(exp(2*m1((N2-3):(N2+3))')+1)];
    end;

    end;

    if (nsplits==2)
    for i=1:N
        for j=(i+1):N
            if (sigma(i,j) >=  m1(N1+1))
            %if (sigma(i,j) >=  sqrt(h) * 0.3)
                BinAdj(i,j,t)=1;
            else
                BinAdj(i,j,t)=0;
            end
            BinAdj(j,i,t)=BinAdj(i,j,t)  ;
        end
    end
    end;
    
    %     eig_A(t,1)=log(max(eig(BinAdj(:,:,t)'*BinAdj(:,:,t))));   % find the log of max eigenvalue of A'A
%  Eigenvector centrality

[V,D]=eig(BinAdj(:,:,t));
%V=round(V(:,N));
K=max(abs(V(:,1)));
Mmax=find(abs(V(:,1))==K);
Centralnode(Mmax,t)=1;
end;



for t=1:(T-h)    

    if (Fisher==1)
        sigma = 0.5* log((1+S(:,:,t))./(1-S(:,:,t)));
    end; 
    
    if (nsplits==3)
    for i=1:N
        for j=(i+1):N
            if (sigma(i,j) <=  0.5* log((1+breaksNg(t,5))./(1-breaksNg(t,5))) )
            %if (sigma(i,j) >=  sqrt(h) * 0.3)
                BinAdj(i,j,t)=0.5;
            end;
            if (sigma(i,j) >=  0.5* log((1+breaksNg(t,13))./(1-breaksNg(t,13))) )
                BinAdj(i,j,t)=1;
            end;
            if  ((sigma(i,j) >  0.5* log((1+breaksNg(t,5))./(1-breaksNg(t,5))) ) && (sigma(i,j) <  0.5* log((1+breaksNg(t,13))./(1-breaksNg(t,13)))))
                BinAdj(i,j,t)=0;    
            end;
            BinAdj(j,i,t)=BinAdj(i,j,t)  ;
        end;
    end
    end;
end




%%%%%%%% spacings and ordered data
    
    t=1;m1=[];
    for i=1:N;
        m1=[m1;S(i,i+1:N,1)'];
    end;
    
    sub3 = figure
    figure(sub3)
    plot(1:NN,m1);
    title('Correlations for t=1', 'FontSize', 12);
    set(sub3,'PaperPosition',[0 0, 14 7]) 
    set(sub3, 'PaperSize', [14 7]);
    if (Fisher==1) saveas(sub3,fullfile(fpath,strcat('t1F_corr')),'pdf') ;end;
    
    sub3 = figure
    figure(sub3)
    plot(1:NN,sort(m1));
    title('Ordered correlations for t=1', 'FontSize', 12);
    set(sub3,'PaperPosition',[0 0, 14 7]) 
    set(sub3, 'PaperSize', [14 7]);
    if (Fisher==1) saveas(sub3,fullfile(fpath,strcat('t1F_corr')),'pdf') ;end;
    
    
    
    sigma = 0.5* log((1+S(:,:,t))./(1-S(:,:,t)));
    rm1=[];m1=[];m1true=[];
    for i=1:N;
        m1=[m1;sigma(i,i+1:N)'];
    end;
    
    ym=normcdf(sqrt(h-3)*sort(m1)); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    


if (which.one==2) SS = BinAdj; end
if (which.one==1) SS = S; end
if (which.one==3)
    SS=zeros(N,N,(T-h));
    for t=1:(T-h)
      SS(:,:,t) = ones(N);
    end;
end;

    


    
% remove NAs

for i=1:Nx
    if (length(find(isnan(X(:,i))))>0) 
       X_inter = interp1(1:Tx, X(:,i), find(isnan(X(:,i))),'previous');
       X(find(isnan(X(:,i))),i)  = X_inter;
    end
end

% standardize X variables
X(:,2) = (X(:,2) - mean(X((h+1):(T-1),2)))/sqrt(var(X((h+1):(T-1),2)));
X(:,1) = (X(:,1) - mean(X((h+1):(T-1),1)))/sqrt(var(X((h+1):(T-1),1)));
X(:,3) = (X(:,3) - mean(X((h+1):(T-1),3)))/sqrt(var(X((h+1):(T-1),3)));



pv = 0.1:0.01:0.9;
b = zeros(T-h, Nx+1+1+1, length(pv));
b_sd = zeros(T-h, Nx+1+1+1, length(pv));



    years = unique(year((1+h):T));
    ind=zeros(length(years),1);   %  Expected shortfall at p%
    for i=1:length(years)
        ind(i) = find(year((1+h):T) == years(i),1,'first');
    end;
    years = years(2:end);
    ind=ind(2:end);


    
    
    
% full sample estimation 


b_full = zeros(Nx-1+1+1+1, length(pv));
b_full_sd = zeros(Nx-1+1+1+1, length(pv));


%for t=(h+1):(T-2*h)
%for t=251:261
     Ypart = Return((h+1):(T),:);
     Ypartlag = Return((h):(T-1),:);
     Xpart = X((h):(T-1),1:2);
     YpartV = Ypart(:);
     Xtemp=zeros(T-h,N);
     for ii=1:(T-h)
       if (which.one==2) Xtemp(ii,:) =  (SS(:,:,ii)*Ypartlag(ii,:)')' ./ sum(SS(:,:,ii),2)';end;
       if (which.one==1) Xtemp(ii,:) =  (SS(:,:,ii)*Ypartlag(ii,:)')' ./ N;end;
     end;

     save('Network.mat', 'Xtemp');
    
     Xtemp = Xtemp(:);
     Xall = [ones((T-2*h)*N,1) Xtemp Ypartlag(:) repmat(Xpart,N,1)];
     
     save('Xall.mat','Xall') ;
     save('YpartV.mat','YpartV');
     save('pv.mat','pv');
     RunRcode(RscriptFileName, Rpath);
     fileID = fopen('rq_result.txt','r');
     fromR = fscanf(fileID,'%f', [length(pv)*2 Inf]);
     fromR=fromR';
     fclose(fileID);
     
     Xall_n = size(Xall);Xall_n=Xall_n(1);
     fileResid = fopen('rq_resid.txt','r');
     %fromResid = fscanf(fileResid,'%f');
     fromResid = fscanf(fileResid,'%f', [17 Xall_n]);
     fromResid=fromResid';
     fclose(fileResid);
     
     
     for ii = 1:length(pv)    
        b_full(:,ii) = fromR(:, (ii-1)*2+1);
        b_full_sd(:,ii) = fromR(:,(ii-1)*2+2);
     end;

    
     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% geographic location
 
pv = 0.1:0.025:0.9;
for iCL=1:3

    b_fullCL = zeros(Nx-1+1+1+1, length(pv));
    b_fullCL_sd = zeros(Nx-1+1+1+1, length(pv));

    if (iCL==1) which_cluster = find(geogr_loc==1);end;
    if (iCL==2) which_cluster = find(geogr_loc==2);end;
    if (iCL==3) which_cluster = find(geogr_loc==3);end;
    
     Ypart = abs(Return((2*h+1):(T),which_cluster));
     Ypartlag = abs(Return((2*h):(T-1),which_cluster));
     Xpart = X((2*h):(T-1),1:2);
     YpartV = Ypart(:);
     Xtemp=zeros(T-2*h,length(which_cluster));
     for ii=1:(T-2*h)
       if (which.one==2) Xtemp(ii,:) =  (SS(which_cluster,which_cluster,ii)*Ypartlag(ii,:)')' ./ sum(SS(which_cluster,which_cluster,ii),2)';end;
       if (which.one==1) Xtemp(ii,:) =  (SS(which_cluster,which_cluster,ii)*Ypartlag(ii,:)')' ./ length(which_cluster);end;
     end;
     Xtemp = Xtemp(:);
     Xall = [ones((T-2*h)*length(which_cluster),1) Xtemp Ypartlag(:) repmat(Xpart,length(which_cluster),1)];
     
     save('Xall.mat','Xall') ;
     save('YpartV.mat','YpartV');
     save('pv.mat','pv');
     RunRcode(RscriptFileName, Rpath);
     fileID = fopen('rq_result.txt','r');
     fromR = fscanf(fileID,'%f', [length(pv)*2 Inf]);
     fromR=fromR';
     fclose(fileID);
     
     for ii = 1:length(pv)    
        b_fullCL(:,ii) = fromR(:, (ii-1)*2+1);
        b_fullCL_sd(:,ii) = fromR(:,(ii-1)*2+2);
     end;

    sub3 = figure
    figure(sub3)
    plot(pv,b_fullCL(2:end,:));
    h_legend=legend(varnames(2:end), 'Location','NorthEastOutside');
    set(h_legend,'FontSize',9);
    % set(gca, 'YLim', [0 1])
    title('Full sample estimation');
    set(sub3,'PaperPosition',[-1 0, 14.5 10]) 
    set(sub3, 'PaperSize', [13 10]);
    if (which.one==2)  saveas(sub3,fullfile(fpath,strcat('geog_ABSallDN', num2str(iCL))),'pdf') ; end;
    if (which.one==1) saveas(sub3,fullfile(fpath,strcat('geog_ABSall', num2str(iCL))),'pdf'); end; 
end;



% getting residuals 

b_fullLR = zeros(6, N);
b_full_pv_LR = zeros(6, N);
b_full_R2_LR = zeros(1, N);
residLR = zeros(T-h,N);

for i=1:N
     Ypart = (Return((h+1):(T),i));
     YpartlagLR = (Return((h):(T-1),i));
     Xpartnode_asset = nodespec_asset((h):(T-1),i);
     Xpartnode_debt = nodespec_debt((h):(T-1),i);
     Xpart = X(h:(T-1),1:2);
    whichstats = {'beta', 'covb', 'r', 'rsquare','tstat'};
     mdl = regstats(Ypart(:),[YpartlagLR Xpart Xpartnode_asset(:) Xpartnode_debt(:)] ,'linear', whichstats);
     residLR(:,i) = mdl.r;
     b_fullLR(:,i) = mdl.beta;
     b_full_pv_LR(:,i) = mdl.tstat.pval;
     b_full_R2_LR(i) = mdl.rsquare;
end;

save('residLR.mat','residLR')

b_fullLRCL = zeros(6, 3);
b_full_pv_LRCL = zeros(6, 3);
b_full_R2_LRCL = zeros(1, 3);

for iCL=1:3

    if (iCL==1) which_cluster = find(geogr_loc==1);end;
    if (iCL==2) which_cluster = find(geogr_loc==2);end;
    if (iCL==3) which_cluster = find(geogr_loc==3);end;
    
     Ypart = (Return((h+1):(T),which_cluster));
     YpartlagLR = (Return((h):(T-1),which_cluster));
     Xpartnode_asset = nodespec_asset((h):(T-1),which_cluster);
     Xpartnode_debt = nodespec_debt((h):(T-1),which_cluster);
     Xpart = X(h:(T-1),1:2);

     whichstats = {'beta', 'covb', 'r', 'rsquare','tstat'};
     mdl = regstats(Ypart(:),[YpartlagLR(:) repmat(Xpart,length(which_cluster),1) Xpartnode_asset(:) Xpartnode_debt(:)] ,'linear', whichstats);
     b_fullLRCL(:,iCL) = mdl.beta;
     b_full_pv_LRCL(:,iCL) = mdl.tstat.pval;
     b_full_R2_LRCL(iCL) = mdl.rsquare;
end;



% geographic location & residuals 
 
pv = 0.03:0.01:0.97;
pq = [0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9];
b_fullCL_all = zeros(length(pv), 4);
b_fullCL_all_sd = zeros(length(pv), 4);
b_fullCL_band_all = zeros(length(pv)-2, 4);
b_fullCL_band = zeros(length(pv)-2, 1);
q_hat01_geog = zeros(4,T-h);
q_hat05_geog = zeros(4,T-h);

b_fullLR = zeros(N, 6);
b_full_pv_LR = zeros(N, 6);
b_full_R2_LR = zeros(N,1);
residLR = zeros(T-h,N);

impulses = zeros(4, 4, length(pq), length(pv), N);
impulses1 = zeros(4, length(pq), length(pv), N,N);
b_QRvsLR = zeros(4,1);


for iCL=1:4

    b_fullCL = zeros(1+1, length(pv));
    b_fullCL_sd = zeros(1+1, length(pv));
    
    if (iCL==1) which_cluster = find(geogr_loc==1);region='US';end;
    if (iCL==2) which_cluster = find(geogr_loc==2);region='Europe';end;
    if (iCL==3) which_cluster = find(geogr_loc==3);region='Asia';end;
    if (iCL==4) which_cluster = find(geogr_loc>0);region='World';end;
    
     % each asset individually 
        
     Ypart = (Return((h+1):(T),which_cluster));
     Ypartlag = (Return((h):(T-1),:));
     YpartlagLR = (Return((h):(T-1),which_cluster));
     Xpartnode_asset = nodespec_asset((h):(T-1),which_cluster);
     Xpartnode_debt = nodespec_debt((h):(T-1),which_cluster);
     Xpart = X(h:(T-1),1:2);
     
     
     
     YpartV = zeros(size(Ypart));
     
     whichstats = {'beta', 'covb', 'r', 'rsquare','tstat'};
     for i = 1:length(which_cluster)
        mdl = regstats(Ypart(:,i),[YpartlagLR(:,i) Xpart Xpartnode_asset(:,i) Xpartnode_debt(:,i)] ,'linear', whichstats);
        %mdl = regstats(Ypart(:,i),[YpartlagLR(:,i) Xpart ] ,'linear', whichstats);
        YpartV(:,i) = mdl.r;
        b_fullLR(which_cluster(i),:) = mdl.beta;
        b_full_pv_LR(which_cluster(i),:) = mdl.tstat.pval;
        b_full_R2_LR(which_cluster(i)) = mdl.rsquare;
        disp(firmnames(which_cluster(i)));
        disp(mdl.rsquare);
     end;
     YpartV=YpartV(:);
     
     
     Xtemp=zeros(T-h,length(which_cluster));
     for ii=1:(T-h)
       if (which.one==2) Xtemp(ii,:) =  (SS(which_cluster,:,ii)*Ypartlag(ii,:)')' ./ sum(SS(which_cluster,:,ii),2)';end;
       if (which.one==1) Xtemp(ii,:) =  (SS(which_cluster,:,ii)*Ypartlag(ii,:)')' ./ N;end;
     end;
     Xtemp = Xtemp(:);
     Xall = [ones((T-h)*length(which_cluster),1) Xtemp];

     
     disp(iCL);
     disp(corr(YpartV,Xall));
    
     mdlLR = regstats(YpartV,[Xall(:,2)] ,'linear', whichstats);
     b_QRvsLR(iCL) = mdlLR.beta(2);
     
     
     save(strcat('Xall_', num2str(iCL),'.mat'),'Xall'); 
     save(strcat('YpartV_', num2str(iCL),'.mat'), 'YpartV');
 
     save('Xall.mat','Xall') ;
     save('YpartV.mat','YpartV');
     save('pv.mat','pv');
     RunRcode(RscriptFileName, Rpath);
     fileID = fopen('rq_result.txt','r');
     fromR = fscanf(fileID,'%f', [length(pv)*2 Inf]);
     fromR=fromR';
     fclose(fileID);
     
     for ii = 1:length(pv)    
        b_fullCL(:,ii) = fromR(:, (ii-1)*2+1);
        b_fullCL_sd(:,ii) = fromR(:,(ii-1)*2+2);
     end;

     b_fullCL_all(:,iCL) = b_fullCL(2,:)';
     b_fullCL_all_sd(:,iCL) = b_fullCL_sd(2,:)';
          

    
    minNet = 0.7*min(Xtemp);
    maxNet = 0.7*max(Xtemp);
    NetV = minNet:0.5:maxNet;
    colorspec = [0, 0, 1];
    linespec = {'-'};
    sub3 = figure
    figure(sub3)
    hold on;
    scatter(Xtemp, YpartV, '.');
    %plot(NetV,ones(length(NetV),1) * b_fullCL(1,[1,5,9,13,17,21,25,29,33])+ NetV'*b_fullCL(2,[1,5,9,13,17,21,25,29,33]));
    plot(NetV,ones(length(NetV),1) * b_fullCL(1,[1,4,8,12,16,20,24,28,32,36,39])+ NetV'*b_fullCL(2,[1,4,8,12,16,20,24,28,32,36,39]));
    hold off;
    %plot((h+1):(T-2*h),bmatrix_sd);
    %ax = gca;
    %set(ax,'XTickLabel',years)
    %set(ax,'XTick',ind)
    %h_legend=legend(strread(num2str([0.1 0.5 0.9]),'%s'), 'Location','NorthEastOutside');
    %set(h_legend,'FontSize',9);
    set(gca, 'XLim', [minNet maxNet])
    set(gca, 'YLim', [-30 30])
    %title(strcat(varnames(i)));
    set(sub3,'PaperPosition',[0 0, 14 7]) 
    set(sub3, 'PaperSize', [14 7]);
    if (which.one==2) saveas(sub3,fullfile(fpath,strcat('geog_DNparallel_', region)),'pdf');end; 
    if (which.one==1) saveas(sub3,fullfile(fpath,strcat('geog_parallel_', region)),'pdf');end; 
    
    
     
    
    end;  

% plotting factors

Ypartlag = (Return((h):(T-1),:));
years = unique(year(h:(T-1)));
ind=zeros(length(years),1);   
for i=1:length(years)
   ind(i) = find(year(h:(T-1)) == years(i),1,'first');
end;
years = years(2:end);
ind=ind(2:end);
which_sifi = [1,13,20];

for j = 1:3
    
    Xtemp=zeros(T-h,length(which_sifi));
    for ii=1:(T-h)
        if (which.one==2) Xtemp(ii,j) =  (SS(which_sifi(j),:,ii)*Ypartlag(ii,:)')' ./ sum(SS(which_sifi(j),:,ii),2)';end;
        if (which.one==1) Xtemp(ii,j) =  (SS(which_sifi(j),:,ii)*Ypartlag(ii,:)')' ./ N;end;
    end;
    
      sub3 = figure
    figure(sub3)
     plot((h):(T-1),Xtemp(:,j));
     %plot((h+1):(T-2*h),bmatrix_sd);
     ax = gca;
     set(ax,'XTickLabel',years)
     set(ax,'XTick',ind+h)
     %h_legend=legend(strread(num2str([0.1 0.5 0.9]),'%s'), 'Location','NorthEastOutside');
     %set(h_legend,'FontSize',9);
      %set(gca, 'YLim', [-15 20])
     title(firmnames(which_sifi(j)));
     set(sub3,'PaperPosition',[0 0, 14 7]) 
     set(sub3, 'PaperSize', [14 7]);
     if (which.one==2)  saveas(sub3,fullfile(fpath,strcat('network_factor_DN_',num2str(which_sifi(j)))),'pdf') ; end;
     if (which.one==1) saveas(sub3,fullfile(fpath,strcat('network_factor_',num2str(which_sifi(j)))),'pdf'); end;     
end;
    
   
    


% full sample & residuals

pv = 0.1:0.025:0.9;

    b_fullCL = zeros(1+1, length(pv));
    b_fullCL_sd = zeros(1+1, length(pv));

    which_cluster = 1:N;
     Ypart = (Return((h+1):(T),which_cluster ));
     YpartlagLR = (Return((h):(T-1),which_cluster));
     Ypartlag = (Return((h):(T-1),which_cluster));
     Xpart = X((h):(T-1),1:3);
     Xpartnode_asset = nodespec_asset((h):(T-1),which_cluster);
     Xpartnode_debt = nodespec_debt((h):(T-1),which_cluster);
     Xpartnode_asset = Xpartnode_asset(:);
     Xpartnode_debt = Xpartnode_debt(:);     
     Xpartnode = [Xpartnode_asset Xpartnode_debt];
     YpartV = Ypart(:);
     
     whichstats = {'beta', 'covb', 'r', 'rsquare', 'tstat'};
     mdl = regstats(YpartV,[YpartlagLR(:) repmat(Xpart,length(which_cluster),1), Xpartnode] ,'linear', whichstats);
     
     disp('full sample: beta');
     disp(mdl.beta');
     disp('full sample: p-values');
     disp(mdl.tstat.pval');
     disp('full sample: R2');
     disp(mdl.rsquare);
     
     YpartV = (mdl.r);
     
     Xtemp=zeros(T-h,length(which_cluster));
     for ii=1:(T-h)
       if (which.one==2) Xtemp(ii,:) =  (SS(which_cluster,which_cluster,ii)*Ypartlag(ii,:)')' ./ sum(SS(which_cluster,which_cluster,ii),2)';end;
       if (which.one==1) Xtemp(ii,:) =  (SS(which_cluster,which_cluster,ii)*Ypartlag(ii,:)')' ./ length(which_cluster);end;
     end;
     save('Network.mat', 'Xtemp');
     Xtemp = Xtemp(:);
     Xall = [ones((T-h)*length(which_cluster),1) Xtemp];
    
     save('Xall.mat','Xall') ;
     save('YpartV.mat','YpartV');
     save('pv.mat','pv');
     RunRcode(RscriptFileName, Rpath);
     fileID = fopen('rq_result.txt','r');
     fromR = fscanf(fileID,'%f', [length(pv)*2 Inf]);
     fromR=fromR';
     fclose(fileID);
     
     for ii = 1:length(pv)    
        b_fullCL(:,ii) = fromR(:, (ii-1)*2+1);
        b_fullCL_sd(:,ii) = fromR(:,(ii-1)*2+2);
     end;

     
    sub3 = figure
    figure(sub3)
    plot(pv,b_fullCL(2:end,:));
    h_legend=legend(varnames(2:end), 'Location','NorthEastOutside');
    set(h_legend,'FontSize',9);
    % set(gca, 'YLim', [0 1])
    title('Full sample estimation');
    set(sub3,'PaperPosition',[-1 0, 14.5 10]) 
    set(sub3, 'PaperSize', [13 10]);
    if (which.one==2)  saveas(sub3,fullfile(fpath,strcat('geog_ABSDN_resid_')),'pdf') ; end;
    if (which.one==1) saveas(sub3,fullfile(fpath,strcat('geog_ABS_resid_')),'pdf'); end; 

colorspec = [0, 0, 1];
linespec = {'-'};
    sub3 = figure
    figure(sub3)
    hold on;
    plot(pv,b_fullCL(2:end,:));
    shadedErrorBar(pv,b_fullCL(2:end,:), 1.96*b_fullCL_sd(2:end,:), {'-','color',colorspec}, 0.5)
    hold off;
    %plot((h+1):(T-2*h),bmatrix_sd);
    %ax = gca;
    %set(ax,'XTickLabel',years)
    %set(ax,'XTick',ind)
    %h_legend=legend(strread(num2str([0.1 0.5 0.9]),'%s'), 'Location','NorthEastOutside');
    %set(h_legend,'FontSize',9);
    %set(gca, 'YLim', [0 1])
    %title(strcat(varnames(i)));
    set(sub3,'PaperPosition',[0 0, 14 7]) 
    set(sub3, 'PaperSize', [14 7]);
    if (which.one==2) saveas(sub3,fullfile(fpath,strcat('geog_ABSDN_resid_NET_shaded_')),'pdf');end; 
    if (which.one==1) saveas(sub3,fullfile(fpath,strcat('geog_ABS_resid_NET_shaded_')),'pdf');end; 

    
    whichstats = {'beta', 'covb', 'r', 'rsquare', 'tstat'};
     mdl = regstats(b_fullCL(2,:)', b_fullCL_all ,'linear', whichstats);







