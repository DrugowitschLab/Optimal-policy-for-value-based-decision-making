function valueDecisionBoundaryRR()
% function valueDecisionBoundaryRR()
%
% This code generates the figures presented in the paper by Tajima, Drugowitsch & Pouget (2016) [1] and runs some extended simulations.
% 
% CITATION:
% [1] Satohiro Tajima*, Jan Drugowitsch*, and Alexandre Pouget.
% Optimal policy for value-based decision-making. 
% Nature Communications, 7:12400, (2016). 
% *Equally contributed.

tic;
Smax = 4;      % Grid range of states space (now we assume: S = [(Rhat1+Rhat2)/2, (Rhat1-Rhat2)/2]); Rhat(t) = (varR*X(t)+varX)/(t*varR+varX) )
resS = 201;      % Grid resolution of state space
tmax = 3;       % Time limit
dt   = .05;       % Time step
c    = 0;       % Cost of evidene accumulation
tNull = .25;     % Non-decision time + inter trial interval
g{1}.meanR = 1; % Prior mean of state (dimension 1)
g{1}.varR  = 5; % Prior variance of stte
g{1}.varX  = 2; % Observation noise variance
g{2}.meanR = 0; % Prior mean of state (dimension 2)
g{2}.varR  = 5; % Prior variance of state
g{2}.varX  = 2; % Observation noise variance
t = 0:dt:tmax;
Sscale = linspace(-Smax,Smax,resS);
[S{1},S{2}] = meshgrid(Sscale, Sscale);
iS0 = [findnearest(g{1}.meanR, Sscale) findnearest(g{2}.meanR, Sscale)];

%% Utility functions:
utilityFunction = @(x) x;               % Linear utility function (for Fig. 3)
% utilityFunction = @(x) tanh(x);       % Saturating utility function (for Fig. 6)

%% Reward rate, Average-adjusted value, Decision:
Slabel = {'r_1^{hat}', 'r_2^{hat}'};

Rh{1} = utilityFunction(S{1});                                                                              % Expected reward for option 1
Rh{2} = utilityFunction(S{2});                                                                              % Expected reward for option 2
RhMax = max_({Rh{1}, Rh{2}});                                                                               % Expected reward for decision
rho_ = fzero(@(rho) backwardInduction(rho,c,tNull,g,Rh,S,t,dt,iS0), g{1}.meanR, optimset('MaxIter',10));    % Reward rate
[V0, V, D, EVnext, rho, Ptrans, iStrans] = backwardInduction(rho_,c,tNull,g,Rh,S,t,dt,iS0);                 % Average-adjusted value, Decision, Transition prob. etc.
dbS2 = detectBoundary(D,S,t);

%% Transform to the space of accumulated evidence:
dbX = transformDecBound(dbS2,Sscale,t,g);


%% - Show results -
figure; clf; colormap bone;
iS2 = findnearest(.5, Sscale, -1);
iTmax = length(t);
rect = [-1 1 -1 1 -2.3 .5];

%% t=0:
subplotXY(5,4,2,1); [r1Max,r2Max,vMax] = plotSurf(Sscale, V(:,:,1)                , iS2, [0 0 0], Slabel); axis(rect); title('V(0)');
%                     plot3(g{1}.meanR, g{2}.meanR, V0, 'g.', 'MarkerSize',15);
subplotXY(5,4,3,1); [r1Acc,r2Acc,vAcc] = plotSurf(Sscale, EVnext(:,:,1)-(rho+c)*dt, iS2, [1 0 0], Slabel); axis(rect); title('<V(\deltat)|R^{hat}(0)> - (\rho+c)\deltat');
subplotXY(5,4,4,1); [r1Dec,r2Dec,vDec] = plotSurf(Sscale, RhMax-rho*tNull         , iS2, [0 0 1], Slabel); axis(rect); title('max(R_1^{hat},R_2^{hat}) - \rho t_{Null}');
subplotXY(5,4,5,1); hold on;
    plot((r1Max-r2Max)/2, vMax, 'k:', (r1Acc-r2Acc)/2, vAcc, 'r', (r1Dec-r2Dec)/2, vDec, 'b');
    xlabel(['(' Slabel{1} '-' Slabel{2} ')/2']); xlim(rect(1:2));
subplotXY(5,4,1,1); imagesc(Sscale, Sscale, D(:,:,  1), [1 3]); axis square; axis xy;
    title(['D(0) \rho=' num2str(rho_,3)]); xlabel(Slabel{1}); ylabel(Slabel{2}); hold on; axis(rect(1:4));
                    plot(r1Max, r2Max, 'r-');
%                     plot(g{1}.meanR, g{2}.meanR, 'g.');

%% t=0 (superimposed & difference):
subplotXY(5,4,3,2); plotSurf(Sscale, EVnext(:,:,1)-(rho+c)*dt, iS2, [1 0 0], Slabel); hold on;
                    plotSurf(Sscale, RhMax-rho*tNull         , iS2, [0 0 1], Slabel); axis(rect);
subplotXY(5,4,4,2); plotSurf(Sscale, RhMax-rho*tNull - (EVnext(:,:,1)-(rho+c)*dt), iS2, [0 1 0], Slabel); xlim(rect(1:2)); ylim(rect(1:2));

%% t=dt:
subplotXY(5,4,2,2); plotSurf(Sscale, V(:,:,2),      iS2, [0 0 0], Slabel); axis(rect); title('V(\deltat)');
subplotXY(5,4,1,2); imagesc(Sscale, Sscale, D(:,:,  2), [1 3]); axis square; axis xy; title('D(\deltat)'); xlabel(Slabel{1}); ylabel(Slabel{2}); hold on; axis(rect(1:4));

%% t=T-dt:
subplotXY(5,4,1,3); imagesc(Sscale, Sscale, D(:,:,iTmax-1), [1 3]); axis square; axis xy;
    title('D(T-\deltat)'); hold on; rectangle('Position',[rect(1) rect(3) rect(2)-rect(1) rect(4)-rect(3)]); axis(rect);
subplotXY(5,4,2,3); [r1Max,r2Max,vMax] = plotSurf(Sscale, V(:,:,iTmax-1)                , iS2, [0 0 0], Slabel); axis(rect); title('V(T-\deltat)')
subplotXY(5,4,3,3); [r1Acc,r2Acc,vAcc] = plotSurf(Sscale, EVnext(:,:,iTmax-1)-(rho+c)*dt, iS2, [1 0 0], Slabel); axis(rect); title('<V(T)|R^{hat}(T-\deltat)> - (\rho+c) \deltat');
subplotXY(5,4,4,3); [r1Dec,r2Dec,vDec] = plotSurf(Sscale, RhMax-rho*tNull               , iS2, [0 0 1], Slabel); axis(rect); title('max(R_1^{Hat},R_2^{Hat}) - \rho t_{Null}');
subplotXY(5,4,5,3); hold on;
    plot((r1Max-r2Max)/2, vMax, 'k:', (r1Acc-r2Acc)/2, vAcc, 'r', (r1Dec-r2Dec)/2, vDec, 'b');
    xlabel(['(' Slabel{1} '-' Slabel{2} ')/2']); xlim(rect(1:2));
    
%% t=T:
subplotXY(5,4,1,4); imagesc(Sscale, Sscale, D(:,:,iTmax), [1 3]); axis square; axis xy; title('D(T)'); hold on; axis(rect(1:4));
subplotXY(5,4,2,4); plotSurf(Sscale, V(:,:,iTmax), iS2, [0 0 0], Slabel); title('V(T) = max(R_1^{hat},R_2^{hat}) - \rho t_{Null}'); axis(rect);

toc;


function [V0, V, D, EVnext, rho, Ptrans, iStrans] = backwardInduction(rho_,c,tNull,g,Rh,S,t,dt,iS0)
k = 0;
rho = k*S{1}/tNull + (1-k)*rho_;                                                                        % Reward rate estimate
[V(:,:,length(t)), D(:,:,length(t))] = max_({Rh{1}-rho*tNull, Rh{2}-rho*tNull});                        % Max V~ at time tmax
for iT = length(t)-1:-1:1
    [EVnext(:,:,iT), Ptrans{iT}, iStrans{iT}] = E(V(:,:,iT+1),S,t(iT),dt,g);                            % <V~(t+1)|S(t)> for waiting
%    disp(size(Rh{1}));
%    disp(size(Rh{2}));
%    disp(size(EVnext(:,:,iT)));
    [V(:,:,iT), D(:,:,iT)] = max_({Rh{1}-rho*tNull, Rh{2}-rho*tNull, EVnext(:,:,iT)-(rho+c)*dt});       % [Average-adjusted value (V~), decision] at time t
%     fprintf('%d/%d\t',iT,length(t)-1); toc;
end
disp(iS0(1));
%V0 = mean(vector(V(iS0(1),iS0(2),1)));
V0 = V(iS0(1),iS0(2),1); % JARM (8th May '19)
fprintf('rho = %d\tV0 = %d\t', rho_, V0); toc;

function R = extrap(mat, varargin)
    % JARM (9th May '19) original function missing; identity function used and code fixed around this
    R = mat;

function [EV, Ptrans, iStrans] = E(V,S,t,dt,g)
g{1}.varRh = g{1}.varR * g{1}.varX / (t * g{1}.varR + g{1}.varX);
g{2}.varRh = g{2}.varR * g{2}.varX / (t * g{2}.varR + g{2}.varX);
v1 = varTrans(g{1}.varRh, g{1}.varR, g{1}.varX, t, dt);
v2 = varTrans(g{2}.varRh, g{2}.varR, g{2}.varX, t, dt);
aSscale = abs(S{1}(1,:));
iStrans{1} = find(aSscale<3*sqrt(v1));
iStrans{2} = find(aSscale<3*sqrt(v2));
Ptrans = normal2({S{1}(iStrans{2},iStrans{1}),S{2}(iStrans{2},iStrans{1})}, [0 0], [v1 0; 0 v2]);
mgn = ceil(size(Ptrans)/2);
V = extrap(V,mgn,[5 5]); % JARM (8th May '19) ???
EV = conv2(V,Ptrans,'same'); % JARM (8th May '19) marginalise expected value over probabilities of future states
%EV = EV(mgn(1)+1:end-mgn(1), mgn(2)+1:end-mgn(2)); % JARM (8th May '19) select central sub-region of larger expected value array

function v = varTrans(varRh, varR, varX, t, dt)
% v = (varR * (varX + varRh)) / ((1 + t/dt) * varR + varX / dt);
v = (varR ./ (varR*(t+dt) + varX)).^2 .* (varX + varRh * dt) * dt;

function prob = normal2(x, m, C)
d1 = x{1} - m(1);
d2 = x{2} - m(2);
H = -1/2*(C\eye(2)); prob = exp(bsxfun(@plus,d1.*d1*H(1,1), d1.*d2*H(1,2)) + d2.*d1*H(2,1) + d2.*d2*H(2,2));
% prob = exp(-(d1.^2/C(1,1)/2 + d2.^2/C(2,2))/2);
prob = prob ./ sum(prob(:));

function [V, D] = max_(x)
x_ = zeros(size(x{1},1), size(x{1},2), length(x));
for k = 1:length(x)
    x_(:,:,k) = x{k};
end
[V, D] = max(x_,[],3);
D(x{1}==x{2} & D==1) = 1.5;

function dbS2 = detectBoundary(D,S,t)
dS = diff(S{2}(1:2,1));
S_ = repmat(S{2},[1 1 length(t)]); S_(D~=1 & D~=1.5) =  Inf; dbS2(:,:,1) = max(squeeze(min(S_))-dS, 0);                % Decision boundary [min(S2;dec=1); max(S2;dec=2)]
S_ = repmat(S{2},[1 1 length(t)]); S_(D~=2 & D~=1.5) = -Inf; dbS2(:,:,2) = min(squeeze(max(S_))+dS, 0);                %  ... bndS2(iS1, iTime, iDec)
mgn = 1; [sm{1},sm{2}] = meshgrid(-mgn:mgn,-mgn:mgn);
for k=1:2;
    %% Extrapolating:
    db_ = dbS2(:,:,k); db_(~isfinite(db_) & isfinite([db_(:,2:end) db_(:,end)])) = (-1)^(k+1)*max(max(S{1}));  dbS2(:,:,k) = db_; % JARM (8th May '19) changed vector() call to max()
    
    %% Smoothing:
    db_ = conv2(extrap(dbS2(:,:,k),mgn),normal2(sm,[0 0],[1 0; 0 1]),'same'); 
%    dbS2(:,:,k) = db_(mgn+1:end-mgn,mgn+1:end-mgn);
    dbS2(:,:,k) = db_;     % JARM (8th May '19)
end

function [dbX, dbR] = transformDecBound(dbS2,Sscale,t,g)
S1 = repmat(Sscale',[1 size(dbS2,2) size(dbS2,3)]);
t_ = repmat(t,[size(dbS2,1) 1 size(dbS2,3)]);
for k=1:2;  mR{k}=g{k}.meanR;  vR{k}=g{k}.varR;  vX{k}=g{k}.varX;  end
dbX(:,:,:,1) = (t_+(vX{1}+vX{2})./(vR{1}+vR{2})) .* (S1+dbS2) - (vX{1}+vX{2})./(vR{1}+vR{2}) .* (mR{1}+mR{2});          % X1 (iS1, iTime, iDec, 1)
dbX(:,:,:,2) = (t_+(vX{1}+vX{2})./(vR{1}+vR{2})) .* (S1-dbS2) - (vX{1}+vX{2})./(vR{1}+vR{2}) .* (mR{1}-mR{2});          % X2 (iS1, iTime, iDec, 2)
dbR(:,:,:,1) = (S1+dbS2);          % R1 (iS1, iTime, iDec, 1)
dbR(:,:,:,2) = (S1-dbS2);          % R2 (iS1, iTime, iDec, 2)

function [x_,y_,v_] = plotSurf(Sscale, Val, iS, col, Slabel)
[x,y] = meshgrid(1:length(Sscale), 1:length(Sscale));
x_ = Sscale(x(x+y==iS+round(length(Sscale)/2)));
y_ = Sscale(y(x+y==iS+round(length(Sscale)/2)));
v_ = Val(x+y==iS+round(length(Sscale)/2));
h = surfl(Sscale, Sscale, Val); hold on; %camproj perspective;
set(h,'FaceColor', col, 'EdgeColor','none'); camlight left; lighting phong; alpha(0.7) % JARM (8th May '19) replaced sat(.5,col) with col
if ischar(col);  plot3(x_, y_, v_,         col); hold on;
else             plot3(x_, y_, v_, 'Color',col); hold on;  end
xlabel(Slabel{1}); ylabel(Slabel{2}); %zlim([-50 50]);

