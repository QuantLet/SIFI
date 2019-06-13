function [brk1 brk2]=splitreg(y,x,trim);

[T,nk]=size(y);
T1=ceil(trim * T);
beta=x\y;
e=y-x*beta;
ee=e'*e;
minsrr = 999;

for t=T1:(T-2*T1);
    for t1 = (t+T1):T-T1;
  y1=y(1:t,:);x1=x(1:t,:);
  y2=y(t+1:t1);x2=x(t+1:t1);
  y3=y(t1+1:T);x3=x(t1+1:T);
  b1=x1\y1; b2=x2\y2;b3=x3\y3;
  e1=y1-x1*b1;e2=y2-x2*b2;e3=y3-x3*b3;
  ssr_tt1=e1'*e1+e2'*e2+e3'*e3;
  if (ssr_tt1<minsrr)
      a1=t; a2 = t1;
      minsrr=ssr_tt1;
  end;
  end;
end;
%[a1,a2]=min(ssr);
brk1 = a1;
brk2 = a2;
