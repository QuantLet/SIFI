function brk=splitreg(y,x,trim);

[T,nk]=size(y);
T1=ceil(trim * T);
beta=x\y;
e=y-x*beta;
ee=e'*e;
ssr=ones(T,1)*999;

for t=T1:T-T1;
  y1=y(1:t,:);x1=x(1:t,:);y2=y(t+1:T);x2=x(t+1:T);
  b1=x1\y1; b2=x2\y2;
  e1=y1-x1*b1;e2=y2-x2*b2;
  ssr(t)=e1'*e1+e2'*e2;
end;
[a1,a2]=min(ssr);
brk=a2;

