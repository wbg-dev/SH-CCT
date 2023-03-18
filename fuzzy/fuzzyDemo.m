clear all
clc
load data
c=zeros(20,3);
for k=1:size(x,2)
    for j=1:size(x,1)
        c(j,k)=fuzzyFunction(x(j,k),x(:,k),y);
    end
end
c1=max(c(:,1))-min(c(:,1));
c2=max(c(:,2))-min(c(:,2));
c3=max(c(:,3))-min(c(:,3));