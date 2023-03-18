function result = fuzzyFunction(a,x,y)
length=max(x)-min(x);
length=0.2*length;
lower=0;
upper=0;
for i =1 :size(x,1)
    lower=lower+exp(-((x(i,1)-a)/length)^2);
    upper=upper+exp(-((x(i,1)-a)/length)^2)*y(i,1);
end
result=upper/lower;
end

