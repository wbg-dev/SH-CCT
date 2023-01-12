function[sol,val]=gabpEval(sol,options)
global codeLength
for i=1:codeLength
x(i)=sol(i);
end
[W1,B1,W2,B2,val] = gadecod(x);
