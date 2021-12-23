function [ y ] = binaryzation( d , t )
d = d';
y=zeros(size(d,1),size(d,2));
[B,I]=sort(d,1,'descend');

for n=1:size(d,2)
    temp=0;
    for m=1:size(B,1)
        temp=temp+B(m,n);
        if temp>t
              break;
        end
    end
    for z=1:m
        y(I(z,n),n)=1;
    end
end

y=y';
end
