function [PF,residue] = lmd(x)
c = x;
N = length(x);
A = ones(1,N);
PF = [];
aii = 2*A;

while(1)

  si = c;
  a = 1;
  
   while(1)
    h = si;
    
      maxVec = [];
      minVec = [];
      
   % look for max and min point（寻找极大值点和极小值点）
      for i = 2: N - 1
         if h (i - 1) < h (i) & h (i) > h (i + 1)
            maxVec = [maxVec i]; 		
         end
         if h (i - 1) > h (i) & h (i) < h (i + 1)
            minVec = [minVec i]; 		
         end         
      end
      
   % check if it is residual（检查是否有残余）
      if (length (maxVec) + length (minVec)) < 2
         break;
      end
           
  % handle end point(原始信号中的两边两个点的判断) 
      lenmax=length(maxVec);
      lenmin=length(minVec);
      %left end point
      if h(1)>0
          if(maxVec(1)<minVec(1))
              yleft_max=h(maxVec(1));
              yleft_min=-h(1);
          else
              yleft_max=h(1);
              yleft_min=h(minVec(1));
          end
      else
          if (maxVec(1)<minVec(1))
              yleft_max=h(maxVec(1));
              yleft_min=h(1);
          else
              yleft_max=-h(1);
              yleft_min=h(minVec(1));
          end
      end
      %right end point
      if h(N)>0
          if(maxVec(lenmax)<minVec(lenmin))
             yright_max=h(N);
             yright_min=h(minVec(lenmin));
          else
              yright_max=h(maxVec(lenmax));
              yright_min=-h(N);
          end
      else
          if(maxVec(lenmax)<minVec(lenmin))
              yright_max=-h(N);
              yright_min=h(minVec(lenmin));
          else
              yright_max=h(maxVec(lenmax));
              yright_min=h(N);
          end
      end
      %get envelop of maxVec and minVec using（使用三次样条插值方法，对极大值向量和极小值向量进行插值）
      %spline interpolate
      maxEnv=spline([1 maxVec N],[yleft_max h(maxVec) yright_max],1:N);
      minEnv=spline([1 minVec N],[yleft_min h(minVec) yright_min],1:N);
      
    mm = (maxEnv + minEnv)/2;%得到局部均值函数
    aa = abs(maxEnv - minEnv)/2;%得到包络函数
    
    mmm = mm;
    aaa = aa;

    preh = h;
    h = h-mmm;%从原始信号中分离处局部均值函数
    si = h./aaa;%对分离出的信号进行解调
    a = a.*aaa;    
    
aii = aaa;

    B = length(aii);
    C = ones(1,B);
    bb = norm(aii-C);%返回aii-C的最大奇异值，aii就是那个包络函数
    if(bb < 1000)%如果bb<1000，就得到了纯调频函数
        break;
    end     
    
   end %分解1个Pf分量在这结束
   
  pf = a.*si;%包络函数和纯调频函数相乘，得到PF分量
  
  PF = [PF; pf];
  
  bbb = length (maxVec) + length (minVec);
 % check if it is residual
      if (length (maxVec) + length (minVec)) < 20
         break;
      end
           
  c = c-pf;

end