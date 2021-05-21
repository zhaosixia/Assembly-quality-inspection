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
      
   % look for max and min point��Ѱ�Ҽ���ֵ��ͼ�Сֵ�㣩
      for i = 2: N - 1
         if h (i - 1) < h (i) & h (i) > h (i + 1)
            maxVec = [maxVec i]; 		
         end
         if h (i - 1) > h (i) & h (i) < h (i + 1)
            minVec = [minVec i]; 		
         end         
      end
      
   % check if it is residual������Ƿ��в��ࣩ
      if (length (maxVec) + length (minVec)) < 2
         break;
      end
           
  % handle end point(ԭʼ�ź��е�������������ж�) 
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
      %get envelop of maxVec and minVec using��ʹ������������ֵ�������Լ���ֵ�����ͼ�Сֵ�������в�ֵ��
      %spline interpolate
      maxEnv=spline([1 maxVec N],[yleft_max h(maxVec) yright_max],1:N);
      minEnv=spline([1 minVec N],[yleft_min h(minVec) yright_min],1:N);
      
    mm = (maxEnv + minEnv)/2;%�õ��ֲ���ֵ����
    aa = abs(maxEnv - minEnv)/2;%�õ����纯��
    
    mmm = mm;
    aaa = aa;

    preh = h;
    h = h-mmm;%��ԭʼ�ź��з��봦�ֲ���ֵ����
    si = h./aaa;%�Է�������źŽ��н��
    a = a.*aaa;    
    
aii = aaa;

    B = length(aii);
    C = ones(1,B);
    bb = norm(aii-C);%����aii-C���������ֵ��aii�����Ǹ����纯��
    if(bb < 1000)%���bb<1000���͵õ��˴���Ƶ����
        break;
    end     
    
   end %�ֽ�1��Pf�����������
   
  pf = a.*si;%���纯���ʹ���Ƶ������ˣ��õ�PF����
  
  PF = [PF; pf];
  
  bbb = length (maxVec) + length (minVec);
 % check if it is residual
      if (length (maxVec) + length (minVec)) < 20
         break;
      end
           
  c = c-pf;

end