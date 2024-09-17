a1 = float(input('nhap a1:'))
b1 = float(input('nhap b1:'))
c1 = float(input('nhap c1:'))
a2 = float(input('nhap a2:'))
b2 = float(input('nhap b2:'))
c2 = float(input('nhap c2:'))
D = a1*b2 - a2*b1
Dx = c1*b2 - c2*b1
Dy = a1*c2 - a2*c1
if D!=0:
    x=Dx/D
    y=Dy/D
    print('nghiem x,y =',x,y)
else:
    if Dx!=0 and Dy!=0:
        print('hpt vo nghiem')
    else:
        print(' he pt vo so nghiem')

