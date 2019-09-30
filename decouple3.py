import numpy as np
import cv2
import math
from math import log
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out
def bilateral(im):
    img=im2double(im)
    #image = np.zeros((999, 781))
    print(np.min(img))
    print(np.max(img))
    #image=cv2.normalize(img,  image, 0, 255, cv2.NORM_MINMAX)

    print(np.min(img))
    print(np.max(img))
    r=input("enter no. of kernel rows")
    c=input("enter no. of kernel columns")
    r1=r//2
    c1=c//2
    p=r*c
    im=np.pad(img,(r1,c1),'edge')
    s=input("enter the value of sigma")
    s1=input("enter the value of sigma-1")
    d1=float(2*s1*s1)
    g11=float(1/math.sqrt(2*3.14*s1*s1))
    d=float(2*s*s)
    g1=float(1/(2*3.14*s*s))
    k=np.empty([r,c])
    for w in range(r-1):
        for q in range(c-1):
            diff1=r1-w
            diff2=c1-q
        
            e=(diff1*diff1)+(diff2*diff2)
            g2=float(math.exp(-e/d))
            g=float(g1*g2)
            k[w][q]=g
    print(k)

    rows,cols=im.shape
    print(rows)
    print(cols)
    for i in range(rows-r):
        for j in range(cols-c):
            sum=0
            k1=np.empty([r,c])
            for x in range(r-1):
                for  y in range(c-1):
                
                    diff=float((im[i+r1][j+c1]-im[i+x][j+y]))
                    e1=float(diff*diff)
                    g21=float(math.exp(-e1/d1))
                    gg=float(g11*g21)
                    k1[x][y]=gg
                    sum=sum+(im[i+x][j+y]*k[x][y])*k1[x][y]
                
            sum1=float(sum/p)
            img[i][j]=sum1
    image = np.zeros((999, 781))
    image=cv2.normalize(img,  image, 0, 255, cv2.NORM_MINMAX)
    print(np.min(image))
    print(np.max(image))
    return image
    #photo=cv2.imread('intensityf.png',0)
    #o=photo-img
    #print(o.dtype)
    #print(o)
    #cv2.imshow("o",o)
    #cv2.imshow("i",img)
    #cv2.imshow('im',photo)
im= cv2.imread('teddy1.jpg',1)
flash=im2double(im)
img= cv2.imread('teddy2.jpg',1)
noflash=im2double(img)
row,col,ch=flash.shape
bf,gf,rf=cv2.split(flash)
bn,gn,rn=cv2.split(noflash)
intensityf=np.empty([row,col])
intensityn=np.empty([row,col])
bf1=np.empty([row,col])
gf1=np.empty([row,col])
rf1=np.empty([row,col])
bn1=np.empty([row,col])
gn1=np.empty([row,col])
rn1=np.empty([row,col])
f=np.empty([row,col])
n=np.empty([row,col])
for i in range(row-1):
    for j in range(col-1):
        a=float((bf[i][j]*bf[i][j])+(gf[i][j]*gf[i][j])+(rf[i][j]*rf[i][j]))
        b=(bf[i][j]+gf[i][j]+rf[i][j])
        c=float((bn[i][j]*bn[i][j])+(gn[i][j]*gn[i][j])+(rn[i][j]*rn[i][j]))
        d=(bn[i][j]+gn[i][j]+rn[i][j])
        intensityf[i][j]=(a/b)
        intensityn[i][j]=(c/d)
        bf1[i][j]=float(bf[i][j]/intensityf[i][j])
        gf1[i][j]=float(gf[i][j]/intensityf[i][j])
        rf1[i][j]=float(rf[i][j]/intensityf[i][j])
        bn1[i][j]=float(bn[i][j]/intensityn[i][j])
        gn1[i][j]=float(gn[i][j]/intensityn[i][j])
        rn1[i][j]=float(rn[i][j]/intensityn[i][j])
f=cv2.merge([bf1,gf1,rf1])
n=cv2.merge([bn1,gn1,rn1])
x=cv2.imread('intensityf.png',0)
y=cv2.imread('intensityn.png',0)
blurf=bilateral(x)
blurf=im2double(blurf)
blurn=bilateral(y)
blurn=im2double(blurn)
imagef = np.zeros((999, 781))
imagef=cv2.normalize(blurf,  imagef, 0, 255, cv2.NORM_MINMAX)
imagen = np.zeros((999, 781))
imagen=cv2.normalize(blurn,  imagen, 0, 255, cv2.NORM_MINMAX)
xf=np.log10(x)
bff=np.log10(blurf)
xn=np.log10(y)
bfn=np.log10(blurn)
af=np.empty([row,col])
an=np.empty([row,col])
for i in range(row-1):
    for j in range(col-1):
        af[i][j]=xf[i][j]-bff[i][j]
        an[i][j]=xn[i][j]-bfn[i][j]
#blurf=cv2.bilateralFilter(x,10,30,30)
#blurn=cv2.bilateralFilter(y,10,30,30)
#blurf=im2double(blurf)
#intensityf=im2double(intensityf)
#detailf=np.empty([row,col])
#for i in range(row-1):
#    for j in range(col-1):
#detailf=intensityf-blurf
#print(detailf.dtype)
#cv2.imshow('detailf',detailf)
cv2.imshow('blurf',blurf)
cv2.imshow('blurn',blurn)
cv2.imshow('t1',intensityf)
cv2.imshow('t2',intensityn)
#cv2.imshow('normf',imagef)
#cv2.imshow('normn',imagen)
cv2.imshow('f',f)
cv2.imshow('n',n)
cv2.waitKey(0)
cv2.destroyAllWindows()
