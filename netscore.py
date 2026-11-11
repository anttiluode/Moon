import numpy as np, json, sys
import exp
from exp import *
ACT={"id":lambda v:v,"relu":lambda v:np.maximum(v,0),"exp":np.exp,"log":np.log,"recip":lambda v:1/v}
def run_net(net,x):
    for L in net["layers"]:
        y=np.empty((len(x),len(L["units"])))
        for k,u in enumerate(L["units"]):
            z=np.full(len(x),float(u["b"]))
            for i,v in u["w"].items(): z+=v*x[:,int(i)]
            z=ACT[u["act"]](z)
            if "g" in u:
                gz=np.full(len(x),float(u["c"]))
                for i,v in u["g"].items(): gz+=v*x[:,int(i)]
                z=z*gz
            y[:,k]=z
        x=y
    return x
def inputs(F):
    M=len(F['pts']);s=F['s'];p=F['pts'].astype(float)
    return np.concatenate([F['R'].reshape(M,-1),F['H'].reshape(M,-1),F['TS'].reshape(M,-1),F['Rs'],F['Hs'],F['Ra'],F['Ha'],
        np.tile(s,(M,1)),F['cs'][:,None],(p@s)[:,None],p],1).astype(float)
if __name__=="__main__":
    net=json.load(open('network.json'));split=sys.argv[1];n=12 if split=='ref' else 6;res=[];dmax=0
    for i in range(n):
        F=dict(np.load(f'cache/{split}{i}.npz'));m=loadref(split,i,int(F['N']));s=F['s']
        J=run_net(net,inputs(F));Jm=model(F,{'beta':0.6,'lam':0.7});dmax=max(dmax,np.abs(J-Jm).max())
        im=render(F,J);Jss=model(F,{'ss':1});Jss[:,1:]=0;base=render(F,Jss);mask=m[...,3]<0.5
        e=lambda x,y:np.abs(x-y)[mask].sum()/y[mask].sum()
        a=np.fromfile(f"{split}/{i}.bin",dtype=np.float32).reshape(96,96,8).astype(float).reshape(32,3,32,3,8).mean((1,3))
        res.append([e(combine(im,s),combine(m,s)),e(im[...,0],m[...,0]),e(im[...,1],m[...,1]),e(im[...,2],m[...,2]),
                    e(combine(base,s),combine(m,s)),e(base[...,0],m[...,0]),e(combine(a[...,:4],s),combine(a[...,4:],s))/2])
        print(i,' '.join('%.3f'%v for v in res[-1]))
    print(split,'MEAN net: comb %.3f sun %.3f sky %.3f gnd %.3f | single-scatter: comb %.3f sun %.3f | MC noise %.3f'%tuple(np.mean(res,0)))
    print('max |network - reference formula| =',dmax)
