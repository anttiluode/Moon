import numpy as np, sys
from feat import *
from score import combine
G=0.5;I4=1/(4*np.pi)
LEB=np.array([1/15]*6+[3/40]*8)
def hgp(g,c): return (1-g*g)*I4/(1+g*g-2*g*c)**1.5
def loadref(split,i,N):
    a=np.fromfile(f"{split}/{i}.bin",dtype=np.float32).reshape(96,96,8).astype(float)
    k=96//N;a=a.reshape(N,k,N,k,8).mean((1,3));return (a[...,:4]+a[...,4:])/2
def model(F,p):
    a=p.get('a',1-G);q=p.get('q',2/3)
    R=F['R'];H=F['H'];pts=F['pts'];s=F['s'];ps=pts@s
    tau=(R*DE).sum(-1);ln=(H*DE).sum(-1)
    cum=np.cumsum(R*DE,-1)-0.5*R*DE
    ts=(F['Rs']*DE).sum(-1);ta=(F['Ra']*DE).sum(-1);ls=(F['Hs']*DE).sum(-1);la=(F['Ha']*DE).sum(-1)
    T=np.exp(-ts);Tst=np.exp(-a*ts)
    b=p.get('b',a)
    S2=((a*R*np.exp(-b*F['TS'])*np.exp(-a*cum)*DE).sum(-1)*LEB).sum(-1)
    Pesc=(np.exp(-a*tau)*LEB).sum(-1)
    # harmonic exit weights
    cg=np.maximum(pts@DIRS.T+ln,0);w=LEB*14/(a*tau*cg+q)
    wS=1/(a*ts*np.maximum(ps+ls,0)+q);wA=1/(a*ta*np.maximum(-ps+la,0)+q)
    Nn=w.sum(-1)+wS+wA
    ny=pts[:,1:2]+ln*DIRS[:,1];sv=(1+ny)/2
    Hsky=((w*sv).sum(-1)+wS*(1+pts[:,1]+ls*s[1])/2+wA*(1+pts[:,1]-la*s[1])/2)/Nn
    u=(DIRS[:,1]>0)+0.5*(DIRS[:,1]==0)
    A1=(np.exp(-a*tau)*LEB*u).sum(-1);A1g=Pesc-A1
    cs=F['cs'];J=np.zeros((len(pts),3))
    be=p.get('beta',1.0)
    S2b=((a*R*np.exp(-b*F['TS'])*np.exp(-be*a*cum)*DE).sum(-1)*LEB).sum(-1)
    ms=I4*(S2+(1-Pesc)/np.maximum(Pesc,1e-3)*S2b)
    gl=p.get('glow',1)*(T*ts*(hgp(G**2,cs)-I4)+T*ts*ts/2*(hgp(G**3,cs)-I4))
    J[:,0]=hgp(G,cs)*T+p.get('k1',1)*(np.exp(-b*ts)-T)*I4+ms+gl
    if p.get('ss'):J[:,0]=hgp(G,cs)*T
    lam=p.get('lam',1.0);Hsky=0.5+lam*(Hsky-0.5);J[:,1]=A1+(1-Pesc)*Hsky;J[:,2]=A1g+(1-Pesc)*(1-Hsky)
    if p.get('amb')=='h':J[:,1]=Hsky;J[:,2]=1-Hsky
    return J
def render(F,J):
    st=int(F['steps']);J=J.reshape(-1,st,3);rho=F['rho'];a=1-np.exp(-rho*F['dt'][:,None])
    Tv=np.cumprod(np.concatenate([np.ones((len(a),1)),1-a[:,:-1]],1),1)
    L=(Tv[...,None]*a[...,None]*J).sum(1);N=int(F['N']);out=np.zeros((N*N,3));out[F['hit']]=L;return out.reshape(N,N,3)
CACHE={}
def run(p,split='ref',cases=range(12),verbose=True):
    res=[]
    for i in cases:
        if (split,i) not in CACHE:
            F=dict(np.load(f'cache/{split}{i}.npz'));CACHE[(split,i)]=(F,loadref(split,i,int(F['N'])))
        F,m=CACHE[(split,i)];im=render(F,model(F,p));mask=m[...,3]<0.5;s=F['s']
        e=lambda x,y:np.abs(x-y)[mask].sum()/y[mask].sum()
        res.append([e(combine(im,s),combine(m,s)),e(im[...,0],m[...,0]),e(im[...,1],m[...,1]),e(im[...,2],m[...,2]),im[...,0][mask].sum()/m[...,0][mask].sum()])
        if verbose:print(i,' '.join('%.3f'%v for v in res[-1]))
    r=np.mean(res,0);print(p,'MEAN comb %.3f sun %.3f sky %.3f gnd %.3f sunratio %.2f'%tuple(r));return np.array(res)
if __name__=="__main__":
    run({});run({'amb':'h'},verbose=False);run({'ss':1},verbose=False)
