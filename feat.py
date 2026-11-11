import numpy as np, sys, os
from score import dens, hull, norm
from pairs import REF, HELD
EDGES=np.array([0,.04,.1,.18,.28,.4,.55,.75,1.0,1.3,1.65,2.0]);MID=(EDGES[1:]+EDGES[:-1])/2;DE=np.diff(EDGES)
s3=1/np.sqrt(3)
DIRS=np.array([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]+[(a*s3,b*s3,c*s3) for a in(1,-1) for b in(1,-1) for c in(1,-1)])
def feats(cd,s,N=32,steps=40):
    cd=norm(cd);s=norm(s);cam=3.4*cd;f=-cd
    rt=norm([-f[2],0,f[0]]);up=np.cross(rt,f)
    u=((np.arange(N)+.5)/N*2-1)*0.36;v=(1-(np.arange(N)+.5)/N*2)*0.36
    U,V=np.meshgrid(u,v);d=f+U[...,None]*rt+V[...,None]*up;d/=np.linalg.norm(d,axis=-1,keepdims=True)
    d=d.reshape(-1,3);b=d@cam;disc=b*b-(cam@cam-1);hit=disc>0
    dh=d[hit];sq=np.sqrt(disc[hit]);t0=-b[hit]-sq;dt=2*sq/steps
    ts=t0[:,None]+(np.arange(steps)+.5)[None,:]*dt[:,None]
    P=(cam+ts[...,None]*dh[:,None,:]);rho=dens(P);pts=P.reshape(-1,3).astype(np.float32);M=len(pts)
    def tsun(x,sg=1.0): return sum(DE[j]*dens(x+sg*MID[j]*s) for j in range(11))
    X=pts[:,None,None,:]+MID[None,None,:,None]*DIRS[None,:,None,:]      # M,14,11,3
    R=dens(X).astype(np.float32);H=hull(X).astype(np.float32)
    TS=np.zeros((M,14,11),np.float32)
    for i in range(14): TS[:,i,:]=tsun(X[:,i].reshape(-1,3)).reshape(M,11)
    Xs=pts[:,None,:]+MID[None,:,None]*s; Xa=pts[:,None,:]-MID[None,:,None]*s
    return dict(hit=hit,dt=dt.astype(np.float32),rho=rho.astype(np.float32),pts=pts,cs=np.repeat(dh@s,steps).astype(np.float32),s=s,
        R=R,H=H,TS=TS,Rs=dens(Xs).astype(np.float32),Hs=hull(Xs).astype(np.float32),Ra=dens(Xa).astype(np.float32),Ha=hull(Xa).astype(np.float32),
        TSs=tsun(Xs.reshape(-1,3)).reshape(M,11).astype(np.float32),TSa=tsun(Xa.reshape(-1,3)).reshape(M,11).astype(np.float32),N=N,steps=steps)
if __name__=="__main__":
    split=sys.argv[1];L=REF if split=="ref" else HELD
    os.makedirs('cache',exist_ok=True)
    for i in [int(v) for v in sys.argv[2].split(',')]:
        np.savez(f'cache/{split}{i}.npz',**feats(*L[i]));print(i,flush=True)
