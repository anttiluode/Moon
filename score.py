import numpy as np, json, sys
from pairs import REF,HELD
SIGMAX=12.0;KQ=11.0;PHI=1.6180339887
AX=np.array([[0,1,PHI],[0,-1,PHI],[1,PHI,0],[-1,PHI,0],[PHI,0,1],[PHI,0,-1]]);AX/=np.linalg.norm(AX,axis=1)[:,None]
def sstep(a,b,x):
    t=np.clip((x-a)/(b-a),0,1);return t*t*(3-2*t)
def dens(p):
    r=np.linalg.norm(p,axis=-1);m=1-sstep(0.86,1.0,r)
    q=np.cos(KQ*(p@AX.T)).sum(-1)
    return SIGMAX*m*(0.12+0.88*sstep(-0.3,1.8,q))
def hull(p): return 1-sstep(0.86,1.0,np.linalg.norm(p,axis=-1))
ACT={"id":lambda v:v,"relu":lambda v:np.maximum(v,0),"exp":np.exp,"log":np.log,"recip":lambda v:1/v}
def run(net,x):
    for L in net["layers"]:
        W=np.array(L["W"]);G=np.array(L["G"]);z=x@W.T+np.array(L["b"]);gz=x@G.T+np.array(L["c"])
        y=np.empty_like(z)
        for k,a in enumerate(L["act"]):
            y[:,k]=ACT[a](z[:,k])*(gz[:,k] if L["gate"][k] else 1)
        x=y
    return x
def norm(v):
    v=np.array(v,float);return v/np.linalg.norm(v)
def render(net,cd,s,N=48,steps=48,mode="net"):
    cd=norm(cd);s=norm(s);cam=3.4*cd;f=-cd
    rt=norm([-f[2],0,f[0]]);up=np.cross(rt,f)
    u=((np.arange(N)+.5)/N*2-1)*0.36;v=(1-(np.arange(N)+.5)/N*2)*0.36
    U,V=np.meshgrid(u,v);d=f+U[...,None]*rt+V[...,None]*up;d/=np.linalg.norm(d,axis=-1,keepdims=True)
    d=d.reshape(-1,3);b=d@cam;disc=b*b-(cam@cam-1);hit=disc>0
    out=np.zeros((N*N,4));out[:,3]=1
    dh=d[hit];sq=np.sqrt(disc[hit]);t0=-b[hit]-sq;dt=2*sq/steps
    ts=t0[:,None]+(np.arange(steps)+.5)[None,:]*dt[:,None]
    P=cam+ts[...,None]*dh[:,None,:];rho=dens(P)
    pts=P.reshape(-1,3);M=len(pts);cs=np.repeat(dh@s,steps)
    cols=[]
    for pr in net["probes"]:
        dr=np.array(pr["dir"]) if pr["frame"]=="world" else (s if pr["frame"]=="sun" else -s)
        cols.append((dens if pr["field"]=="dens" else hull)(pts+pr["dist"]*dr))
    x=np.stack(cols+[np.full(M,s[0]),np.full(M,s[1]),np.full(M,s[2]),cs,pts@s,pts[:,0],pts[:,1],pts[:,2]],1)
    if mode=="net": J=run(net,x)
    else:
        g=net["g"];nc=len(net["edges"])-1;de=np.diff(net["edges"])
        tau=sum(de[j]*cols[14*nc+j] for j in range(nc))
        J=np.zeros((M,3));J[:,0]=np.exp(-tau)*(1-g*g)/(4*np.pi*(1+g*g-2*g*cs)**1.5)
    J=J.reshape(-1,steps,3);a=1-np.exp(-rho*dt[:,None])
    Tv=np.cumprod(np.concatenate([np.ones((len(dh),1)),1-a[:,:-1]],1),1)
    L=(Tv[...,None]*a[...,None]*J).sum(1)
    o=np.zeros((hit.sum(),4));o[:,:3]=L;o[:,3]=np.prod(1-a,1);out[hit]=o
    return out.reshape(N,N,4)
E_SUN=3.0;L_SKY=0.3;ALB=0.25
def combine(im,s):
    s=norm(s);lg=ALB*(E_SUN*max(s[1],0)+np.pi*L_SKY)/np.pi
    return E_SUN*im[...,0]+L_SKY*im[...,1]+lg*im[...,2]
def load(path):
    a=np.fromfile(path,dtype=np.float32).reshape(96,96,8).astype(float)
    a=a.reshape(48,2,48,2,8).mean((1,3));return (a[...,:4]+a[...,4:])/2,a[...,:4],a[...,4:]
def score(net,split="ref",verbose=True,mode="net"):
    L=REF if split=="ref" else HELD;res=[]
    for i,(c,s) in enumerate(L):
        m,A,B=load(f"{split}/{i}.bin");im=render(net,c,s,mode=mode);mask=m[...,3]<0.5
        e=lambda x,y:np.abs(x-y)[mask].sum()/y[mask].sum()
        r=[e(combine(im,s),combine(m,s)),e(im[...,0],m[...,0]),e(im[...,1],m[...,1]),e(im[...,2],m[...,2]),
           e(combine(A,s),combine(B,s))/2, (combine(im,s)[mask].sum()/combine(m,s)[mask].sum()), im[...,0][mask].sum()/m[...,0][mask].sum()]
        res.append(r)
        if verbose:print(i,' '.join('%.3f'%v for v in r))
    res=np.array(res);print(split,mode,"MEAN comb %.3f sun %.3f sky %.3f gnd %.3f noise %.3f | ratio comb %.2f sun %.2f"%tuple(res.mean(0)))
    return res
if __name__=="__main__":
    net=json.load(open("network.json"))
    score(net,sys.argv[1] if len(sys.argv)>1 else "ref",mode=sys.argv[2] if len(sys.argv)>2 else "net")
