"""Crystal-moon light-transport network. Every weight is set here by formula or by a
constant chosen by reasoning; there is no training data and no optimiser.
Rerunning this script reproduces network.json byte for byte."""
import json, math
G    = 0.5            # Henyey-Greenstein asymmetry of the medium (scene constant)
A    = 1.0 - G        # similarity scaling: transport optical depth tau* = (1-g) tau
Q    = 2.0/3.0        # diffusion extrapolation length, in transport mean free paths
BETA = 0.6            # HAND-SET against the 12 reference views: orders >=3 attenuate at 0.6 of tau*
LAM  = 0.7            # HAND-SET against the 12 reference views: ambient exit contrast kept after mixing
I4   = 1/(4*math.pi)
EDGES=[0,.04,.1,.18,.28,.4,.55,.75,1.0,1.3,1.65,2.0]; nc=len(EDGES)-1
MID=[(EDGES[j]+EDGES[j+1])/2 for j in range(nc)]; DE=[EDGES[j+1]-EDGES[j] for j in range(nc)]
s3=1/math.sqrt(3)
DIRS=[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]+[(a*s3,b*s3,c*s3) for a in(1,-1) for b in(1,-1) for c in(1,-1)]
LEB=[1/15]*6+[3/40]*8                      # Lebedev 14-point quadrature weights on the sphere (sum 1)
SKYU=[1.0 if d[1]>0 else (0.5 if d[1]==0 else 0.0) for d in DIRS]   # does direction i look at the sky
# ---- inputs: three stencils along 14 world directions, two along the sun axis, plus geometry scalars
groups=[]; names=[]
def grp(field,frame,tag):
    groups.append({"field":field,"frame":frame,"offset":len(names),"n":(14 if frame=="world" else 1)*nc})
    if frame=="world":
        for i in range(14):
            for j in range(nc): names.append(f"{tag}{i}_{j}")
    else:
        for j in range(nc): names.append(f"{tag}_{j}")
grp("dens","world","R"); grp("hull","world","H"); grp("beam","world","TS")
grp("dens","sun","Rs"); grp("hull","sun","Hs"); grp("dens","antisun","Ra"); grp("hull","antisun","Ha")
scal=["sx","sy","sz","cs","ps","px","py","pz"]; names+=scal
layers=[]
def layer(prev,units):
    idx={k:i for i,k in enumerate(prev)}; out=[]
    for u in units:
        o={"n":u["n"],"act":u.get("act","id"),"w":{str(idx[k]):v for k,v in u.get("w",{}).items() if v!=0},"b":u.get("b",0.0)}
        if "g" in u or "c" in u: o["g"]={str(idx[k]):v for k,v in u.get("g",{}).items() if v!=0}; o["c"]=u.get("c",0.0)
        out.append(o)
    layers.append({"n_in":len(prev),"units":out}); return [u["n"] for u in units]
def P(*ks): return [{"n":k,"w":{k:1.0}} for k in ks]
Rn=[f"R{i}_{j}" for i in range(14) for j in range(nc)]; TSn=[f"TS{i}_{j}" for i in range(14) for j in range(nc)]
LEN=[f"len{i}" for i in range(14)]
# L1 quadrature. tau_i: optical depth to the surface along direction i. cum_ij: optical depth from p to probe j.
u=[{"n":f"tau{i}","w":{f"R{i}_{j}":DE[j] for j in range(nc)}} for i in range(14)]
u+=[{"n":f"cum{i}_{j}","w":{**{f"R{i}_{k}":DE[k] for k in range(j)},f"R{i}_{j}":DE[j]/2}} for i in range(14) for j in range(nc)]
u+=[{"n":LEN[i],"w":{f"H{i}_{j}":DE[j] for j in range(nc)}} for i in range(14)]
u+=[{"n":"taus","w":{f"Rs_{j}":DE[j] for j in range(nc)}},{"n":"taua","w":{f"Ra_{j}":DE[j] for j in range(nc)}},
    {"n":"lens","w":{f"Hs_{j}":DE[j] for j in range(nc)}},{"n":"lena","w":{f"Ha_{j}":DE[j] for j in range(nc)}}]
x=layer(names,u+P(*Rn,*TSn,*scal))
# L2 second-scatter integrand: reduced source a*rho*exp(-a*tau_sun) at the probe, attenuated back to p
u=[]
for i in range(14):
    for j in range(nc):
        k=f"{i}_{j}"; src={"R"+k:A*DE[j]*LEB[i]}
        u.append({"n":"E"+k,"act":"exp","w":{"cum"+k:-A,"TS"+k:-A},"g":src})
        u.append({"n":"F"+k,"act":"exp","w":{"cum"+k:-A*BETA,"TS"+k:-A},"g":src})
u+=[{"n":f"e{i}","act":"exp","w":{f"tau{i}":-A}} for i in range(14)]
u+=[{"n":f"tc{i}","act":"relu","w":{"px":DIRS[i][0],"py":DIRS[i][1],"pz":DIRS[i][2],LEN[i]:1.0},"g":{f"tau{i}":A}} for i in range(14)]
u+=[{"n":"tcS","act":"relu","w":{"ps":1.0,"lens":1.0},"g":{"taus":A}},{"n":"tcA","act":"relu","w":{"ps":-1.0,"lena":1.0},"g":{"taua":A}},
    {"n":"T","act":"exp","w":{"taus":-1.0}},{"n":"Tb","act":"exp","w":{"taus":-A}},{"n":"ts","w":{"taus":1.0}}]
u+=[{"n":f"l{n}","act":"log","w":{"cs":-2*G**n},"b":1+G**(2*n)} for n in(1,2,3)]
x=layer(x,u+P(*LEN,"lens","lena","sy","py"))
# L3 sums over the stencil; escape probability; diffusion exit weights 1/(tau* cos + q); HG phase functions
u=[{"n":"S2","w":{f"E{i}_{j}":1.0 for i in range(14) for j in range(nc)}},
   {"n":"S2b","w":{f"F{i}_{j}":1.0 for i in range(14) for j in range(nc)}},
   {"n":"Pesc","w":{f"e{i}":LEB[i] for i in range(14)}},
   {"n":"A1","w":{f"e{i}":LEB[i]*SKYU[i] for i in range(14)}}]
u+=[{"n":f"w{i}","act":"recip","w":{f"tc{i}":1.0},"b":Q} for i in range(14)]
u+=[{"n":"wS","act":"recip","w":{"tcS":1.0},"b":Q},{"n":"wA","act":"recip","w":{"tcA":1.0},"b":Q}]
u+=[{"n":f"P{n}","act":"exp","w":{f"l{n}":-1.5},"b":math.log((1-G**(2*n))*I4)} for n in(1,2,3)]
u+=[{"n":"a1","w":{"T":1.0},"g":{"ts":1.0}}]
x=layer(x,u+P("T","Tb","ts",*LEN,"lens","lena","sy","py"))
# L4
u=[{"n":"invP","act":"recip","w":{"Pesc":1.0}},{"n":"omP","w":{"Pesc":-1.0},"b":1.0},
   {"n":"N","w":{**{f"w{i}":14*LEB[i] for i in range(14)},"wS":1.0,"wA":1.0}}]
u+=[{"n":f"wl{i}","w":{f"w{i}":14*LEB[i]},"g":{LEN[i]:1.0}} for i in range(14)]
u+=[{"n":"wlS","w":{"wS":1.0},"g":{"lens":1.0}},{"n":"wlA","w":{"wA":1.0},"g":{"lena":1.0}},
    {"n":"ss","w":{"T":1.0},"g":{"P1":1.0}},
    {"n":"iso1","w":{"Tb":I4,"T":-I4}},
    {"n":"gl1","w":{"a1":1.0},"g":{"P2":1.0},"c":-I4},
    {"n":"a2","w":{"a1":1.0},"g":{"ts":0.5}},
    {"n":"P3m","w":{"P3":1.0},"b":-I4}]
x=layer(x,u+P("S2","S2b","A1","Pesc","sy","py"))
# L5 build-up (1-P)/P; sky visibility of every exit, (1+n_y)/2 with n_y = py + len_i d_iy
u=[{"n":"r","w":{"omP":1.0},"g":{"invP":1.0}},{"n":"invN","act":"recip","w":{"N":1.0}},
   {"n":"SkA","w":{"N":1.0},"g":{"py":0.5},"c":0.5},
   {"n":"SkB","w":{f"wl{i}":DIRS[i][1]/2 for i in range(14)}},
   {"n":"SkC","w":{"wlS":1.0,"wlA":-1.0},"g":{"sy":0.5}},
   {"n":"gl2","w":{"a2":1.0},"g":{"P3m":1.0}}]
x=layer(x,u+P("S2","S2b","A1","Pesc","omP","ss","iso1","gl1"))
# L6
u=[{"n":"hi","w":{"r":1.0},"g":{"S2b":I4}},{"n":"Hs","w":{"SkA":1.0,"SkB":1.0,"SkC":1.0},"g":{"invN":1.0}}]
x=layer(x,u+P("S2","A1","Pesc","omP","ss","iso1","gl1","gl2"))
# L7
u=[{"n":"mix","w":{"omP":1.0},"g":{"Hs":LAM},"c":0.5*(1-LAM)}]
x=layer(x,u+P("S2","hi","A1","Pesc","omP","ss","iso1","gl1","gl2"))
# L8 readout: source function per unit sun irradiance / sky radiance / ground radiance
x=layer(x,[{"n":"sun","w":{"ss":1.0,"iso1":1.0,"S2":I4,"hi":1.0,"gl1":1.0,"gl2":1.0}},
           {"n":"sky","w":{"A1":1.0,"mix":1.0}},
           {"n":"gnd","w":{"Pesc":1.0,"A1":-1.0,"omP":1.0,"mix":-1.0}}])
net={"g":G,"edges":EDGES,"dirs":[list(d) for d in DIRS],"inputs":names,"groups":groups,"layers":layers}
json.dump(net,open("network.json","w"),separators=(",",":"))
tot=nz=0
for L in layers:
    for uu in L["units"]:
        tot+=2*L["n_in"]+2; nz+=len(uu["w"])+len(uu.get("g",{}))+(uu["b"]!=0)+(uu.get("c",0)!=0)
print("inputs",len(names),"params",tot,"nonzero",nz,"units",[len(L["units"]) for L in layers])
