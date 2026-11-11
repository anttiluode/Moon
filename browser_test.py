import asyncio, json, sys, numpy as np
from playwright.async_api import async_playwright
from pairs import REF, HELD
from score import combine
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(args=["--use-gl=angle","--use-angle=swiftshader","--enable-unsafe-swiftshader","--ignore-gpu-blocklist"])
        pg=await b.new_page(viewport={"width":1200,"height":900})
        logs=[];pg.on("console",lambda m:logs.append(m.text));pg.on("pageerror",lambda e:logs.append("ERR "+str(e)))
        await pg.goto("file:///home/claude/moon/page/moon.html");await pg.wait_for_timeout(6000)
        print("logs",logs[:5]);print("read:",await pg.inner_text("#read") if await pg.query_selector("#read") else "none")
        print("stats:",await pg.inner_text("#stats"))
        await pg.screenshot(path="/tmp/claude-0/-home-claude/b4489859-d02b-575c-b904-fa93e949d105/scratchpad/shot.png")
        if len(sys.argv)>1:
            for split,L in (("ref",REF),("held",HELD)):
                res=[]
                for i,(c,s) in enumerate(L):
                    ch=[]
                    for k in (1,2,3,4):
                        a=np.array(await pg.evaluate(f"window.__moon.channel({list(c)},{list(s)},{k})"),dtype=float).reshape(96,96,4)[::-1]
                        ch.append((a[...,0]+a[...,1]/255)/255*(1/3 if k==1 else 1))
                    im=np.stack(ch,-1)
                    m=np.fromfile(f"{split}/{i}.bin",dtype=np.float32).reshape(96,96,8).astype(float);m=(m[...,:4]+m[...,4:])/2
                    blk=lambda x:x.reshape(32,3,32,3,-1).mean((1,3))
                    im=blk(im);m=blk(m);mask=m[...,3]<0.5;sn=np.array(s)/np.linalg.norm(s)
                    e=lambda x,y:np.abs(x-y)[mask].sum()/y[mask].sum()
                    res.append([e(combine(im,sn),combine(m,sn)),e(im[...,0],m[...,0]),e(im[...,1],m[...,1])])
                print(split,"browser MEAN comb %.3f sun %.3f sky %.3f"%tuple(np.mean(res,0)))
        await b.close()
asyncio.run(main())
