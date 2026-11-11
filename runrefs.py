import subprocess,os
from pairs import REF,HELD
os.makedirs('ref',exist_ok=True);os.makedirs('held',exist_ok=True)
for name,L in(('ref',REF),('held',HELD)):
    for i,(c,s) in enumerate(L):
        subprocess.run(['./pt',*map(str,c),*map(str,s),'96','1536',f'{name}/{i}.bin'],check=True)
print('done')
