"""Bounded subprocess reproduction of the missing-Chrome export failure."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args()
env=dict(os.environ,PYTHONPATH=str(Path(a.source).resolve()))
code='''import io,json,zipfile
import plotly.graph_objects as go
import qemx.bundle as B
PATCH
blob=B.make_zip('export',tables={'values':[{'x':1,'y':2}]},figures={'line':go.Figure(go.Scatter(x=[1,2],y=[2,3]))},images=True)
z=zipfile.ZipFile(io.BytesIO(blob))
print(json.dumps({'members':z.namelist(),'notes':{n:z.read(n).decode() for n in z.namelist() if n.lower().endswith('.txt')}}))
'''
results=[]
for patched in [False,True]:
 start=time.monotonic()
 try:
  r=subprocess.run([sys.executable,'-c',code.replace('PATCH','B._server=lambda start: False' if patched else '')],env=env,capture_output=True,text=True,timeout=20)
  row=dict(patched=patched,returncode=r.returncode,seconds=time.monotonic()-start,stdout=r.stdout,stderr=r.stderr)
 except subprocess.TimeoutExpired as e:
  row=dict(patched=patched,timed_out=True,seconds=time.monotonic()-start,stdout=(e.stdout or b'').decode() if isinstance(e.stdout,bytes) else e.stdout,stderr=(e.stderr or b'').decode() if isinstance(e.stderr,bytes) else e.stderr)
 results.append(row);Path(a.out).write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
