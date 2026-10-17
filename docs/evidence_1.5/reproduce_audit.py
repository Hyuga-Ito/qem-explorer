"""Focused executable reproductions for the QEM Explorer 1.4.2 QA report.
No external accounts, credentials, or hardware jobs are used.
"""
import argparse, json, sys, warnings, math
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--reps',type=int,default=6000)
a=p.parse_args();sys.path.insert(0,str(Path(a.source).resolve()))
from qemx import compile as CP, templates as T, budget as BG
from qemx.noise import NoiseSpec
from qemx.methods import Recipe
from qemx.recipes import compare, summary, table_rows
from qemx.methods.pec import depol_quasi, gamma_of
from qemx.methods.nil import nil_fit

report={'pec':[],'parity':[],'learning':[],'boundaries':[],'readout':[]}
def clean(value):
 if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
 if isinstance(value,list):return [clean(v) for v in value]
 if isinstance(value,float) and not math.isfinite(value):return str(value)
 return value
def save():Path(a.out).write_text(json.dumps(clean(report),indent=2,default=float,allow_nan=False))
def compile_tpl(t):return CP.build(t.spec,relabel=t.relabel)
for p2 in [0.,.02,.1,.3,.95]:
 t=T.bell();ideal,outs,_=compare(compile_tpl(t),NoiseSpec(p2=p2),t.observable,[Recipe.make('raw'),Recipe.make('pec'),Recipe.make('pec',construction='compensation'),Recipe.make('zne',rule='linear'),Recipe.make('zne',rule='richardson')],shots=4096,seed=17)
 assert abs(outs[0].result.mean-(1-p2))<1e-10
 for j,g in [(1,(15/(1-p2)-7)/8),(2,(1+p2)/(1-p2))]:
  assert abs(outs[j].result.mean-1)<1e-9
  assert np.isclose(outs[j].result.gamma,g,rtol=1e-12)
  assert np.isclose(outs[j].result.var1,g*g-1,rtol=1e-10,atol=1e-12)
 report['pec'].append(dict(p2=p2,methods=[dict(method=o.title,mean=o.result.mean,var1=o.result.var1,gamma=o.result.gamma) for o in outs]))
save()
for key,params in [('bell',{}),('ghz',{'n':3}),('qft',{'n':3})]:
 t=T.build(key,params);noise=NoiseSpec(p1=.003,p2=.02,amp_damp=.01,phase_damp=.02,over_rot=.01,ro01=.03,ro10=.04);values={}
 for backend in ['numpy','aer']:
  _,outs,_=compare(compile_tpl(t),noise,t.observable,[Recipe.make('raw'),Recipe.make('raw',readout='tensored')],backend=backend,shots=8192,seed=9)
  values[backend]=[o.result.mean for o in outs]
 error=float(np.max(np.abs(np.array(values['numpy'])-values['aer'])))
 assert error<1e-9
 report['parity'].append(dict(circuit=key,max_error=error,**values))
save()

# Reference-fit moments versus finite-shot refitting. Two independent MC streams.
t=T.build('qft',{'n':3});co=compile_tpl(t);B=100000;R=a.reps
cases=[(.03,64,Recipe.make('learning',features='cdr',model='ols',n_train=8,train_shots=64)),(.03,1024,Recipe.make('learning',features='nil',model='ridge',alpha=.001,n_train=8,train_shots=1024)),(.2,1024,Recipe.make('nil',model='l1',l1_bound=2.,n_train=8,n_neighbors=4,train_shots=1024))]
for p2,train,recipe in cases:
 ideal,outs,_=compare(co,NoiseSpec(p1=p2/10,p2=p2),t.observable,[recipe],shots=B,seed=13)
 o=outs[0];r=o.result;assert r is not None,o.error
 mu,v=r.train_resampler(train,R,np.random.default_rng(4101));mu,v=np.asarray(mu),np.asarray(v)
 x=np.asarray(BG.draw(r,B,R,np.random.default_rng(5101)))
 assert np.isfinite(mu).all() and np.isfinite(v).all() and np.isfinite(x).all()
 N,_=BG.split(BG.CostModel.of(r),B)
 m=float(mu.mean());u=float(mu.var(ddof=1));vbar=float(v.mean());sd=float(np.sqrt(u+vbar/N))
 ex=float(x.mean());esd=float(x.std(ddof=1));z=(ex-m)/np.sqrt(u/R+esd*esd/R)
 row=dict(method=o.title,ideal=ideal,p1=p2/10,p2=p2,train_shots=train,budget=B,reps=R,reference_mean=r.mean,reported_rmse=summary(o,ideal,B,250,x)['RMSE'],empirical_mean=ex,empirical_sd=esd,empirical_rmse=float(np.sqrt(np.mean((x-ideal)**2))),corrected_mean=m,corrected_var1=vbar,corrected_u=u,corrected_sd=sd,corrected_rmse=float(np.sqrt((m-ideal)**2+sd*sd)),mean_difference_mcse=float(z))
 assert abs(z)<5 and abs(sd-esd)<.1*esd
 report['learning'].append(row);save();print(json.dumps(row),flush=True)

for bad in [-.01,1.,float('nan')]:
 try:
  qs=depol_quasi(bad,1,'inverse');row=dict(p=str(bad),accepted=True,gamma=gamma_of(qs))
 except Exception as e:row=dict(p=str(bad),accepted=False,error=str(e))
 report['boundaries'].append(row)
X=np.array([[1.,1.,0.],[0.,0.,1.],[1.,1.,1.],[0.,0.,0.]])
y=np.array([1.,-.5,.5,0.])
for bound in [1.,0.,-.1]:
 c,b,info=nil_fit(X,y,'l1',l1_bound=bound)
 report['boundaries'].append(dict(nil_bound=bound,l1=float(np.abs(c).sum()),ok=info['ok']))

t=T.bell()
for pflip in [.49999,.5,.6]:
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always')
  ideal,outs,_=compare(compile_tpl(t),NoiseSpec(ro01=pflip,ro10=pflip),t.observable,[Recipe.make('raw'),Recipe.make('raw',readout='tensored')],shots=4096)
  _,rows=table_rows(outs,ideal,4096,250,repeats=500)
  report['readout'].append(dict(flip=pflip,rows=rows,errors=[o.error for o in outs if o.error],warnings=[str(w.message) for w in ws]))
save();print('DONE',flush=True)
