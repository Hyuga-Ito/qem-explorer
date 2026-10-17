"""AppTest journeys. These are not browser rendering/accessibility tests."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.source).resolve();sys.path.insert(0,str(root))
import qiskit_ibm_runtime.fake_provider
from streamlit.testing.v1 import AppTest
from qemx.methods import Recipe
events=[]
def record(name,at,**fields):
 e=dict(name=name,exceptions=[x.value for x in at.exception],errors=[x.value for x in at.error],warnings=[x.value for x in at.warning],**fields)
 events.append(e);Path(a.out).write_text(json.dumps(events,indent=2,default=str));assert not at.exception
def w(at,kind,key):return next(x for x in getattr(at,kind) if x.key==key)
at=AppTest.from_file(str(root/'app.py'),default_timeout=600).run();record('home',at)
at.switch_page('pages/9_Learn.py').run();record('guides',at,guide_buttons=[b.key for b in at.button if str(b.key).startswith('guide_')])
next(b for b in at.button if str(b.key).startswith('guide_')).click().run()
at.switch_page('pages/1_Playground.py').run()
w(at,'button','sb_run_run_requested').click().run()
record('first_guide_run',at,template=at.session_state['template'],has_results=at.session_state['results'] is not None)
at.session_state['custom_recipes']=[Recipe.make('zne',rule='linear',scales=(1.,3.)).to_json()]
at.switch_page('pages/9_Learn.py').run();next(b for b in at.button if str(b.key).startswith('guide_')).click().run()
at.switch_page('pages/1_Playground.py').run();record('guide_keeps_custom',at,custom=at.session_state['custom_recipes'])
at=AppTest.from_file(str(root/'app.py'),default_timeout=600).run();at.switch_page('pages/1_Playground.py').run()
at.session_state['w_pg_tabs']='Mitigate & compare';at.run()
pick=w(at,'multiselect','w_rc_pick');nil=next(s for s in pick.options if s.startswith('NIL'))
pick.set_value(['Unmitigated',nil]).run();w(at,'button','sb_run_run_requested').click().run()
rows=at.session_state['results']['rows']
record('nil_default_budget',at,budget=at.session_state['shots'],rows=[{k:r[k] for k in ('method','training shots','target shots','failed repeats','status')} for r in rows])
w(at,'number_input','w_shots_num').set_value(12345).run();at.switch_page('pages/3_Bias_Variance.py').run();at.switch_page('pages/1_Playground.py').run()
record('budget_persists',at,budget=at.session_state['shots']);assert at.session_state['shots']==12345
print('PASS: learner journeys completed')
