import numpy as np
def assignment_inverse_gamma(e,h):
 e,h=float(e),float(h)
 if not(np.isfinite([e,h]).all() and 0<=e<=1 and 0<=h<=1):raise ValueError('Invalid readout rates')
 d=abs(1-e-h)
 return float('inf') if d==0 else (1+abs(e-h))/d
for e,h in [(0.,0.),(.03,.04),(.2,.7),(.6,.6),(.7,.6),(1.,1.),(.5,.5)]:
 g=assignment_inverse_gamma(e,h)
 if e+h==1:assert np.isinf(g)
 else:assert np.isclose(g,np.linalg.norm(np.linalg.inv([[1-e,h],[e,1-h]]),1),rtol=1e-12)
assert np.isclose(assignment_inverse_gamma(.6,.6)**2,25.)
for e,h in [(-.1,.1),(float('nan'),.1),(1.1,.1)]:
 try:assignment_inverse_gamma(e,h)
 except ValueError:pass
 else:raise AssertionError('Invalid input accepted')
print('PASS: 7 channel cases and 3 invalid-input cases')
