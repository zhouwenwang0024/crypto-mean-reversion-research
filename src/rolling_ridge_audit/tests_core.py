"""Actual-function assertions; new live reference never becomes fictitious cash."""
import numpy as np,pandas as pd
try:
    from .engine6 import simulate6
    from .models6 import fit_ridge, price_means, state_filter
except ImportError:
    from engine6 import simulate6
    from models6 import fit_ridge, price_means, state_filter

def toy(policy=0,fees=0,shock=None,dynamic=True):
 n=3;N=100
 op=np.full((N,n),100.);cl=op.copy()
 if shock is not None:op[shock[0]:,0]=shock[1];cl[shock[0]:,0]=shock[1]
 lc=np.log(cl);ends=np.arange(5,N,5,dtype=np.int64);D=np.zeros((N+1,n));D[:,0]=.03;D[:5,0]=0.
 D[20:,0]=0. if dynamic else .03
 SD=np.full_like(D,.01);center=lc[ends-1]-D[ends]
 # frozen anchor at entry is log100 - .03, dynamic signal becomes zero at20
 W=np.eye(n)[None,:,:];wi=np.zeros(len(ends),np.int64)
 return simulate6(op,cl,lc,np.zeros((1,1)),ends,wi,W,D[ends],center,SD[ends],np.ones(len(ends),bool),0,90,
  entry_z=2,gap_min=.015,cost_bp=fees,hold_min=60,exit_type=policy,record=True,live_dev=D,live_scale=SD)

def test_ridge_direct_equals_precision():
 r=np.random.default_rng(87).normal(size=(672,6));r[:,1:]+=.5*r[:,:1];r*=np.arange(1,7)
 W,_=fit_ridge(r);sd=r.std(0,ddof=1);z=(r-r.mean(0))/sd;co=z.T@z/(len(z)-1)
 for j in range(6):
  p=np.delete(np.arange(6),j);b=np.linalg.solve(co[np.ix_(p,p)]+.1*np.eye(5),co[p,j]);assert np.max(abs(W[j,p]+b*sd[j]/sd[p]))<1e-10

def test_future_fit_inputs_isolation():
 rng=np.random.default_rng(44);r=rng.normal(size=(1000,5));r2=r.copy();r2[700:]+=100
 assert np.allclose(fit_ridge(r[:672])[0],fit_ridge(r2[:672])[0])

def test_normalized_penalty():
 r=np.random.default_rng(20).normal(size=(1000,4));w,_=fit_ridge(r)
 w2,_=fit_ridge(np.repeat(r,2,axis=0));assert np.max(abs(w-w2))<1e-10

def test_rolling_excludes_current():
 x=np.arange(400.)[:,None];m=price_means(x,3);assert m[200,0]==np.mean(x[20:200,0])
 x2=x.copy();x2[201:]+=1e5;assert np.allclose(m[:201],price_means(x2,3)[:201],equal_nan=True)

def test_ewm_lagmatched():
 x=np.zeros((500,1));x[200,0]=1;m=price_means(x,3,'ewm');alpha=2/181
 assert abs(m[201,0]-alpha)<1e-12

def test_state_prior_not_current():
 x=np.zeros(200);x[100:]=2;x2=x.copy();x2[100]=100
 m,*_=state_filter(x,.99,.001,.0001);m2,*_=state_filter(x2,.99,.001,.0001)
 assert np.allclose(m[:101],m2[:101]);assert not np.isclose(m[101],m2[101])

def test_state_future_prefix():
 x=np.sin(np.arange(500)/20);a=state_filter(x[:300],.995,.002,.00002)[0];b=state_filter(x,.995,.002,.00002)[0]
 assert np.allclose(a,b[:300])

def test_live_exit_not_frozen():
 a=toy(0);b=toy(7);assert a[4][0,1]==66 and b[4][0,1]==21

def test_no_phantom_repricing_profit():
 a=toy(7,5);assert abs(a[4][0,7])<1e-12;assert abs(a[0][-1]-(1-a[1].sum()*.0005))<1e-12

def test_sign_correct():
 a=toy(0,0,shock=(10,99));assert a[4][0,7]>0;assert a[4][0,18]<0

def test_order_time_and_future_price():
 a=toy(7);b=toy(7,shock=(6,120));assert np.allclose(a[0][:6],b[0][:6]);assert np.allclose(a[4][0,18:21],b[4][0,18:21])
 assert np.all(a[5][:6]==0)

def test_true_timeout():
 a=toy(0);assert a[4][0,1]-a[4][0,0]==60

def test_fixed_price_cost():
 a=toy(0,10);assert np.max(np.diff(a[0]))<1e-12;assert abs(a[0][-1]-1+a[1].sum()*.001)<1e-12

def test_zero_cross_jump():
 a=toy(7);assert a[4][0,1]==21

def test_independent_account():
 a=toy(7,5,shock=(10,99));q=a[5];px=np.full_like(q,100.);px[10:,0]=99.
 out=np.ones(len(q));prevq=np.zeros(3);prevpx=px[0];cash=1.
 for t in range(len(q)):
  cash+=np.dot(prevq,px[t]-prevpx)-.0005*np.abs((q[t]-prevq)*px[t]).sum();out[t]=cash;prevq=q[t];prevpx=px[t]
 assert np.allclose(out,a[0],atol=1e-12)

if __name__=='__main__':
 import json,traceback
 try:
  from .data6 import ROOT
 except ImportError:
  from data6 import ROOT
 res=[]
 for name,f in list(globals().items()):
  if name.startswith('test_'):
   try:f();res.append(dict(test=name,passed=True));print(name,'PASS')
   except Exception as e:res.append(dict(test=name,passed=False,error=repr(e)));traceback.print_exc()
 (ROOT/'results/tests6_initial.json').write_text(json.dumps(res,indent=2));assert all(x['passed'] for x in res)
