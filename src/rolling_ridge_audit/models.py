"""Causal peer-implied reference prices. State at integer t sees close[t-1]."""
from pathlib import Path
import json,time
import numpy as np,pandas as pd
from scipy.optimize import minimize
try:
    from sklearn.covariance import LedoitWolf
except ImportError:  # optional; the primary Ridge replay does not need sklearn
    LedoitWolf = None
from numba import njit
try:
    from .data6 import ROOT,N,load,SYMBOLS
except ImportError:
    from data6 import ROOT,N,load,SYMBOLS
C=ROOT/'tmp'/'rolling_ridge_cache';R=ROOT/'results'

SPECS=[
 {'id':'ridge_sma3','fit':'ridge','update':1440,'hours':3,'anchor':'sma'},
 {'id':'pca2_sma3','fit':'pca2','update':1440,'hours':3,'anchor':'sma'},
 {'id':'pca3_sma3','fit':'pca3','update':1440,'hours':3,'anchor':'sma'},
 {'id':'pca5_sma3','fit':'pca5','update':1440,'hours':3,'anchor':'sma'},
 {'id':'ridge_recent_hourly','fit':'weighted','update':60,'hours':3,'anchor':'sma'},
 {'id':'ridge_huber','fit':'huber','update':1440,'hours':3,'anchor':'sma'},
 {'id':'ridge_multiscale','fit':'blend','update':1440,'hours':3,'anchor':'sma'},
 {'id':'ledoit_wolf','fit':'lw','update':1440,'hours':3,'anchor':'sma'},
 {'id':'ridge_sma5','fit':'ridge','update':1440,'hours':5,'anchor':'sma'},
 {'id':'ridge_ewm_matched','fit':'ridge','update':1440,'hours':3,'anchor':'ewm'},
 {'id':'ridge_partial_state','fit':'ridge','update':1440,'hours':3,'anchor':'state'},
]

def fit_ridge(ret,method='ridge'):
 n,p=ret.shape
 sd=ret.std(0,ddof=1);mu=ret.mean(0);assert np.all(sd>0) and np.isfinite(ret).all()
 z=(ret-mu)/sd
 if method=='weighted':
  ww=2.**(-np.arange(n-1,-1,-1)/168);ww/=ww.sum();mu=np.sum(ret*ww[:,None],0)
  sd=np.sqrt(np.sum((ret-mu)**2*ww[:,None],0)/(1-np.sum(ww*ww)))
  z=(ret-mu)/sd;co=(z*ww[:,None]).T@z/(1-np.sum(ww*ww))
 elif method=='lw':
  if LedoitWolf is None:
   raise RuntimeError('ledoit_wolf requires scikit-learn; install requirements.txt')
  obj=LedoitWolf(assume_centered=True).fit(z);co=obj.covariance_;meta={'shrinkage':float(obj.shrinkage_)}
 else:co=z.T@z/(n-1)
 if method!='huber':
  K=np.linalg.inv(co+ (0.0 if method=='lw' else .1)*np.eye(p))
  W=K/np.diag(K)[:,None]*sd[:,None]/sd[None,:]
  return W, (meta if method=='lw' else {})
 # Huber residual-reweighted loss; regularization is normalized, no forward clipping.
 W=np.eye(p);diag=[]
 for j in range(p):
  ids=np.delete(np.arange(p),j);x=z[:,ids];y=z[:,j];b=np.linalg.solve(co[np.ix_(ids,ids)]+.1*np.eye(p-1),co[ids,j]);a=0.
  for _ in range(5):
   e=y-a-x@b;es=max(1e-6,1.4826*np.median(np.abs(e-np.median(e))))
   wt=np.minimum(1.,1.5*es/np.maximum(np.abs(e),1e-12));wt/=wt.sum()
   xm=wt@x;ym=wt@y;xx=x-xm
   b=np.linalg.solve((xx*wt[:,None]).T@xx+.1*np.eye(p-1),(xx*wt[:,None]).T@(y-ym));a=ym-xm@b
  W[j,ids]=-b*sd[j]/sd[ids];diag.append(float((np.abs(e)>1.5*es).mean()))
 return W,{'huber_downweight_fraction':float(np.mean(diag))}

def fit_pca(ret,k):
 """Leave-one-target-out PCA residual hedge on standardized returns.

 The factor fit uses only the same 28-day hourly history as Ridge.  The
 normalized covariance penalty is 0.1, expressed in the unnormalized
 least-squares system as 0.1*(n-1).  Because the inputs are demeaned, the
 cumulative level representation needs no look-ahead intercept.
 """
 n,p=ret.shape
 sd=ret.std(0,ddof=1);mu=ret.mean(0)
 if not np.all(sd>0) or not np.isfinite(ret).all():
  raise ValueError('invalid PCA fit window')
 z=(ret-mu)/sd;W=np.zeros((p,p));ev=[]
 for j in range(p):
  ids=np.delete(np.arange(p),j);zp=z[:,ids]
  _,sv,vh=np.linalg.svd(zp,full_matrices=False)
  kk=min(int(k),vh.shape[0]);v=vh[:kk].T
  factors=zp@v; y=z[:,j]
  # penalty=.1 on covariance (F'F/(n-1)), same scale as fit_ridge.
  beta=np.linalg.solve(factors.T@factors + .1*(n-1)*np.eye(kk),
                       factors.T@y)
  W[j,j]=1.;W[j,ids]=-(sd[j]/sd[ids])*(v@beta)
  ev.append(float(np.sum(sv[:kk]**2)/np.sum(sv**2)))
 return W,{'pca_factors':int(k),'pca_explained_variance':float(np.mean(ev))}

def price_means(lc,hours,kind='sma'):
 n=int(hours*60)
 if kind=='ewm':return pd.DataFrame(lc).ewm(span=n,adjust=False,min_periods=n).mean().shift(1).to_numpy()
 return pd.DataFrame(lc).rolling(n,min_periods=n).mean().shift(1).to_numpy()

@njit(cache=True)
def state_nll(s,rho,qm,qr):
 # x=(mean-reverting, permanent); s=x0+x1+observation error.
 x0=0.;x1=s[0];p00=qm/max(1-rho*rho,1e-5);p11=1.;p01=0.;out=0.
 obs=1e-8
 for i in range(1,len(s)):
  x0=rho*x0;a=rho*rho*p00+qm;b=rho*p01;c=p11+qr
  v=a+2*b+c+obs;e=s[i]-x0-x1
  k0=(a+b)/v;k1=(b+c)/v
  x0+=k0*e;x1+=k1*e
  p00=a-k0*(a+b);p01=b-k0*(b+c);p11=c-k1*(b+c)
  out+=np.log(v)+e*e/v
 return out/(len(s)-1)

@njit(cache=True)
def state_filter(s,rho,qm,qr):
 x0=0.;x1=s[0];p00=qm/max(1-rho*rho,1e-8);p11=1.;p01=0.;obs=1e-10
 prior_mean=np.empty(len(s));filtered_transitory=np.empty(len(s));prior_sd=np.empty(len(s))
 for i in range(len(s)):
  x0=rho*x0;a=rho*rho*p00+qm;b=rho*p01;c=p11+qr
  # Reference at i uses permanent prediction BEFORE seeing target s[i].
  prior_mean[i]=x1;prior_sd[i]=np.sqrt(max(0.,c))
  v=max(1e-14,a+2*b+c+obs);e=s[i]-x0-x1;k0=(a+b)/v;k1=(b+c)/v
  x0+=k0*e;x1+=k1*e
  p00=max(0.,a-k0*(a+b));p01=b-k0*(b+c);p11=max(0.,c-k1*(b+c))
  filtered_transitory[i]=x0
 return prior_mean,filtered_transitory,prior_sd

def fit_state(s):
 # Parameters fit only to past hourly observations, conditional on a past-fitted hedge.
 dr=np.diff(s);v=max(float(np.var(dr)),1e-10);s=(s-s[0])/np.sqrt(v)
 f=lambda x:state_nll(s,float(x[0]),float(np.exp(x[1])),float(np.exp(x[2])))
 bounds=[(.01,.99999),(-12.,5.),(-12.,5.)]
 starts=[(.7,np.log(.5),np.log(.5)),(.98,np.log(.2),np.log(.8))]
 fits=[minimize(f,np.array(x),method='L-BFGS-B',bounds=bounds,options={'maxiter':60,'ftol':1e-7}) for x in starts]
 res=min(fits,key=lambda x:x.fun)
 return float(res.x[0]),float(np.exp(res.x[1])*v),float(np.exp(res.x[2])*v),bool(res.success),float(res.fun)

def make(spec,force=False):
 name=spec['id'];out=C/(name+'.npz')
 C.mkdir(parents=True,exist_ok=True)
 tag=json.dumps(spec,sort_keys=True,separators=(',',':'))
 if out.exists() and not force:
  try:
   with np.load(out,allow_pickle=False) as old:
    if str(old['spec_tag'].item()) == tag:return out
  except (KeyError,ValueError, OSError):
   pass
 t0=time.time();op,cl,lc,vol,buy=load();p=cl.shape[1]
 up=spec['update'];times=np.arange(29*1440,N+1,up,dtype=np.int64)
 W=[];fitmeta=[]
 for t in times:
  def hist(f):
   ee=np.arange(f,t+1,f,dtype=np.int64)[-(28*1440//f+1):]
   assert len(ee)==28*1440//f+1
   return np.diff(lc[ee-1],axis=0)
  if spec['fit']=='blend':
   w=np.mean(np.stack([fit_ridge(hist(f))[0] for f in (15,60,240)]),axis=0);mm={}
  elif spec['fit'].startswith('pca'):
   w,mm=fit_pca(hist(60),int(spec['fit'][3:]))
  else:w,mm=fit_ridge(hist(60),spec['fit'])
  assert np.isfinite(w).all() and np.allclose(np.diag(w),1.)
  W.append(w);fitmeta.append(dict(time=int(t),day=t//1440,model=name,gross_median=float(np.median(abs(w).sum(1))),**mm))
 W=np.stack(W)
 dev=np.full((N+1,p),np.nan);ctr=np.full_like(dev,np.nan);sig=np.full_like(dev,np.nan)
 wi=np.maximum(0,np.searchsorted(times,np.arange(N+1),side='right')-1)
 if spec['anchor']!='state':
  means=price_means(lc,spec['hours'],spec['anchor']);delta=lc-means
  covs={d:np.cov(delta[d*1440-7*1440:d*1440],rowvar=False) for d in range(31,185)}
  for k,t in enumerate(times):
   if t<31*1440:continue
   stop=min(t+up,N+1);idx=np.arange(t,stop);ii=idx-1;d=t//1440;w=W[k]
   dev[idx]=delta[ii]@w.T;ctr[idx]=means[ii]@w.T
   sig[idx]=np.sqrt(np.maximum(1e-18,np.einsum('ij,jk,ik->i',w,covs[d],w)))
 else:
  sm=[]
  for k,t in enumerate(times):
   if t<31*1440 or t>=N:continue
   stop=min(t+1440,N+1);ii=np.arange(t,stop)-1;w=W[k];past_ee=np.arange(60,t+1,60)[-673:]
   hs=lc[past_ee-1]@w.T
   # Run causal filter on prior 7 complete days, then the current day in current coordinates.
   lo=t-7*1440-1;histidx=np.arange(lo,stop-1);s=lc[histidx]@w.T
   for j in range(p):
    rho,qm,qr,ok,nll=fit_state(hs[:,j]);ph=rho**(1/60);qm_m=qm*(1-ph*ph)/(1-rho*rho);qr_m=qr/60
    m,tr,unc=state_filter(s[:,j],ph,qm_m,qr_m)
    res=s[:,j]-m
    ss=np.std(res[1:7*1440+1],ddof=1)
    take=7*1440
    ctr[ii+1,j]=m[take:take+len(ii)];dev[ii+1,j]=res[take:take+len(ii)];sig[ii+1,j]=max(ss,1e-8)
    sm.append(dict(day=t//1440,symbol=SYMBOLS[j],rho_hour=rho,q_mr=qm,q_rw=qr,half_life_hours=np.log(.5)/np.log(rho),fit_ok=ok,nll=nll,mr_increment_share=(2*qm/(1+rho))/(2*qm/(1+rho)+qr)))
   if k%15==0:print('statefit',t//1440,round(time.time()-t0,1),flush=True)
  pd.DataFrame(sm).to_csv(R/'state_model_parameters.csv',index=False)
 np.savez_compressed(out,dev=dev,center=ctr,scale=sig,W=W,times=times,wi=wi,spec_tag=np.array(tag))
 pd.DataFrame(fitmeta).to_csv(R/(name+'_coefficients.csv'),index=False)
 print('FEATURE',name,'seconds',round(time.time()-t0,1),flush=True)
 return out

def load_feat(name):return dict(np.load(C/(name+'.npz')))
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='all');a=ap.parse_args()
 for s in SPECS:
  if a.model in ('all',s['id']):make(s)
