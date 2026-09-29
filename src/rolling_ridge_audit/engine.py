"""Minute-clock futures quantity ledger. Research assumptions, not exchange fills.
Signal completion t -> fixed quantity using close[t-1] -> fill open[t+1].
Frozen residual monitor may continue moving because peer prices keep moving.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def simulate6(op,cl,logcl,fund,ends,day,W,dev,center,scale,valid,
             start,end,entry_z=2.,gap_min=.005,structure=0,exit_type=0,
             repair=.5,hold_min=720,profit_bp=25.,cost_bp=2.,include_funding=False,
             allowed=None,record=False,stop_fraction=.03,entry_confirm_minutes=0,
             eligible=None,hedge_fraction=0.,repair_profit_bp=0.,
             stale_minutes=0,stale_loss_bp=50.,gap_max=1.,
             cont_policy=None,cont_minute=60,entry_delay=1,
             live_dev=None,live_scale=None,live_pred=None,exit_gap_bp=25.,
             entry_pred_min_bp=-1e10,exit_pred_bp=0.,anchor_logmeans=None):
    # structure 0=three singles, 1=two targets in one pair, 2=half hedge, 3=full hedge
    # exit_type 0=frozen residual, 1=fixed time, 2=actual lot PnL target, 3=rolling z control
    n=op.shape[1]; ns=end-start;slots=1 if structure==1 else 3
    qty=np.zeros((slots,n));qw=np.zeros((slots,n));entrypx=np.zeros((slots,n))
    qm=np.zeros(slots);qd=np.zeros(slots);qz=np.zeros(slots);qgap=np.zeros(slots)
    qtarget=np.full(slots,-1);qother=np.full(slots,-1);qt=np.zeros(slots,np.int64);qsignal=np.zeros(slots,np.int64)
    active=np.zeros(slots,np.bool_);due=np.full(slots,-1,np.int64);kind=np.zeros(slots,np.int64)
    entryfee=np.zeros(slots);gros=np.zeros(slots);fpnl=np.zeros(slots)
    minp=np.zeros(slots);maxp=np.zeros(slots);reason=np.zeros(slots,np.int64)
    cont_checked=np.zeros(slots,np.bool_)
    armed=np.ones(n,np.bool_)
    netqty=np.zeros(n);nav=1.;eq=np.ones(ns+1);tv=np.zeros(ns+1);gross=np.zeros(ns+1);net=np.zeros(ns+1)
    tape=np.zeros((ns+1 if record else 0,n))
    # Each signal can open at most 3 lots. Closed-lot records include full quantities and reference weights.
    rec=np.zeros((3*(np.searchsorted(ends,end)-np.searchsorted(ends,start)+10),18+2*n));nr=0
    k=np.searchsorted(ends,start)
    prev=op[start].copy();halt=False
    for t in range(start,end+1):
        it=t-start
        px=cl[end-1] if t==end else op[t]
        nav += np.dot(netqty,px-prev)
        prev=px.copy()
        if include_funding and t<end:
            fc=0.
            for a in range(slots):
                if active[a]:
                    f=np.dot(qty[a],fund[t]);fpnl[a]+=f;fc+=f
            nav+=fc
        # Terminal close is an administrative boundary, predeclared, not a signal-dependent favorable fill.
        if t==end:
            for a in range(slots):
                if active[a]:due[a]=t;kind[a]=-1;reason[a]=4
                else:due[a]=-1;kind[a]=0
        delta=np.zeros(n);actgross=np.zeros(slots);gs=0.
        for a in range(slots):
            if due[a]==t:
                sgn=1. if kind[a]==1 else -1.
                delta+=sgn*qty[a];gg=np.sum(np.abs(qty[a]*px));actgross[a]=gg;gs+=gg
        turn=np.sum(np.abs(delta*px));fee=turn*cost_bp/10000.;nav-=fee;tv[it]=turn
        for a in range(slots):
            if due[a]!=t:continue
            af=fee*actgross[a]/gs if gs>0 else 0.
            if kind[a]==1:
                active[a]=True;entrypx[a]=px.copy();qt[a]=t;entryfee[a]=af
                gros[a]=actgross[a];fpnl[a]=0.;minp[a]=0.;maxp[a]=0.;cont_checked[a]=False
            else:
                pnl=np.dot(qty[a],px-entrypx[a]);den=max(gros[a],1e-30)
                minp[a]=min(minp[a],pnl/den);maxp[a]=max(maxp[a],pnl/den)
                rec[nr,0]=qt[a];rec[nr,1]=t;rec[nr,2]=qtarget[a];rec[nr,3]=qother[a]
                rec[nr,4]=qz[a];rec[nr,5]=qgap[a];rec[nr,6]=gros[a];rec[nr,7]=pnl
                rec[nr,8]=entryfee[a]+af;rec[nr,9]=fpnl[a];rec[nr,10]=reason[a]
                rec[nr,11]=minp[a];rec[nr,12]=maxp[a];rec[nr,13]=qd[a]
                rec[nr,14]=qsignal[a];rec[nr,15]=qm[a];rec[nr,16]=np.sum(np.abs(qty[a]*px))+gros[a]
                rec[nr,17]=np.sum(qty[a]*entrypx[a])
                rec[nr,18:18+n]=qty[a];rec[nr,18+n:18+2*n]=qw[a];nr+=1
                active[a]=False;qty[a]=0.
            due[a]=-1;kind[a]=0
        netqty[:]=0.
        for a in range(slots):
            if active[a]:netqty+=qty[a]
        eq[it]=nav
        if nav>0:
            gross[it]=np.sum(np.abs(netqty*px))/nav;net[it]=np.sum(netqty*px)/nav
        else:halt=True
        if record:tape[it]=netqty
        if t==end:break
        # Minute-close risk/exit monitoring. A new fill at t cannot be marked to a pre-fill close.
        for a in range(slots):
            if not active[a] or due[a]>=0 or qt[a]>=t:continue
            pval=np.dot(qty[a],cl[t-1]-entrypx[a]);pfrac=pval/max(gros[a],1e-30)
            minp[a]=min(minp[a],pfrac);maxp[a]=max(maxp[a],pfrac)
            tim=t+1-qt[a]>=hold_min
            hit=False
            if exit_type==0 or exit_type==4:
                dcur=np.dot(qw[a],logcl[t-1])-qm[a]
                hit=np.sign(qd[a])*dcur <= (1.-repair)*abs(qd[a])
                if exit_type==4: hit=hit and pfrac>=repair_profit_bp/10000.
            elif exit_type==2:hit=pfrac>=profit_bp/10000.
            elif exit_type==5:  # current relationship, fixed fraction threshold only
                support=np.sign(qd[a])*live_dev[t,qtarget[a]]
                hit=support <= (1.-repair)*abs(qd[a])
            elif exit_type==6:  # fully current fair and current risk scale, no entry anchor
                support=np.sign(qd[a])*live_dev[t,qtarget[a]]
                hit=support <= .5*live_scale[t,qtarget[a]]
            elif exit_type==7:
                support=np.sign(qd[a])*live_dev[t,qtarget[a]]
                hit=support<=0.
            elif exit_type==8:
                support=np.sign(qd[a])*live_dev[t,qtarget[a]]
                hit=support <= np.log1p(exit_gap_bp/10000.)
            elif exit_type==9 and live_pred is not None:
                # future actual target-price return forecast, not a tradable model gap
                forecast=-np.sign(qd[a])*live_pred[t,qtarget[a]]
                hit=np.isfinite(forecast) and forecast<=exit_pred_bp/10000.
            elif exit_type==10 and anchor_logmeans is not None: # frozen coefficients, but rolling historical center: diagnostic only
                dcur=np.dot(qw[a],logcl[t-1]-anchor_logmeans[t-1])
                hit=np.sign(qd[a])*dcur <= (1.-repair)*abs(qd[a])
            if stale_minutes>0 and t-qt[a]>=stale_minutes and maxp[a]<=0 and pfrac<=-stale_loss_bp/10000.:
                reason[a]=5;hit=True
            # Treat the economic stop boundary as inclusive despite binary
            # representation differences after quote-unit changes.
            if halt or pfrac<=-stop_fraction+1e-12:reason[a]=1;hit=True
            elif tim:reason[a]=2;hit=True
            elif hit and not (stale_minutes>0 and t-qt[a]>=stale_minutes and maxp[a]<=0 and pfrac<=-stale_loss_bp/10000.):reason[a]=0 if exit_type in (0,4,5,6,7,8,9,10) else 3
            if cont_policy is not None and not cont_checked[a] and t-qt[a]>=cont_minute:
                cont_checked[a]=True
                dr=np.dot(qw[a],logcl[t-1])-qm[a]
                progress=1.-np.sign(qd[a])*dr/max(abs(qd[a]),1e-20)
                b=0 if pfrac<-.01 else (1 if pfrac<0 else (2 if pfrac<.01 else 3))
                b2=1 if progress>=.25 else 0
                if cont_policy[b,b2] and not hit:hit=True;reason[a]=6
            if hit and t+1<end:due[a]=t+1;kind[a]=-1
        if k>=len(ends) or ends[k]!=t:continue
        kk=k;k+=1
        if t+1>=end or halt or not valid[kk]:continue
        if not np.isfinite(dev[kk]).all() or not np.isfinite(scale[kk]).all():continue
        z=dev[kk]/scale[kk];used=np.zeros(n,np.bool_)
        for a in range(slots):
            if active[a] or kind[a]==1:
                used[qtarget[a]]=True
                if qother[a]>=0:used[qother[a]]=True
            if exit_type==3 and active[a] and due[a]<0:
                if abs(z[qtarget[a]])<.5 and t+1<end:
                    due[a]=t+1;kind[a]=-1;reason[a]=0
        for j in range(n):
            if not used[j] and abs(z[j])<1.:armed[j]=True
        ok=np.zeros(n,np.bool_)
        for j in range(n):
            ok[j]=armed[j] and not used[j] and abs(z[j])>=entry_z and abs(np.expm1(dev[kk,j]))>=gap_min
            if allowed is not None and not allowed[day[kk],j]:ok[j]=False
            if eligible is not None and not eligible[kk,j]:ok[j]=False
            if abs(np.expm1(dev[kk,j]))>gap_max:ok[j]=False
            if live_pred is not None and entry_pred_min_bp>-1e9:
                fore=-np.sign(z[j])*live_pred[t,j]
                if not np.isfinite(fore) or fore<entry_pred_min_bp/10000.:ok[j]=False
            if ok[j] and entry_confirm_minutes>0:
                if t<=entry_confirm_minutes:ok[j]=False
                else:
                    recent=np.dot(W[day[kk],j],logcl[t-1]-logcl[t-1-entry_confirm_minutes])
                    if recent*np.sign(dev[kk,j])>=0:ok[j]=False
        # Entry sizing uses prices and account equity at the last completed minute, never next fill.
        known=cl[t-1];eqknown=nav+np.dot(netqty,known-px)
        existgross=0.
        for a in range(slots):
            if active[a] or kind[a]==1:existgross+=np.sum(np.abs(qty[a]*known))
        if eqknown<=0:continue
        if structure==1:
            if active[0] or kind[0]!=0:continue
            cheap=-1;rich=-1
            for j in range(n):
                if ok[j] and z[j]<0 and (cheap<0 or z[j]<z[cheap]):cheap=j
                if ok[j] and z[j]>0 and (rich<0 or z[j]>z[rich]):rich=j
            if cheap<0 or rich<0:continue
            w=np.zeros(n);w[cheap]=.5;w[rich]=-.5
            qty[0]=.9*eqknown*w/known
            qw[0]=W[day[kk],rich]-W[day[kk],cheap]
            qm[0]=center[kk,rich]-center[kk,cheap];qd[0]=dev[kk,rich]-dev[kk,cheap]
            qz[0]=min(abs(z[rich]),abs(z[cheap]));qgap[0]=min(abs(np.expm1(dev[kk,rich])),abs(np.expm1(dev[kk,cheap])))
            qtarget[0]=cheap;qother[0]=rich;qsignal[0]=t;kind[0]=1;due[0]=t+entry_delay
            armed[cheap]=False;armed[rich]=False
        else:
            order=np.argsort(-np.abs(z))
            for j in order:
                if not ok[j]:continue
                free=-1
                for a in range(slots):
                    if not active[a] and kind[a]==0:free=a;break
                if free<0:break
                budget=max(0.,min(.3*eqknown,.9*eqknown-existgross))
                if budget<=eqknown*1e-8:break
                w=np.zeros(n);w[j]=1.
                if structure in (2,3) or hedge_fraction>0:
                    hedge=hedge_fraction if hedge_fraction>0 else (.5 if structure==2 else 1.)
                    for jj in range(n):
                        if jj!=j:w[jj]=hedge*W[day[kk],j,jj]
                w*= -np.sign(z[j]);w/=np.sum(np.abs(w))
                a=free;qty[a]=budget*w/known;existgross+=budget
                qw[a]=W[day[kk],j].copy();qm[a]=center[kk,j];qd[a]=dev[kk,j]
                qz[a]=z[j];qgap[a]=np.expm1(dev[kk,j]);qtarget[a]=j;qother[a]=-1;qsignal[a]=t
                due[a]=t+entry_delay;kind[a]=1;armed[j]=False
    return eq,tv,gross,net,rec[:nr],tape
