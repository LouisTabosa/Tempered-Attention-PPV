"""
Pilot test with field data (one campaign in an open-pit iron-ore mine).
One signature hole (1 hole, 153 kg) and four blasts of 6 and 15 holes (153 kg per delay),
recorded at the same 7 stations. Uses the segments already cut and corrected in the
Waveform-Simulation-PPV package (1.Data/RESULTS/segments_signature) and blast_data.xlsx.

Usage: python real_data_test.py <path to Waveform-Simulation-PPV-main>
"""
import os, sys
REPO = sys.argv[1] if len(sys.argv) > 1 else 'Waveform-Simulation-PPV-main'
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
# per-segment table: PPV, signal-to-noise ratio, dominant and centroid frequencies
import numpy as np, json, pandas as pd
SEG=os.path.join(REPO,'1.Data','RESULTS','segments_signature')
idx=json.load(open(SEG+'/index.json'))
N={'Signature':1,'Test_2':6,'Test_1':15,'Test_3':15,'Test_4':15}
rows=[]
for e in idx:
    d=np.genfromtxt(f"{SEG}/{e['rec']}_{e['blast']}.csv",delimiter=',',names=True)
    t=d['Time_s']; fs=e['fs_hz']
    comps=[d[c]-np.mean(d[c][:int(0.3*fs)]) for c in ('Tran_mmps','Vert_mmps','Long_mmps')]
    pk=max(np.abs(c).max() for c in comps)
    noise=max(np.abs(c[:int(0.3*fs)]).max() for c in comps)
    # dominant frequency of the component with the peak
    k=int(np.argmax([np.abs(c).max() for c in comps])); c=comps[k]
    sp=np.abs(np.fft.rfft(c*np.hanning(len(c)))); fr=np.fft.rfftfreq(len(c),1/fs)
    m=(fr>2)&(fr<100); fdom=fr[m][np.argmax(sp[m])]
    # spectral centroid 2-100 Hz
    fc=(fr[m]*sp[m]**2).sum()/(sp[m]**2).sum()
    rows.append(dict(rec=e['rec'],blast=e['blast'],N=N[e['blast']],D=e['xlsx_geo']['dist_m'],
                     xd=e['xlsx_geo']['xd'],yd=e['xlsx_geo']['yd'],ppv=pk,ppv_seg=max(e['ppv_seg'].values()),
                     snr=pk/noise,fdom=fdom,fcent=fc))
df=pd.DataFrame(rows); df.to_csv('field_table.csv',index=False)
pd.set_option('display.width',200)
print(df.sort_values(['rec','blast']).round(2).to_string())

# single-hole site law, equal-delay attention law (tau fitted leaving one blast out) and metrics
import pandas as pd, numpy as np, json
df=pd.read_csv('field_table.csv')
xl=pd.read_excel(os.path.join(REPO,'1.Data','inputs','blast_data.xlsx'))
xl['ppv']=xl[['vt','vv','vl']].max(axis=1)
# column names below are the original (Portuguese) headers of blast_data.xlsx
A=xl[(xl.Local.str.startswith('Area_A'))&(xl['Nº de furos']==1)&(xl.Data.dt.date.astype(str)!='2021-06-22')]
X=np.column_stack([np.ones(len(A)),np.log(A['Distância']/np.sqrt(A['Q/retardo']))])
lnK,mb=np.linalg.lstsq(X,np.log(A.ppv),rcond=None)[0]; beta=-mb
print(f'Single-hole site law (n={len(A)}): lnK={lnK:.2f}, beta={beta:.2f}, alpha=0.5 fixed')
good=~df.rec.isin(['UM10457','UM10458'])
sig=df[(df.blast=='Signature')&good].set_index('rec')
T=df[(df.blast!='Signature')&good].copy()
W=153.0
T['law']=lnK-beta*np.log(T.D/np.sqrt(W))
T['ref']=[np.log(sig.loc[r,'ppv'])-beta*np.log(d/sig.loc[r,'D']) for r,d in zip(T.rec,T.D)]
T['y']=np.log(T.ppv); T['lnN']=np.log(T.N)
ab=0.5*beta
def lobo_tau(base):
    p=np.empty(len(T))
    for b in T.blast.unique():
        tr=(T.blast!=b)&(T.D>=30); te=(T.blast==b).values
        tau=np.sum((T.y[tr]-T[base][tr])*T.lnN[tr])/np.sum(T.lnN[tr]**2)
        p[te]=T[base][te]+tau*T.lnN[te]
    return p
P={
 'Site law, W per delay (usual USBM)': T.law,
 'Site law, total W': T.law+ab*T.lnN,
 'Site law + attention (tau, LOBO)': lobo_tau('law'),
 'Signature hole at station, W per delay': T.ref,
 'Signature hole at station, total W': T.ref+ab*T.lnN,
 'Signature hole + attention (tau, LOBO)': lobo_tau('ref'),
}
res={}
for sub,m in [('D>=30 m (n=%d)'%(T.D>=30).sum(),T.D>=30),('all (n=%d)'%len(T),T.D>0)]:
    print('\n',sub)
    res[sub]={}
    for k,p in P.items():
        e=np.asarray(p)[m.values]-T.y[m].values
        ef=np.exp(np.median(np.abs(e))); r=np.sqrt(np.mean(e**2)); bias=np.mean(e)
        res[sub][k]=dict(EF=round(ef,2),RMSE=round(r,2),bias=round(bias,2))
        print(f'  {k:45s} EF {ef:5.2f}  RMSE {r:4.2f}  bias {bias:+.2f}')
m=T.D>=30
tau_all=np.sum((T.y[m]-T.ref[m])*T.lnN[m])/np.sum(T.lnN[m]**2)
taus=(T.y-T.ref)/T.lnN
from scipy.stats import spearmanr
fs=T.rec.map(sig.fcent)
print('\nglobal tau (station reference, D>=30):',round(tau_all,3))
print('tau per station-blast pair: median',round(taus[m].median(),2),'IQR',taus[m].quantile([.25,.75]).round(2).tolist(),'min',round(taus[m].min(),2),'max',round(taus[m].max(),2))
print('Spearman tau vs signature-hole centroid frequency at the station:',spearmanr(taus[m],fs[m]))
print('Spearman tau vs distance:',spearmanr(taus[m],T.D[m]))
T['tau']=taus; T['f_sig']=fs; T.to_csv('field_eval.csv',index=False)
json.dump(dict(law=dict(lnK=lnK,beta=beta,n=len(A)),tau_global=tau_all,metrics=res),open('real_results.json','w'),indent=1,ensure_ascii=False)
