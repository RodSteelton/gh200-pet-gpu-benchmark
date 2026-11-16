import json,glob,os,csv,re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=os.path.expanduser('~/mnt/Claude-analysis/neo_bench/results'); O=os.path.expanduser('~/mnt/Claude-analysis/paper_figs/draft2')
C={'GH200':'#0072B2','A100':'#E69F00','B200':'#009E73','Afton CPU (32 thr)':'#555555','Grace CPU (72 thr)':'#CC79A7'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'figure.dpi':100})
def save(fig,n):
    fig.savefig(f'{O}/{n}.png',dpi=300,bbox_inches='tight'); fig.savefig(f'{O}/{n}.svg',bbox_inches='tight'); plt.close(fig)
def lj(p): return json.load(open(p))
# ---------- Fig 1: MC throughput
A={}
for f in glob.glob(R+'/bench_A_*.json')+glob.glob(R+'/rivanna/bench_A_*.json'):
    d=lj(f); g=d['env'].get('gpu','')
    if 'gpu_trues' in d:
        k='GH200' if 'GH200' in g else 'A100' if 'A100' in g else 'B200' if 'B200' in g else None
        if k: A.setdefault(k,[]).append(d['gpu_trues']['emissions_per_s']/1e6)
    if 'cpu_trues' in d: A.setdefault('cpu1',[]).append(d['cpu_trues']['emissions_per_s']/1e6)
cpu1=np.mean(A['cpu1']); print('Fig1',{k:v for k,v in A.items()})
fig,ax=plt.subplots(figsize=(5.0,3.2))
names=['GH200','A100','B200']; vals=[np.mean(A[n]) for n in names]
b=ax.bar(names,vals,color=[C[n] for n in names],width=.55)
for r,v in zip(b,vals): ax.text(r.get_x()+r.get_width()/2,v*1.12,f'{v:.0f}',ha='center',fontsize=8)
ax.axhline(cpu1,color='#333',ls='--',lw=1); ax.text(2.36,cpu1*1.15,f'1 CPU process\n{cpu1:.2f}',ha='left',va='bottom',fontsize=8)
# measured rate of the production pool of 30 CPU worker processes on full open ring scans:
# old Rivanna logs (32 core tasks, jobs 20537059 and 20558471) and the Grace CPU runs D (jobs 641162, 648558)
LOGS=os.path.expanduser('~/mnt/Claude-analysis/rivanna_scratch_backup/logs')
_pat=re.compile(r'\[R280_slot (?:rest|stress) r\d\] ([\d.e+]+) decays -> [\d,]+ trues in (\d+) s')
def pool(files): return [float(a)/int(b)/1e6 for f in files for a,b in _pat.findall(open(f,errors='ignore').read())]
px=pool(sorted(glob.glob(LOGS+'/dynamic_2*_*.log')+glob.glob(LOGS+'/dynamic320_2*_*.log'))); pg=pool([R+'/runDorig_648558.out',R+'/runCDE_641162.out'])
print('pool x86 n=%d min %.1f max %.1f median %.1f mean %.1f | Grace n=%d min %.1f max %.1f'%(len(px),min(px),max(px),np.median(px),np.mean(px),len(pg),min(pg),max(pg)))
ax.axhspan(min(px),max(px),color='#777',alpha=.28,lw=0,zorder=0); ax.text(2.36,max(px)*1.12,f'30 CPU workers (measured)\n{min(px):.0f} to {max(px):.0f}',ha='left',va='bottom',fontsize=8)
ax.set_yscale('log'); ax.set_ylim(1,1200); ax.set_xlim(-0.5,4.0); ax.set_ylabel('Emitted decays per second (millions)'); ax.grid(axis='x',visible=False)
save(fig,'Fig2_mc_throughput')
# ---------- Fig 2: OSEM time vs events (fast mode)
pts={}
def add(k,n,t): pts.setdefault(k,{}).setdefault(n,[]).append(t)
for f in glob.glob(R+'/bench_B_*.json')+glob.glob(R+'/rivanna/bench_B_*.json'):
    d=lj(f); e=d['env']; ct=d['ctr']
    if list(ct.keys())!=['200']: continue
    n=d['n_events']; v=ct['200']; g=e.get('gpu','')
    if tuple(d['grid'])!=(186,186,80): continue
    if 'gpu' in v and g:
        k='GH200' if 'GH200' in g else 'A100' if 'A100' in g else 'B200'
        # exclude non-fast GH200 runs: v11-27 host files
        if 'v11-27' in f: continue
        if k=='GH200' and n==300000000 and '214453' in f: continue
        add(k,n,v['gpu']['median_s'])
    if 'cpu' in v:
        if 'v11-27' in f: continue
        if e['host'].startswith('cdc') and e['numba_threads']==72: add('Grace CPU (72 thr)',n,v['cpu']['median_s'])
        if e['host'].startswith('udc-ba05') and e['numba_threads']==32 and n==10**8 and v['cpu']['median_s']<40: add('Afton CPU (32 thr)',n,v['cpu']['median_s'])
        if e['host'].startswith('udc-ba05') and e['numba_threads']==32 and n==10**7: add('Afton CPU (32 thr)',n,v['cpu']['median_s'])
print('Fig2',{k:{n:np.round(v,2).tolist() for n,v in d.items()} for k,d in pts.items()})
json.dump({k:{str(n):v for n,v in d.items()} for k,d in pts.items()},open(O+'/fig2_data.json','w'))
fig,ax=plt.subplots(figsize=(5.4,3.6))
for k in ['Afton CPU (32 thr)','Grace CPU (72 thr)','A100','B200','GH200']:
    if k not in pts: continue
    xs=sorted(pts[k]); ys=[np.median(pts[k][x]) for x in xs]
    ax.plot(xs,ys,marker='o' if 'CPU' not in k else 's',ms=4,lw=1.4,color=C[k],label=k,ls='--' if 'CPU' in k else '-')
ax.axvline(3.1e9,color=C['A100'],ls=':',lw=1); ax.text(3.1e9*1.05,0.9,'A100 80 GB\nlimit ~3.1e9',fontsize=7,color=C['A100'])
ax.axvline(5.4e9,color=C['GH200'],ls=':',lw=1); ax.text(5.4e9*1.05,0.5,'GH200 limit\n~5.4e9',fontsize=7,color=C['GH200'])
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(5e6,1.5e10); ax.set_xlabel('List-mode events'); ax.set_ylabel('OSEM time, 4 iterations x 10 subsets (s)')
ax.legend(frameon=False,fontsize=8); save(fig,'Fig3_osem_vs_events')
# ---------- scans
S={}
for name,dd in [('GH200','FS'),('A100','FS_a100'),('B200','FS_b200')]:
    L=[]
    for c in ('c200','c320'):
        for f in glob.glob(f'{R}/{dd}/dynamic_gpu_{c}/*.json'):
            d=lj(f); d['ctr']=c; d['file']=os.path.basename(f); L.append(d)
    S[name]=L
    print(name,len(L))
def sel(L,cfg='R280_slot',ctr='c200',state=None,scale=1.0): return [d for d in L if d['config']==cfg and d['ctr']==ctr and (state is None or d['state']==state) and abs(d.get('scale',1)-scale)<1e-9]
# ---------- Fig 3 stage breakdown
fig,ax=plt.subplots(figsize=(5.2,2.6)); rows=[]
stg=[('simulate','Monte Carlo','#0072B2'),('osem_all_frames','OSEM, all frames','#E69F00'),('fit_and_other','Fit and other','#999999')]
for i,k in enumerate(['GH200','A100','B200']):
    L=sel(S[k]); m={s:np.mean([d['seconds'][s] for d in L]) for s,_,_ in stg}; tot=np.mean([d['seconds']['total'] for d in L])
    rows.append((k,len(L),m,tot)); left=0
    for s,lab,col in stg:
        ax.barh(i,m[s],left=left,color=col,edgecolor='white',height=.55,label=lab if i==0 else None); left+=m[s]
    ax.text(tot+3,i,f'{tot:.0f} s',va='center',fontsize=8)
ax.set_yticks(range(3)); ax.set_yticklabels(['GH200','A100','B200']); ax.invert_yaxis(); ax.set_xlabel('Wall time per scan (s)'); ax.set_xlim(0,max(r[3] for r in rows)*1.18)
ax.legend(frameon=False,fontsize=7.5,ncol=3,loc='lower center',bbox_to_anchor=(0.5,1.0)); ax.grid(axis='y',visible=False); save(fig,'Fig5_stage_breakdown'); print('Fig3',[(k,n,{a:round(b,1) for a,b in m.items()},round(t,1)) for k,n,m,t in rows])
# ---------- Fig 4 MBF
fig,axs=plt.subplots(1,2,figsize=(6.4,3.2)); old={}
for j,(key,truth,ttl) in enumerate([('mbf_normal_mean',3.0,'Stress MBF, normal segments'),('mbf_defect',1.5,'Stress MBF, defect segment')]):
    ax=axs[j]
    for i,k in enumerate(['GH200','A100','B200']):
        L=sel(S[k],state='stress'); y=np.array([d[key] for d in L]); x=i+(np.random.RandomState(1).rand(len(y))-.5)*.25
        ax.scatter(x,y,s=9,color=C[k],alpha=.7,lw=0); ax.errorbar(i,y.mean(),y.std(ddof=1),color='k',capsize=3,lw=1,zorder=3)
        for d in L:
            if 'old_reference' in d: old.setdefault(key,set()).add(json.dumps(d['old_reference']))
    ax.axhline(truth,color='#333',ls='--',lw=1); ax.text(2.45,truth,' truth',va='bottom',ha='right',fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels(['GH200','A100','B200']); ax.set_title(ttl,fontsize=9); ax.set_ylabel('MBF (mL/min/g)'); ax.grid(axis='x',visible=False)
oldst=[ (d['state'],d.get('old_reference')) for d in S['GH200'] if d['real']==0 and d['config']=='R280_slot' and d['ctr']=='c200' and abs(d.get('scale',1)-1)<1e-9]
print('old_ref',oldst)
# old CPU stress scan 0 from OLD-SYSTEM-RESULTS: normal 3.122, defect 1.591 (open ring, 200 ps)
for j,v in enumerate([3.122,1.591]): axs[j].scatter([-.42],[v],marker='D',s=22,color='k',zorder=4,label='old CPU, scan 0' if j==0 else None)
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,frameon=False,fontsize=7,loc='lower center',ncol=1,bbox_to_anchor=(.5,-.04)); fig.tight_layout(rect=(0,.06,1,1)); save(fig,'Fig6_mbf_agreement')
# ---------- Fig 5 GPU memory
mem={}
for f in sorted(glob.glob(R+'/bench_B_*cdc-v11-31-s_202611122051*.json')+glob.glob(R+'/bench_B_*cdc-v11-31-s_20261112205[3-7]*.json')): pass
mem_pts=[(4.0e9,54.09),(4.8e9,57.72),(5.2e9,64.27)]
# memory used from G5 csv: peak
rows=[l.strip().split(', ') for l in open(R+'/gpu_util_G5_639721.csv').read().splitlines()[1:] if l.strip()]
mu=[int(r[3].split()[0]) for r in rows]; ts=np.arange(len(mu))*5/60.   # nvidia-smi -l 5: one sample every 5 s, shown in minutes
fig,ax=plt.subplots(figsize=(5.4,3.0)); ax.plot(ts,np.array(mu)/1024,color=C['GH200'],lw=1.2); ax.axhline(146831/1024,color='#333',ls='--',lw=1); ax.text(0,146831/1024+2,'GH200 capacity 143.4 GiB',fontsize=7)
ax.axhline(81920/1024,color=C['A100'],ls='--',lw=1); ax.text(0,81920/1024+2,'A100 80 GB capacity 80 GiB',fontsize=7,color=C['A100'])
ax.set_xlabel('Time (min), job G5: runs of 4.0e9, 4.8e9 and 5.2e9 events'); ax.set_ylabel('GPU memory used (GiB)'); ax.set_ylim(0,165); save(fig,'Fig4_gpu_memory')
print('Fig5 peak MiB',max(mu),'samples',len(mu))
