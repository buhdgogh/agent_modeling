import numpy as np, pandas as pd
from scipy import stats
from scipy.spatial import distance_matrix

df = pd.read_csv('temp/virtual_boreholes_points.csv')
fm = pd.read_csv('temp/auto_generated_formation.csv')
fn = dict(zip(fm['formation_code'].astype(str), fm['formation']))

bh_s = df.sort_values(['borehole_id','z'], ascending=[True,False])
bh_s['next_z'] = bh_s.groupby('borehole_id')['z'].shift(-1)
bh_s['thick'] = bh_s['z'] - bh_s['next_z']
thick = bh_s['thick'].dropna()

print(f'Thickness: n={len(thick)}, mean={thick.mean():.2f}m, SD={thick.std():.2f}m, CV={100*thick.std()/thick.mean():.1f}%')
print(f'Range: {thick.min():.1f}-{thick.max():.1f}m')

t_stat, t_pval = stats.ttest_1samp(thick, 30.0)
d = abs(thick.mean()-30)/thick.std()
print(f't-test: t={t_stat:.3f}, p={t_pval:.4f}, Cohen d={d:.3f}')

np.random.seed(42)
bm = np.array([np.mean(np.random.choice(thick,size=len(thick),replace=True)) for _ in range(10000)])
ci = np.percentile(bm, [2.5,97.5])
print(f'Bootstrap CI: [{ci[0]:.2f}, {ci[1]:.2f}], 30m inside: {ci[0]<=30<=ci[1]}')

sw, sw_p = stats.shapiro(thick)
print(f'Shapiro-Wilk: W={sw:.3f}, p={sw_p:.3f}')

ff = df['formation_code'].value_counts().sort_index()
chi2, chi2_p = stats.chisquare(ff.values, np.full(len(ff), len(df)/len(ff)))
print(f'Chi2: {chi2:.1f}, p={chi2_p:.6f}')

for fc_i in sorted(df['formation_code'].unique()):
    vals = bh_s.loc[bh_s['formation_code']==fc_i, 'thick'].dropna()
    if len(vals)>0:
        bm_fc = np.array([np.mean(np.random.choice(vals,size=len(vals),replace=True)) for _ in range(10000)])
        ci_fc = np.percentile(bm_fc, [2.5,97.5])
        ok = 'OK' if ci_fc[0]<=30<=ci_fc[1] else 'OUT'
        print(f'FC{int(fc_i)} {fn[str(int(fc_i))][:8]}: n={len(vals)}, mean={vals.mean():.1f}, CI=[{ci_fc[0]:.1f},{ci_fc[1]:.1f}], 30m: {ok}')

bs = df.groupby('borehole_id').agg(nl=('z','count'))
multi = bs[bs['nl']>1]
ok = sum(1 for bid in multi.index if all(
    df[df['borehole_id']==bid].sort_values('z',ascending=False)['formation_code'].values[i] <=
    df[df['borehole_id']==bid].sort_values('z',ascending=False)['formation_code'].values[i+1]
    for i in range(len(df[df['borehole_id']==bid])-1)))
print(f'Layer monotonicity: {ok}/{len(multi)}')

bs2 = df.groupby('borehole_id').agg(zt=('z','max'),x=('x','first'),y=('y','first'))
coords = bs2[['x','y']].values; z_s = bs2['zt'].values; n=len(z_s)
z_c = z_s-z_s.mean()
dist = distance_matrix(coords, coords)
W = np.where((dist>0)&(dist<2000), 1.0/(dist+1e-9), 0)
W_sum = W.sum()
mi = n*np.sum(W*np.outer(z_c,z_c))/(W_sum*np.sum(z_c**2))
print(f"Moran I={mi:.4f}, E[I]0={-1/(n-1):.4f}")
