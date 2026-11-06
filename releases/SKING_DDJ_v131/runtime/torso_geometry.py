"""Local geometric evidence; no model/provider-specific input names."""
import numpy as np
from scipy import ndimage
from torso_models import VIEWS, FB_IDS
def fit_error(a,ids,ys,xs,dx,dy):
    # Independent checkerboard folds prevent scoring each pixel against its own fitted colour.
    z=ids[ys,xs];v=a[ys+dy,xs+dx].astype(np.float64)/255
    fold=(ys+2*xs)%4<2;err=np.zeros(len(z));sizes=np.zeros(len(z))
    for train in [fold,~fold]:
        count=np.bincount(z[train],minlength=4097);sums=np.stack([np.bincount(z[train],weights=v[train,c],minlength=4097) for c in range(3)],1);mean=sums/np.maximum(count[:,None],1)
        test=~train;err[test]=np.square(v[test]-mean[z[test]]).mean(1);sizes[test]=count[z[test]]
    good=sizes>=3
    return err,good
class FastRaster:
    def __init__(self,renderer):
        self.views=[]
        for v in VIEWS:
            state={}
            for key in ['inner','outer','composite','geometry']:
                suffix='_grid' if key in ['inner','outer'] else '_grid_layers'
                grid=getattr(renderer,v+'_'+key+suffix).detach().cpu().numpy();xy=(grid+1)*31.5
                assert np.abs(xy-np.rint(xy)).max()<.001
                xy=np.rint(xy).astype(np.int32).clip(0,63);ids=xy[...,1]*64+xy[...,0]
                msuffix='_mask' if key in ['inner','outer'] else '_mask_layers'
                masks=getattr(renderer,v+'_'+key+msuffix).detach().cpu().numpy()>.5
                state[key]=np.where(masks,ids,4096).astype(np.int16)
            state['decor']=getattr(renderer,v+'_composite_is_decor_layers').cpu().numpy()
            state['order']=getattr(renderer,v+'_geometry_sort_indices').cpu().numpy()
            self.views.append(state)
    def __call__(self,uv):
        alpha=np.concatenate([uv[...,3].reshape(-1)>127,[False]])
        views=[]
        for s in self.views:
            simple=np.where(alpha[s['outer']],s['outer'],np.where(alpha[s['inner']],s['inner'],4096))
            comp=s['composite'];vis=alpha[comp];first=vis.argmax(0);anyvis=vis.any(0);ci=np.take_along_axis(comp,first[None],axis=0)[0]
            decor=np.take_along_axis(s['decor'],first[None],axis=0)[0]
            trust=anyvis&(decor|(first<=1));final=np.where(trust,ci,simple)
            geom=np.take_along_axis(s['geometry'],s['order'],axis=0);gv=alpha[geom];last=geom.shape[0]-1-gv[::-1].argmax(0);gi=np.take_along_axis(geom,last[None],axis=0)[0]
            fallback=(comp[0]!=4096)&s['decor'][0]&~alpha[comp[0]]&gv.any(0)
            final=np.where(fallback,gi,final);views.append(final)
        return np.concatenate(views,axis=1)
def components(uv):
    out=[]
    for face,ox in enumerate([20,32]):
        mask=uv[36:48,ox:ox+8,3]>127;labels,n=ndimage.label(mask)
        for j in range(1,n+1):
            yy,xx=np.nonzero(labels==j);ids=(yy+36)*64+xx+ox
            out.append(dict(face=face,ids=ids.tolist(),size=len(ids),total=int(mask.sum()),width=int(xx.max()-xx.min()+1),height=int(yy.max()-yy.min()+1)))
    return out

def support(ids,face):
    ox=[20,32][face];y,x=ids//64,ids%64
    return (x>=ox)&(x<ox+8)&(((y>=36)&(y<48))|((y>=20)&(y<32)))

def component_features(a,m0,m1,c):
    shared=support(m0,c['face'])&support(m1,c['face'])
    # Ignore an image-space boundary band where adjacent body parts could contaminate evidence.
    shared=ndimage.binary_erosion(shared,iterations=1)
    changed=(m0!=m1)&shared;local=ndimage.binary_dilation(changed,iterations=3)&shared
    yy,xx=np.nonzero(shared);sub=(yy+xx)%2==0;yy,xx=yy[sub],xx[sub];ll=local[yy,xx]
    vals=[np.log1p(c['size']),c['size']/max(1,c['total']),c['width']/8,c['height']/12,c['size']/(c['width']*c['height']),c['total']/96,np.log1p(changed.sum()),float(ll.mean()) if len(ll) else 0]
    goodcase=len(yy)>200 and int(ll.sum())>=8
    if not goodcase:return vals+[0.]*38,False
    for sigma in [0,.6,1.2]:
        im=a if sigma==0 else ndimage.gaussian_filter(a.astype(float),(sigma,sigma,0))
        ee=[]
        for m in [m0,m1]:
            err,g=fit_error(im,m,yy,xx,0,0);g=g&ll;ee.append(err[g]);vals.extend([np.log1p(10000*err[g].mean()),np.log1p(10000*np.quantile(err[g],.8)),np.log1p(10000*err.mean())])
        v0,v1=[e.mean() for e in ee];vals.extend([(v1-v0)/(v1+v0+1e-6),(v1-v0)*1000])
    # Sensitivity of local evidence to a shared +/-1-pixel registration, no separate fit freedoms.
    ratios=[]
    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ee=[]
        for m in [m0,m1]:
            e,g=fit_error(a,m,yy,xx,dx,dy);ee.append(e[g&ll].mean())
        ratios.append((ee[1]-ee[0])/(ee[1]+ee[0]+1e-6))
    vals+=ratios+[np.mean(ratios),np.std(ratios)]
    z0=m0[yy,xx];z1=m1[yy,xx];vals.extend([len(np.unique(z0[ll]))/100,len(np.unique(z1[ll]))/100,float(a[yy[ll],xx[ll]].std()/255),float(changed.sum()/max(1,c['size']))/200])
    # Total=42; invalid vectors are fixed below from declared dimension.
    assert len(vals)==42,len(vals)
    return vals,True

def analyze(g,a,uv):
    cc=components(uv)
    if not cc:return []
    candidates=[uv]
    for c in cc:
        alt=uv.copy();alt.reshape(-1,4)[c['ids']]=0;candidates.append(alt)
    maps=[g(a) for a in candidates];out=[]
    for c,m in zip(cc,maps[1:]):
        vals,valid=component_features(a,maps[0],m,c)
        if not valid:vals=[0.]*42
        out.append(dict(**c,features=vals,observable=valid))
    return out
def remove_components(components,policy):
    ids=[]
    for c in components:
        if not c['observable'] or c['size']>policy['max_component'] or c['total']>policy['max_face']:continue
        f=c['features'];evidence=max(f[14],f[22],f[30])
        if evidence<=policy['max_relative_error']:ids+=c['ids']
    return set(ids)
