"""Classic 64x64 atlas: explicit face/part/layer identity, no semantic heuristics."""
import numpy as np

def layout():
    region=np.full((64,64),-1,np.int64);face=np.full_like(region,-1);local=np.zeros((64,64,2),np.float32)
    boxes=[(0,0,8,8,8,32,0),(16,16,8,4,12,0,16),(32,48,4,4,12,16,0),
           (40,16,4,4,12,0,16),(16,48,4,4,12,-16,0),(0,16,4,4,12,0,16)]
    for part,(u,v,w,d,h,dx,dy) in enumerate(boxes):
        rects=[(u+d,v,w,d),(u+d+w,v,w,d),(u,v+d,d,h),(u+d,v+d,w,h),(u+d+w,v+d,d,h),(u+2*d+w,v+d,w,h)]
        for layer in range(2):
            for fi,(x,y,ww,hh) in enumerate(rects):
                x+=layer*dx;y+=layer*dy
                region[y:y+hh,x:x+ww]=part+layer*6;face[y:y+hh,x:x+ww]=(part+layer*6)*6+fi
                yy,xx=np.meshgrid(np.linspace(-1,1,hh),np.linspace(-1,1,ww),indexing='ij')
                local[y:y+hh,x:x+ww]=np.stack([xx,yy],-1)
    return region,face,local

REGION,FACE,LOCAL=layout();VALID=REGION>=0;INNER=(REGION>=0)&(REGION<6);OUTER=REGION>=6

def canonical(a):
    a=a.copy();a[~VALID]=0;a[a[:,:,3]==0,:3]=0
    return a
