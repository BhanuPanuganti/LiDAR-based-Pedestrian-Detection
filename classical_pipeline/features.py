import numpy as np
from sklearn.decomposition import PCA

def extract_features(pts):
    try:
        xyz = pts[:,:3]
        n = len(xyz)

        xmin,xmax = xyz[:,0].min(), xyz[:,0].max()
        ymin,ymax = xyz[:,1].min(), xyz[:,1].max()
        zmin,zmax = xyz[:,2].min(), xyz[:,2].max()

        H = zmax - zmin
        W = ymax - ymin
        D = xmax - xmin
        vol = max(H*W*D, 1e-6)

        cx, cy = (xmin+xmax)/2, (ymin+ymax)/2

        if n >= 3:
            ev = PCA(n_components=min(3,n)).fit(xyz).explained_variance_
            ev = np.sort(np.pad(ev,(0,max(0,3-len(ev))),constant_values=1e-9))[::-1]
            lin = (ev[0]-ev[1])/(ev[0]+1e-9)
            pln = (ev[1]-ev[2])/(ev[0]+1e-9)
            sct = ev[2]/(ev[0]+1e-9)
        else:
            lin = pln = sct = 0.0

        return {
            'n_points': n,
            'height': H,
            'width': W,
            'depth': D,
            'volume': vol,
            'density': n/vol,
            'dist': np.sqrt(cx**2 + cy**2),
            'z_var': float(np.var(xyz[:,2])),
            'refl_mean': float(pts[:,3].mean()),
            'linearity': lin,
            'planarity': pln,
            'scattering': sct,
            '_xmin': xmin, '_xmax': xmax,
            '_ymin': ymin, '_ymax': ymax,
            '_zmin': zmin, '_zmax': zmax
        }

    except Exception as e:
        print(f"[ERROR] feature extraction: {e}")
        return None