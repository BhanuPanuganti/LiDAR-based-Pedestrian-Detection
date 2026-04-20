ROI = dict(x_min=0, x_max=50, y_min=-15, y_max=15, z_min=-3, z_max=3)

def roi_filter(pc):
    try:
        m = (
            (pc[:,0]>=ROI['x_min']) & (pc[:,0]<=ROI['x_max']) &
            (pc[:,1]>=ROI['y_min']) & (pc[:,1]<=ROI['y_max']) &
            (pc[:,2]>=ROI['z_min']) & (pc[:,2]<=ROI['z_max'])
        )
        return pc[m]
    except Exception as e:
        print(f"[ERROR] roi_filter: {e}")
        return None