from sklearn.cluster import DBSCAN

EPS = 0.22
MIN_PTS = 18

def run_dbscan(pc):
    try:
        return DBSCAN(
            eps=EPS,
            min_samples=MIN_PTS,
            algorithm='ball_tree',
            n_jobs=-1
        ).fit_predict(pc[:,:3])
    except Exception as e:
        print(f"[ERROR] DBSCAN failed: {e}")
        return None