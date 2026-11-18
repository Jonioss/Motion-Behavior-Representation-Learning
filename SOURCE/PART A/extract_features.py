import numpy as np
import pandas as pd
from pathlib import Path
# Argoverse 2 object_category integer mapping:
# 0 = UNSCORED_TRACK
# 1 = SCORED_TRACK
# 2 = FOCAL_TRACK
# 3 = TRACK_FRAGMENT

VALID_CATEGORIES = {1, 2}           # SCORED + FOCALa
MIN_TIMESTEPS    = 50

def apply_labels_per_track(group):
    speed = group["speed"].values
    x_vals = group["position_x"].values
    y_vals = group["position_y"].values
    
    direction, speed_label = get_human_labels(speed, x_vals, y_vals)
    
    return pd.Series({"direction": direction, "speed_label": speed_label})

def get_human_labels(speed, x_vals, y_vals):
    if len(speed) < 5:
        return "Invalid", "Invalid"
    # Total displacement start -> end
    dx_total = x_vals[-1] - x_vals[0]
    dy_total = y_vals[-1] - y_vals[0]
    #if dx_total+dy_total<=0.01:
     #   return "Non-Moving","Non_moving"
    dist = np.hypot(dx_total, dy_total) + 1e-9

    # Perpendicular (lateral) deviation from straight line
    cross = np.abs((dx_total * (y_vals - y_vals[0]) - dy_total * (x_vals - x_vals[0])) / dist)
    max_lateral = np.max(cross)

    # Total path length
    path_length = np.sum(np.hypot(np.diff(x_vals), np.diff(y_vals))) + 1e-9

    # Straightness ratio: how straight is the path vs total displacement
    straightness = dist / path_length  # 1.0 = perfectly straight, <1 = curved

    k = min(2, len(x_vals) - 1) 
    dx_init = x_vals[k] - x_vals[0]
    dy_init = y_vals[k] - y_vals[0]
    
    if np.hypot(dx_init, dy_init) < 1e-5:
        k = min(4, len(x_vals) - 1)
        dx_init = x_vals[k] - x_vals[0]
        dy_init = y_vals[k] - y_vals[0]
    # Signed lateral displacement at the END relative to heading direction
    # positive = left of heading, negative = right
    heading_angle = np.arctan2(dy_init, dx_init)
    cos_h, sin_h = np.cos(heading_angle), np.sin(heading_angle)
    # rotate all points into heading frame
    local_y = -(x_vals - x_vals[0]) * sin_h + (y_vals - y_vals[0]) * cos_h
    final_lateral = local_y[-1]  # positive = left, negative = right

    # Classify direction
    if straightness > 0.92 and max_lateral < 0.5:
        direction = "straight"
    elif straightness < 0.35:
        direction = "u turn"
    elif final_lateral > 0.3:
        direction = "left turn"
    elif final_lateral < -0.3:
        direction = "right turn"
    else:
        direction = "unspecified"

    # acceleration 
    third_idx = max(1, len(speed) // 3)
    start_speed = np.mean(speed[:third_idx])
    end_speed   = np.mean(speed[-third_idx:])

    if end_speed > start_speed * 1.15:
        speed_label = "accelerate"
    elif end_speed < start_speed * 0.85:
        speed_label = "decelerate"
    else:
        speed_label = "constant speed"

    return direction, speed_label

def process_scenario(parquet_path: Path) -> pd.DataFrame:

    df = pd.read_parquet(parquet_path)

    #this is used to take all the non-vehicles from the train and validation datasets
    #comment out for the whole dataset
    df1 = df[df["object_type"]!='vehicle']
    df2=df[df["track_id"]== 'AV']
    df=pd.concat([df1, df2], axis=0)    

    #remove some invalid data
    df = df[df["object_category"].isin(VALID_CATEGORIES)]

    if "observed" in df.columns:
        df = df[df["observed"] == True]

    counts = df.groupby("track_id")["timestep"].count()
    valid_tracks = counts[counts >= MIN_TIMESTEPS].index
    df = df[df["track_id"].isin(valid_tracks)]

    if df.empty:
        return pd.DataFrame(),[0,0,0,0]
    df = df.sort_values(["track_id", "timestep"]).copy()
    # Derived features
    df["speed"]        = np.sqrt(df["velocity_x"]**2 + df["velocity_y"]**2)
    keep_cols = [
        "scenario_id", "track_id", "object_type",
        "timestep", "position_x","position_y", "speed_label","direction"
    ]
    labels_df = df.groupby("track_id").apply(apply_labels_per_track).reset_index()
    df = df.merge(labels_df, on="track_id")
    available = [c for c in keep_cols if c in df.columns]
    x_norm = df.loc[
    (df["track_id"] == 'AV') & 
    (df["timestep"] == 0.0), 
    "position_x"
    ]
    y_norm = df.loc[
    (df["track_id"] == 'AV') & 
    (df["timestep"] == 0.0), 
    "position_y"
    ]
    if x_norm.empty:
        return df[available],[0,0,0,0]
    else:
        xn=x_norm.item()
    if y_norm.empty:
        return df[available],[0,0,0,0]
    else:
        yn=y_norm.item()
    norm_list=[df["position_x"].max()-xn,df["position_x"].min()-xn,df["position_y"].max()-yn,df["position_y"].min()-yn]
    return df[available], norm_list


def extract_all(root_dir: str, output_path: str):
    root = Path(root_dir)

    parquet_files = list(root.rglob("scenario_*.parquet"))
    if not parquet_files:
        parquet_files = list(root.rglob("*.parquet"))

    print(f"Found {len(parquet_files)} parquet file(s).")

    all_dfs     = []
    empty_count = 0
    for pf in parquet_files:
        try:
            result,norm_list = process_scenario(pf)
            if not result.empty:
                all_dfs.append(result)#new datapoint
            else:
                empty_count += 1
        except Exception as e:
            print(f"  Error in {pf.name}: {e}")

    print(f"Empty after filter: {empty_count} / {len(parquet_files)}")

    if all_dfs:
        final = pd.concat(all_dfs, ignore_index=True)
        final.to_parquet(output_path, index=False)
        print(f"Saved {len(final)} rows -> {output_path}")
    else:
        print("No data extracted.")


if __name__ == "__main__":
    extract_all("test","output_test.parquet")
    #extract_all("train","output_train.parquet")
    #extract_all("val","output_val.parquet")


