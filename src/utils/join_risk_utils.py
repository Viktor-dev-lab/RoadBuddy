from bisect import bisect_left

def build_time_index(records, key_fields):
  by_key = {}
  for r in records:
      key = tuple(r[k] for k in key_fields)
      by_key.setdefault(key, []).append(r)
  for key, lst in by_key.items():
      lst.sort(key=lambda x: x["time_s"])
  return by_key


def find_nearest_time(record_list, t, max_dt=0.2):
  if not record_list: return None
  times = [r["time_s"] for r in record_list]
  i = bisect_left(times, t)
  
  candidates = []
  if i < len(times): candidates.append(record_list[i])
  if i > 0: candidates.append(record_list[i - 1])

  best = min(candidates, key=lambda can: abs(can["time_s"] - t))
  if abs(best["time_s"] - t) <= max_dt: return best
  return None


def join_risk(tracks_sorted, track_lane, ego_motion_sorted):
    track_lane_idx = build_time_index(track_lane, ["track_id"])

    fused_samples = []
    for tr in tracks_sorted:
        t = tr["time_s"]
        tid = tr["track_id"]

        lane_series = track_lane_idx.get((tid,), [])
        lane_rec = find_nearest_time(lane_series, t)
        ego_rec = find_nearest_time(ego_motion_sorted, t)

        fused = {
            "time_s": t,
            "track_id": tid,
            "class": tr["class"],
            "bbox": tr["bbox"],
            "track_quality": tr["track_quality"],
            "lane_id": lane_rec["lane_id"] if lane_rec else None,
            "lane_conf": lane_rec["lane_conf"] if lane_rec else None,
            "ego_speed_mps": ego_rec["speed_mps"] if ego_rec else None,
            "ego_accel_mps2": ego_rec["accel_mps2"] if ego_rec else None,
            "ego_yaw_rate_deg_s": ego_rec["yaw_rate_deg_s"] if ego_rec else None,
            "ego_motion_method": ego_rec["method"] if ego_rec else None,
            "ego_motion_conf": ego_rec.get("confidence") if ego_rec else None, 
        }

        fused_samples.append(fused)

    return fused_samples