from collections import defaultdict

def estimate_distance_from_bbox(sample):
  x1, y1, x2, y2 = sample["bbox"]
  h = max(y2 - y1, 1)
  return round(1000.0 / h, 1)

def compute_headway(distance_m, ego_speed_mps):
  if ego_speed_mps is None or ego_speed_mps <= 0: return None
  return round(distance_m / ego_speed_mps, 2)

def compute_ttc(distance_m, rel_speed_mps):
  if rel_speed_mps is None or rel_speed_mps <= 0: return None
  return round(distance_m / rel_speed_mps, 2)

def infer_lane_change_state(track_samples):
  prev_lane = None
  for s in track_samples:
      lane = s["lane_id"]
      if prev_lane is None or lane is None: s["lane_change_state"] = "steady"
      else:
          if lane == prev_lane: s["lane_change_state"] = "steady"
          else: s["lane_change_state"] = "changing_in"
      prev_lane = lane


def compute_lane_departure_score(sample):
  if sample["lane_id"] == "ego_lane": return 0.05
  else: return 0.10


def build_risk_signals(fused_samples):
  by_track = defaultdict(list)
  for s in fused_samples:
      by_track[s["track_id"]].append(s)

  for tid, lst in by_track.items():
      lst.sort(key=lambda x: x["time_s"])
      infer_lane_change_state(lst)

  risk_signals = []

  for tid, lst in by_track.items():
    for s in lst:
      ego_v = s["ego_speed_mps"]

      distance_m = estimate_distance_from_bbox(s)
      # demo: xe trước đi ~50% tốc độ ego
      track_speed_approx = 0.5 * ego_v if ego_v is not None else 0.0
      rel_speed_mps = ego_v - track_speed_approx if ego_v is not None else None

      headway_s = compute_headway(distance_m, ego_v)
      ttc_s = compute_ttc(distance_m, rel_speed_mps)

      lane_departure_score = compute_lane_departure_score(s)

      # demo risk_score: càng gần (distance nhỏ) + headway nhỏ → risk cao
      if headway_s is not None: risk_score = max(0.0, min(1.0, 1.5 - headway_s))  # toy
      else: risk_score = 0.0

      # estimation_method + quality_flags demo
      # giả sử: track_quality tốt + lane_conf tốt → tracking_stable, lane_conf_ok
      if s["track_quality"] >= 0.8: tracking_flag = "tracking_stable"
      else: tracking_flag = "tracking_unstable"

      if s["lane_conf"] is not None and s["lane_conf"] >= 0.8: lane_flag = "lane_conf_ok"
      else: lane_flag = "lane_conf_low"

      # demo: nếu có ego_motion_method thì map sang estimation_method
      if s["ego_motion_method"] == "gps_imu_fusion": estimation_method = "mono_depth_proxy" 
      else: estimation_method = "2d_proxy"

      risk_sample = {
        "time_s": s["time_s"],
        "track_id": s["track_id"],
        "class": s["class"],
        "bbox": s["bbox"],
        "track_quality": s["track_quality"],
        "lane_id": s["lane_id"],
        "lane_conf": s["lane_conf"],
        "ego_speed_mps": s["ego_speed_mps"],
        "ego_accel_mps2": s["ego_accel_mps2"],
        "ego_yaw_rate_deg_s": s["ego_yaw_rate_deg_s"],
        "distance_m": distance_m,
        "rel_speed_mps": round(rel_speed_mps, 2) if rel_speed_mps is not None else None,
        "headway_s": headway_s,
        "ttc_s": ttc_s,
        "lane_change_state": s["lane_change_state"],
        "lane_departure_score": lane_departure_score,
        "risk_score": round(risk_score, 2),
        "estimation_method": estimation_method,
        "quality_flags": [tracking_flag, lane_flag],
      }

      risk_signals.append(risk_sample)

  risk_signals.sort(key=lambda x: (x["track_id"], x["time_s"]))
  return risk_signals
  