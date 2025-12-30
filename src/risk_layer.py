import json
import sys
import os
import time
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# Setup environment & path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

load_dotenv()

from src.utils.join_risk_utils import join_risk
from src.analytics.risk_engine import build_risk_signals
from google import genai


def track_to_text(track: dict) -> str:
    track_id = track["track_id"]
    records = track["records"]

    cls = records[0].get("class", "N/A")
    start_time = records[0]["time_s"]
    end_time = records[-1]["time_s"]
    duration = end_time - start_time

    speeds = [
        r.get("ego_speed_mps", 0)
        for r in records
        if r.get("ego_speed_mps") is not None
    ]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
    max_risk = max(r.get("risk_score", 0.0) for r in records)
    lanes = sorted(set(r.get("lane_id") for r in records if r.get("lane_id")))
    last_lane_change = records[-1].get("lane_change_state", "unknown")

    text = f"""
        Phân tích quỹ đạo xe (Track ID: {track_id})

        - Loại phương tiện: {cls}
        - Thời gian quan sát: từ {start_time:.2f}s đến {end_time:.2f}s
        - Tổng thời gian theo dõi: {duration:.2f} giây
        - Tốc độ trung bình của xe chủ: {avg_speed:.2f} m/s
        - Các làn đường đã ghi nhận: {', '.join(lanes) if lanes else 'Không xác định'}
        - Trạng thái thay đổi làn cuối cùng: {last_lane_change}
        - Điểm rủi ro cao nhất trong toàn bộ quỹ đạo: {max_risk:.2f}

        Dữ liệu chi tiết theo từng khung hình (JSON):
        {json.dumps(records, ensure_ascii=False)}
        """
    return text.strip()

def llm_reponse(output_file: Path, tracks: list, client: genai.Client):
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, track in enumerate(tracks, start=1):
            track_text = track_to_text(track)

            prompt = f"""
                Bạn là một chuyên gia AI về an toàn giao thông.

                Dưới đây là dữ liệu phân tích chi tiết của một phương tiện
                được ghi nhận từ camera hành trình (dashcam).
                Tất cả các thông số đều được hệ thống tự động tính toán.

                {track_text}

                Yêu cầu:
                1. Mô tả hành vi di chuyển của phương tiện.
                2. Đánh giá mức độ rủi ro: An toàn / Cảnh báo / Nguy hiểm.
                3. Giải thích rõ ràng dựa trên các con số cụ thể.
                4. Đưa ra lời khuyên cho tài xế xe chủ.

                Phản hồi bằng tiếng Việt, văn bản thuần, không markdown,
                không chào hỏi, không lan man.
                """

            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )

            # ===== Write to file =====
            f.write("\n" + "=" * 90 + "\n")
            f.write(f"TRACK {idx} | Track ID: {track['track_id']}\n")
            f.write("=" * 90 + "\n")
            f.write(response.text.strip())
            f.write("\n\n")

            # ===== Console log =====
            print(f"✓ Đã phân tích Track ID {track['track_id']}")

            # tránh rate limit
            time.sleep(0.2)

    print(f"\n✅ Hoàn tất. Kết quả đã lưu tại:\n{output_file}")


if __name__ == "__main__":

    telemetry_path = Path(r"D:\Workstation\My Project\RoadBuddy\output\data\telemetry_data.json")

    with open(telemetry_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tracks_raw = data["object_tracks"]
    ego_motion = data["ego_motion"]
    track_lane = data["lane_assignments"]


    tracks_sorted = sorted(tracks_raw, key=lambda x: (x["track_id"], x["time_s"]))
    ego_motion_sorted = sorted(ego_motion, key=lambda x: x["time_s"])
    track_lane_sorted = sorted(track_lane, key=lambda x: (x["track_id"], x["time_s"]))

    joined = join_risk(tracks_sorted, track_lane_sorted, ego_motion_sorted)
    risk_signals = build_risk_signals(joined)


    grouped_by_track = defaultdict(list)
    for r in risk_signals:
        grouped_by_track[r["track_id"]].append(r)

    tracks = [
        {"track_id": tid, "records": recs}
        for tid, recs in grouped_by_track.items()
    ]


    output_dir = ROOT_DIR / "output" / "llm"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "gemini_track_analysis.txt"

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    llm_reponse(output_file, tracks[18:], client)
