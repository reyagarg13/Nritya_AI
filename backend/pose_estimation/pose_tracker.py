import cv2
import json
import os
import numpy as np
from datetime import datetime
import mediapipe as mp
from fastapi import WebSocket
import base64
import logging
from pathlib import Path
from typing import List, Dict, Any

# Import EnhancedPoseAnalyzer from main.py
from main import EnhancedPoseAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def save_landmarks_to_json(landmarks: List[Dict], confidence: float, pose_analysis: Dict, output_dir: str = "saved_poses") -> str:
    """Save pose landmarks with additional metadata to JSON."""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"pose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    data = {
        "landmarks": landmarks,
        "confidence": confidence,
        "pose_analysis": pose_analysis,
        "timestamp": datetime.now().isoformat()
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"[INFO] Landmarks saved to {filename}")
    return filename

async def analyze_pose_realtime(websocket: WebSocket, analyzer: EnhancedPoseAnalyzer):
    """Analyze poses in real-time using webcam and send results via WebSocket."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await websocket.send_text(json.dumps({"error": "Could not open webcam"}))
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await websocket.send_text(json.dumps({"error": "Failed to grab frame"}))
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = analyzer.pose_video.process(frame_rgb)

            response = {
                "type": "pose_result",
                "data": {
                    "landmarks": [],
                    "confidence": 0.0,
                    "pose_analysis": None,
                    "timestamp": datetime.now().isoformat()
                },
                "frame_count": analyzer.frame_count
            }

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                landmarks = [
                    {"id": i, "x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                    for i, lm in enumerate(results.pose_landmarks.landmark)
                ]
                confidence = float(np.mean([lm.visibility for lm in results.pose_landmarks.landmark]))
                pose_analysis = analyzer.analyze_pose_characteristics(landmarks)
                response["data"] = {
                    "landmarks": landmarks,
                    "confidence": confidence,
                    "pose_analysis": pose_analysis,
                    "timestamp": datetime.now().isoformat()
                }
                analyzer.pose_history.append(response["data"])
                analyzer.frame_count += 1

            cv2.imshow('Pose Tracker - Press Q to Quit', frame)
            await websocket.send_text(json.dumps(response))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    except Exception as e:
        await websocket.send_text(json.dumps({"error": f"Real-time analysis failed: {str(e)}"}))
    finally:
        cap.release()
        cv2.destroyAllWindows()

async def capture_pose(image_data: str, analyzer: EnhancedPoseAnalyzer) -> Dict:
    """Capture and analyze a single pose from base64 image data."""
    try:
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"error": "Invalid image data"}
        
        results = analyzer.pose_static.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            landmarks = [
                {"id": i, "x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for i, lm in enumerate(results.pose_landmarks.landmark)
            ]
            confidence = float(np.mean([lm.visibility for lm in results.pose_landmarks.landmark]))
            pose_analysis = analyzer.analyze_pose_characteristics(landmarks)
            saved_path = save_landmarks_to_json(landmarks, confidence, pose_analysis)
            return {
                "landmarks": landmarks,
                "confidence": confidence,
                "pose_analysis": pose_analysis,
                "timestamp": datetime.now().isoformat(),
                "saved_to": saved_path
            }
        else:
            return {
                "landmarks": [],
                "confidence": 0.0,
                "pose_analysis": None,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error capturing pose: {e}")
        return {"error": f"Capture failed: {str(e)}"}

if __name__ == "__main__":
    import asyncio
    async def main():
        analyzer = EnhancedPoseAnalyzer()
        # Simulate WebSocket for testing
        class MockWebSocket:
            async def send_text(self, text):
                print(f"Mock WebSocket sent: {text}")
            async def accept(self):
                pass
        await analyze_pose_realtime(MockWebSocket(), analyzer)
    asyncio.run(main())