from fastapi import status
# --- Add POST endpoint for /choreography/move to support frontend POST requests ---
from fastapi import status
# --- Add POST endpoint for /choreography/move to support frontend POST requests ---
from fastapi import Form
import sys
import traceback

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    print("Uncaught exception:", ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))

sys.excepthook = handle_exception

import os
import json
import argparse
import tempfile
import cv2
import numpy as np
import asyncio
import base64
from datetime import datetime
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import mediapipe as mp
import logging
from pathlib import Path
from pose_tracker import compare_poses

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Nritya AI API",
    description="Real-time motion capture and dance choreography generation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

# Pydantic models for request/response
class PoseRequest(BaseModel):
    image_data: str  # Base64 encoded image
from fastapi import status
from fastapi import Form
    # timestamp: Optional[str] = None

class PoseRequest(BaseModel):
    image_data: str  # Base64 encoded image
    timestamp: Optional[str] = None

class ChoreographyRequest(BaseModel):
    style: str
    duration: Optional[int] = 10  # seconds
    tempo: Optional[str] = "medium"  # slow, medium, fast
    complexity: Optional[str] = "intermediate"  # beginner, intermediate, advanced

class PoseData(BaseModel):
    landmarks: List[Dict[str, float]]
    timestamp: str
    confidence: float

class PoseFeedbackRequest(BaseModel):
    user_landmarks: list
    expected_landmarks: list

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                disconnected.append(connection)
        
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

class EnhancedPoseAnalyzer:
    def __init__(self):
        self.pose_static = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7
        )
        self.pose_video = mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_history = []
        self.frame_count = 0
        
    def analyze_image(self, image_data: str) -> Dict[str, Any]:
        """Analyze pose from base64 image data"""
        try:
            # Decode base64 image
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"error": "Invalid image data"}
            
            # Process with MediaPipe
            results = self.pose_static.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                landmarks = []
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    landmarks.append({
                        "id": i,
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility
                    })
                
                # Calculate pose confidence
                confidence = np.mean([lm.visibility for lm in results.pose_landmarks.landmark])
                
                # Analyze pose characteristics
                pose_analysis = self.analyze_pose_characteristics(landmarks)
                
                return {
                    "landmarks": landmarks,
                    "confidence": float(confidence),
                    "pose_analysis": pose_analysis,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "landmarks": [],
                    "confidence": 0.0,
                    "pose_analysis": None,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def analyze_pose_characteristics(self, landmarks: List[Dict]) -> Dict[str, Any]:
        """Analyze pose characteristics for dance classification"""
        if not landmarks:
            return {}
        
        # Key pose points
        nose = landmarks[0]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        # Calculate pose metrics
        shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
        hip_width = abs(left_hip['x'] - right_hip['x'])
        body_height = abs(nose['y'] - (left_ankle['y'] + right_ankle['y']) / 2)
        
        # Determine stance and movement
        weight_distribution = self.calculate_weight_distribution(landmarks)
        arm_position = self.analyze_arm_position(landmarks)
        leg_position = self.analyze_leg_position(landmarks)
        
        # Pass computed arm/leg positions to avoid recursion
        pose_type = self.classify_pose_type(arm_position, leg_position)
        
        return {
            "shoulder_width": shoulder_width,
            "hip_width": hip_width,
            "body_height": body_height,
            "weight_distribution": weight_distribution,
            "arm_position": arm_position,
            "leg_position": leg_position,
            "pose_type": pose_type
        }
    
    def calculate_weight_distribution(self, landmarks: List[Dict]) -> str:
        """Calculate weight distribution between left and right leg"""
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Simple weight distribution based on ankle positions
        left_weight = left_ankle['visibility'] * (1 - abs(left_ankle['x'] - left_hip['x']))
        right_weight = right_ankle['visibility'] * (1 - abs(right_ankle['x'] - right_hip['x']))
        
        if left_weight > right_weight * 1.2:
            return "left"
        elif right_weight > left_weight * 1.2:
            return "right"
        else:
            return "balanced"
    
    def analyze_arm_position(self, landmarks: List[Dict]) -> str:
        """Analyze arm position for dance classification"""
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        
        # Calculate arm angles
        left_arm_raised = left_wrist['y'] < left_shoulder['y']
        right_arm_raised = right_wrist['y'] < right_shoulder['y']
        
        if left_arm_raised and right_arm_raised:
            return "both_raised"
        elif left_arm_raised:
            return "left_raised"
        elif right_arm_raised:
            return "right_raised"
        else:
            return "neutral"
    
    def analyze_leg_position(self, landmarks: List[Dict]) -> str:
        """Analyze leg position for dance classification"""
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Check for wide stance
        hip_distance = abs(left_hip['x'] - right_hip['x'])
        ankle_distance = abs(left_ankle['x'] - right_ankle['x'])
        
        if ankle_distance > hip_distance * 1.5:
            return "wide_stance"
        elif ankle_distance < hip_distance * 0.7:
            return "narrow_stance"
        else:
            return "neutral_stance"
    
    def classify_pose_type(self, arm_pos: str, leg_pos: str) -> str:
        """Classify the type of dance pose based on arm and leg positions"""
        if arm_pos == 'both_raised' and leg_pos == 'wide_stance':
            return "expressive"
        elif arm_pos == 'both_raised':
            return "celebratory"
        elif leg_pos == 'wide_stance':
            return "grounded"
        elif arm_pos == 'left_raised':
            return "left_celebration"
        elif arm_pos == 'right_raised':
            return "right_celebration"
        else:
            return "neutral"

class ChoreographyGenerator:
    def __init__(self):
        # Use PhantomDance dataset if available, else fallback to built-in
        self.pose_sequences = {}
        if PHANTOMDANCE and "styles" in PHANTOMDANCE:
            self.pose_sequences = PHANTOMDANCE["styles"]
        else:
            self.pose_sequences = {
                "bollywood": [
                    {"name": "Thumka", "duration": 2, "keyframes": self.generate_thumka_keyframes()},
                    {"name": "Bhangra Arms", "duration": 2, "keyframes": self.generate_bhangra_keyframes()},
                    {"name": "Classical Pose", "duration": 3, "keyframes": self.generate_classical_keyframes()},
                    {"name": "Spin Move", "duration": 2, "keyframes": self.generate_spin_keyframes()},
                    {"name": "Hand Gestures", "duration": 2, "keyframes": self.generate_mudra_keyframes()}
                ],
                "kathak": [
                    {"name": "Chakkars", "duration": 3, "keyframes": self.generate_chakkar_keyframes()},
                    {"name": "Hasta Mudra", "duration": 2, "keyframes": self.generate_mudra_keyframes()},
                    {"name": "Aramandi", "duration": 2, "keyframes": self.generate_aramandi_keyframes()},
                    {"name": "Tatkar", "duration": 2, "keyframes": self.generate_tatkar_keyframes()},
                    {"name": "Bhramari", "duration": 3, "keyframes": self.generate_bhramari_keyframes()}
                ],
                "hiphop": [
                    {"name": "Pop & Lock", "duration": 2, "keyframes": self.generate_pop_lock_keyframes()},
                    {"name": "Wave", "duration": 2, "keyframes": self.generate_wave_keyframes()},
                    {"name": "Freeze", "duration": 1, "keyframes": self.generate_freeze_keyframes()},
                    {"name": "Breakdance", "duration": 3, "keyframes": self.generate_breakdance_keyframes()},
                    {"name": "Isolation", "duration": 2, "keyframes": self.generate_isolation_keyframes()}
                ],
                "contemporary": [
                    {"name": "Spiral", "duration": 3, "keyframes": self.generate_spiral_keyframes()},
                    {"name": "Contraction", "duration": 2, "keyframes": self.generate_contraction_keyframes()},
                    {"name": "Release", "duration": 2, "keyframes": self.generate_release_keyframes()},
                    {"name": "Floor Work", "duration": 3, "keyframes": self.generate_floor_keyframes()},
                    {"name": "Leap", "duration": 2, "keyframes": self.generate_leap_keyframes()}
                ]
            }

    def generate_choreography(self, style: str, duration: int = 10, tempo: str = "medium", complexity: str = "intermediate") -> Dict[str, Any]:
        """Generate a choreography sequence"""
        if style not in self.pose_sequences:
            return {"error": f"Style '{style}' not supported"}

        available_moves = self.pose_sequences[style]

        # Adjust for tempo
        tempo_multiplier = {"slow": 1.5, "medium": 1.0, "fast": 0.7}.get(tempo, 1.0)

        # Generate sequence
        sequence = []
        current_time = 0

        while current_time < duration:
            # Select move based on complexity
            move = self.select_move_by_complexity(available_moves, complexity)
            adjusted_duration = move["duration"] * tempo_multiplier

            if current_time + adjusted_duration <= duration:
                sequence.append({
                    **move,
                    "start_time": current_time,
                    "end_time": current_time + adjusted_duration,
                    "adjusted_duration": adjusted_duration
                })
                current_time += adjusted_duration
            else:
                break

        return {
            "style": style,
            "duration": duration,
            "tempo": tempo,
            "complexity": complexity,
            "sequence": sequence,
            "total_moves": len(sequence),
            "generated_at": datetime.now().isoformat()
        }

    def select_move_by_complexity(self, moves: List[Dict], complexity: str) -> Dict:
        """Select move based on complexity level"""
        if not moves:
            return {}
        if complexity == "beginner":
            # Prefer shorter, simpler moves
            return min(moves, key=lambda x: x.get("duration", 2))
        elif complexity == "advanced":
            # Prefer longer, complex moves
            return max(moves, key=lambda x: x.get("duration", 2))
        else:
            # Random selection for intermediate
            import random
            return random.choice(moves)

    # Keyframe generation methods (simplified - in practice these would be more sophisticated)
    def generate_thumka_keyframes(self):
        return [
            {"frame": 0, "hip_rotation": 0, "shoulder_position": "neutral"},
            {"frame": 30, "hip_rotation": 15, "shoulder_position": "right"},
            {"frame": 60, "hip_rotation": -15, "shoulder_position": "left"}
        ]
    
    def generate_bhangra_keyframes(self):
        return [
            {"frame": 0, "arm_position": "down", "shoulder_bounce": 0},
            {"frame": 20, "arm_position": "up", "shoulder_bounce": 10},
            {"frame": 40, "arm_position": "side", "shoulder_bounce": 5}
        ]
    
    def generate_classical_keyframes(self):
        return [
            {"frame": 0, "mudra": "anjali", "stance": "samapada"},
            {"frame": 45, "mudra": "abhaya", "stance": "aramandi"},
            {"frame": 90, "mudra": "varada", "stance": "samapada"}
        ]
    
    def generate_spin_keyframes(self):
        return [
            {"frame": 0, "rotation": 0, "arm_position": "preparation"},
            {"frame": 30, "rotation": 180, "arm_position": "extended"},
            {"frame": 60, "rotation": 360, "arm_position": "close"}
        ]
    
    def generate_mudra_keyframes(self):
        return [
            {"frame": 0, "hand_gesture": "alapadma", "expression": "neutral"},
            {"frame": 30, "hand_gesture": "pataka", "expression": "focused"},
            {"frame": 60, "hand_gesture": "ardhachandra", "expression": "graceful"}
        ]
    
    def generate_chakkar_keyframes(self):
        return [
            {"frame": 0, "spin_speed": 0, "arm_position": "first"},
            {"frame": 45, "spin_speed": 5, "arm_position": "second"},
            {"frame": 90, "spin_speed": 0, "arm_position": "close"}
        ]
    
    def generate_aramandi_keyframes(self):
        return [
            {"frame": 0, "knee_bend": 0, "back_posture": "straight"},
            {"frame": 30, "knee_bend": 45, "back_posture": "slight_forward"},
            {"frame": 60, "knee_bend": 0, "back_posture": "straight"}
        ]
    
    def generate_tatkar_keyframes(self):
        return [
            {"frame": 0, "foot_position": "flat", "rhythm": "start"},
            {"frame": 15, "foot_position": "heel", "rhythm": "beat1"},
            {"frame": 30, "foot_position": "toe", "rhythm": "beat2"}
        ]
    
    def generate_bhramari_keyframes(self):
        return [
            {"frame": 0, "rotation": 0, "arm_flow": "start"},
            {"frame": 60, "rotation": 360, "arm_flow": "continuous"},
            {"frame": 90, "rotation": 450, "arm_flow": "end"}
        ]
    
    def generate_pop_lock_keyframes(self):
        return [
            {"frame": 0, "tension": "relaxed", "joint_position": "neutral"},
            {"frame": 20, "tension": "locked", "joint_position": "sharp"},
            {"frame": 40, "tension": "pop", "joint_position": "explosive"}
        ]
    
    def generate_wave_keyframes(self):
        return [
            {"frame": 0, "wave_position": "start", "body_part": "hand"},
            {"frame": 30, "wave_position": "middle", "body_part": "arm"},
            {"frame": 60, "wave_position": "end", "body_part": "shoulder"}
        ]
    
    def generate_freeze_keyframes(self):
        return [
            {"frame": 0, "position": "dynamic", "stability": "moving"},
            {"frame": 30, "position": "freeze", "stability": "locked"}
        ]
    
    def generate_breakdance_keyframes(self):
        return [
            {"frame": 0, "ground_contact": "feet", "power": "building"},
            {"frame": 45, "ground_contact": "hands", "power": "explosive"},
            {"frame": 90, "ground_contact": "back", "power": "controlled"}
        ]
    
    def generate_isolation_keyframes(self):
        return [
            {"frame": 0, "isolated_part": "head", "movement": "neutral"},
            {"frame": 30, "isolated_part": "shoulders", "movement": "independent"},
            {"frame": 60, "isolated_part": "chest", "movement": "isolated"}
        ]
    
    def generate_spiral_keyframes(self):
        return [
            {"frame": 0, "spiral_direction": "up", "intensity": "gentle"},
            {"frame": 45, "spiral_direction": "peak", "intensity": "full"},
            {"frame": 90, "spiral_direction": "down", "intensity": "release"}
        ]
    
    def generate_contraction_keyframes(self):
        return [
            {"frame": 0, "core_engagement": "neutral", "breath": "inhale"},
            {"frame": 30, "core_engagement": "contracted", "breath": "exhale"},
            {"frame": 60, "core_engagement": "release", "breath": "inhale"}
        ]
    
    def generate_release_keyframes(self):
        return [
            {"frame": 0, "energy": "held", "flow": "contained"},
            {"frame": 30, "energy": "releasing", "flow": "flowing"},
            {"frame": 60, "energy": "free", "flow": "extended"}
        ]
    
    def generate_floor_keyframes(self):
        return [
            {"frame": 0, "level": "standing", "transition": "prepare"},
            {"frame": 45, "level": "floor", "transition": "smooth"},
            {"frame": 90, "level": "standing", "transition": "rise"}
        ]
    
    def generate_leap_keyframes(self):
        return [
            {"frame": 0, "preparation": "ground", "energy": "gathering"},
            {"frame": 30, "preparation": "air", "energy": "explosive"},
            {"frame": 60, "preparation": "landing", "energy": "controlled"}
        ]

def get_pose_landmark_names():
    """Return a mapping of landmark indices to names for MediaPipe pose."""
    try:
        return {i: name for i, name in enumerate(mp_pose.PoseLandmark)}
    except Exception:
        # Fallback for older mediapipe versions
        return {i: f"LANDMARK_{i}" for i in range(33)}

# --- PhantomDance dataset integration ---
def load_phantomdance_dataset():
    """
    Load PhantomDance reference choreography dataset.
    Expects a JSON file at 'data/phantomdance.json' with structure:
    {
        "styles": {
            "bollywood": [ { "name": ..., "landmarks": [...], ... }, ... ],
            ...
        }
    }
    """
    dataset_path = Path("data/phantomdance.json")
    if not dataset_path.exists():
        logger.warning("PhantomDance dataset not found at data/phantomdance.json")
        return {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)

PHANTOMDANCE = load_phantomdance_dataset()

# Initialize components
pose_analyzer = EnhancedPoseAnalyzer()
choreography_generator = ChoreographyGenerator()

# Create output directory
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "Nritya AI API",
        "version": "1.0.0",
        "endpoints": {
            "/pose": "POST - Analyze pose from image",
            "/pose/batch": "POST - Analyze batch of poses",
            "/choreography": "POST - Generate choreography sequence",
            "/choreography/preview": "POST - Preview choreography moves",
            "/pose/landmark-names": "GET - Get landmark names",
            "/health": "GET - Health check",
            "/ws": "WebSocket - Real-time pose tracking",
            "/about": "GET - About the Nritya AI app",
            "/feedback": "POST - Compare user pose to expected pose"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "pose_analysis": "active",
            "choreography_generation": "active",
            "websocket": "active"
        }
    }

@app.post("/pose")
async def analyze_pose_endpoint(request: PoseRequest):
    """Analyze pose from uploaded image."""
    try:
        result = pose_analyzer.analyze_image(request.image_data)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Save results if landmarks found
        if result["landmarks"]:
            output_path = OUTPUT_DIR / f"pose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            result["saved_to"] = str(output_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in pose analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/pose/batch")
async def analyze_pose_batch_endpoint(
    requests: List[PoseRequest] = Body(..., description="List of pose requests")
):
    """
    Analyze a batch of images for pose detection.
    Returns a list of results, one per image.
    """
    results = []
    for req in requests:
        result = pose_analyzer.analyze_image(req.image_data)
        results.append(result)
    return {"results": results, "count": len(results)}

@app.get("/pose/landmark-names")
async def get_landmark_names():
    """
    Get the list of MediaPipe pose landmark names and their indices.
    """
    names = get_pose_landmark_names()
    return {"landmark_names": names, "total": len(names)}

@app.post("/choreography")
async def generate_choreography_endpoint(request: ChoreographyRequest):
    """Generate choreography sequence using PhantomDance if available."""
    try:
        # Use PhantomDance dataset for steps if available
        if PHANTOMDANCE and "styles" in PHANTOMDANCE and request.style in PHANTOMDANCE["styles"]:
            available_moves = PHANTOMDANCE["styles"][request.style]
            # Simple sequence: select moves up to duration
            sequence = []
            total_time = 0
            tempo_multiplier = {"slow": 1.5, "medium": 1.0, "fast": 0.7}[request.tempo]
            for move in available_moves:
                move_duration = move.get("duration", 2) * tempo_multiplier
                if total_time + move_duration > request.duration:
                    break
                sequence.append({**move, "adjusted_duration": move_duration})
                total_time += move_duration
            result = {
                "style": request.style,
                "duration": request.duration,
                "tempo": request.tempo,
                "complexity": request.complexity,
                "sequence": sequence,
                "total_moves": len(sequence),
                "generated_at": datetime.now().isoformat(),
                "source": "phantomdance"
            }
        else:
            # fallback to built-in generator
            result = choreography_generator.generate_choreography(
                style=request.style,
                duration=request.duration,
                tempo=request.tempo,
                complexity=request.complexity
            )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        # Save choreography
        output_path = OUTPUT_DIR / f"choreography_{request.style}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        result["saved_to"] = str(output_path)
        return result
    except Exception as e:
        logger.error(f"Error in choreography generation: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/choreography/preview")
async def preview_choreography_moves(
    style: str = Body(..., embed=True, description="Dance style"),
):
    """
    Preview all moves and their keyframes for a given style from PhantomDance if available.
    Returns a list of full move objects (not just names).
    """
    logger.info(f"/choreography/preview called for style: {style}")
    if PHANTOMDANCE and "styles" in PHANTOMDANCE and style in PHANTOMDANCE["styles"]:
        moves = PHANTOMDANCE["styles"][style]
        return {"style": style, "moves": moves, "total_moves": len(moves), "source": "phantomdance"}
    if style not in choreography_generator.pose_sequences:
        raise HTTPException(status_code=400, detail=f"Style '{style}' not supported")
    moves = choreography_generator.pose_sequences[style]
    return {"style": style, "moves": moves, "total_moves": len(moves)}

@app.get("/choreography/styles")
async def get_available_styles():
    """Get available dance styles."""
    return {
        "styles": list(choreography_generator.pose_sequences.keys()),
        "total": len(choreography_generator.pose_sequences)
    }

@app.get("/choreography/motions")
async def get_motion_labels(style: str, enhanced: Optional[int] = 0):
    """
    Get unique motion labels (move names) for a given style.
    If enhanced=1, add description/icon for dropdown.
    Returns: [{label: "Kick", value: "kick", desc: "...", icon: "🦵"}, ...]
    """
    motions = []
    # Use built-in moves only, not PhantomDance
    moves = choreography_generator.pose_sequences.get(style, [])
    seen = set()
    # Example: add icons/descriptions for some common moves
    ICONS = {
        "kick": "🦵", "spin": "🔄", "slide": "🕺", "bodywave": "🌊", "handroll": "🤲",
        "thumka": "💃", "bhangra arms": "👐", "pop & lock": "🧊", "wave": "🌊", "freeze": "❄️",
        "spiral": "🌀", "leap": "🦘", "floor work": "🛏️"
    }
    DESCS = {
        "kick": "Energetic leg kick", "spin": "360° rotation", "slide": "Smooth side movement",
        "bodywave": "Wave through body", "handroll": "Rolling hand gesture",
        "thumka": "Hip movement", "bhangra arms": "Upward arm bounce", "pop & lock": "Sharp isolation",
        "wave": "Fluid arm wave", "freeze": "Sudden stop", "spiral": "Twisting motion",
        "leap": "Jump in air", "floor work": "Dance on floor"
    }
    for move in moves:
        name = move.get("name", "")
        if name and name.lower() not in seen:
            label = name
            value = name
            desc = DESCS.get(name.lower(), "")
            icon = ICONS.get(name.lower(), "")
            if enhanced:
                motions.append({"label": label, "value": value, "desc": desc, "icon": icon})
            else:
                motions.append({"label": label, "value": value})
            seen.add(name.lower())
    return {"motions": motions, "total": len(motions)}

@app.get("/choreography/move")
async def get_move(style: str, name: str):
    """
    Get full move data for a given style and move name from PhantomDance.
    """
    if PHANTOMDANCE and "styles" in PHANTOMDANCE and style in PHANTOMDANCE["styles"]:
        moves = PHANTOMDANCE["styles"][style]
        for move in moves:
            if move.get("name", "") == name:
                return move
    # fallback: use built-in moves
    moves = choreography_generator.pose_sequences.get(style, [])
    for move in moves:
        if move.get("name", "") == name:
            return move
    raise HTTPException(status_code=404, detail="Move not found")

@app.get("/about")
async def about_info():
    """About page info for frontend."""
    return {
        "app": "Nritya AI",
        "description": (
            "Nritya AI is an innovative platform blending Indian dance heritage with modern AI. "
            "Our mission is to empower dancers, choreographers, and learners with real-time motion capture and creative choreography tools."
        ),
        "team": [
            {"name": "Reya Garg", "role": "Full Stack Developer"}
        ],
        "heritage": [
            {"style": "Bollywood", "desc": "Vibrant, expressive, and energetic"},
            {"style": "Kathak", "desc": "Graceful spins and intricate footwork"},
            {"style": "Hip-Hop", "desc": "Urban, freestyle, and powerful"},
            {"style": "Contemporary", "desc": "Fluid, creative, and modern"}
        ],
        "mission": (
            "To democratize dance education and creativity using AI. "
            "We blend tradition with technology to make learning, practicing, and choreographing dance accessible to all."
        )
    }

@app.get("/choreography/move")
async def get_move(style: str, name: str):
    """
    Get full move data for a given style and move name from PhantomDance.
    """
    if PHANTOMDANCE and "styles" in PHANTOMDANCE and style in PHANTOMDANCE["styles"]:
        moves = PHANTOMDANCE["styles"][style]
        for move in moves:
            if move.get("name", "") == name:
                return move
    # fallback: use built-in moves
    moves = choreography_generator.pose_sequences.get(style, [])
    for move in moves:
        if move.get("name", "") == name:
            return move
    raise HTTPException(status_code=404, detail="Move not found")

# --- POST endpoint for /choreography/move ---
@app.post("/choreography/move")
async def post_move(request: Request):
    """
    POST endpoint to get move details for a given style and move name.
    Accepts JSON body: {"style": ..., "name": ...}
    """
    try:
        data = await request.json()
        style = data.get("style")
        name = data.get("name")
        if not style or not name:
            raise HTTPException(status_code=400, detail="Missing style or name in request body")
        # Reuse logic from GET endpoint
        if PHANTOMDANCE and "styles" in PHANTOMDANCE and style in PHANTOMDANCE["styles"]:
            moves = PHANTOMDANCE["styles"][style]
            for move in moves:
                if move.get("name", "") == name:
                    return move
        moves = choreography_generator.pose_sequences.get(style, [])
        for move in moves:
            if move.get("name", "") == name:
                return move
        raise HTTPException(status_code=404, detail="Move not found")
    except Exception as e:
        logger.error(f"Error in POST /choreography/move: {e}")
    raise HTTPException(status_code=500, detail=f"Failed to get move: {str(e)}")

@app.post("/feedback")
async def pose_feedback(request: PoseFeedbackRequest):
    """
    Compare user pose to expected pose and return score, feedback, and details.
    Uses PhantomDance reference if available.
    """
    # Optionally: could fetch expected_landmarks from PhantomDance if only move name is given
    score, feedback, details = compare_poses(request.user_landmarks, request.expected_landmarks)
    return {
        "score": score,
        "feedback": feedback,
        "details": details
    }

# Remove any existing cleanup/signal code and replace with this:
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing API resources...")
    # Suppress TensorFlow Lite and absl warnings as early as possible
    try:
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except Exception:
        pass

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Cleaning up API resources...")
    try:
        # Close MediaPipe resources
        if hasattr(pose_analyzer, 'pose_static') and pose_analyzer.pose_static:
            try:
                pose_analyzer.pose_static.close()
            except Exception:
                pass
        if hasattr(pose_analyzer, 'pose_video') and pose_analyzer.pose_video:
            try:
                pose_analyzer.pose_video.close()
            except Exception:
                pass
        # Do not close logger handlers or flush sys.stdout/sys.stderr here
        # Let Python's runtime handle it to avoid sys.excepthook/atexit errors
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def main():
    parser = argparse.ArgumentParser(description="Nritya AI API")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])

    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    try:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level=args.log_level)
    except Exception as e:
        logger.error(f"Error starting server: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()