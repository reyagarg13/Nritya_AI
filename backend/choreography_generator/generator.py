from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import json
import os
import random
import logging
from pathlib import Path
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pose Choreography API",
    description="API for generating dance pose sequences and choreography",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class DurationLevel(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

class PoseSequenceRequest(BaseModel):
    style: str = Field(default="bollywood", description="Dance style")
    count: Optional[int] = Field(default=None, description="Number of sequences to generate")
    min_poses: int = Field(default=1, description="Minimum poses per sequence")
    max_poses: Optional[int] = Field(default=None, description="Maximum poses per sequence")
    difficulty: Optional[DifficultyLevel] = Field(default=None, description="Difficulty level")
    duration: Optional[DurationLevel] = Field(default=None, description="Duration preference")

class ChoreographyRequest(BaseModel):
    style: str = Field(default="bollywood", description="Dance style")
    target_length: int = Field(default=8, ge=1, le=50, description="Target number of poses")
    mix_sequences: bool = Field(default=True, description="Mix poses from different sequences")
    tempo: Optional[str] = Field(default="medium", description="Choreography tempo")
    theme: Optional[str] = Field(default=None, description="Choreography theme")

class PoseSequenceLoader:
    """Enhanced pose sequence loader with caching, filtering, and validation."""
    
    def __init__(self, base_path: Optional[str] = None):
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path(__file__).parent / "poses_library"
        
        self._cache = {}
        self.logger = logging.getLogger(__name__)
    
    def get_available_styles(self) -> List[str]:
        """Get list of available pose styles."""
        if not self.base_path.exists():
            return []
        
        styles = []
        for file_path in self.base_path.glob("*.json"):
            styles.append(file_path.stem)
        return sorted(styles)
    
    def load_pose_data(self, style: str, use_cache: bool = True) -> Dict:
        """Load and validate pose data for a given style."""
        style_key = style.lower()
        
        if use_cache and style_key in self._cache:
            return self._cache[style_key]
        
        file_path = self.base_path / f"{style_key}.json"
        
        if not file_path.exists():
            available = self.get_available_styles()
            return {
                "error": f"No pose data found for style '{style}'.",
                "available_styles": available
            }
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not self._validate_pose_data(data):
                return {"error": f"Invalid pose data structure in '{style}' file."}
            
            if use_cache:
                self._cache[style_key] = data
            
            return data
            
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON in '{style}' file: {str(e)}"}
        except Exception as e:
            return {"error": f"Error loading '{style}' data: {str(e)}"}
    
    def _validate_pose_data(self, data: Dict) -> bool:
        """Validate the structure of pose data."""
        required_keys = ["sequences"]
        
        for key in required_keys:
            if key not in data:
                return False
        
        if not isinstance(data["sequences"], list) or not data["sequences"]:
            return False
        
        for seq in data["sequences"]:
            if not isinstance(seq, list):
                return False
        
        return True
    
    def load_pose_sequence(
        self, 
        style: str = "bollywood",
        count: Optional[int] = None,
        min_poses: int = 1,
        max_poses: Optional[int] = None,
        difficulty: Optional[str] = None,
        duration: Optional[str] = None
    ) -> Dict:
        """Load a pose sequence with enhanced filtering options."""
        
        pose_data = self.load_pose_data(style)
        
        if "error" in pose_data:
            return pose_data
        
        sequences = pose_data["sequences"]
        
        filtered_sequences = self._filter_sequences(
            sequences, min_poses, max_poses, difficulty, duration, pose_data
        )
        
        if not filtered_sequences:
            return {
                "error": "No sequences found matching the specified criteria.",
                "style": style,
                "total_available": len(sequences)
            }
        
        if count is None:
            selected_sequence = random.choice(filtered_sequences)
            return {
                "style": style,
                "count": len(selected_sequence),
                "sequence": selected_sequence,
                "metadata": self._get_sequence_metadata(selected_sequence, pose_data)
            }
        else:
            selected_sequences = random.sample(
                filtered_sequences, 
                min(count, len(filtered_sequences))
            )
            return {
                "style": style,
                "count": len(selected_sequences),
                "sequences": selected_sequences,
                "total_poses": sum(len(seq) for seq in selected_sequences),
                "metadata": [self._get_sequence_metadata(seq, pose_data) for seq in selected_sequences]
            }
    
    def _filter_sequences(
        self, 
        sequences: List, 
        min_poses: int,
        max_poses: Optional[int],
        difficulty: Optional[str],
        duration: Optional[str],
        pose_data: Dict
    ) -> List:
        """Filter sequences based on specified criteria."""
        filtered = []
        
        for seq in sequences:
            if len(seq) < min_poses:
                continue
            
            if max_poses and len(seq) > max_poses:
                continue
            
            if difficulty and "metadata" in pose_data:
                seq_meta = self._get_sequence_metadata(seq, pose_data)
                if seq_meta.get("difficulty", "").lower() != difficulty.lower():
                    continue
            
            if duration and "metadata" in pose_data:
                seq_meta = self._get_sequence_metadata(seq, pose_data)
                if seq_meta.get("duration", "").lower() != duration.lower():
                    continue
            
            filtered.append(seq)
        
        return filtered
    
    def _get_sequence_metadata(self, sequence: List, pose_data: Dict) -> Dict:
        """Extract metadata for a sequence if available."""
        metadata = {
            "pose_count": len(sequence),
            "estimated_duration": f"{len(sequence) * 3}-{len(sequence) * 5} seconds"
        }
        
        if "metadata" in pose_data:
            metadata.update(pose_data.get("metadata", {}))
        
        return metadata
    
    def create_choreography(
        self, 
        style: str,
        target_length: int,
        mix_sequences: bool = True,
        tempo: str = "medium",
        theme: Optional[str] = None
    ) -> Dict:
        """Create choreography with enhanced features."""
        pose_data = self.load_pose_data(style)
        
        if "error" in pose_data:
            return pose_data
        
        if mix_sequences:
            all_poses = []
            for sequence in pose_data["sequences"]:
                all_poses.extend(sequence)
            
            if len(all_poses) < target_length:
                return {
                    "error": f"Not enough poses available. Requested: {target_length}, Available: {len(all_poses)}",
                    "style": style
                }
            
            choreography = random.sample(all_poses, target_length)
        else:
            sequence = random.choice(pose_data["sequences"])
            choreography = []
            
            while len(choreography) < target_length:
                remaining = target_length - len(choreography)
                if remaining >= len(sequence):
                    choreography.extend(sequence)
                else:
                    choreography.extend(random.sample(sequence, remaining))
        
        # Add tempo-based timing
        timing = self._calculate_timing(choreography, tempo)
        
        return {
            "style": style,
            "choreography": choreography,
            "length": len(choreography),
            "tempo": tempo,
            "theme": theme,
            "timing": timing,
            "metadata": {
                "total_duration": sum(timing),
                "difficulty": self._estimate_difficulty(choreography),
                "energy_level": self._estimate_energy_level(tempo, len(choreography))
            }
        }
    
    def _calculate_timing(self, choreography: List, tempo: str) -> List[float]:
        """Calculate timing for each pose based on tempo."""
        base_timing = {
            "slow": 6.0,
            "medium": 4.0,
            "fast": 2.5,
            "very_fast": 1.5
        }
        
        beat_duration = base_timing.get(tempo, 4.0)
        
        # Add some variation to timing
        timings = []
        for _ in choreography:
            variation = random.uniform(0.8, 1.2)
            timings.append(beat_duration * variation)
        
        return timings
    
    def _estimate_difficulty(self, choreography: List) -> str:
        """Estimate difficulty based on choreography complexity."""
        if len(choreography) <= 5:
            return "beginner"
        elif len(choreography) <= 12:
            return "intermediate"
        else:
            return "advanced"
    
    def _estimate_energy_level(self, tempo: str, length: int) -> str:
        """Estimate energy level based on tempo and length."""
        energy_map = {
            "slow": 1,
            "medium": 2,
            "fast": 3,
            "very_fast": 4
        }
        
        tempo_energy = energy_map.get(tempo, 2)
        length_factor = min(length / 10, 2)  # Cap at 2x multiplier
        
        total_energy = tempo_energy * length_factor
        
        if total_energy <= 2:
            return "low"
        elif total_energy <= 4:
            return "medium"
        else:
            return "high"

# Initialize the pose loader
pose_loader = PoseSequenceLoader()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Pose Choreography API",
        "version": "1.0.0",
        "endpoints": {
            "pose_sequence": "/pose/sequence",
            "generate_choreography": "/pose/generate_choreography",
            "available_styles": "/pose/styles",
            "style_info": "/pose/styles/{style}",
            "random_pose": "/pose/random"
        }
    }

@app.get("/pose/styles")
async def get_available_styles():
    """Get all available dance styles."""
    try:
        styles = pose_loader.get_available_styles()
        return {
            "styles": styles,
            "count": len(styles)
        }
    except Exception as e:
        logger.error(f"Error fetching styles: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/pose/styles/{style}")
async def get_style_info(style: str):
    """Get detailed information about a specific style."""
    try:
        pose_data = pose_loader.load_pose_data(style)
        
        if "error" in pose_data:
            raise HTTPException(status_code=404, detail=pose_data["error"])
        
        sequences = pose_data["sequences"]
        all_poses = [pose for seq in sequences for pose in seq]
        
        info = {
            "style": style,
            "total_sequences": len(sequences),
            "total_poses": len(all_poses),
            "unique_poses": len(set(all_poses)),
            "sequence_lengths": [len(seq) for seq in sequences],
            "avg_sequence_length": sum(len(seq) for seq in sequences) / len(sequences),
            "min_sequence_length": min(len(seq) for seq in sequences),
            "max_sequence_length": max(len(seq) for seq in sequences)
        }
        
        if "metadata" in pose_data:
            info["metadata"] = pose_data["metadata"]
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching style info for {style}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/pose/sequence")
@app.post("/pose/sequence")
async def get_pose_sequence(
    style: str = Query(default="bollywood", description="Dance style"),
    count: Optional[int] = Query(default=None, description="Number of sequences"),
    min_poses: int = Query(default=1, description="Minimum poses per sequence"),
    max_poses: Optional[int] = Query(default=None, description="Maximum poses per sequence"),
    difficulty: Optional[DifficultyLevel] = Query(default=None, description="Difficulty level"),
    duration: Optional[DurationLevel] = Query(default=None, description="Duration preference")
):
    """Get pose sequences with filtering options."""
    try:
        result = pose_loader.load_pose_sequence(
            style=style,
            count=count,
            min_poses=min_poses,
            max_poses=max_poses,
            difficulty=difficulty.value if difficulty else None,
            duration=duration.value if duration else None
        )
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating pose sequence: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/pose/generate_choreography")
@app.post("/pose/generate_choreography")
async def generate_choreography(
    style: str = Query(default="bollywood", description="Dance style"),
    target_length: int = Query(default=8, ge=1, le=50, description="Target number of poses"),
    mix_sequences: bool = Query(default=True, description="Mix poses from different sequences"),
    tempo: str = Query(default="medium", description="Choreography tempo"),
    theme: Optional[str] = Query(default=None, description="Choreography theme")
):
    """Generate choreography with specified parameters."""
    try:
        result = pose_loader.create_choreography(
            style=style,
            target_length=target_length,
            mix_sequences=mix_sequences,
            tempo=tempo,
            theme=theme
        )
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating choreography: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/pose/random")
async def get_random_pose(
    style: str = Query(default="bollywood", description="Dance style")
):
    """Get a single random pose from a style."""
    try:
        pose_data = pose_loader.load_pose_data(style)
        
        if "error" in pose_data:
            raise HTTPException(status_code=404, detail=pose_data["error"])
        
        all_poses = []
        for sequence in pose_data["sequences"]:
            all_poses.extend(sequence)
        
        if not all_poses:
            raise HTTPException(status_code=404, detail=f"No poses found in '{style}' data.")
        
        return {
            "style": style,
            "pose": random.choice(all_poses),
            "total_available": len(all_poses)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting random pose: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": [
                "/pose/sequence",
                "/pose/generate_choreography", 
                "/pose/styles",
                "/pose/styles/{style}",
                "/pose/random"
            ]
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)