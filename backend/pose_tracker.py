import numpy as np

def compare_poses(user_landmarks, expected_landmarks):
    """
    Compare two sets of pose landmarks (list of dicts with x, y, z, visibility).
    Returns a similarity score (0-100), feedback string, and per-joint details.
    """
    if not user_landmarks or not expected_landmarks or len(user_landmarks) != len(expected_landmarks):
        return 0, "Pose data incomplete.", {}

    # Convert to numpy arrays (x, y, z)
    user = np.array([[lm['x'], lm['y'], lm.get('z', 0)] for lm in user_landmarks])
    expected = np.array([[lm['x'], lm['y'], lm.get('z', 0)] for lm in expected_landmarks])

    # Euclidean distance per landmark
    distances = np.linalg.norm(user - expected, axis=1)
    avg_dist = np.mean(distances)
    max_dist = np.max(distances)

    # Cosine similarity (overall pose direction)
    def cosine_sim(a, b):
        a_flat = a.flatten()
        b_flat = b.flatten()
        if np.linalg.norm(a_flat) == 0 or np.linalg.norm(b_flat) == 0:
            return 0
        return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    cos_sim = cosine_sim(user, expected)

    # Similarity: lower distance = higher score, combine with cosine similarity
    dist_score = max(0, 1 - avg_dist / 0.3)
    cos_score = max(0, cos_sim)
    score = int((0.7 * dist_score + 0.3 * cos_score) * 100)

    # Feedback: find largest difference and suggest adjustment
    max_idx = int(np.argmax(distances))
    # MediaPipe Pose 33 landmarks (short names)
    landmark_names = [
        "nose", "left eye inner", "left eye", "left eye outer", "right eye inner", "right eye", "right eye outer",
        "left ear", "right ear", "mouth left", "mouth right", "left shoulder", "right shoulder", "left elbow",
        "right elbow", "left wrist", "right wrist", "left pinky", "right pinky", "left index", "right index",
        "left thumb", "right thumb", "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle",
        "left heel", "right heel", "left foot index", "right foot index"
    ]
    name = landmark_names[max_idx] if max_idx < len(landmark_names) else f"landmark {max_idx}"

    # More detailed feedback
    feedback = "Great job! Keep it up."
    if avg_dist > 0.15:
        feedback = f"Try to match the pose more closely, especially your {name}."
    elif distances[max_idx] > 0.12:
        feedback = f"Adjust your {name} for better alignment."
    elif cos_score < 0.7:
        feedback = "Try to match the overall direction of the pose."

    # Per-joint feedback (for frontend visualization)
    joint_feedback = []
    for i, d in enumerate(distances):
        joint_feedback.append({
            "landmark": landmark_names[i] if i < len(landmark_names) else f"landmark {i}",
            "distance": float(d),
            "user": user_landmarks[i],
            "expected": expected_landmarks[i]
        })

    return score, feedback, {
        "avg_distance": float(avg_dist),
        "max_distance": float(max_dist),
        "cosine_similarity": float(cos_score),
        "joint_feedback": joint_feedback
    }
