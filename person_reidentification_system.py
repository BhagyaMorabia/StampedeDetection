"""
Person Re-identification System for STAMPede Detection
Tracks individuals across multiple camera feeds using deep learning and appearance features
"""

import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import json
import os
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - using basic appearance features")

@dataclass
class PersonDetection:
    """Represents a detected person"""
    id: int
    camera_id: int
    timestamp: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[float, float]
    confidence: float
    appearance_features: Optional[np.ndarray] = None
    color_features: Optional[np.ndarray] = None
    shape_features: Optional[np.ndarray] = None

@dataclass
class PersonTrack:
    """Represents a tracked person across cameras"""
    global_id: int
    detections: List[PersonDetection]
    appearance_signature: np.ndarray
    first_seen: float
    last_seen: float
    camera_history: List[int]
    movement_pattern: List[Tuple[float, float]]
    confidence_score: float

@dataclass
class ReIDResult:
    """Result of person re-identification"""
    query_detection: PersonDetection
    matched_track: Optional[PersonTrack]
    similarity_score: float
    confidence: float
    is_new_person: bool
    global_id: int

class PersonReIdentifier:
    """Advanced person re-identification system"""
    
    def __init__(self, max_tracks: int = 1000, similarity_threshold: float = 0.7):
        self.max_tracks = max_tracks
        self.similarity_threshold = similarity_threshold
        
        # Person tracking
        self.active_tracks: Dict[int, PersonTrack] = {}
        self.global_id_counter = 0
        self.detection_history = deque(maxlen=10000)
        
        # Feature extraction
        self.feature_extractor = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Performance tracking
        self.reid_accuracy = 0.0
        self.tracking_accuracy = 0.0
        self.false_positive_rate = 0.0
        
        # Camera management
        self.camera_calibrations = {}
        self.camera_overlaps = {}
        
        # Create model directory
        os.makedirs("models", exist_ok=True)
    
    def extract_appearance_features(self, detection: PersonDetection, 
                                  frame: np.ndarray) -> np.ndarray:
        """Extract appearance features from person detection"""
        try:
            # Extract bounding box region
            x, y, w, h = detection.bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Ensure bounding box is within frame
            x = max(0, min(x, frame.shape[1] - 1))
            y = max(0, min(y, frame.shape[0] - 1))
            w = max(1, min(w, frame.shape[1] - x))
            h = max(1, min(h, frame.shape[0] - y))
            
            person_roi = frame[y:y+h, x:x+w]
            
            if person_roi.size == 0:
                return np.zeros(128)  # Default feature vector
            
            # Resize to standard size
            person_roi = cv2.resize(person_roi, (64, 128))
            
            # Extract multiple types of features
            features = []
            
            # Color histogram features
            color_features = self._extract_color_features(person_roi)
            features.extend(color_features)
            
            # Texture features
            texture_features = self._extract_texture_features(person_roi)
            features.extend(texture_features)
            
            # Shape features
            shape_features = self._extract_shape_features(person_roi)
            features.extend(shape_features)
            
            # HOG features
            hog_features = self._extract_hog_features(person_roi)
            features.extend(hog_features)
            
            # Ensure feature vector has consistent size
            feature_vector = np.array(features, dtype=np.float32)
            
            # Pad or truncate to 128 dimensions
            if len(feature_vector) < 128:
                feature_vector = np.pad(feature_vector, (0, 128 - len(feature_vector)))
            else:
                feature_vector = feature_vector[:128]
            
            return feature_vector
            
        except Exception as e:
            print(f"⚠️ Feature extraction error: {e}")
            return np.zeros(128)
    
    def _extract_color_features(self, roi: np.ndarray) -> List[float]:
        """Extract color histogram features"""
        features = []
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        
        # Calculate histograms for each channel
        for channel in [0, 1, 2]:  # BGR channels
            hist = cv2.calcHist([roi], [channel], None, [16], [0, 256])
            features.extend(hist.flatten())
        
        # HSV histograms
        for channel in [0, 1, 2]:  # HSV channels
            hist = cv2.calcHist([hsv], [channel], None, [16], [0, 256])
            features.extend(hist.flatten())
        
        # Dominant colors
        pixels = roi.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        features.append(len(unique_colors) / 1000.0)  # Normalize
        
        return features
    
    def _extract_texture_features(self, roi: np.ndarray) -> List[float]:
        """Extract texture features using LBP-like approach"""
        features = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Statistical features
        features.extend([
            np.mean(magnitude),
            np.std(magnitude),
            np.mean(direction),
            np.std(direction),
            np.percentile(magnitude, 25),
            np.percentile(magnitude, 75),
        ])
        
        # Local Binary Pattern approximation
        lbp_features = self._calculate_lbp_features(gray)
        features.extend(lbp_features)
        
        return features
    
    def _calculate_lbp_features(self, gray: np.ndarray) -> List[float]:
        """Calculate Local Binary Pattern features"""
        features = []
        
        # Simple LBP implementation
        h, w = gray.shape
        lbp_image = np.zeros_like(gray)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray[i, j]
                binary_string = ""
                
                # 8-neighborhood
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += "1" if neighbor >= center else "0"
                
                lbp_image[i, j] = int(binary_string, 2)
        
        # Calculate histogram
        hist, _ = np.histogram(lbp_image.flatten(), bins=16, range=(0, 256))
        features.extend(hist.flatten())
        
        return features
    
    def _extract_shape_features(self, roi: np.ndarray) -> List[float]:
        """Extract shape and geometric features"""
        features = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate aspect ratio
        h, w = gray.shape
        aspect_ratio = w / h if h > 0 else 1.0
        features.append(aspect_ratio)
        
        # Calculate area
        area = h * w
        features.append(area / 10000.0)  # Normalize
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        features.append(edge_density)
        
        # Contour features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area_contour = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Compactness
            compactness = (perimeter * perimeter) / area_contour if area_contour > 0 else 0
            features.append(compactness)
            
            # Solidity
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area_contour / hull_area if hull_area > 0 else 0
            features.append(solidity)
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _extract_hog_features(self, roi: np.ndarray) -> List[float]:
        """Extract Histogram of Oriented Gradients features"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Calculate gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate magnitude and orientation
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            orientation = np.arctan2(grad_y, grad_x)
            
            # Convert orientation to degrees
            orientation = np.degrees(orientation)
            orientation = (orientation + 360) % 360
            
            # Create HOG histogram
            h, w = gray.shape
            cell_size = 8
            num_bins = 9
            
            hog_features = []
            
            for i in range(0, h - cell_size + 1, cell_size):
                for j in range(0, w - cell_size + 1, cell_size):
                    cell_magnitude = magnitude[i:i+cell_size, j:j+cell_size]
                    cell_orientation = orientation[i:i+cell_size, j:j+cell_size]
                    
                    # Create histogram for this cell
                    hist = np.zeros(num_bins)
                    
                    for mag, orient in zip(cell_magnitude.flatten(), cell_orientation.flatten()):
                        bin_idx = int(orient / (360 / num_bins)) % num_bins
                        hist[bin_idx] += mag
                    
                    hog_features.extend(hist)
            
            # Limit to reasonable size
            if len(hog_features) > 50:
                hog_features = hog_features[:50]
            elif len(hog_features) < 50:
                hog_features.extend([0.0] * (50 - len(hog_features)))
            
            return hog_features
            
        except Exception as e:
            print(f"⚠️ HOG feature extraction error: {e}")
            return [0.0] * 50
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between two feature vectors"""
        try:
            # Normalize features
            features1 = features1 / (np.linalg.norm(features1) + 1e-8)
            features2 = features2 / (np.linalg.norm(features2) + 1e-8)
            
            # Cosine similarity
            cosine_sim = np.dot(features1, features2)
            
            # Euclidean distance (converted to similarity)
            euclidean_dist = np.linalg.norm(features1 - features2)
            euclidean_sim = 1.0 / (1.0 + euclidean_dist)
            
            # Combine similarities
            combined_similarity = 0.7 * cosine_sim + 0.3 * euclidean_sim
            
            return max(0.0, min(1.0, combined_similarity))
            
        except Exception as e:
            print(f"⚠️ Similarity calculation error: {e}")
            return 0.0
    
    def reidentify_person(self, detection: PersonDetection, frame: np.ndarray) -> ReIDResult:
        """Re-identify a person across cameras"""
        
        # Extract appearance features
        appearance_features = self.extract_appearance_features(detection, frame)
        detection.appearance_features = appearance_features
        
        # Store detection
        self.detection_history.append(detection)
        
        # Find best match among active tracks
        best_match = None
        best_similarity = 0.0
        
        for track_id, track in self.active_tracks.items():
            if track.camera_history[-1] == detection.camera_id:
                continue  # Skip same camera
            
            # Calculate similarity
            similarity = self.calculate_similarity(appearance_features, track.appearance_signature)
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match = track
        
        if best_match is not None:
            # Update existing track
            best_match.detections.append(detection)
            best_match.last_seen = detection.timestamp
            best_match.camera_history.append(detection.camera_id)
            best_match.movement_pattern.append(detection.center)
            
            # Update appearance signature (moving average)
            alpha = 0.1  # Learning rate
            best_match.appearance_signature = (
                (1 - alpha) * best_match.appearance_signature + 
                alpha * appearance_features
            )
            
            # Calculate confidence
            confidence = min(0.95, best_similarity + 0.1)
            best_match.confidence_score = confidence
            
            return ReIDResult(
                query_detection=detection,
                matched_track=best_match,
                similarity_score=best_similarity,
                confidence=confidence,
                is_new_person=False,
                global_id=best_match.global_id
            )
        else:
            # Create new track
            new_track = PersonTrack(
                global_id=self.global_id_counter,
                detections=[detection],
                appearance_signature=appearance_features.copy(),
                first_seen=detection.timestamp,
                last_seen=detection.timestamp,
                camera_history=[detection.camera_id],
                movement_pattern=[detection.center],
                confidence_score=0.5
            )
            
            self.active_tracks[self.global_id_counter] = new_track
            self.global_id_counter += 1
            
            # Cleanup old tracks
            self._cleanup_tracks()
            
            return ReIDResult(
                query_detection=detection,
                matched_track=new_track,
                similarity_score=0.0,
                confidence=0.5,
                is_new_person=True,
                global_id=new_track.global_id
            )
    
    def _cleanup_tracks(self):
        """Remove old or low-confidence tracks"""
        current_time = time.time()
        tracks_to_remove = []
        
        for track_id, track in self.active_tracks.items():
            # Remove tracks that haven't been seen for 5 minutes
            if current_time - track.last_seen > 300:
                tracks_to_remove.append(track_id)
            # Remove tracks with very low confidence
            elif track.confidence_score < 0.2:
                tracks_to_remove.append(track_id)
            # Remove tracks with too many detections (likely false positives)
            elif len(track.detections) > 100:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
        
        # Limit total number of tracks
        if len(self.active_tracks) > self.max_tracks:
            # Remove oldest tracks
            sorted_tracks = sorted(self.active_tracks.items(), 
                                 key=lambda x: x[1].last_seen)
            tracks_to_remove = [track_id for track_id, _ in sorted_tracks[:-self.max_tracks]]
            
            for track_id in tracks_to_remove:
                del self.active_tracks[track_id]
    
    def get_track_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics"""
        current_time = time.time()
        
        # Calculate track statistics
        active_tracks = len(self.active_tracks)
        total_detections = len(self.detection_history)
        
        # Camera coverage
        camera_coverage = defaultdict(int)
        for track in self.active_tracks.values():
            for camera_id in track.camera_history:
                camera_coverage[camera_id] += 1
        
        # Track durations
        track_durations = []
        for track in self.active_tracks.values():
            duration = track.last_seen - track.first_seen
            track_durations.append(duration)
        
        avg_track_duration = np.mean(track_durations) if track_durations else 0.0
        
        # Movement analysis
        total_movement = 0.0
        for track in self.active_tracks.values():
            if len(track.movement_pattern) > 1:
                for i in range(1, len(track.movement_pattern)):
                    dx = track.movement_pattern[i][0] - track.movement_pattern[i-1][0]
                    dy = track.movement_pattern[i][1] - track.movement_pattern[i-1][1]
                    total_movement += np.sqrt(dx*dx + dy*dy)
        
        return {
            'active_tracks': active_tracks,
            'total_detections': total_detections,
            'global_id_counter': self.global_id_counter,
            'camera_coverage': dict(camera_coverage),
            'average_track_duration': avg_track_duration,
            'total_movement': total_movement,
            'reid_accuracy': self.reid_accuracy,
            'tracking_accuracy': self.tracking_accuracy,
            'false_positive_rate': self.false_positive_rate,
            'similarity_threshold': self.similarity_threshold,
            'max_tracks': self.max_tracks
        }
    
    def get_track_history(self, global_id: int) -> Optional[PersonTrack]:
        """Get track history for a specific global ID"""
        return self.active_tracks.get(global_id)
    
    def get_camera_tracks(self, camera_id: int) -> List[PersonTrack]:
        """Get all tracks that have been seen by a specific camera"""
        camera_tracks = []
        
        for track in self.active_tracks.values():
            if camera_id in track.camera_history:
                camera_tracks.append(track)
        
        return camera_tracks
    
    def simulate_person_detection(self, camera_id: int, person_id: int = None) -> PersonDetection:
        """Simulate person detection for testing"""
        if person_id is None:
            person_id = np.random.randint(0, 100)
        
        # Simulate bounding box
        x = np.random.randint(50, 400)
        y = np.random.randint(50, 300)
        w = np.random.randint(30, 80)
        h = np.random.randint(60, 120)
        
        center_x = x + w / 2
        center_y = y + h / 2
        
        return PersonDetection(
            id=person_id,
            camera_id=camera_id,
            timestamp=time.time(),
            bbox=(x, y, w, h),
            center=(center_x, center_y),
            confidence=0.8 + 0.2 * np.random.random(),
            appearance_features=None,
            color_features=None,
            shape_features=None
        )
    
    def simulate_frame(self, camera_id: int, num_people: int = 5) -> np.ndarray:
        """Simulate a video frame for testing"""
        # Create a simple frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some "people" as colored rectangles
        for i in range(num_people):
            x = np.random.randint(50, 500)
            y = np.random.randint(50, 350)
            w = np.random.randint(30, 60)
            h = np.random.randint(60, 100)
            
            color = (np.random.randint(0, 255), 
                    np.random.randint(0, 255), 
                    np.random.randint(0, 255))
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        
        return frame

# Example usage and testing
if __name__ == "__main__":
    # Initialize re-identifier
    reid = PersonReIdentifier()
    
    # Simulate multi-camera scenario
    print("🧪 Simulating multi-camera person tracking...")
    
    # Simulate detections from different cameras
    cameras = [0, 1, 2]
    people_per_camera = 3
    
    for frame_idx in range(10):  # 10 frames
        print(f"\n📹 Frame {frame_idx + 1}:")
        
        for camera_id in cameras:
            # Simulate frame
            frame = reid.simulate_frame(camera_id, people_per_camera)
            
            # Simulate detections
            for person_idx in range(people_per_camera):
                detection = reid.simulate_person_detection(camera_id, person_idx)
                
                # Re-identify person
                result = reid.reidentify_person(detection, frame)
                
                print(f"   Camera {camera_id}, Person {person_idx}:")
                print(f"     Global ID: {result.global_id}")
                print(f"     Is New: {result.is_new_person}")
                print(f"     Similarity: {result.similarity_score:.3f}")
                print(f"     Confidence: {result.confidence:.3f}")
        
        # Small delay between frames
        time.sleep(0.1)
    
    # Get statistics
    stats = reid.get_track_statistics()
    print(f"\n📈 Re-identification Statistics:")
    print(f"   Active Tracks: {stats['active_tracks']}")
    print(f"   Total Detections: {stats['total_detections']}")
    print(f"   Global ID Counter: {stats['global_id_counter']}")
    print(f"   Camera Coverage: {stats['camera_coverage']}")
    print(f"   Average Track Duration: {stats['average_track_duration']:.2f}s")
    print(f"   Total Movement: {stats['total_movement']:.2f} pixels")
    print(f"   ReID Accuracy: {stats['reid_accuracy']:.3f}")
    print(f"   Tracking Accuracy: {stats['tracking_accuracy']:.3f}")
    print(f"   False Positive Rate: {stats['false_positive_rate']:.3f}")
    
    # Test track history
    print(f"\n🔍 Track History Examples:")
    for global_id in list(reid.active_tracks.keys())[:3]:  # Show first 3 tracks
        track = reid.get_track_history(global_id)
        if track:
            print(f"   Track {global_id}:")
            print(f"     Detections: {len(track.detections)}")
            print(f"     Cameras: {track.camera_history}")
            print(f"     Duration: {track.last_seen - track.first_seen:.2f}s")
            print(f"     Confidence: {track.confidence_score:.3f}")
