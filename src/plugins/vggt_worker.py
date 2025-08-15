#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "numpy",
#     "opencv-python",
#     "mmap-sync",
#     "scikit-learn",
#     "huggingface_hub",
# ]
# ///

"""
VGGT Worker - Python process for real-time gaussian generation using StreamVGGT
Communicates with Rust via shared memory (mmap-sync)
"""

import os
import sys
import time
import signal
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import struct

import numpy as np
import torch
import cv2
from sklearn.neighbors import NearestNeighbors

# Add StreamVGGT to path
sys.path.append(str(Path(__file__).parent.parent.parent / "StreamVGGT" / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[VGGT Worker] %(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Shared Memory Structures (matching Rust definitions)
# ============================================================================

@dataclass
class SharedFrame:
    width: int
    height: int
    timestamp_ms: int
    camera_position: Tuple[float, float, float]
    camera_rotation: Tuple[float, float, float, float]
    frame_data: np.ndarray

@dataclass
class SharedGaussian:
    position: Tuple[float, float, float]
    quaternion: Tuple[float, float, float, float]
    scale: Tuple[float, float, float]
    opacity: float
    color: Tuple[float, float, float]
    confidence: float

@dataclass
class SharedGaussianBatch:
    count: int
    timestamp_ms: int
    camera_position: Tuple[float, float, float]
    gaussians: List[SharedGaussian]

# ============================================================================
# Memory-mapped communication
# ============================================================================

class MemorySync:
    """Handles shared memory communication with Rust"""
    
    def __init__(self, frame_path: str, gaussian_path: str, control_path: str):
        # Import mmap_sync
        try:
            from mmap_sync import Synchronizer
            self.Synchronizer = Synchronizer
        except ImportError:
            logger.error("mmap_sync not found. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mmap-sync"])
            from mmap_sync import Synchronizer
            self.Synchronizer = Synchronizer
        
        self.frame_reader = Synchronizer(frame_path)
        self.gaussian_writer = Synchronizer(gaussian_path)
        self.control_reader = Synchronizer(control_path)
        
        logger.info(f"Memory sync initialized:")
        logger.info(f"  Frame path: {frame_path}")
        logger.info(f"  Gaussian path: {gaussian_path}")
        logger.info(f"  Control path: {control_path}")
    
    def read_frame(self) -> Optional[SharedFrame]:
        """Read frame from shared memory"""
        try:
            result = self.frame_reader.read()
            if result and result.data:
                # Parse the binary data
                data = result.data
                # This is simplified - you'd need proper deserialization
                # For now, return None to indicate implementation needed
                return None
        except Exception as e:
            logger.debug(f"No frame available: {e}")
            return None
    
    def write_gaussians(self, batch: SharedGaussianBatch):
        """Write gaussian batch to shared memory"""
        try:
            # Serialize the batch (simplified - needs proper implementation)
            # For now, we'll write a placeholder
            data = {
                'count': batch.count,
                'timestamp_ms': batch.timestamp_ms,
                'camera_position': batch.camera_position,
                'gaussians': [
                    {
                        'position': g.position,
                        'quaternion': g.quaternion,
                        'scale': g.scale,
                        'opacity': g.opacity,
                        'color': g.color,
                        'confidence': g.confidence
                    }
                    for g in batch.gaussians
                ]
            }
            
            grace_duration = 0.01  # 10ms
            self.gaussian_writer.write(data, grace_duration)
            logger.info(f"Wrote {batch.count} gaussians to shared memory")
        except Exception as e:
            logger.error(f"Failed to write gaussians: {e}")

# ============================================================================
# StreamVGGT Integration
# ============================================================================

class VGGTProcessor:
    """Handles StreamVGGT model loading and inference"""
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Try to import StreamVGGT
        try:
            from streamvggt.models.streamvggt import StreamVGGT
            from streamvggt.utils.load_fn import load_and_preprocess_images
            from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
            
            self.StreamVGGT = StreamVGGT
            self.load_and_preprocess_images = load_and_preprocess_images
            self.pose_encoding_to_extri_intri = pose_encoding_to_extri_intri
        except ImportError as e:
            logger.error(f"Failed to import StreamVGGT: {e}")
            logger.error("Please ensure StreamVGGT is in the correct path")
            raise
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        # Streaming state
        self.frame_buffer = []
        self.max_buffer_size = 5
        self.last_process_time = 0
        self.min_process_interval = 0.5  # Process at most every 500ms
    
    def _load_model(self, checkpoint_path: Optional[str]):
        """Load StreamVGGT model"""
        model = self.StreamVGGT()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(ckpt, strict=True)
            del ckpt
        else:
            # Try to download from HuggingFace
            try:
                from huggingface_hub import hf_hub_download
                logger.info("Downloading model from HuggingFace...")
                path = hf_hub_download(
                    repo_id="lch01/StreamVGGT",
                    filename="checkpoints.pth",
                    revision="main"
                )
                ckpt = torch.load(path, map_location="cpu")
                model.load_state_dict(ckpt, strict=True)
                del ckpt
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                logger.warning("Using randomly initialized model for testing")
        
        return model.to(self.device)
    
    def process_frame(self, frame: SharedFrame) -> Optional[SharedGaussianBatch]:
        """Process a single frame and extract gaussians"""
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_process_time < self.min_process_interval:
            return None
        
        # Convert frame data to image
        if frame.frame_data is None or len(frame.frame_data) == 0:
            return None
        
        # Reshape RGBA data to image
        try:
            image = np.frombuffer(frame.frame_data, dtype=np.uint8)
            image = image.reshape((frame.height, frame.width, 4))
            # Convert RGBA to RGB
            image = image[:, :, :3]
            
            # Add to buffer
            self.frame_buffer.append(image)
            if len(self.frame_buffer) > self.max_buffer_size:
                self.frame_buffer.pop(0)
            
            # Process buffer
            gaussians = self._extract_gaussians_from_buffer()
            
            if gaussians:
                self.last_process_time = current_time
                return SharedGaussianBatch(
                    count=len(gaussians),
                    timestamp_ms=frame.timestamp_ms,
                    camera_position=frame.camera_position,
                    gaussians=gaussians
                )
        except Exception as e:
            logger.error(f"Failed to process frame: {e}")
            logger.error(traceback.format_exc())
        
        return None
    
    def _extract_gaussians_from_buffer(self) -> List[SharedGaussian]:
        """Extract gaussians from frame buffer using StreamVGGT"""
        
        if len(self.frame_buffer) < 2:
            return []
        
        try:
            # Stack frames for batch processing
            images = np.stack(self.frame_buffer[-3:])  # Use last 3 frames
            
            # Convert to tensor and preprocess
            images_tensor = torch.from_numpy(images).float()
            images_tensor = images_tensor.permute(0, 3, 1, 2)  # NHWC to NCHW
            images_tensor = images_tensor.to(self.device)
            
            # Normalize (simplified - should match StreamVGGT preprocessing)
            images_tensor = images_tensor / 255.0
            
            # Prepare frames for model
            frames = []
            for i in range(images_tensor.shape[0]):
                frames.append({"img": images_tensor[i:i+1]})
            
            # Run inference
            with torch.no_grad():
                dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                with torch.cuda.amp.autocast(dtype=dtype):
                    output = self.model.inference(frames)
            
            # Extract predictions
            gaussians = []
            for res in output.ress:
                pts3d = res['pts3d_in_other_view'].squeeze(0).cpu().numpy()
                conf = res['conf'].squeeze(0).cpu().numpy()
                
                # Sample points (simplified)
                h, w = pts3d.shape[:2]
                sample_rate = 100  # Sample every 100th point
                
                for y in range(0, h, sample_rate):
                    for x in range(0, w, sample_rate):
                        if conf[y, x] > 0.3:  # Confidence threshold
                            position = pts3d[y, x]
                            
                            # Generate random quaternion for rotation
                            quat = np.random.randn(4)
                            quat = quat / np.linalg.norm(quat)
                            
                            # Scale based on confidence
                            scale_value = 0.01 * (2.0 - conf[y, x])
                            
                            gaussian = SharedGaussian(
                                position=tuple(position),
                                quaternion=tuple(quat),
                                scale=(scale_value, scale_value, scale_value),
                                opacity=float(0.95 * conf[y, x]),
                                color=(0.5, 0.5, 0.5),  # Default gray
                                confidence=float(conf[y, x])
                            )
                            gaussians.append(gaussian)
            
            logger.info(f"Extracted {len(gaussians)} gaussians from buffer")
            return gaussians[:1000]  # Limit to 1000 gaussians per batch
            
        except Exception as e:
            logger.error(f"Failed to extract gaussians: {e}")
            logger.error(traceback.format_exc())
            return []

# ============================================================================
# Fallback: Simple gaussian generator (when StreamVGGT unavailable)
# ============================================================================

class SimpleGaussianGenerator:
    """Fallback gaussian generator for testing"""
    
    def __init__(self):
        logger.warning("Using simple gaussian generator (StreamVGGT not available)")
        self.frame_count = 0
    
    def process_frame(self, frame: SharedFrame) -> Optional[SharedGaussianBatch]:
        """Generate random gaussians for testing"""
        
        self.frame_count += 1
        
        # Generate some random gaussians
        num_gaussians = min(100, self.frame_count * 10)
        gaussians = []
        
        for i in range(num_gaussians):
            # Random position in view frustum
            position = (
                np.random.uniform(-10, 10),
                np.random.uniform(-5, 5),
                np.random.uniform(-10, 10)
            )
            
            # Random rotation (quaternion)
            quat = np.random.randn(4)
            quat = quat / np.linalg.norm(quat)
            
            gaussian = SharedGaussian(
                position=position,
                quaternion=tuple(quat),
                scale=(0.05, 0.05, 0.05),
                opacity=0.8,
                color=(
                    np.random.uniform(0, 1),
                    np.random.uniform(0, 1),
                    np.random.uniform(0, 1)
                ),
                confidence=np.random.uniform(0.5, 1.0)
            )
            gaussians.append(gaussian)
        
        if gaussians:
            return SharedGaussianBatch(
                count=len(gaussians),
                timestamp_ms=int(time.time() * 1000),
                camera_position=frame.camera_position if frame else (0, 0, 0),
                gaussians=gaussians
            )
        
        return None

# ============================================================================
# Main Worker Loop
# ============================================================================

class VGGTWorker:
    """Main worker process"""
    
    def __init__(self):
        self.running = True
        self.setup_signal_handlers()
        
        # Get paths from environment
        frame_path = os.environ.get('VGGT_FRAME_PATH', '/tmp/vggt/frames')
        gaussian_path = os.environ.get('VGGT_GAUSSIAN_PATH', '/tmp/vggt/gaussians')
        control_path = os.environ.get('VGGT_CONTROL_PATH', '/tmp/vggt/control')
        checkpoint = os.environ.get('VGGT_MODEL_CHECKPOINT', None)
        
        # Initialize components
        self.memory_sync = MemorySync(frame_path, gaussian_path, control_path)
        
        # Try to initialize VGGT processor, fall back to simple generator
        try:
            self.processor = VGGTProcessor(checkpoint)
        except Exception as e:
            logger.error(f"Failed to initialize VGGTProcessor: {e}")
            self.processor = SimpleGaussianGenerator()
        
        logger.info("VGGT Worker initialized and ready")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown"""
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
    
    def shutdown(self, signum, frame):
        """Graceful shutdown"""
        logger.info("Shutting down VGGT worker...")
        self.running = False
    
    def run(self):
        """Main processing loop"""
        logger.info("Starting VGGT worker main loop")
        
        frame_count = 0
        last_frame_time = 0
        
        while self.running:
            try:
                # Read frame from shared memory
                frame = self.memory_sync.read_frame()
                
                if frame:
                    frame_count += 1
                    current_time = time.time()
                    
                    # Log frame rate
                    if current_time - last_frame_time > 1.0:
                        logger.info(f"Processing frame {frame_count}")
                        last_frame_time = current_time
                    
                    # Process frame and generate gaussians
                    gaussian_batch = self.processor.process_frame(frame)
                    
                    if gaussian_batch:
                        # Write gaussians to shared memory
                        self.memory_sync.write_gaussians(gaussian_batch)
                else:
                    # No frame available, sleep briefly
                    time.sleep(0.01)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(0.1)
        
        logger.info("VGGT worker stopped")

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting VGGT Worker Process")
    
    try:
        worker = VGGTWorker()
        worker.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)