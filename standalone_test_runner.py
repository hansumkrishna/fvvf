#!/usr/bin/env python3
"""
Standalone File-Video Conversion Test Runner
Contains all classes in one file for easy testing
"""

import os
import cv2
import numpy as np
import struct
import zlib
import hashlib
from typing import List, Tuple, Optional, Dict, Any
import argparse
import json
from pathlib import Path
from collections import defaultdict
import tempfile
import shutil
import time
import random
import string

# ============================================================================
# ENCODER CLASS
# ============================================================================

class FileToVideoEncoder:
    def __init__(self, 
                 frame_width: int = 1280, 
                 frame_height: int = 720, 
                 fps: int = 20,
                 cell_size: int = 8):
        """
        Initialize the encoder with video parameters
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.cell_size = cell_size
        
        # Calculate grid dimensions
        self.grid_width = frame_width // cell_size
        self.grid_height = frame_height // cell_size
        
        # Reserve space for headers and sync patterns
        self.header_rows = 3  # Top rows for sync pattern and metadata
        self.data_rows = self.grid_height - self.header_rows - 1  # Bottom row for frame CRC
        self.data_cols = self.grid_width - 2  # Side columns for alignment
        
        # Calculate payload capacity
        self.data_bits_per_frame = self.data_rows * self.data_cols
        self.data_bytes_per_frame = self.data_bits_per_frame // 8
        
        # Account for frame metadata (sequence, chunk info, etc.)
        self.metadata_bytes = 16  # 8 bytes for sequence + chunk info + flags
        self.frame_crc_bytes = 4
        self.payload_bytes_per_frame = self.data_bytes_per_frame - self.metadata_bytes - self.frame_crc_bytes
        
        print(f"Encoder initialized:")
        print(f"  Grid: {self.grid_width}x{self.grid_height}")
        print(f"  Data area: {self.data_cols}x{self.data_rows}")
        print(f"  Payload per frame: {self.payload_bytes_per_frame} bytes")
        print(f"  Effective bitrate: {self.payload_bytes_per_frame * 8 * fps} bps")

    def generate_sync_pattern(self) -> np.ndarray:
        """Generate a recognizable sync pattern for frame detection"""
        pattern = np.zeros((self.header_rows, self.grid_width), dtype=np.uint8)
        
        # Create alternating pattern for sync
        for i in range(self.grid_width):
            if i % 4 < 2:
                pattern[0, i] = 255
        
        # Frame start marker
        pattern[1, :8] = [255, 0, 255, 0, 255, 255, 0, 0]
        
        return pattern

    def encode_metadata(self, frame_seq: int, chunk_id: int, total_chunks: int, 
                       is_last_frame: bool, chunk_size: int) -> bytes:
        """Encode frame metadata into bytes"""
        flags = 0
        if is_last_frame:
            flags |= 1
        
        # Handle special chunk_id for metadata frames (-1 -> 0xFFFFFFFF)
        if chunk_id == -1:
            chunk_id = 0xFFFFFFFF
            
        metadata = struct.pack('>IIHHI', 
                              frame_seq,     # 4 bytes: frame sequence number
                              chunk_id,      # 4 bytes: chunk ID (0xFFFFFFFF for metadata)
                              total_chunks,  # 2 bytes: total chunks
                              flags,         # 2 bytes: flags
                              chunk_size)    # 4 bytes: chunk size
        return metadata

    def bytes_to_bits(self, data: bytes) -> List[int]:
        """Convert bytes to list of bits"""
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> (7-i)) & 1)
        return bits

    def create_frame(self, payload: bytes, frame_seq: int, chunk_id: int, 
                    total_chunks: int, is_last_frame: bool, chunk_size: int) -> np.ndarray:
        """Create a video frame with encoded data"""
        frame = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        
        # Add sync pattern
        sync_pattern = self.generate_sync_pattern()
        frame[:self.header_rows, :] = sync_pattern
        
        # Encode metadata
        metadata = self.encode_metadata(frame_seq, chunk_id, total_chunks, 
                                      is_last_frame, chunk_size)
        
        # Combine metadata + payload
        frame_data = metadata + payload
        
        # Calculate frame CRC
        frame_crc = zlib.crc32(frame_data) & 0xffffffff
        frame_data += struct.pack('>I', frame_crc)
        
        # Pad to fill frame if necessary
        if len(frame_data) < self.data_bytes_per_frame:
            frame_data += b'\x00' * (self.data_bytes_per_frame - len(frame_data))
        
        # Convert to bits
        bits = self.bytes_to_bits(frame_data)
        
        # Fill data area
        bit_idx = 0
        for row in range(self.header_rows, self.grid_height - 1):
            for col in range(1, self.grid_width - 1):
                if bit_idx < len(bits):
                    frame[row, col] = 255 if bits[bit_idx] else 0
                    bit_idx += 1
        
        # Add border for alignment
        frame[:, 0] = 255  # Left border
        frame[:, -1] = 255  # Right border
        frame[-1, :] = 255  # Bottom border
        
        return frame

    def grid_to_image(self, grid: np.ndarray) -> np.ndarray:
        """Convert grid to full resolution image"""
        image = np.repeat(grid, self.cell_size, axis=0)
        image = np.repeat(image, self.cell_size, axis=1)
        return image

    def split_file(self, filepath: str, chunk_size: int = None) -> List[Tuple[bytes, str]]:
        """Split file into chunks with CRC"""
        if chunk_size is None:
            # Calculate optimal chunk size based on frame capacity
            chunk_size = self.payload_bytes_per_frame * 100  # 100 frames per chunk
        
        chunks = []
        with open(filepath, 'rb') as f:
            chunk_id = 0
            while True:
                chunk_data = f.read(chunk_size)
                if not chunk_data:
                    break
                
                chunk_crc = hashlib.md5(chunk_data).hexdigest()
                chunks.append((chunk_data, chunk_crc))
                chunk_id += 1
        
        return chunks

    def encode_file_to_video(self, input_file: str, output_video: str, 
                           chunk_size: int = None) -> str:
        """
        Encode a file into a video
        
        Returns metadata JSON string with file info
        """
        # Get file info
        file_path = Path(input_file)
        file_size = file_path.stat().st_size
        file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
        
        # Split file into chunks
        chunks = self.split_file(input_file, chunk_size)
        total_chunks = len(chunks)
        
        print(f"Encoding {input_file} ({file_size} bytes) into {total_chunks} chunks")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video, fourcc, self.fps, 
                                     (self.frame_width, self.frame_height), False)
        
        frame_seq = 0
        
        # Add file header frames
        file_metadata = {
            'filename': file_path.name,
            'file_size': file_size,
            'file_hash': file_hash,
            'total_chunks': total_chunks,
            'encoder_version': '1.0'
        }
        
        metadata_json = json.dumps(file_metadata).encode('utf-8')
        metadata_chunks = [metadata_json[i:i+self.payload_bytes_per_frame] 
                          for i in range(0, len(metadata_json), self.payload_bytes_per_frame)]
        
        # Write metadata frames
        for i, meta_chunk in enumerate(metadata_chunks):
            grid = self.create_frame(
                meta_chunk, frame_seq, -1, len(metadata_chunks), 
                i == len(metadata_chunks) - 1, len(metadata_json)
            )
            image = self.grid_to_image(grid)
            video_writer.write(image)
            frame_seq += 1
        
        # Write data frames
        for chunk_id, (chunk_data, chunk_crc) in enumerate(chunks):
            # Add chunk CRC to beginning of chunk
            chunk_with_crc = chunk_crc.encode('utf-8') + b'|' + chunk_data
            
            # Split chunk into frames
            frames_in_chunk = []
            for i in range(0, len(chunk_with_crc), self.payload_bytes_per_frame):
                frame_payload = chunk_with_crc[i:i+self.payload_bytes_per_frame]
                frames_in_chunk.append(frame_payload)
            
            # Write frames for this chunk
            for i, frame_payload in enumerate(frames_in_chunk):
                is_last_frame = (i == len(frames_in_chunk) - 1)
                
                grid = self.create_frame(
                    frame_payload, frame_seq, chunk_id, total_chunks,
                    is_last_frame, len(chunk_with_crc)
                )
                image = self.grid_to_image(grid)
                video_writer.write(image)
                frame_seq += 1
            
            print(f"Encoded chunk {chunk_id + 1}/{total_chunks}")
        
        video_writer.release()
        print(f"Video saved: {output_video}")
        print(f"Total frames: {frame_seq}")
        
        return json.dumps(file_metadata, indent=2)

# ============================================================================
# DECODER CLASS
# ============================================================================

class VideoToFileDecoder:
    def __init__(self, 
                 frame_width: int = 1280, 
                 frame_height: int = 720,
                 cell_size: int = 8,
                 adaptive_threshold: bool = True):
        """
        Initialize the decoder with video parameters
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cell_size = cell_size
        self.adaptive_threshold = adaptive_threshold
        
        # Calculate grid dimensions (must match encoder)
        self.grid_width = frame_width // cell_size
        self.grid_height = frame_height // cell_size
        
        # Frame structure (must match encoder)
        self.header_rows = 3
        self.data_rows = self.grid_height - self.header_rows - 1
        self.data_cols = self.grid_width - 2
        
        self.data_bits_per_frame = self.data_rows * self.data_cols
        self.data_bytes_per_frame = self.data_bits_per_frame // 8
        
        self.metadata_bytes = 16
        self.frame_crc_bytes = 4
        self.payload_bytes_per_frame = self.data_bytes_per_frame - self.metadata_bytes - self.frame_crc_bytes
        
        print(f"Decoder initialized:")
        print(f"  Grid: {self.grid_width}x{self.grid_height}")
        print(f"  Data area: {self.data_cols}x{self.data_rows}")
        print(f"  Payload per frame: {self.payload_bytes_per_frame} bytes")

    def image_to_grid(self, image: np.ndarray) -> np.ndarray:
        """Convert full resolution image to grid by averaging cells"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize if necessary
        if image.shape != (self.frame_height, self.frame_width):
            image = cv2.resize(image, (self.frame_width, self.frame_height))
        
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                # Extract cell
                y1 = row * self.cell_size
                y2 = (row + 1) * self.cell_size
                x1 = col * self.cell_size
                x2 = (col + 1) * self.cell_size
                
                cell = image[y1:y2, x1:x2]
                grid[row, col] = np.mean(cell)
        
        return grid

    def detect_sync_pattern(self, grid: np.ndarray) -> Tuple[bool, float]:
        """
        Detect sync pattern in grid
        Returns (found, confidence)
        """
        if grid.shape[0] < self.header_rows:
            return False, 0.0
        
        # Check for alternating pattern in first row
        threshold = 128  # Middle threshold
        first_row = grid[0, :]
        
        # Binarize first row
        binary_row = (first_row > threshold).astype(int)
        
        # Expected pattern: alternating groups of 2
        expected_pattern = []
        for i in range(self.grid_width):
            expected_pattern.append(1 if (i % 4) < 2 else 0)
        
        # Calculate correlation
        matches = sum(1 for i in range(min(len(binary_row), len(expected_pattern))) 
                     if binary_row[i] == expected_pattern[i])
        confidence = matches / len(expected_pattern)
        
        # Check frame start marker in second row
        if grid.shape[0] > 1 and grid.shape[1] >= 8:
            second_row = grid[1, :8]
            binary_second = (second_row > threshold).astype(int)
            expected_marker = [1, 0, 1, 0, 1, 1, 0, 0]
            
            marker_matches = sum(1 for i in range(8) 
                               if binary_second[i] == expected_marker[i])
            marker_confidence = marker_matches / 8
            
            # Combine confidences
            confidence = (confidence + marker_confidence) / 2
        
        return confidence > 0.7, confidence

    def grid_to_bits(self, grid: np.ndarray) -> List[int]:
        """Convert grid data area to bits"""
        if self.adaptive_threshold:
            # Use adaptive threshold based on grid statistics
            data_area = grid[self.header_rows:-1, 1:-1]
            threshold = np.mean(data_area)
        else:
            threshold = 128
        
        bits = []
        for row in range(self.header_rows, self.grid_height - 1):
            for col in range(1, self.grid_width - 1):
                bit = 1 if grid[row, col] > threshold else 0
                bits.append(bit)
        
        return bits

    def bits_to_bytes(self, bits: List[int]) -> bytes:
        """Convert list of bits to bytes"""
        # Pad to byte boundary
        while len(bits) % 8 != 0:
            bits.append(0)
        
        bytes_data = bytearray()
        for i in range(0, len(bits), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(bits):
                    byte_val |= (bits[i + j] << (7 - j))
            bytes_data.append(byte_val)
        
        return bytes(bytes_data)

    def decode_frame_metadata(self, frame_bytes: bytes) -> Optional[Dict[str, Any]]:
        """Decode frame metadata from bytes"""
        if len(frame_bytes) < self.metadata_bytes:
            return None
        
        try:
            metadata_bytes = frame_bytes[:self.metadata_bytes]
            frame_seq, chunk_id, total_chunks, flags, chunk_size = struct.unpack('>IIHHI', metadata_bytes)
            
            # Handle special chunk_id for metadata frames (0xFFFFFFFF -> -1)
            if chunk_id == 0xFFFFFFFF:
                chunk_id = -1
            
            is_last_frame = bool(flags & 1)
            
            return {
                'frame_seq': frame_seq,
                'chunk_id': chunk_id,
                'total_chunks': total_chunks,
                'is_last_frame': is_last_frame,
                'chunk_size': chunk_size
            }
        except struct.error:
            return None

    def verify_frame_crc(self, frame_bytes: bytes) -> bool:
        """Verify frame CRC"""
        if len(frame_bytes) < self.frame_crc_bytes:
            return False
        
        data = frame_bytes[:-self.frame_crc_bytes]
        crc_bytes = frame_bytes[-self.frame_crc_bytes:]
        
        try:
            expected_crc = struct.unpack('>I', crc_bytes)[0]
            actual_crc = zlib.crc32(data) & 0xffffffff
            return expected_crc == actual_crc
        except struct.error:
            return False

    def decode_frame(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Decode a single frame
        Returns frame data dict or None if failed
        """
        # Convert to grid
        grid = self.image_to_grid(image)
        
        # Check sync pattern
        sync_found, confidence = self.detect_sync_pattern(grid)
        if not sync_found:
            return None
        
        # Convert to bits and bytes
        bits = self.grid_to_bits(grid)
        frame_bytes = self.bits_to_bytes(bits)
        
        # Verify frame CRC
        if not self.verify_frame_crc(frame_bytes):
            return None
        
        # Decode metadata
        metadata = self.decode_frame_metadata(frame_bytes)
        if not metadata:
            return None
        
        # Extract payload
        payload_start = self.metadata_bytes
        payload_end = len(frame_bytes) - self.frame_crc_bytes
        payload = frame_bytes[payload_start:payload_end]
        
        return {
            'metadata': metadata,
            'payload': payload,
            'confidence': confidence
        }

    def reconstruct_chunks(self, frames: List[Dict[str, Any]]) -> Dict[int, bytes]:
        """Reconstruct chunks from decoded frames"""
        chunks = defaultdict(list)
        file_metadata = None
        
        # Separate metadata frames and data frames
        metadata_frames = []
        data_frames = []
        
        for frame in frames:
            chunk_id = frame['metadata']['chunk_id']
            if chunk_id == -1:  # Metadata frame
                metadata_frames.append(frame)
            else:
                data_frames.append(frame)
        
        # Reconstruct file metadata
        if metadata_frames:
            metadata_frames.sort(key=lambda x: x['metadata']['frame_seq'])
            metadata_payload = b''.join(f['payload'] for f in metadata_frames)
            # Remove null padding
            metadata_payload = metadata_payload.rstrip(b'\x00')
            try:
                file_metadata = json.loads(metadata_payload.decode('utf-8'))
                print(f"Decoded file metadata: {file_metadata['filename']}")
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"Failed to decode file metadata: {e}")
        
        # Group data frames by chunk
        chunk_frames = defaultdict(list)
        for frame in data_frames:
            chunk_id = frame['metadata']['chunk_id']
            chunk_frames[chunk_id].append(frame)
        
        # Reconstruct each chunk
        reconstructed_chunks = {}
        for chunk_id, frames_list in chunk_frames.items():
            # Sort frames by sequence
            frames_list.sort(key=lambda x: x['metadata']['frame_seq'])
            
            # Combine payloads
            chunk_data = b''.join(f['payload'] for f in frames_list)
            
            # Remove null padding
            chunk_data = chunk_data.rstrip(b'\x00')
            
            # Extract CRC and data
            if b'|' in chunk_data:
                crc_part, data_part = chunk_data.split(b'|', 1)
                expected_crc = crc_part.decode('utf-8')
                actual_crc = hashlib.md5(data_part).hexdigest()
                
                if expected_crc == actual_crc:
                    reconstructed_chunks[chunk_id] = data_part
                    print(f"Chunk {chunk_id}: CRC verified ✓")
                else:
                    print(f"Chunk {chunk_id}: CRC mismatch ✗ (expected: {expected_crc}, got: {actual_crc})")
            else:
                print(f"Chunk {chunk_id}: No CRC found")
        
        return reconstructed_chunks, file_metadata

    def decode_video_to_file(self, input_video: str, output_file: str = None) -> bool:
        """
        Decode video back to original file
        
        Returns True if successful
        """
        if not os.path.exists(input_video):
            print(f"Error: Video file not found: {input_video}")
            return False
        
        print(f"Decoding video: {input_video}")
        
        # Open video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            print("Error: Could not open video")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {total_frames}")
        
        decoded_frames = []
        frame_count = 0
        successful_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Decode frame
            decoded_frame = self.decode_frame(frame)
            if decoded_frame:
                decoded_frames.append(decoded_frame)
                successful_frames += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames, "
                      f"decoded {successful_frames} successfully")
        
        cap.release()
        
        print(f"Successfully decoded {successful_frames}/{frame_count} frames")
        
        if not decoded_frames:
            print("Error: No frames could be decoded")
            return False
        
        # Reconstruct chunks
        chunks, file_metadata = self.reconstruct_chunks(decoded_frames)
        
        if not chunks:
            print("Error: No chunks could be reconstructed")
            return False
        
        # Determine output filename
        if output_file is None:
            if file_metadata and 'filename' in file_metadata:
                output_file = file_metadata['filename']
            else:
                output_file = 'decoded_file.bin'
        
        # Reconstruct complete file
        print(f"Reconstructing file: {output_file}")
        
        with open(output_file, 'wb') as f:
            # Write chunks in order
            chunk_ids = sorted(chunks.keys())
            total_size = 0
            
            for chunk_id in chunk_ids:
                chunk_data = chunks[chunk_id]
                f.write(chunk_data)
                total_size += len(chunk_data)
                print(f"Wrote chunk {chunk_id}: {len(chunk_data)} bytes")
        
        print(f"File reconstructed: {output_file} ({total_size} bytes)")
        
        # Verify file integrity if metadata available
        if file_metadata:
            if 'file_size' in file_metadata:
                expected_size = file_metadata['file_size']
                if total_size == expected_size:
                    print(f"File size verification: ✓ ({total_size} bytes)")
                else:
                    print(f"File size mismatch: expected {expected_size}, got {total_size}")
            
            if 'file_hash' in file_metadata:
                with open(output_file, 'rb') as f:
                    actual_hash = hashlib.md5(f.read()).hexdigest()
                expected_hash = file_metadata['file_hash']
                
                if actual_hash == expected_hash:
                    print(f"File integrity verification: ✓")
                    print(f"MD5: {actual_hash}")
                else:
                    print(f"File integrity mismatch:")
                    print(f"Expected MD5: {expected_hash}")
                    print(f"Actual MD5:   {actual_hash}")
                    return False
        
        return True

# ============================================================================
# TEST SUITE
# ============================================================================

class FileVideoTestSuite:
    def __init__(self, temp_dir: str = None):
        """Initialize test suite with temporary directory"""
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix='file_video_test_')
        self.test_files = []
        print(f"Test directory: {self.temp_dir}")

    def cleanup(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print("Cleanup completed")

    def create_test_file(self, filename: str, size_bytes: int = 1024, 
                        content_type: str = 'random') -> str:
        """Create a test file with specified characteristics"""
        filepath = os.path.join(self.temp_dir, filename)
        
        with open(filepath, 'wb') as f:
            if content_type == 'random':
                # Random binary data
                f.write(os.urandom(size_bytes))
            elif content_type == 'text':
                # Random text data
                text = ''.join(random.choices(string.ascii_letters + string.digits + '\n ',
                                            k=size_bytes))
                f.write(text.encode('utf-8')[:size_bytes])
            elif content_type == 'binary_pattern':
                # Repeating pattern
                pattern = bytes(range(256))
                full_patterns = size_bytes // 256
                remainder = size_bytes % 256
                f.write(pattern * full_patterns + pattern[:remainder])
            elif content_type == 'zeros':
                # All zeros
                f.write(b'\x00' * size_bytes)
            else:
                raise ValueError(f"Unknown content type: {content_type}")
        
        self.test_files.append(filepath)
        return filepath

    def run_encoding_test(self, input_file: str, cell_size: int = 8, fps: int = 20) -> str:
        """Run encoding test and return video path"""
        encoder = FileToVideoEncoder(cell_size=cell_size, fps=fps)
        
        video_path = input_file + '.mp4'
        metadata = encoder.encode_file_to_video(input_file, video_path)
        
        print(f"Encoded {input_file} -> {video_path}")
        return video_path

    def run_decoding_test(self, video_path: str, cell_size: int = 8) -> str:
        """Run decoding test and return decoded file path"""
        decoder = VideoToFileDecoder(cell_size=cell_size)
        
        output_file = video_path + '.decoded'
        success = decoder.decode_video_to_file(video_path, output_file)
        
        if success:
            print(f"Decoded {video_path} -> {output_file}")
            return output_file
        else:
            raise Exception("Decoding failed")

    def verify_file_integrity(self, original_file: str, decoded_file: str) -> bool:
        """Verify that decoded file matches original"""
        if not os.path.exists(decoded_file):
            print("❌ Decoded file does not exist")
            return False
        
        # Compare file sizes
        orig_size = os.path.getsize(original_file)
        decoded_size = os.path.getsize(decoded_file)
        
        if orig_size != decoded_size:
            print(f"❌ Size mismatch: {orig_size} vs {decoded_size}")
            return False
        
        # Compare file hashes
        with open(original_file, 'rb') as f:
            orig_hash = hashlib.md5(f.read()).hexdigest()
        
        with open(decoded_file, 'rb') as f:
            decoded_hash = hashlib.md5(f.read()).hexdigest()
        
        if orig_hash != decoded_hash:
            print(f"❌ Hash mismatch: {orig_hash} vs {decoded_hash}")
            return False
        
        print(f"✅ File integrity verified: {orig_hash}")
        return True

    def run_comprehensive_test(self, test_name: str, file_size: int, 
                             content_type: str, cell_size: int = 8, fps: int = 20):
        """Run a complete encode->decode->verify test"""
        print(f"\n{'='*60}")
        print(f"Running test: {test_name}")
        print(f"File size: {file_size} bytes, Content: {content_type}")
        print(f"Cell size: {cell_size}px, FPS: {fps}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Create test file
            test_file = self.create_test_file(f"test_{test_name}.bin", 
                                            file_size, content_type)
            print(f"Created test file: {os.path.basename(test_file)}")
            
            # Encode to video
            encode_start = time.time()
            video_file = self.run_encoding_test(test_file, cell_size, fps)
            encode_time = time.time() - encode_start
            
            video_size = os.path.getsize(video_file)
            print(f"Video file size: {video_size:,} bytes")
            print(f"Encoding time: {encode_time:.2f} seconds")
            
            # Decode back to file
            decode_start = time.time()
            decoded_file = self.run_decoding_test(video_file, cell_size)
            decode_time = time.time() - decode_start
            
            print(f"Decoding time: {decode_time:.2f} seconds")
            
            # Verify integrity
            success = self.verify_file_integrity(test_file, decoded_file)
            
            total_time = time.time() - start_time
            
            # Calculate efficiency metrics
            compression_ratio = video_size / file_size
            effective_bitrate = (file_size * 8) / (encode_time + decode_time)
            
            print(f"\n📊 Test Results:")
            print(f"   Status: {'✅ PASSED' if success else '❌ FAILED'}")
            print(f"   Total time: {total_time:.2f} seconds")
            print(f"   Video/File size ratio: {compression_ratio:.2f}x")
            print(f"   Effective bitrate: {effective_bitrate:.0f} bps")
            
            return success
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🚀 Starting File-Video Conversion Test Suite")
        
        test_cases = [
            # (test_name, file_size, content_type, cell_size, fps)
            ("small_random", 1024, "random", 8, 20),
            ("medium_text", 10240, "text", 8, 20),
            ("large_pattern", 51200, "binary_pattern", 8, 20),
            ("robust_small_cells", 2048, "random", 4, 20),
            ("robust_large_cells", 2048, "random", 16, 20),
            ("high_fps", 5120, "random", 8, 30),
            ("zeros_test", 4096, "zeros", 8, 20),
        ]
        
        passed = 0
        total = len(test_cases)
        
        for test_name, file_size, content_type, cell_size, fps in test_cases:
            success = self.run_comprehensive_test(test_name, file_size, content_type, 
                                                cell_size, fps)
            if success:
                passed += 1
        
        print(f"\n{'='*60}")
        print(f"🏁 Test Suite Complete: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! System is working correctly.")
        else:
            print(f"⚠️  {total - passed} tests failed. Check the logs above.")
        
        return passed == total

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def create_sample_files():
    """Create sample files for manual testing"""
    samples_dir = "sample_files"
    os.makedirs(samples_dir, exist_ok=True)
    
    # Text file
    with open(os.path.join(samples_dir, "sample.txt"), 'w') as f:
        f.write("Hello, World!\nThis is a test file for the File-Video conversion system.\n")
        f.write("It contains some sample text that will be encoded into a video.\n")
        f.write("The system should be able to recover this exact text after decoding.\n")
    
    # Binary file with pattern
    with open(os.path.join(samples_dir, "pattern.bin"), 'wb') as f:
        pattern = bytes(range(256))
        f.write(pattern * 10)  # 2560 bytes
    
    print(f"Sample files created in {samples_dir}/")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='File-Video Conversion System')
    parser.add_argument('command', choices=['encode', 'decode', 'test', 'samples'], 
                       help='Command to run')
    parser.add_argument('--input', help='Input file (for encode) or video (for decode)')
    parser.add_argument('--output', help='Output file/video name')
    parser.add_argument('--cell-size', type=int, default=8, help='Cell size in pixels')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second')
    
    args = parser.parse_args()
    
    if args.command == 'test':
        # Run automated tests
        test_suite = FileVideoTestSuite()
        try:
            success = test_suite.run_all_tests()
            exit(0 if success else 1)
        finally:
            test_suite.cleanup()
    
    elif args.command == 'samples':
        # Create sample files
        create_sample_files()
    
    elif args.command == 'encode':
        if not args.input:
            print("Error: --input required for encode command")
            exit(1)
        
        output = args.output or (args.input + '.mp4')
        encoder = FileToVideoEncoder(cell_size=args.cell_size, fps=args.fps)
        metadata = encoder.encode_file_to_video(args.input, output)
        
        # Save metadata
        with open(output + '.meta.json', 'w') as f:
            f.write(metadata)
        print(f"Metadata saved: {output}.meta.json")
    
    elif args.command == 'decode':
        if not args.input:
            print("Error: --input required for decode command")
            exit(1)
        
        decoder = VideoToFileDecoder(cell_size=args.cell_size)
        success = decoder.decode_video_to_file(args.input, args.output)
        
        if success:
            print("✅ File successfully decoded and verified!")
        else:
            print("❌ Failed to decode file")
            exit(1)

if __name__ == '__main__':
    main()
