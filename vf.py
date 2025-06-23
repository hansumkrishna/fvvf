#!/usr/bin/env python3
"""
Video to File Decoder - Receiver Side
Converts recorded videos back into original files with error checking
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
import math

class VideoToFileDecoder:
    def __init__(self, 
                 frame_width: int = 1280, 
                 frame_height: int = 720,
                 cell_size: int = 8,
                 adaptive_threshold: bool = True):
        """
        Initialize the decoder with video parameters
        
        Args:
            frame_width: Expected video frame width
            frame_height: Expected video frame height  
            cell_size: Size of each data cell in pixels
            adaptive_threshold: Use adaptive thresholding for robustness
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

def main():
    parser = argparse.ArgumentParser(description='Decode videos back into original files')
    parser.add_argument('input_video', help='Input video file to decode')
    parser.add_argument('--output-file', help='Output file name (auto-detected if not specified)')
    parser.add_argument('--cell-size', type=int, default=8, help='Cell size in pixels (default: 8)')
    parser.add_argument('--adaptive-threshold', action='store_true', default=True, 
                       help='Use adaptive thresholding (default: True)')
    
    args = parser.parse_args()
    
    decoder = VideoToFileDecoder(
        cell_size=args.cell_size,
        adaptive_threshold=args.adaptive_threshold
    )
    
    success = decoder.decode_video_to_file(args.input_video, args.output_file)
    
    if success:
        print("✓ File successfully decoded and verified!")
    else:
        print("✗ Failed to decode file")
        exit(1)

if __name__ == '__main__':
    main()
