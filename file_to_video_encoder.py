#!/usr/bin/env python3
"""
File to Video Encoder - Sender Side
Converts files into videos with robust visual encoding for HDMI transmission
"""

import os
import cv2
import numpy as np
import struct
import zlib
import hashlib
from typing import List, Tuple, Optional
import argparse
import json
from pathlib import Path

class FileToVideoEncoder:
    def __init__(self, 
                 frame_width: int = 1280, 
                 frame_height: int = 720, 
                 fps: int = 20,
                 cell_size: int = 8):
        """
        Initialize the encoder with video parameters
        
        Args:
            frame_width: Video frame width (default 1280 for 720p)
            frame_height: Video frame height (default 720 for 720p)
            fps: Frames per second
            cell_size: Size of each data cell in pixels (larger = more robust)
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
            
        metadata = struct.pack('>IIHHI', 
                              frame_seq,     # 4 bytes: frame sequence number
                              chunk_id,      # 4 bytes: chunk ID
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

def main():
    parser = argparse.ArgumentParser(description='Encode files into videos for HDMI transmission')
    parser.add_argument('input_file', help='Input file to encode')
    parser.add_argument('output_video', help='Output video file')
    parser.add_argument('--cell-size', type=int, default=8, help='Cell size in pixels (default: 8)')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second (default: 20)')
    parser.add_argument('--chunk-size', type=int, help='Chunk size in bytes (auto if not specified)')
    
    args = parser.parse_args()
    
    encoder = FileToVideoEncoder(cell_size=args.cell_size, fps=args.fps)
    metadata = encoder.encode_file_to_video(args.input_file, args.output_video, args.chunk_size)
    
    # Save metadata
    metadata_file = args.output_video + '.meta.json'
    with open(metadata_file, 'w') as f:
        f.write(metadata)
    print(f"Metadata saved: {metadata_file}")

if __name__ == '__main__':
    main()
