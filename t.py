#!/usr/bin/env python3
"""
File-Video Conversion Test Suite and Usage Examples
Demonstrates end-to-end file encoding/decoding with comprehensive testing
"""

import os
import tempfile
import shutil
import hashlib
import time
import random
import string
from pathlib import Path
import subprocess
import sys

# Import our encoder and decoder classes
# (In practice, these would be in separate files)

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
        """
        Create a test file with specified characteristics
        
        Args:
            filename: Name of the test file
            size_bytes: Size in bytes
            content_type: 'random', 'text', 'binary_pattern', or 'zeros'
        """
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
        from file_to_video_encoder import FileToVideoEncoder
        
        encoder = FileToVideoEncoder(cell_size=cell_size, fps=fps)
        
        video_path = input_file + '.mp4'
        metadata = encoder.encode_file_to_video(input_file, video_path)
        
        print(f"Encoded {input_file} -> {video_path}")
        return video_path

    def run_decoding_test(self, video_path: str, cell_size: int = 8) -> str:
        """Run decoding test and return decoded file path"""
        from video_to_file_decoder import VideoToFileDecoder
        
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
    
    # Small image (if PIL available)
    try:
        from PIL import Image
        import numpy as np
        
        # Create a small test image
        img_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(os.path.join(samples_dir, "test_image.png"))
    except ImportError:
        print("PIL not available, skipping image sample")
    
    print(f"Sample files created in {samples_dir}/")

def usage_example():
    """Show usage examples"""
    print("""
🎬 File-Video Conversion System Usage Examples

1. ENCODING (Sender Side):
   python file_to_video_encoder.py input.txt output.mp4
   python file_to_video_encoder.py --cell-size 4 --fps 30 document.pdf video.mp4

2. DECODING (Receiver Side):
   python video_to_file_decoder.py recorded_video.mp4
   python video_to_file_decoder.py --output-file restored.txt recorded.mp4

3. COMPLETE WORKFLOW:
   
   Sender:
   1. python file_to_video_encoder.py my_document.pdf encoded_video.mp4
   2. Play encoded_video.mp4 through HDMI output
   
   Receiver:
   1. Record the HDMI video stream as recorded_video.mp4
   2. python video_to_file_decoder.py recorded_video.mp4
   3. Verify the decoded file matches the original

4. PARAMETERS:
   --cell-size: Larger = more robust to recording artifacts (4-16 pixels)
   --fps: Frame rate (10-30 fps recommended)
   --adaptive-threshold: Improves robustness to lighting changes

5. BEST PRACTICES:
   - Use high contrast display settings
   - Ensure stable recording setup
   - Test with small files first
   - Use cell-size 8-16 for noisy recording conditions
   - Keep room lighting consistent during recording
""")

def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # Run automated tests
            test_suite = FileVideoTestSuite()
            try:
                success = test_suite.run_all_tests()
                sys.exit(0 if success else 1)
            finally:
                test_suite.cleanup()
        
        elif sys.argv[1] == "samples":
            # Create sample files
            create_sample_files()
        
        elif sys.argv[1] == "usage":
            # Show usage examples
            usage_example()
        
        else:
            print("Unknown command. Available commands: test, samples, usage")
    
    else:
        print("""
File-Video Conversion System Test Suite

Commands:
  python test_suite.py test     - Run automated tests
  python test_suite.py samples - Create sample files for manual testing
  python test_suite.py usage   - Show usage examples

For encoding: python file_to_video_encoder.py <input_file> <output_video>
For decoding: python video_to_file_decoder.py <input_video> [output_file]
""")

if __name__ == '__main__':
    main()
