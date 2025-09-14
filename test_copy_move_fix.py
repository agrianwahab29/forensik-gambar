#!/usr/bin/env python3
"""
Test script untuk memverifikasi perbaikan copy-move detection
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from copy_move_detection import detect_copy_move_advanced
from feature_detection import extract_multi_detector_features
from main import preprocess_image

def test_copy_move_detection(image_path):
    """Test copy-move detection pada gambar tertentu"""
    print(f"Testing copy-move detection pada: {image_path}")
    
    try:
        # Load and preprocess image
        image_pil = Image.open(image_path)
        print(f"  Original size: {image_pil.size}")
        
        # Preprocess
        preprocessed_pil = preprocess_image(image_pil)
        print(f"  Preprocessed size: {preprocessed_pil.size}")
        
        # Simulate ELA results (for testing purposes)
        ela_image = preprocessed_pil.copy()
        ela_mean = 30.0
        ela_std = 15.0
        
        # Extract features
        print("  Extracting features...")
        feature_sets, roi_mask, enhanced_gray = extract_multi_detector_features(
            preprocessed_pil, ela_image, ela_mean, ela_std, sift_nfeatures=500
        )
        
        sift_kps, sift_descs = feature_sets.get('sift', ([], None))
        print(f"  SIFT keypoints found: {len(sift_kps)}")
        print(f"  SIFT descriptors: {'Available' if sift_descs is not None else 'None'}")
        
        if sift_descs is None or len(sift_kps) < 10:
            print("  ❌ Tidak cukup keypoints untuk copy-move detection")
            return False
        
        # Test copy-move detection
        print("  Running copy-move detection...")
        matches, inliers, transform, total_matches = detect_copy_move_advanced(
            feature_sets, preprocessed_pil.size
        )
        
        print(f"  Results:")
        print(f"    Total matches: {total_matches}")
        print(f"    RANSAC inliers: {inliers}")
        print(f"    Transform type: {transform[0] if transform else 'None'}")
        
        if inliers > 0:
            print(f"  ✅ Copy-move terdeteksi dengan {inliers} inliers")
            return True
        elif total_matches > 0:
            print(f"  ⚠️  Matches ditemukan ({total_matches}) tapi tidak ada RANSAC inliers")
            return False
        else:
            print("  ❌ Tidak ada matches yang ditemukan")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("COPY-MOVE DETECTION TEST SUITE")
    print("=" * 60)
    
    # Test dengan berbagai gambar
    test_images = []
    
    # Cari gambar di direktori saat ini
    for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']:
        for file in os.listdir('.'):
            if file.lower().endswith(ext):
                test_images.append(file)
    
    if not test_images:
        print("Tidak ada gambar yang ditemukan untuk testing")
        print("Silakan tambahkan gambar ke direktori ini dan jalankan lagi")
        return
    
    print(f"Found {len(test_images)} images for testing")
    
    results = []
    for image_path in test_images:
        success = test_copy_move_detection(image_path)
        results.append((image_path, success))
        print("-" * 40)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    successful_tests = sum(1 for _, success in results if success)
    total_tests = len(results)
    
    print(f"Total tests: {total_tests}")
    print(f"Successful detections: {successful_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
    
    print("\nDetailed results:")
    for image_path, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {image_path}")

if __name__ == "__main__":
    main()