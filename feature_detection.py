"""
Feature detection and matching functions
"""

import numpy as np
import cv2
try:
    from sklearn.preprocessing import normalize as sk_normalize
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    def sk_normalize(arr, norm='l2', axis=1):
        denom = np.linalg.norm(arr, ord=2 if norm=='l2' else 1, axis=axis, keepdims=True)
        denom[denom == 0] = 1
        return arr / denom
from config import *

def extract_multi_detector_features(image_pil, ela_image_pil, ela_mean, ela_stddev, sift_nfeatures=SIFT_FEATURES):
    """Extract features using multiple detectors (SIFT, ORB, SURF)"""
    ela_np = np.array(ela_image_pil)
    
    # Adaptive thresholding with multiple methods
    thresholds = [
        ela_mean + 1.2 * ela_stddev,  # Standard threshold
        ela_mean + 1.0 * ela_stddev,  # Lower threshold for more sensitive detection
        np.percentile(ela_np, 75),    # 75th percentile
    ]
    
    # Try multiple thresholds and select the one that gives reasonable ROI size
    best_roi_mask = None
    best_roi_pixels = 0
    
    for thresh_val in thresholds:
        thresh_val = max(min(thresh_val, 200), 20)  # Clamp between 20-200
        temp_mask = (ela_np > thresh_val).astype(np.uint8) * 255
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_CLOSE, kernel)
        temp_mask = cv2.morphologyEx(temp_mask, cv2.MORPH_OPEN, kernel)
        
        roi_pixels = np.sum(temp_mask > 0)
        total_pixels = temp_mask.size
        roi_percentage = roi_pixels / total_pixels
        
        # Prefer masks that cover 10-40% of image
        if 0.1 <= roi_percentage <= 0.4 and roi_pixels > best_roi_pixels:
            best_roi_mask = temp_mask
            best_roi_pixels = roi_pixels
    
    # Fallback to original method if no good mask found
    if best_roi_mask is None:
        threshold = ela_mean + 1.5 * ela_stddev
        threshold = max(min(threshold, 180), 30)
        roi_mask = (ela_np > threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)
    else:
        roi_mask = best_roi_mask
    
    # Convert to grayscale with enhancement
    original_image_np = np.array(image_pil.convert('RGB'))
    gray_original = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2GRAY)
    
    # Multiple enhancement techniques
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray_original)
    
    # Extract features using multiple detectors
    feature_sets = {}
    
    # 1. SIFT with multiple parameter attempts
    sift_kps = []
    sift_descs = None
    
    # Try multiple parameter combinations for better feature detection
    param_combinations = [
        (sift_nfeatures, SIFT_CONTRAST_THRESHOLD, SIFT_EDGE_THRESHOLD),  # Default
        (sift_nfeatures * 2, SIFT_CONTRAST_THRESHOLD * 0.5, SIFT_EDGE_THRESHOLD),  # More features, lower contrast threshold
        (sift_nfeatures, SIFT_CONTRAST_THRESHOLD, SIFT_EDGE_THRESHOLD * 2),  # Higher edge threshold
    ]
    
    for nfeat, contrast_thresh, edge_thresh in param_combinations:
        try:
            sift = cv2.SIFT_create(nfeatures=nfeat, 
                                  contrastThreshold=contrast_thresh, 
                                  edgeThreshold=edge_thresh)
            kp, desc = sift.detectAndCompute(gray_enhanced, mask=roi_mask)
            if desc is not None and len(kp) > len(sift_kps):
                sift_kps = kp
                sift_descs = desc
        except Exception:
            continue
    
    # If no features found with mask, try without mask
    if len(sift_kps) < 10:
        try:
            sift = cv2.SIFT_create(nfeatures=sift_nfeatures, 
                                  contrastThreshold=SIFT_CONTRAST_THRESHOLD * 0.3, 
                                  edgeThreshold=SIFT_EDGE_THRESHOLD)
            kp, desc = sift.detectAndCompute(gray_enhanced, None)
            if desc is not None and len(kp) > len(sift_kps):
                sift_kps = kp
                sift_descs = desc
        except Exception:
            pass
    
    feature_sets['sift'] = (sift_kps, sift_descs)
    
    # 2. ORB
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES, 
                        scaleFactor=ORB_SCALE_FACTOR, 
                        nlevels=ORB_LEVELS)
    kp_orb, desc_orb = orb.detectAndCompute(gray_enhanced, mask=roi_mask)
    feature_sets['orb'] = (kp_orb, desc_orb)
    
    # 3. AKAZE
    try:
        akaze = cv2.AKAZE_create()
        kp_akaze, desc_akaze = akaze.detectAndCompute(gray_enhanced, mask=roi_mask)
        feature_sets['akaze'] = (kp_akaze, desc_akaze)
    except:
        feature_sets['akaze'] = ([], None)
    
    return feature_sets, roi_mask, gray_enhanced

def match_sift_features(keypoints, descriptors, ratio_thresh, min_distance, ransac_thresh, min_inliers):
    """Enhanced SIFT matching"""
    # Handle empty descriptors
    if descriptors is None or len(descriptors) == 0:
        return [], 0, None
        
    descriptors_norm = sk_normalize(descriptors, norm='l2', axis=1)
    
    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(descriptors_norm, descriptors_norm, k=8)
    
    good_matches = []
    match_pairs = []
    
    for i, match_list in enumerate(matches):
        for m in match_list[1:]:  # Skip self-match
            pt1 = keypoints[i].pt
            pt2 = keypoints[m.trainIdx].pt
            
            spatial_dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            
            if spatial_dist > min_distance and m.distance < ratio_thresh:
                good_matches.append(m)
                match_pairs.append((i, m.trainIdx))
    
    if len(match_pairs) < min_inliers:
        return good_matches, 0, None
    
    # RANSAC verification
    src_pts = np.float32([keypoints[i].pt for i, _ in match_pairs]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints[j].pt for _, j in match_pairs]).reshape(-1, 1, 2)
    
    best_inliers = 0
    best_transform = None
    best_mask = None
    
    # Try different transformations
    for transform_type in ['affine', 'homography', 'similarity']:
        try:
            if transform_type == 'affine':
                M, mask = cv2.estimateAffine2D(src_pts, dst_pts,
                                             method=cv2.RANSAC,
                                             ransacReprojThreshold=ransac_thresh)
            elif transform_type == 'homography':
                M, mask = cv2.findHomography(src_pts, dst_pts,
                                           cv2.RANSAC, ransac_thresh)
            else:  # similarity
                M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts,
                                                    method=cv2.RANSAC,
                                                    ransacReprojThreshold=ransac_thresh)
            
            if M is not None:
                inliers = np.sum(mask)
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_transform = (transform_type, M)
                    best_mask = mask
        except:
            continue
    
    if best_mask is not None and best_inliers >= min_inliers:
        ransac_matches = [good_matches[i] for i in range(len(good_matches))
                         if best_mask[i][0] == 1]
        return ransac_matches, best_inliers, best_transform
    
    return good_matches, 0, None

def match_orb_features(keypoints, descriptors, min_distance, ransac_thresh, min_inliers):
    """ORB feature matching"""
    # Hamming distance matcher for ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors, descriptors, k=6)
    
    good_matches = []
    match_pairs = []
    
    for i, match_list in enumerate(matches):
        for m in match_list[1:]:  # Skip self-match
            pt1 = keypoints[i].pt
            pt2 = keypoints[m.trainIdx].pt
            
            spatial_dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            
            if spatial_dist > min_distance and m.distance < 80:  # Hamming distance threshold
                good_matches.append(m)
                match_pairs.append((i, m.trainIdx))
    
    if len(match_pairs) < min_inliers:
        return good_matches, 0, None
    
    # Simple geometric verification
    return good_matches, len(match_pairs), ('orb_matches', None)

def match_akaze_features(keypoints, descriptors, min_distance, ransac_thresh, min_inliers):
    """AKAZE feature matching"""
    if descriptors is None:
        return [], 0, None
    
    # Hamming distance for AKAZE
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors, descriptors, k=6)
    
    good_matches = []
    match_pairs = []
    
    for i, match_list in enumerate(matches):
        if len(match_list) > 1:
            for m in match_list[1:]:  # Skip self-match
                pt1 = keypoints[i].pt
                pt2 = keypoints[m.trainIdx].pt
                
                spatial_dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                
                if spatial_dist > min_distance and m.distance < 100:
                    good_matches.append(m)
                    match_pairs.append((i, m.trainIdx))
    
    return good_matches, len(match_pairs), ('akaze_matches', None)