"""
Copy-Move Detection Module
Implements feature-based and block-based copy-move forgery detection
"""

import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans, DBSCAN
from config import RATIO_THRESH, MIN_DISTANCE, RANSAC_THRESH, MIN_INLIERS, BLOCK_SIZE


def detect_copy_move_advanced(feature_sets, image_shape):
    """
    Advanced copy-move detection using feature matching with RANSAC verification.
    
    Args:
        feature_sets: Dictionary containing feature keypoints and descriptors
        image_shape: Tuple of (width, height) of the image
    
    Returns:
        Tuple of (ransac_matches, ransac_inliers, geometric_transform, total_matches)
    """
    try:
        # Get SIFT features if available
        sift_features = feature_sets.get('sift', ([], None))
        keypoints = sift_features[0]
        descriptors = sift_features[1]
        
        if descriptors is None or len(keypoints) < 10:
            return [], 0, None, 0
        
        # Create matcher - use FLANN for better performance with SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Match descriptors with k=3 to ensure we have at least 2 neighbors
        matches = flann.knnMatch(descriptors, descriptors, k=min(3, len(descriptors)))
        
        # Apply ratio test and filter self-matches
        good_matches = []
        unique_pairs = set()  # To avoid duplicate pairs
        
        for match_list in matches:
            if len(match_list) < 2:
                continue
                
            # Find first non-self match
            valid_matches = [m for m in match_list if m.queryIdx != m.trainIdx]
            
            if len(valid_matches) >= 2:
                # Apply Lowe's ratio test on the two best non-self matches
                m, n = valid_matches[0], valid_matches[1]
                
                if m.distance < RATIO_THRESH * n.distance:
                    # Check minimum spatial distance
                    kp1 = keypoints[m.queryIdx]
                    kp2 = keypoints[m.trainIdx]
                    distance = np.sqrt((kp1.pt[0] - kp2.pt[0])**2 + (kp1.pt[1] - kp2.pt[1])**2)
                    
                    if distance > MIN_DISTANCE:
                        # Avoid duplicate pairs (i->j and j->i)
                        pair = tuple(sorted([m.queryIdx, m.trainIdx]))
                        if pair not in unique_pairs:
                            unique_pairs.add(pair)
                            good_matches.append(m)
            elif len(valid_matches) == 1:
                # If we only have one valid match, use a fixed threshold
                m = valid_matches[0]
                if m.distance < 100:  # Reasonable threshold for SIFT
                    kp1 = keypoints[m.queryIdx]
                    kp2 = keypoints[m.trainIdx]
                    distance = np.sqrt((kp1.pt[0] - kp2.pt[0])**2 + (kp1.pt[1] - kp2.pt[1])**2)
                    
                    if distance > MIN_DISTANCE:
                        pair = tuple(sorted([m.queryIdx, m.trainIdx]))
                        if pair not in unique_pairs:
                            unique_pairs.add(pair)
                            good_matches.append(m)
        
        total_matches = len(good_matches)
        
        if len(good_matches) < MIN_INLIERS:
            # Fallback: try to find at least some matches even if below threshold
            if len(good_matches) > 0:
                # Return the matches we have, even if below MIN_INLIERS
                return good_matches, len(good_matches), ('fallback', None), total_matches
            return good_matches, 0, None, total_matches
        
        # Prepare points for RANSAC
        src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Try different transformation models
        best_inliers = 0
        best_matches = good_matches
        best_transform = None
        
        # Try homography if we have enough points
        if len(src_pts) >= 4:
            try:
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH, maxIters=2000, confidence=0.95)
                if mask is not None:
                    inliers = np.sum(mask)
                    if inliers > best_inliers:
                        best_inliers = int(inliers)
                        matches_mask = mask.ravel().tolist()
                        best_matches = [m for i, m in enumerate(good_matches) if i < len(matches_mask) and matches_mask[i]]
                        best_transform = ('homography', M)
            except Exception:
                pass
        
        # Try affine transformation as fallback
        if best_inliers < MIN_INLIERS and len(src_pts) >= 3:
            try:
                M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESH, confidence=0.95, maxIters=2000)
                if mask is not None:
                    inliers = np.sum(mask)
                    if inliers > best_inliers:
                        best_inliers = int(inliers)
                        matches_mask = mask.ravel().tolist()
                        best_matches = [m for i, m in enumerate(good_matches) if i < len(matches_mask) and matches_mask[i]]
                        best_transform = ('affine', M)
            except Exception:
                pass
        
        # Additional fallback: try fundamental matrix estimation
        if best_inliers < MIN_INLIERS and len(src_pts) >= 8:
            try:
                M, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, RANSAC_THRESH, 0.95)
                if mask is not None:
                    inliers = np.sum(mask)
                    if inliers > best_inliers:
                        best_inliers = int(inliers)
                        matches_mask = mask.ravel().tolist()
                        best_matches = [m for i, m in enumerate(good_matches) if i < len(matches_mask) and matches_mask[i]]
                        best_transform = ('fundamental', M)
            except Exception:
                pass
        
        return best_matches, best_inliers, best_transform, total_matches
        
    except Exception as e:
        print(f"Error in detect_copy_move_advanced: {e}")
        import traceback
        traceback.print_exc()
        return [], 0, None, 0


def detect_copy_move_blocks(image_pil, block_size=BLOCK_SIZE):
    """
    Detect copy-move forgery using block-based method.
    
    Args:
        image_pil: PIL Image object
        block_size: Size of blocks to compare
    
    Returns:
        List of matched block pairs
    """
    try:
        # Convert to grayscale numpy array
        if image_pil.mode != 'L':
            gray_img = np.array(image_pil.convert('L'))
        else:
            gray_img = np.array(image_pil)
        
        h, w = gray_img.shape
        block_matches = []
        
        # Skip if image is too small
        if h < block_size * 2 or w < block_size * 2:
            return []
        
        # Extract blocks
        blocks = []
        positions = []
        
        # Sliding window with stride
        stride = block_size // 2
        for y in range(0, h - block_size + 1, stride):
            for x in range(0, w - block_size + 1, stride):
                block = gray_img[y:y+block_size, x:x+block_size]
                blocks.append(block.flatten())
                positions.append((x, y))
        
        if len(blocks) < 2:
            return []
        
        blocks_array = np.array(blocks)
        
        # Use DBSCAN clustering to find similar blocks
        if len(blocks_array) > 1000:
            # Sample for performance
            indices = np.random.choice(len(blocks_array), 1000, replace=False)
            sampled_blocks = blocks_array[indices]
            sampled_positions = [positions[i] for i in indices]
        else:
            sampled_blocks = blocks_array
            sampled_positions = positions
        
        # Compute pairwise distances and find matches
        for i in range(len(sampled_blocks)):
            for j in range(i + 1, len(sampled_blocks)):
                # Calculate normalized cross-correlation
                block1 = sampled_blocks[i].reshape(block_size, block_size)
                block2 = sampled_blocks[j].reshape(block_size, block_size)
                
                # Normalize blocks
                block1_norm = (block1 - np.mean(block1)) / (np.std(block1) + 1e-10)
                block2_norm = (block2 - np.mean(block2)) / (np.std(block2) + 1e-10)
                
                # Compute correlation
                correlation = np.sum(block1_norm * block2_norm) / (block_size * block_size)
                
                # Check if blocks are similar enough and far apart
                if correlation > 0.95:  # High correlation threshold
                    pos1 = sampled_positions[i]
                    pos2 = sampled_positions[j]
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    if distance > MIN_DISTANCE:
                        block_matches.append({
                            'block1_pos': pos1,
                            'block2_pos': pos2,
                            'correlation': float(correlation),
                            'distance': float(distance)
                        })
        
        # Limit number of matches to avoid overwhelming results
        block_matches = sorted(block_matches, key=lambda x: x['correlation'], reverse=True)[:50]
        
        return block_matches
        
    except Exception as e:
        print(f"Error in detect_copy_move_blocks: {e}")
        return []


def kmeans_tampering_localization(image_pil, ela_array, n_clusters=8):
    """
    Perform K-means clustering on ELA data to localize tampering.
    
    Args:
        image_pil: PIL Image object (original image)
        ela_array: Numpy array of ELA values
        n_clusters: Number of clusters for K-means
    
    Returns:
        Dictionary containing localization map and tampering mask
    """
    try:
        # Ensure ela_array is 2D grayscale
        if len(ela_array.shape) == 3:
            ela_gray = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)
        else:
            ela_gray = ela_array
        
        # Flatten for clustering
        pixels = ela_gray.flatten().reshape(-1, 1)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Reshape labels back to image shape
        localization_map = labels.reshape(ela_gray.shape)
        
        # Find suspicious clusters (those with highest ELA values)
        cluster_means = []
        for i in range(n_clusters):
            cluster_pixels = pixels[labels == i]
            if len(cluster_pixels) > 0:
                cluster_means.append((i, np.mean(cluster_pixels)))
        
        # Sort clusters by mean ELA value
        cluster_means.sort(key=lambda x: x[1], reverse=True)
        
        # Consider top 30% of clusters as potentially tampered
        num_suspicious = max(1, n_clusters // 3)
        suspicious_clusters = [c[0] for c in cluster_means[:num_suspicious]]
        
        # Create tampering mask
        tampering_mask = np.zeros_like(localization_map, dtype=bool)
        for cluster_id in suspicious_clusters:
            tampering_mask[localization_map == cluster_id] = True
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tampering_mask_uint8 = tampering_mask.astype(np.uint8) * 255
        tampering_mask_uint8 = cv2.morphologyEx(tampering_mask_uint8, cv2.MORPH_CLOSE, kernel)
        tampering_mask_uint8 = cv2.morphologyEx(tampering_mask_uint8, cv2.MORPH_OPEN, kernel)
        tampering_mask = tampering_mask_uint8 > 128
        
        return {
            'localization_map': localization_map,
            'tampering_mask': tampering_mask,
            'suspicious_clusters': suspicious_clusters,
            'cluster_means': [c[1] for c in cluster_means]
        }
        
    except Exception as e:
        print(f"Error in kmeans_tampering_localization: {e}")
        # Return empty results
        return {
            'localization_map': np.zeros_like(ela_array),
            'tampering_mask': np.zeros_like(ela_array, dtype=bool),
            'suspicious_clusters': [],
            'cluster_means': []
        }