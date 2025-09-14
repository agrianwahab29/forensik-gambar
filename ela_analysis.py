"""
Enhanced Error Level Analysis (ELA) functions with advanced features
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageStat, ImageFilter
from scipy import ndimage, signal
from scipy.stats import entropy
from config import ELA_QUALITIES, ELA_SCALE_FACTOR
from utils import detect_outliers_iqr

def perform_multi_quality_ela(image_pil, quality_steps=ELA_QUALITIES, scale_factor=ELA_SCALE_FACTOR):
    """Enhanced multi-quality ELA with advanced features for better manipulation detection"""
    temp_filename = "temp_ela_multi.jpg"
    
    if image_pil.mode != 'RGB':
        image_rgb = image_pil.convert('RGB')
    else:
        image_rgb = image_pil
    
    # Convert to numpy for advanced processing
    original_array = np.array(image_rgb)
    h, w = original_array.shape[:2]
    
    ela_results = []
    quality_stats = []
    frequency_features = []
    
    for q in quality_steps:
        # Save and reload with JPEG compression
        image_rgb.save(temp_filename, 'JPEG', quality=q)
        with Image.open(temp_filename) as compressed_rgb:
            # Calculate difference
            diff_rgb = ImageChops.difference(image_rgb, compressed_rgb)
            diff_l = diff_rgb.convert('L')
            ela_np = np.array(diff_l, dtype=float)

        # Advanced ELA processing
        processed_ela = _enhance_ela_signal(ela_np, original_array, scale_factor)
        ela_results.append(processed_ela)
        
        # Enhanced statistics for this quality
        stat = ImageStat.Stat(Image.fromarray(processed_ela.astype(np.uint8)))
        
        # Add frequency domain analysis
        freq_features = _analyze_frequency_domain(processed_ela)
        frequency_features.append(freq_features)
        
        quality_stats.append({
            'quality': q,
            'mean': stat.mean[0],
            'stddev': stat.stddev[0],
            'max': np.max(processed_ela),
            'percentile_95': np.percentile(processed_ela, 95),
            'entropy': entropy(processed_ela.flatten() + 1),  # +1 to avoid log(0)
            'edge_response': _calculate_edge_response(processed_ela),
            'frequency_energy': freq_features['high_freq_energy']
        })
    
    # Cross-quality analysis with enhanced variance computation
    ela_variance = np.var(ela_results, axis=0)
    
    # Adaptive weighted averaging based on image characteristics
    weights = _calculate_adaptive_weights(quality_stats, original_array)
    final_ela = np.average(ela_results, axis=0, weights=weights)
    
    # Multi-scale enhancement for better visualization
    final_ela_enhanced = _apply_multiscale_enhancement(final_ela, original_array)
    final_ela_image = Image.fromarray(final_ela_enhanced.astype(np.uint8), mode='L')
    
    # Enhanced regional analysis with texture awareness
    regional_stats = analyze_ela_regions_enhanced(final_ela_enhanced, ela_variance, original_array)
    
    # Overall statistics
    final_stat = ImageStat.Stat(final_ela_image)
    
    # Add advanced metrics
    regional_stats['frequency_consistency'] = _analyze_frequency_consistency(frequency_features)
    regional_stats['adaptive_weights'] = weights.tolist()
    regional_stats['signal_enhancement_ratio'] = np.mean(final_ela_enhanced) / max(np.mean(final_ela), 1e-6)
    
    try:
        os.remove(temp_filename)
    except:
        pass
    
    return (final_ela_image, final_stat.mean[0], final_stat.stddev[0],
            regional_stats, quality_stats, ela_variance)

def analyze_ela_regions_enhanced(ela_array, ela_variance, original_array=None, block_size=32):
    """Enhanced regional ELA analysis with texture awareness and adaptive thresholding"""
    h, w = ela_array.shape
    regional_means = []
    regional_stds = []
    regional_variances = []
    suspicious_regions = []
    texture_features = []
    
    # Calculate texture features from original if available
    texture_map = None
    if original_array is not None:
        gray_original = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY) if len(original_array.shape) == 3 else original_array
        texture_map = _calculate_local_texture(gray_original, block_size)
    
    for i in range(0, h - block_size, block_size//2):
        for j in range(0, w - block_size, block_size//2):
            block = ela_array[i:i+block_size, j:j+block_size]
            var_block = ela_variance[i:i+block_size, j:j+block_size]
            
            block_mean = np.mean(block)
            block_std = np.std(block)
            block_var = np.mean(var_block)
            
            regional_means.append(block_mean)
            regional_stds.append(block_std)
            regional_variances.append(block_var)
            
            # Texture-aware analysis
            texture_score = 0.0
            if texture_map is not None:
                texture_block = texture_map[i:i+min(block_size, texture_map.shape[0]-i), 
                                         j:j+min(block_size, texture_map.shape[1]-j)]
                texture_score = np.mean(texture_block) if texture_block.size > 0 else 0.0
                texture_features.append(texture_score)
            
            # Adaptive thresholding based on texture
            base_threshold = 15
            texture_adaptive_threshold = base_threshold * (1 + texture_score * 0.5)
            
            # Enhanced suspicious region detection
            is_suspicious = (
                block_mean > texture_adaptive_threshold or 
                block_std > (25 + texture_score * 10) or 
                block_var > (100 + texture_score * 50) or
                _detect_edge_inconsistency(block)
            )
            
            if is_suspicious:
                suspicious_regions.append({
                    'position': (i, j),
                    'mean': block_mean,
                    'std': block_std,
                    'variance': block_var,
                    'texture_score': texture_score,
                    'confidence': _calculate_region_confidence(block, block_var, texture_score)
                })
    
    # Advanced statistical analysis
    regional_means = np.array(regional_means)
    regional_stds = np.array(regional_stds)
    regional_variances = np.array(regional_variances)
    
    # Calculate entropy-based inconsistency
    entropy_inconsistency = 0.0
    if len(regional_means) > 1:
        # Bin the means for entropy calculation
        hist, _ = np.histogram(regional_means, bins=min(20, len(regional_means)//2))
        hist = hist + 1e-10  # Avoid log(0)
        entropy_inconsistency = entropy(hist)
    
    # Detect outlier clusters using improved IQR method
    outlier_indices_mean = detect_outliers_iqr(regional_means)
    outlier_indices_std = detect_outliers_iqr(regional_stds)
    total_outliers = len(set(outlier_indices_mean) | set(outlier_indices_std))
    
    return {
        'mean_variance': np.var(regional_means),
        'std_variance': np.var(regional_stds),
        'outlier_regions': total_outliers,
        'regional_inconsistency': np.std(regional_means) / (np.mean(regional_means) + 1e-6),
        'entropy_inconsistency': entropy_inconsistency,
        'suspicious_regions': suspicious_regions,
        'cross_quality_variance': np.mean(regional_variances),
        'texture_aware_score': np.mean(texture_features) if texture_features else 0.0,
        'adaptive_threshold_used': True,
        'confidence_weighted_score': np.mean([r['confidence'] for r in suspicious_regions]) if suspicious_regions else 0.0
    }


def _enhance_ela_signal(ela_array, original_array, scale_factor):
    """Apply advanced signal enhancement to ELA"""
    # Adaptive scaling based on image characteristics
    gray_original = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY) if len(original_array.shape) == 3 else original_array
    
    # Calculate local contrast to adapt scaling
    local_contrast = cv2.Laplacian(gray_original, cv2.CV_64F)
    contrast_factor = 1.0 + np.std(local_contrast) / 100.0
    
    # Apply adaptive scaling
    adaptive_scale = scale_factor * contrast_factor
    scaled_ela = np.clip(ela_array * adaptive_scale, 0, 255)
    
    # Edge-preserving enhancement using bilateral filter
    enhanced_ela = cv2.bilateralFilter(scaled_ela.astype(np.uint8), 5, 50, 50).astype(float)
    
    # Local contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(enhanced_ela.astype(np.uint8)).astype(float)
    
    return contrast_enhanced


def _analyze_frequency_domain(ela_array):
    """Analyze frequency domain characteristics of ELA"""
    # Apply 2D FFT
    f_transform = np.fft.fft2(ela_array)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # Calculate frequency features
    h, w = ela_array.shape
    center_h, center_w = h // 2, w // 2
    
    # High frequency energy (outer regions)
    mask_high = np.zeros((h, w))
    mask_high[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 1
    mask_high = 1 - mask_high  # Invert to get high frequencies
    
    high_freq_energy = np.sum(magnitude_spectrum * mask_high)
    total_energy = np.sum(magnitude_spectrum)
    
    return {
        'high_freq_energy': high_freq_energy / (total_energy + 1e-6),
        'spectral_centroid': _calculate_spectral_centroid(magnitude_spectrum),
        'spectral_rolloff': _calculate_spectral_rolloff(magnitude_spectrum)
    }


def _calculate_adaptive_weights(quality_stats, original_array):
    """Calculate adaptive weights based on image and quality characteristics"""
    # Determine number of qualities and create appropriate base weights
    num_qualities = len(quality_stats)
    if num_qualities == 0:
        return np.array([1.0])  # Default single weight if no qualities
    
    # Create base weights that match the number of qualities
    if num_qualities == 1:
        base_weights = np.array([1.0])
    elif num_qualities == 2:
        base_weights = np.array([0.5, 0.5])
    elif num_qualities == 3:
        base_weights = np.array([0.25, 0.5, 0.25])
    else:  # 4 or more qualities
        # For 4 qualities, use the original weights
        if num_qualities == 4:
            base_weights = np.array([0.2, 0.3, 0.3, 0.2])
        else:
            # For more than 4, give more weight to middle qualities
            base_weights = np.ones(num_qualities)
            middle_indices = range(num_qualities // 4, 3 * num_qualities // 4)
            base_weights[middle_indices] = 1.5
    
    # Analyze image characteristics
    gray_original = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY) if len(original_array.shape) == 3 else original_array
    image_complexity = np.std(gray_original) / 255.0
    
    # Adjust weights based on quality response consistency
    if num_qualities >= 4:
        quality_means = [q['mean'] for q in quality_stats]
        quality_variation = np.std(quality_means)
        
        # If high variation across qualities, give more weight to extreme qualities
        if quality_variation > 10:
            # Create an adjustment array of appropriate length
            adjustment = np.zeros(num_qualities)
            adjustment[0] = 0.1  # First quality
            adjustment[-1] = 0.1  # Last quality
            # For middle qualities, slightly decrease weight
            middle_indices = range(1, num_qualities - 1)
            adjustment[middle_indices] = -0.05 * (2 / (num_qualities - 2))  # Distribute negative weight
            
            base_weights += adjustment * image_complexity
    
    # Normalize weights
    base_weights = np.clip(base_weights, 0.1, 0.4)
    return base_weights / np.sum(base_weights)


def _apply_multiscale_enhancement(ela_array, original_array):
    """Apply multi-scale enhancement for better visual representation"""
    # Create multiple scales
    scales = [1.0, 0.5, 0.25]
    enhanced_ela = ela_array.copy()
    
    gray_original = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY) if len(original_array.shape) == 3 else original_array
    
    for scale in scales[1:]:
        # Resize for multi-scale analysis
        small_h, small_w = int(ela_array.shape[0] * scale), int(ela_array.shape[1] * scale)
        ela_small = cv2.resize(ela_array.astype(np.uint8), (small_w, small_h))
        original_small = cv2.resize(gray_original, (small_w, small_h))
        
        # Enhance details at this scale
        enhanced_small = _enhance_details_at_scale(ela_small, original_small)
        
        # Resize back and combine
        enhanced_resized = cv2.resize(enhanced_small, (ela_array.shape[1], ela_array.shape[0]))
        enhanced_ela = 0.7 * enhanced_ela + 0.3 * enhanced_resized
    
    return np.clip(enhanced_ela, 0, 255)


def _calculate_edge_response(ela_array):
    """Calculate edge response in ELA"""
    # Sobel edge detection
    sobel_x = cv2.Sobel(ela_array, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(ela_array, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.mean(edge_magnitude)


def _calculate_local_texture(gray_image, block_size):
    """Calculate local texture features using GLCM-inspired approach"""
    h, w = gray_image.shape
    texture_map = np.zeros((h, w))
    
    for i in range(0, h - block_size, block_size//2):
        for j in range(0, w - block_size, block_size//2):
            block = gray_image[i:i+block_size, j:j+block_size]
            
            # Simple texture measure: local standard deviation
            texture_score = np.std(block)
            texture_map[i:i+block_size, j:j+block_size] = texture_score
    
    # Normalize texture map
    return texture_map / (np.max(texture_map) + 1e-6)


def _detect_edge_inconsistency(block):
    """Detect edge inconsistencies in a block"""
    # Calculate gradients
    grad_x = np.gradient(block, axis=1)
    grad_y = np.gradient(block, axis=0)
    
    # Check for unusual gradient patterns
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_std = np.std(grad_magnitude)
    
    # High gradient variance suggests inconsistency
    return grad_std > np.mean(grad_magnitude) * 1.5


def _calculate_region_confidence(block, block_var, texture_score):
    """Calculate confidence score for a suspicious region"""
    # Multiple factors contribute to confidence
    mean_factor = min(1.0, np.mean(block) / 50.0)
    var_factor = min(1.0, block_var / 200.0)
    texture_factor = min(1.0, texture_score)
    
    # Combine factors with weights
    confidence = 0.4 * mean_factor + 0.3 * var_factor + 0.3 * texture_factor
    return min(1.0, confidence)


def _analyze_frequency_consistency(frequency_features):
    """Analyze consistency across different quality levels in frequency domain"""
    if len(frequency_features) < 2:
        return 0.0
    
    high_freq_energies = [f['high_freq_energy'] for f in frequency_features]
    return np.std(high_freq_energies) / (np.mean(high_freq_energies) + 1e-6)


def _calculate_spectral_centroid(magnitude_spectrum):
    """Calculate spectral centroid of magnitude spectrum"""
    h, w = magnitude_spectrum.shape
    freqs_h = np.arange(h)
    freqs_w = np.arange(w)
    
    # Calculate weighted frequency
    total_magnitude = np.sum(magnitude_spectrum)
    if total_magnitude == 0:
        return 0.0
    
    centroid_h = np.sum(np.outer(freqs_h, np.ones(w)) * magnitude_spectrum) / total_magnitude
    centroid_w = np.sum(np.outer(np.ones(h), freqs_w) * magnitude_spectrum) / total_magnitude
    
    return np.sqrt(centroid_h**2 + centroid_w**2)


def _calculate_spectral_rolloff(magnitude_spectrum):
    """Calculate spectral rolloff (frequency below which 85% of energy is contained)"""
    total_energy = np.sum(magnitude_spectrum**2)
    if total_energy == 0:
        return 0.0
    
    # Flatten and sort by magnitude
    flat_spectrum = magnitude_spectrum.flatten()
    sorted_indices = np.argsort(flat_spectrum)[::-1]
    
    cumulative_energy = 0
    for i, idx in enumerate(sorted_indices):
        cumulative_energy += flat_spectrum[idx]**2
        if cumulative_energy >= 0.85 * total_energy:
            return i / len(flat_spectrum)
    
    return 1.0


def _enhance_details_at_scale(ela_small, original_small):
    """Enhance details at a specific scale"""
    # Apply unsharp masking for detail enhancement
    gaussian_blur = cv2.GaussianBlur(ela_small, (3, 3), 1.0)
    unsharp_mask = ela_small.astype(float) - gaussian_blur.astype(float)
    enhanced = ela_small.astype(float) + 0.5 * unsharp_mask
    
    return np.clip(enhanced, 0, 255)