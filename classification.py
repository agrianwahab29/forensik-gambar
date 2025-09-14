# --- START OF FILE classification.py ---

"""
Classification Module for Forensic Image Analysis System
Contains functions for machine learning classification, feature vector preparation, and confidence scoring
"""

import numpy as np
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import normalize as sk_normalize
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    class RandomForestClassifier:
        def __init__(self,*a,**k): pass
        def fit(self,X,y): pass
        def predict_proba(self,X): return np.zeros((len(X),2))
    class SVC:
        def __init__(self,*a,**k): pass
        def fit(self,X,y): pass
        def decision_function(self,X): return np.zeros(len(X))
    def sk_normalize(arr, norm='l2', axis=1):
        denom = np.linalg.norm(arr, ord=2 if norm=='l2' else 1, axis=axis, keepdims=True)
        denom[denom==0]=1
        return arr/denom
import warnings
from uncertainty_classification import UncertaintyClassifier, format_probability_results

warnings.filterwarnings('ignore')

# ======================= Helper Functions =======================

def sigmoid(x):
    """Sigmoid activation function"""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    """Tanh activation function (alternative)"""
    return np.tanh(x)

# ======================= Feature Vector Preparation =======================

def prepare_feature_vector(analysis_results):
    """Prepare comprehensive feature vector for ML classification"""
    features = []
    
    # ELA features (6)
    # Safely get ELA regional stats with defaults
    ela_regional = analysis_results.get('ela_regional_stats', {'mean_variance': 0.0, 'regional_inconsistency': 0.0, 'outlier_regions': 0, 'suspicious_regions': []})

    features.extend([
        analysis_results.get('ela_mean', 0.0),
        analysis_results.get('ela_std', 0.0),
        ela_regional.get('mean_variance', 0.0),
        ela_regional.get('regional_inconsistency', 0.0),
        ela_regional.get('outlier_regions', 0),
        len(ela_regional.get('suspicious_regions', []))
    ])
    
    # SIFT features (3)
    features.extend([
        analysis_results.get('sift_matches', 0),
        analysis_results.get('ransac_inliers', 0),
        1 if analysis_results.get('geometric_transform') else 0
    ])
    
    # Block matching (1)
    features.append(len(analysis_results.get('block_matches', [])))
    
    # Noise analysis (1)
    noise_analysis = analysis_results.get('noise_analysis', {'overall_inconsistency': 0.0})
    features.append(noise_analysis.get('overall_inconsistency', 0.0))
    
    # JPEG analysis (3)
    jpeg_analysis_main = analysis_results.get('jpeg_analysis', {})
    basic_jpeg_analysis = jpeg_analysis_main.get('basic_analysis', {}) # Added for safe access
    features.extend([
        analysis_results.get('jpeg_ghost_suspicious_ratio', 0.0),
        basic_jpeg_analysis.get('response_variance', 0.0),
        basic_jpeg_analysis.get('double_compression_indicator', 0.0)
    ])
    
    # Frequency domain (2)
    freq_analysis = analysis_results.get('frequency_analysis', {'frequency_inconsistency': 0.0, 'dct_stats': {}})
    features.extend([
        freq_analysis.get('frequency_inconsistency', 0.0),
        freq_analysis['dct_stats'].get('freq_ratio', 0.0)
    ])
    
    # Texture analysis (1)
    texture_analysis = analysis_results.get('texture_analysis', {'overall_inconsistency': 0.0})
    features.append(texture_analysis.get('overall_inconsistency', 0.0))
    
    # Edge analysis (1)
    edge_analysis = analysis_results.get('edge_analysis', {'edge_inconsistency': 0.0})
    features.append(edge_analysis.get('edge_inconsistency', 0.0))
    
    # Illumination analysis (1)
    illumination_analysis = analysis_results.get('illumination_analysis', {'overall_illumination_inconsistency': 0.0})
    features.append(illumination_analysis.get('overall_illumination_inconsistency', 0.0))
    
    # Statistical features (5)
    stat_analysis = analysis_results.get('statistical_analysis', {})
    stat_features = [
        stat_analysis.get('R_entropy', 0.0),
        stat_analysis.get('G_entropy', 0.0),
        stat_analysis.get('B_entropy', 0.0),
        stat_analysis.get('rg_correlation', 0.0),
        stat_analysis.get('overall_entropy', 0.0)
    ]
    features.extend(stat_features)
    
    # Metadata score (1)
    metadata = analysis_results.get('metadata', {})
    features.append(metadata.get('Metadata_Authenticity_Score', 0.0))
    
    # Localization features (3)
    if 'localization_analysis' in analysis_results:
        loc_results = analysis_results['localization_analysis']
        # Default empty dict for kmeans_localization if not present, to prevent KeyError
        kmeans_loc = loc_results.get('kmeans_localization', {}) 
        cluster_ela_means = kmeans_loc.get('cluster_ela_means', [])
        
        features.extend([
            loc_results.get('tampering_percentage', 0.0),
            len(cluster_ela_means),
            max(cluster_ela_means) if cluster_ela_means else 0.0
        ])
    else:
        features.extend([0.0, 0, 0.0]) # Add defaults for localization if whole key is missing
    
    return np.array(features, dtype=np.float32) # Ensure float32 for consistency

def validate_feature_vector(feature_vector):
    """
    Validate and clean feature vector.
    Ensure `ransac_inliers` and other count-based features are non-negative.
    """
    if not isinstance(feature_vector, np.ndarray):
        feature_vector = np.array(feature_vector)

    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=0.0)
    feature_vector = np.clip(feature_vector, -1e6, 1e6) # clip large values

    # Apply non-negative constraint for specific indices/features that are counts or percentages
    # Assuming the fixed feature vector structure from `prepare_feature_vector`
    # (Adjust indices if feature vector structure changes significantly)
    
    # Feature 7 (index 6): SIFT Matches (must be >= 0)
    if len(feature_vector) > 6:
        feature_vector[6] = max(0.0, feature_vector[6])
    # Feature 8 (index 7): RANSAC Inliers (must be >= 0)
    if len(feature_vector) > 7:
        feature_vector[7] = max(0.0, feature_vector[7])
    # Feature 10 (index 9): Block Matches (must be >= 0)
    if len(feature_vector) > 9:
        feature_vector[9] = max(0.0, feature_vector[9])
    # Localization Tampering Percentage (index 25, if present)
    if len(feature_vector) > 25:
        feature_vector[25] = np.clip(feature_vector[25], 0.0, 100.0)

    return feature_vector

def normalize_feature_vector(feature_vector):
    """Normalize feature vector for ML processing"""
    # Use sklearn's normalize for more robust L2 normalization if available
    if SKLEARN_AVAILABLE:
        try:
            return sk_normalize(feature_vector.reshape(1, -1), norm='l2', axis=1)[0]
        except Exception as e:
            print(f"  Warning: sklearn normalization failed: {e}, falling back to manual.")

    # Manual Min-Max Scaling (fallback if sklearn is not robust or available)
    feature_min = np.min(feature_vector)
    feature_max = np.max(feature_vector)
    
    if feature_max - feature_min > 0:
        normalized = (feature_vector - feature_min) / (feature_max - feature_min)
    else:
        normalized = np.zeros_like(feature_vector)
    return normalized

# ======================= Machine Learning Classification =======================

def classify_with_ml(feature_vector):
    """Classify using pre-trained models (simplified version)"""
    feature_vector = validate_feature_vector(feature_vector)
    
    # Simplified logic, usually uses actual trained models or complex rules
    copy_move_indicators = [
        feature_vector[7] > 10 if len(feature_vector) > 7 else False, # RANSAC inliers
        feature_vector[9] > 10 if len(feature_vector) > 9 else False, # Block matches
        feature_vector[8] > 0 if len(feature_vector) > 8 else False, # Geometric transform presence
    ]
    
    splicing_indicators = [
        feature_vector[0] > 8 if len(feature_vector) > 0 else False, # ELA Mean
        feature_vector[4] > 3 if len(feature_vector) > 4 else False, # ELA Outlier Regions
        feature_vector[10] > 0.3 if len(feature_vector) > 10 else False, # Noise inconsistency
        feature_vector[11] > 0.15 if len(feature_vector) > 11 else False, # JPEG Ghost ratio
        feature_vector[17] > 0.3 if len(feature_vector) > 17 else False, # Texture inconsistency
        feature_vector[18] > 0.3 if len(feature_vector) > 18 else False, # Edge inconsistency
    ]
    
    copy_move_score = sum(copy_move_indicators) * 20
    splicing_score = sum(splicing_indicators) * 15
    
    return copy_move_score, splicing_score

def classify_with_advanced_ml(feature_vector):
    """Advanced ML classification with multiple algorithms"""
    feature_vector = validate_feature_vector(feature_vector)
    normalized_features = normalize_feature_vector(feature_vector)
    
    scores = {}
    
    rf_copy_move = simulate_random_forest_classification(normalized_features, 'copy_move')
    rf_splicing = simulate_random_forest_classification(normalized_features, 'splicing')
    scores['random_forest'] = (rf_copy_move, rf_splicing)
    
    svm_copy_move = simulate_svm_classification(normalized_features, 'copy_move')
    svm_splicing = simulate_svm_classification(normalized_features, 'splicing')
    scores['svm'] = (svm_copy_move, svm_splicing)
    
    nn_copy_move = simulate_neural_network_classification(normalized_features, 'copy_move')
    nn_splicing = simulate_neural_network_classification(normalized_features, 'splicing')
    scores['neural_network'] = (nn_copy_move, nn_splicing)
    
    copy_move_scores = [scores[model][0] for model in scores]
    splicing_scores = [scores[model][1] for model in scores]
    
    ensemble_copy_move = np.mean(copy_move_scores)
    ensemble_splicing = np.mean(splicing_scores)
    
    return ensemble_copy_move, ensemble_splicing, scores

def simulate_random_forest_classification(features, manipulation_type):
    """Simulate Random Forest classification"""
    # Assuming a feature vector of around 28-30 elements
    if manipulation_type == 'copy_move':
        # Weights focused on geometric & copy-move artifacts
        weights = np.array([0.05, 0.05, 0.1, 0.1, 0.2, 0.1, 0.8, 1.0, 1.0, 0.9, 0.3, 0.2, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.05, 0.01, 0.01, 0.01, 0.05, 0.02, 0.3, 0.6, 0.4, 0.5])
    else: # splicing
        # Weights focused on statistical, compression, and other inconsistencies
        weights = np.array([0.9, 0.9, 0.7, 0.7, 0.8, 0.7, 0.1, 0.05, 0.02, 0.05, 0.9, 0.7, 0.6, 0.7, 0.8, 0.7, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6, 0.5, 0.5, 0.4, 0.7, 0.3, 0.4])
    
    # Pad or truncate weights to match feature vector length
    if len(weights) > len(features):
        weights = weights[:len(features)]
    elif len(weights) < len(features):
        # Default to a small, non-zero weight for padded features
        weights = np.pad(weights, (0, len(features) - len(weights)), 'constant', constant_values=0.1) 
    
    weighted_features = features * weights
    score = np.sum(weighted_features) / len(features) * 100
    
    return min(max(score, 0), 100)

def simulate_svm_classification(features, manipulation_type):
    """Simulate SVM classification - IMPROVED VERSION"""
    # Features will be normalized 0-1 from previous step.
    # Use selected indices to represent SVM decision boundaries for key features.
    
    if manipulation_type == 'copy_move':
        # Prioritize ransac inliers, block matches, geometric transform presence
        # indices for: ransac_inliers(7), block_matches(9), geometric_transform(8), ela_regional_inconsistency(3)
        key_indices = [idx for idx in [7, 9, 8, 3] if idx < len(features)]
        # Add a bias term / threshold
        bias_threshold = 0.4 
    else: # splicing
        # Prioritize ELA mean, noise inconsistency, JPEG ghost ratio, frequency inconsistency
        # indices for: ela_mean(0), ela_std(1), noise_inconsistency(10), jpeg_ghost_ratio(11), frequency_inconsistency(14), texture_inconsistency(16)
        key_indices = [idx for idx in [0, 1, 10, 11, 14, 16] if idx < len(features)]
        bias_threshold = 0.35 
    
    if len(key_indices) > 0:
        # Sum relevant features, high value indicates manipulation
        decision_score_sum = np.sum(features[key_indices]) / len(key_indices) # Average relevant feature values
        
        # Linearly map decision score (adjusted for bias) to 0-100
        # If score is > bias_threshold, it contributes positively, else negatively (after scaling)
        decision_score = (decision_score_sum - bias_threshold) * 200 # Factor 200 to map roughly from -0.x to 100
    else:
        decision_score = 0
    
    return min(max(decision_score, 0), 100) # Clip between 0 and 100

def simulate_neural_network_classification(features, manipulation_type):
    """Simulate Neural Network classification - FIXED VERSION"""
    try:
        # Simple two-layer feedforward network simulation
        # Normalize features first if not already done, though they should be 0-1
        
        # Hidden layer 1: Tanh activation
        # Arbitrary weights and biases for simulation
        hidden1_weights = np.linspace(-0.5, 0.5, len(features)) 
        hidden1_output = tanh_activation(features * hidden1_weights + 0.1) # Added a small bias

        # Hidden layer 2: Sigmoid activation
        hidden2_weights = np.linspace(0.2, 1.2, len(hidden1_output))
        hidden2_output = sigmoid(hidden1_output * hidden2_weights - 0.2) # Added a small negative bias

        # Output layer: weighted sum, then scaled
        output_weights_base = np.ones(len(hidden2_output)) 

        if manipulation_type == 'copy_move':
            # Boost weights for copy-move related features' contributions (e.g., SIFT, Block matches indices)
            if len(hidden2_output) > 9:
                output_weights_base[[7, 8, 9]] *= 3.0 # RANSAC Inliers, Geom Transform, Block Matches
            if len(hidden2_output) > 25: # Localization tampering percentage
                output_weights_base[25] *= 1.5 
        else: # splicing
            # Boost weights for splicing related features' contributions (e.g., ELA, Noise, JPEG ghost indices)
            if len(hidden2_output) > 11:
                output_weights_base[[0, 1, 10, 11]] *= 2.5 # ELA Mean, ELA Std, Noise Inc, JPEG Ghost Ratio
            if len(hidden2_output) > 18:
                 output_weights_base[[14, 16, 17, 18]] *= 2.0 # Freq, Texture, Edge, Illumination inconsistencies

        final_output_sum = np.sum(hidden2_output * output_weights_base)

        # Scale to 0-100 range.
        # Max sum for normalized features and boosted weights. A general factor is enough for simulation.
        max_possible_sum = np.sum(output_weights_base[output_weights_base > 0.0]) # sum positive weights only
        score = (final_output_sum / (max_possible_sum + 1e-9)) * 100 
        
        return min(max(score, 0), 100)
    except Exception as e:
        print(f"  Warning: Neural network simulation failed: {e}")
        # Fallback to a basic aggregated score
        feature_sum_effective = np.mean(features) # Mean of features more stable than raw sum
        if manipulation_type == 'copy_move':
            return min(feature_sum_effective * 50, 100) # Arbitrary scaling
        else:
            return min(feature_sum_effective * 40, 100)

# ======================= Advanced Classification System =======================

def classify_manipulation_advanced(analysis_results):
    """Advanced classification with comprehensive scoring including localization and uncertainty"""
    
    try:
        # Initialize uncertainty classifier
        uncertainty_classifier = UncertaintyClassifier()
        
        # Calculate probabilities with uncertainty
        probabilities = uncertainty_classifier.calculate_manipulation_probability(analysis_results)
        uncertainty_report = uncertainty_classifier.generate_uncertainty_report(probabilities)
        
        # Traditional classification for backward compatibility
        feature_vector = prepare_feature_vector(analysis_results)
        ensemble_copy_move, ensemble_splicing, ml_scores = classify_with_advanced_ml(feature_vector)
        
        ml_copy_move_score = ensemble_copy_move
        ml_splicing_score = ensemble_splicing
        
        # Perhitungan skor heuristik
        copy_move_score_heuristic = 0
        splicing_score_heuristic = 0
        
        ransac_inliers = analysis_results.get('ransac_inliers', 0)
        if ransac_inliers >= 50: copy_move_score_heuristic += 60
        elif ransac_inliers >= 20: copy_move_score_heuristic += 50
        elif ransac_inliers >= 10: copy_move_score_heuristic += 30
        
        block_matches = len(analysis_results.get('block_matches', []))
        if block_matches >= 30: copy_move_score_heuristic += 50
        elif block_matches >= 10: copy_move_score_heuristic += 30

        ela_mean = analysis_results.get('ela_mean', 0)
        if ela_mean > 15.0: splicing_score_heuristic += 50
        elif ela_mean > 8.0: splicing_score_heuristic += 35
        
        noise_inconsistency = analysis_results.get('noise_analysis', {}).get('overall_inconsistency', 0)
        if noise_inconsistency >= 0.7: splicing_score_heuristic += 50
        elif noise_inconsistency > 0.35: splicing_score_heuristic += 35
        
        if analysis_results.get('geometric_transform') is not None: copy_move_score_heuristic += 30
        
        copy_move_score_heuristic = min(copy_move_score_heuristic, 100)
        splicing_score_heuristic = min(splicing_score_heuristic, 100)

        raw_copy_move = (copy_move_score_heuristic * 0.8 + ml_copy_move_score * 0.2)
        raw_splicing = (splicing_score_heuristic * 0.8 + ml_splicing_score * 0.2)
        
        final_copy_move_score = min(max(0, int(raw_copy_move)), 100)
        final_splicing_score = min(max(0, int(raw_splicing)), 100)
        
        # ======================= PEMBARUAN UTAMA DI SINI =======================
        # Hasil utama sekarang diambil langsung dari laporan ketidakpastian
        # `type` dan `confidence` akan diganti dengan `primary_assessment` dan `assessment_reliability`
        
        final_manipulation_type = uncertainty_report.get('primary_assessment', 'N/A').replace('Indikasi: ', '')
        assessment_reliability = uncertainty_report.get('assessment_reliability', 'Sangat Rendah')
        
        # Detail sekarang adalah kombinasi dari indikator keandalan dan koherensi
        final_details = [uncertainty_report.get('indicator_coherence', '')]
        final_details.extend(uncertainty_report.get('reliability_indicators', []))
        
        # Buat dictionary hasil klasifikasi yang baru
        classification_result = {
            'type': final_manipulation_type,
            'confidence': assessment_reliability, # Menggunakan reliabilitas sebagai 'confidence'
            'copy_move_score': final_copy_move_score,
            'splicing_score': final_splicing_score,
            'details': final_details,
            'ml_scores': {
                'copy_move': ml_copy_move_score,
                'splicing': ml_splicing_score,
                'detailed_ml_scores': ml_scores
            },
            'feature_vector': feature_vector.tolist(),
            'traditional_scores': {
                'copy_move': copy_move_score_heuristic,
                'splicing': splicing_score_heuristic
            },
            'uncertainty_analysis': {
                'probabilities': probabilities,
                'report': uncertainty_report,
                'formatted_output': format_probability_results(probabilities, uncertainty_report),
            }
        }

        return classification_result

    except KeyError as e:
        print(f"  Warning: Classification failed due to missing key: {e}. Returning default error.")
        return {
            'type': "Analysis Error", 'confidence': "Error",
            'copy_move_score': 0, 'splicing_score': 0, 'details': [f"Classification error: Missing key {str(e)}."],
            'ml_scores': {}, 'feature_vector': [], 'traditional_scores': {},
            'uncertainty_analysis': {
                'probabilities': {'copy_move_probability': 0.0, 'splicing_probability': 0.0, 'authentic_probability': 1.0, 'uncertainty_level': 1.0, 'confidence_intervals': {'copy_move': {'lower':0, 'upper':0}, 'splicing': {'lower':0, 'upper':0}, 'authentic': {'lower':0, 'upper':0}}},
                'report': {'primary_assessment': 'Error: Data Insufficient', 'assessment_reliability': 'Sangat Rendah', 'indicator_coherence': 'Data tidak memadai.', 'reliability_indicators': [], 'recommendation': 'Jalankan ulang analisis dengan data yang valid.'},
                'formatted_output': "Error: Classification data missing. See logs."
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'type': "Analysis Error", 'confidence': "Error",
            'copy_move_score': 0, 'splicing_score': 0, 'details': [f"Classification error: {str(e)}"],
            'ml_scores': {}, 'feature_vector': [], 'traditional_scores': {},
            'uncertainty_analysis': {
                'probabilities': {'copy_move_probability': 0.0, 'splicing_probability': 0.0, 'authentic_probability': 1.0, 'uncertainty_level': 1.0, 'confidence_intervals': {'copy_move': {'lower':0, 'upper':0}, 'splicing': {'lower':0, 'upper':0}, 'authentic': {'lower':0, 'upper':0}}},
                'report': {'primary_assessment': 'Error: Unknown Issue', 'assessment_reliability': 'Sangat Rendah', 'indicator_coherence': 'Terjadi error tidak terduga.', 'reliability_indicators': [], 'recommendation': 'Jalankan ulang analisis atau hubungi support.'},
                'formatted_output': "Error: Classification process failed. See logs for details."
            }
        }

# ======================= Confidence and Detail Functions (Tidak diubah, tetap sebagai referensi internal) =======================

def get_enhanced_confidence_level(score):
    if score >= 90: return "Sangat Tinggi (>90%)"
    elif score >= 75: return "Tinggi (75-90%)"
    elif score >= 60: return "Sedang (60-75%)"
    elif score >= 45: return "Rendah (45-60%)"
    else: return "Sangat Rendah (<45%)"

def get_enhanced_copy_move_details(results):
    details = []
    ransac_inliers = results.get('ransac_inliers', 0)
    if ransac_inliers > 0: details.append(f"✓ RANSAC verification: {ransac_inliers} geometric matches")
    transform = results.get('geometric_transform')
    if transform: details.append(f"✓ Geometric transformation: {transform[0]}")
    block_matches_len = len(results.get('block_matches', []))
    if block_matches_len > 0: details.append(f"✓ Block matching: {block_matches_len} identical blocks")
    return details

def get_enhanced_splicing_details(results):
    details = []
    ela_outliers = results.get('ela_regional_stats', {}).get('outlier_regions', 0)
    if ela_outliers > 0: details.append(f"⚠ ELA anomalies: {ela_outliers} suspicious regions")
    noise_inconsistency = results.get('noise_analysis', {}).get('overall_inconsistency', 0)
    if noise_inconsistency > 0.25: details.append(f"⚠ Noise inconsistency: {noise_inconsistency:.3f}")
    jpeg_double_comp = results.get('jpeg_analysis', {}).get('basic_analysis', {}).get('compression_inconsistency', False)
    if jpeg_double_comp: details.append("⚠ JPEG compression inconsistency detected")
    return details

def get_enhanced_complex_details(results):
    return get_enhanced_copy_move_details(results) + get_enhanced_splicing_details(results)


# ======================= Classification Utilities (Tidak ada perubahan signifikan) =======================

def generate_classification_report(classification_result, analysis_results):
    """Generate comprehensive classification report"""
    # Safe access to nested dictionary values
    ml_scores = classification_result.get('ml_scores', {})
    detailed_ml_scores = ml_scores.get('detailed_ml_scores', {})
    uncertainty_report = classification_result.get('uncertainty_analysis', {}).get('report', {})

    report = {
        'summary': {
            'detected_type': classification_result['type'],
            'assessment_reliability': classification_result['confidence'],
            'copy_move_score': classification_result['copy_move_score'],
            'splicing_score': classification_result['splicing_score']
        },
        'evidence': {
            'technical_indicators': classification_result['details'],
            'feature_count': len(classification_result['feature_vector']),
        },
        'methodology': {
            'feature_vector_size': len(classification_result['feature_vector']),
            'ml_algorithms_used': ['Random Forest', 'SVM', 'Neural Network'],
        },
        'uncertainty': uncertainty_report
    }
    
    return report

def export_classification_metrics(classification_result, output_filename="classification_metrics.txt"):
    """Export classification metrics to text file"""
    
    report_text = format_probability_results(
        classification_result.get('uncertainty_analysis', {}).get('probabilities', {}),
        classification_result.get('uncertainty_analysis', {}).get('report', {})
    )
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("CLASSIFICATION METRICS EXPORT\n\n")
        f.write(report_text)
        f.write("\n\nADDITIONAL DEBUG INFO:\n")
        f.write(f"Copy-Move Score (Final): {classification_result.get('copy_move_score', 0)}\n")
        f.write(f"Splicing Score (Final): {classification_result.get('splicing_score', 0)}\n")

    print(f"📊 Classification metrics exported to '{output_filename}'")
    return output_filename
# --- END OF FILE classification.py ---