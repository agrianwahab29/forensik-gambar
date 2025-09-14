# validator.py

from PIL import Image
import numpy as np # Import numpy

# Diambil dari app2.py
class ForensicValidator:
    def __init__(self):
        # Bobot algoritma (harus berjumlah 1.0)
        self.weights = {
            'clustering': 0.25,  # K-Means (metode utama)
            'localization': 0.25,  # Lokalisasi tampering (metode utama)
            'ela': 0.20,  # Error Level Analysis (metode pendukung)
            'feature_matching': 0.15,  # SIFT (metode pendukung)
            'metadata': 0.15,  # Metadata analysis (metode pendukung)
        }
        
        # Threshold minimum untuk setiap teknik (0-1 scale)
        self.thresholds = {
            'clustering': 0.60,
            'localization': 0.60,
            'ela': 0.60,
            'feature_matching': 0.60,
            'metadata': 0.40,  # Threshold lebih rendah untuk metadata
        }
    
    def validate_clustering(self, analysis_results):
        """Validasi kualitas clustering K-Means"""
        # Access `kmeans_localization` inside `localization_analysis`
        kmeans_data = analysis_results.get('localization_analysis', {}).get('kmeans_localization', {})

        if not kmeans_data or 'cluster_means' not in kmeans_data:
            return 0.0, "Data clustering tidak tersedia atau tidak lengkap"
            
        cluster_means = kmeans_data.get('cluster_means', [])
        cluster_count = len(cluster_means)

        if cluster_count < 2:
            return 0.4, "Diferensiasi cluster tidak memadai (kurang dari 2 cluster teridentifikasi)"
            
        # 2. Periksa pemisahan cluster (semakin tinggi selisih mean ELA antar cluster semakin baik)
        mean_diff = max(cluster_means) - min(cluster_means) if cluster_means else 0
        mean_diff_score = min(1.0, mean_diff / 20.0)  # Normalisasi: a diff of 20 implies a score of 1.0
        
        # 3. Periksa identifikasi cluster tampering (jika ada cluster dengan ELA tinggi yang ditandai)
        suspicious_clusters = kmeans_data.get('suspicious_clusters', [])
        # Check if highest ELA cluster is marked as suspicious
        tampering_identified = len(suspicious_clusters) > 0 and cluster_means and max(cluster_means) > 5
        
        # 4. Periksa area tampering berukuran wajar (tidak terlalu kecil atau terlalu besar)
        tampering_pct = analysis_results.get('localization_analysis', {}).get('tampering_percentage', 0)
        size_score = 0.0
        if 1.0 < tampering_pct < 50.0:  # Ideal size range for actual tampering
            size_score = 1.0
        elif tampering_pct <= 1.0 and tampering_pct > 0.0:  # Too small but exists
            size_score = tampering_pct # linear interpolation from 0 to 1
        elif tampering_pct >= 50.0: # Too large (might be global effect or full image replacement)
            size_score = max(0.0, 1.0 - ((tampering_pct - 50) / 50.0)) # Linear falloff from 1.0 to 0.0 for 50-100%

        # Gabungkan skor dengan faktor berbobot
        confidence = (
            0.3 * min(cluster_count / 5.0, 1.0)  # Up to 5 clusters, more means better differentiation up to a point
            + 0.3 * mean_diff_score
            + 0.2 * float(tampering_identified)
            + 0.2 * size_score
        )
        
        confidence = min(1.0, max(0.0, confidence)) # Ensure 0-1 range
        
        details = (
            f"Jumlah cluster: {cluster_count}, "
            f"Pemisahan cluster (Max-Min ELA): {mean_diff:.2f}, "
            f"Tampering teridentifikasi: {'Ya' if tampering_identified else 'Tidak'}, "
            f"Area tampering: {tampering_pct:.1f}%"
        )
        
        return confidence, details
    
    def validate_localization(self, analysis_results):
        """Validasi efektivitas lokalisasi tampering"""
        localization_data = analysis_results.get('localization_analysis', {})

        if not localization_data:
            return 0.0, "Data lokalisasi tidak tersedia"
            
        # 1. Periksa apakah mask tampering yang digabungkan telah dihasilkan
        has_combined_mask = 'combined_tampering_mask' in localization_data and localization_data['combined_tampering_mask'] is not None and localization_data['combined_tampering_mask'].size > 0
        if not has_combined_mask:
            return 0.0, "Tidak ada mask tampering gabungan yang dihasilkan"
            
        # 2. Periksa persentase area yang ditandai (harus wajar untuk manipulasi)
        tampering_pct = localization_data.get('tampering_percentage', 0.0)
        area_score = 0.0
        if 0.5 < tampering_pct < 40.0:  # Common range for effective tampering, neither too small nor too large
            area_score = 1.0
        elif 0.0 < tampering_pct <= 0.5:  # Too small, might be noise
            area_score = tampering_pct / 0.5 # Scale from 0 to 1 as it gets to 0.5%
        else: # tampering_pct >= 40.0: # Too large, could be entire image replaced or a global filter
            area_score = max(0.0, 1.0 - ((tampering_pct - 40.0) / 60.0)) # Drops from 1 to 0 for 40% to 100%
        
        # 3. Periksa konsistensi fisik dengan analisis lain
        ela_mean = analysis_results.get('ela_mean', 0.0)
        ela_std = analysis_results.get('ela_std', 0.0)
        noise_inconsistency = analysis_results.get('noise_analysis', {}).get('overall_inconsistency', 0.0)
        jpeg_ghost_ratio = analysis_results.get('jpeg_ghost_suspicious_ratio', 0.0) # Check this exists
        
        # High ELA means stronger splicing signal in general.
        ela_consistency = min(1.0, max(0.0, (ela_mean - 5.0) / 10.0)) # Scores 0 at ELA mean 5, 1 at 15
        ela_consistency = ela_consistency * min(1.0, max(0.0, (ela_std - 10.0) / 15.0)) # Add std influence (scores 0 at 10, 1 at 25)

        # High noise inconsistency (for areas, or globally near manipulated regions)
        noise_consistency = min(1.0, max(0.0, (noise_inconsistency - 0.1) / 0.3)) # Scores 0 at 0.1, 1 at 0.4

        # High JPEG ghost ratio
        jpeg_consistency = min(1.0, max(0.0, jpeg_ghost_ratio / 0.2)) # Scores 0 at 0, 1 at 0.2

        # Combine physical consistency. Max implies if one is strong, it still lends credence.
        physical_consistency = max(ela_consistency, noise_consistency, jpeg_consistency)
        
        # Skor gabungan dengan faktor berbobot
        confidence = (
            0.4 * float(has_combined_mask) # Must have a mask
            + 0.3 * area_score # Quality of area percentage
            + 0.3 * physical_consistency # Agreement with other physical anomalies
        )
        
        confidence = min(1.0, max(0.0, confidence)) # Kalibrasi ke rentang [0,1]
        
        details = (
            f"Mask tampering: {'Ada' if has_combined_mask else 'Tidak ada'}, "
            f"Persentase area: {tampering_pct:.1f}%, "
            f"Konsistensi ELA (Mean, Std): {ela_consistency:.2f}, "
            f"Konsistensi noise: {noise_consistency:.2f}, "
            f"JPEG ghost: {jpeg_consistency:.2f}"
        )
        
        return confidence, details
    
    def validate_ela(self, analysis_results):
        """Enhanced validation for advanced Error Level Analysis"""
        ela_image_obj = analysis_results.get('ela_image')
        # Check if ela_image object itself is a valid PIL Image or has image-like properties that can be converted
        if ela_image_obj is None or (not isinstance(ela_image_obj, Image.Image) and not hasattr(ela_image_obj, 'size') and not hasattr(ela_image_obj, 'ndim')):
             return 0.0, "Tidak ada gambar ELA yang tersedia atau format tidak valid"
            
        ela_mean = analysis_results.get('ela_mean', 0.0)
        ela_std = analysis_results.get('ela_std', 0.0)
        
        # Enhanced adaptive scoring for ELA mean with better low-value handling
        mean_score = self._calculate_adaptive_mean_score(ela_mean, analysis_results)
        
        # Enhanced std scoring with texture awareness
        std_score = self._calculate_adaptive_std_score(ela_std, analysis_results)
            
        # Enhanced regional analysis with new features
        regional_stats = analysis_results.get('ela_regional_stats', {})
        regional_inconsistency = regional_stats.get('regional_inconsistency', 0.0)
        outlier_regions = regional_stats.get('outlier_regions', 0)
        entropy_inconsistency = regional_stats.get('entropy_inconsistency', 0.0)
        texture_aware_score = regional_stats.get('texture_aware_score', 0.0)
        confidence_weighted_score = regional_stats.get('confidence_weighted_score', 0.0)
        
        # Advanced scoring components
        inconsistency_score = min(1.0, regional_inconsistency / 0.4)  # Slightly more sensitive
        outlier_score = min(1.0, outlier_regions / 4.0)  # More sensitive to outliers
        entropy_score = min(1.0, entropy_inconsistency / 2.0)
        texture_score = min(1.0, texture_aware_score)
        confidence_score = confidence_weighted_score
        
        # Enhanced quality analysis with frequency domain features
        quality_stats = analysis_results.get('ela_quality_stats', [])
        quality_variation, frequency_score = self._analyze_enhanced_quality_metrics(quality_stats)
        
        # Adaptive weighting based on signal characteristics
        signal_enhancement_ratio = regional_stats.get('signal_enhancement_ratio', 1.0)
        enhancement_bonus = min(0.2, (signal_enhancement_ratio - 1.0) * 0.5) if signal_enhancement_ratio > 1.0 else 0.0
        
        # Dynamic weight adjustment for low ELA scenarios
        base_weights = self._calculate_dynamic_weights(ela_mean, ela_std, regional_stats)
        
        # Combine scores with adaptive weights
        confidence = (
            base_weights['mean'] * mean_score
            + base_weights['std'] * std_score
            + base_weights['inconsistency'] * inconsistency_score
            + base_weights['outlier'] * outlier_score
            + base_weights['entropy'] * entropy_score
            + base_weights['texture'] * texture_score
            + base_weights['confidence'] * confidence_score
            + base_weights['quality'] * quality_variation
            + base_weights['frequency'] * frequency_score
            + enhancement_bonus
        )
        
        confidence = min(1.0, max(0.0, confidence)) # Ensure 0-1 range
        
        # Enhanced details with preservation of important low-value explanations
        details = self._generate_enhanced_details(
            ela_mean, ela_std, mean_score, std_score, 
            regional_inconsistency, outlier_regions, 
            entropy_inconsistency, texture_aware_score,
            confidence_weighted_score, signal_enhancement_ratio
        )
        
        return confidence, details
    
    def _calculate_adaptive_mean_score(self, ela_mean, analysis_results):
        """Calculate adaptive mean score with better handling of low values"""
        regional_stats = analysis_results.get('ela_regional_stats', {})
        suspicious_regions = regional_stats.get('suspicious_regions', [])
        outlier_regions = regional_stats.get('outlier_regions', 0)
        regional_inconsistency = regional_stats.get('regional_inconsistency', 0.0)
        
        # More sophisticated scoring that doesn't penalize low ELA values as harshly
        if 6.0 <= ela_mean <= 25.0:  # Good range for manipulation
            base_score = 1.0
        elif ela_mean > 25.0:  # High values with gradual penalty
            base_score = max(0.3, 1.0 - (ela_mean - 25.0) / 20.0)
        elif 2.0 <= ela_mean < 6.0:  # Low but potentially valid range
            # Use a logarithmic scale for better low-value handling
            base_score = 0.3 + (0.7 * (ela_mean - 2.0) / 4.0)
            
            # Significant bonus for having suspicious regions or outliers
            if len(suspicious_regions) > 0 or outlier_regions > 0:
                region_bonus = min(0.4, (len(suspicious_regions) * 0.1 + outlier_regions * 0.05))
                base_score = min(1.0, base_score + region_bonus)
            
            # Additional bonus for regional inconsistency
            if regional_inconsistency > 0.1:
                inconsistency_bonus = min(0.3, regional_inconsistency)
                base_score = min(1.0, base_score + inconsistency_bonus)
                
        elif 0.1 <= ela_mean < 2.0:  # Very low values
            # Don't automatically give 0 score - use regional analysis
            base_score = 0.1 + (0.2 * ela_mean / 2.0)
            
            # Strong bonus if there are regional anomalies despite low mean
            if len(suspicious_regions) > 0 or outlier_regions > 0 or regional_inconsistency > 0.1:
                anomaly_score = min(0.6, 
                    len(suspicious_regions) * 0.15 + 
                    outlier_regions * 0.1 + 
                    regional_inconsistency * 2.0
                )
                base_score = min(0.8, base_score + anomaly_score)
        else:
            # Only give 0 if truly no signal
            base_score = max(0.0, ela_mean * 0.1)
            
        return base_score
    
    def _calculate_adaptive_std_score(self, ela_std, analysis_results):
        """Calculate adaptive std score with texture awareness"""
        regional_stats = analysis_results.get('ela_regional_stats', {})
        texture_aware_score = regional_stats.get('texture_aware_score', 0.0)
        
        # Adjust std thresholds based on texture complexity
        texture_factor = 1.0 + texture_aware_score * 0.5
        lower_threshold = 8.0 / texture_factor
        upper_threshold = 35.0 * texture_factor
        
        if lower_threshold <= ela_std <= upper_threshold:
            std_score = 1.0
        elif ela_std > upper_threshold:
            std_score = max(0.0, 1.0 - (ela_std - upper_threshold) / 15.0)
        elif ela_std < lower_threshold and ela_std > 0.0:
            std_score = ela_std / lower_threshold
        else:
            std_score = 0.0
            
        return std_score
    
    def _analyze_enhanced_quality_metrics(self, quality_stats):
        """Analyze enhanced quality metrics including frequency features"""
        if not quality_stats:
            return 0.0, 0.0
            
        # Traditional quality variation
        means = [q.get('mean', 0) for q in quality_stats if 'mean' in q]
        quality_variation = 0.0
        if len(means) > 1:
            quality_variation = max(means) - min(means)
            quality_variation = min(1.0, quality_variation / 12.0)  # More sensitive
        
        # Frequency domain analysis
        frequency_score = 0.0
        if any('frequency_energy' in q for q in quality_stats):
            freq_energies = [q.get('frequency_energy', 0) for q in quality_stats]
            freq_variation = np.std(freq_energies) if len(freq_energies) > 1 else 0.0
            frequency_score = min(1.0, freq_variation / 0.1)
        
        return quality_variation, frequency_score
    
    def _calculate_dynamic_weights(self, ela_mean, ela_std, regional_stats):
        """Calculate dynamic weights based on signal characteristics"""
        # Base weights
        weights = {
            'mean': 0.25,
            'std': 0.15,
            'inconsistency': 0.15,
            'outlier': 0.15,
            'entropy': 0.1,
            'texture': 0.05,
            'confidence': 0.05,
            'quality': 0.05,
            'frequency': 0.05
        }
        
        # Adjust weights for low ELA scenarios
        if ela_mean < 5.0:
            # Increase importance of regional and texture analysis
            weights['inconsistency'] += 0.1
            weights['entropy'] += 0.05
            weights['texture'] += 0.05
            weights['confidence'] += 0.1
            weights['mean'] -= 0.15
            weights['std'] -= 0.15
        
        # Adjust for high texture complexity
        texture_score = regional_stats.get('texture_aware_score', 0.0)
        if texture_score > 0.5:
            weights['texture'] += 0.05
            weights['frequency'] += 0.05
            weights['mean'] -= 0.05
            weights['std'] -= 0.05
        
        return weights
    
    def _generate_enhanced_details(self, ela_mean, ela_std, mean_score, std_score,
                                  regional_inconsistency, outlier_regions,
                                  entropy_inconsistency, texture_aware_score,
                                  confidence_weighted_score, signal_enhancement_ratio):
        """Generate enhanced details with preservation of important explanations"""
        
        # Format scores to ensure they show actual calculated values, not just 0.00
        mean_score_display = mean_score if mean_score > 0.01 else max(0.01, ela_mean * 0.01)
        
        details = (
            f"ELA mean: {ela_mean:.2f} (score: {mean_score:.2f}), "
            f"ELA std: {ela_std:.2f} (score: {std_score:.2f}), "
            f"Inkonsistensi regional: {regional_inconsistency:.3f}, "
            f"Region outlier: {outlier_regions}, "
            f"Entropi inkonsistensi: {entropy_inconsistency:.3f}, "
            f"Tekstur-aware score: {texture_aware_score:.3f}, "
            f"Weighted confidence: {confidence_weighted_score:.3f}, "
            f"Signal enhancement: {signal_enhancement_ratio:.2f}x"
        )
        
        # Add interpretive guidance with score explanation
        if ela_mean < 5.0:
            score_explanation = "rendah namun masih valid" if mean_score > 0.3 else "rendah karena nilai ELA minimal"
            details += f" | CATATAN: Nilai ELA {ela_mean:.2f} tergolong rendah. "
            details += f"Score {mean_score:.2f} ({score_explanation}) karena: "
            
            if outlier_regions > 0 or regional_inconsistency > 0.1:
                details += f"(1) Terdeteksi {outlier_regions} region outlier dan inkonsistensi regional {regional_inconsistency:.3f}, "
                details += f"yang meningkatkan score meskipun ELA mean rendah. "
            else:
                details += f"(1) Tidak ada anomali regional signifikan yang terdeteksi. "
                
            details += f"(2) Nilai ELA rendah dapat mengindikasikan gambar asli atau manipulasi halus. "
            details += f"(3) Analisis tekstur (score: {texture_aware_score:.3f}) dan regional tetap penting untuk konfirmasi."
        elif ela_mean > 25.0:
            details += f" | CATATAN: Nilai ELA tinggi ({ela_mean:.2f}) mungkin mengindikasikan over-processing atau noise berlebih."
        
        return details
    
    def validate_metadata(self, analysis_results):
        """Validasi komprehensif metadata dengan pendekatan yang realistis"""
        metadata = analysis_results.get('metadata', {})
        
        if not metadata:
            return 0.0, "Data metadata tidak tersedia"
        
        # Ambil skor autentisitas metadata
        authenticity_score = metadata.get('Metadata_Authenticity_Score', 0)
        inconsistencies = metadata.get('Metadata_Inconsistency', [])
        
        # Konversi skor ke skala 0-1 dengan threshold yang lebih realistis
        if authenticity_score >= 75:      # Excellent metadata
            base_confidence = 1.0
        elif authenticity_score >= 65:    # Good metadata (lowered from 80)
            base_confidence = 0.9
        elif authenticity_score >= 55:    # Acceptable metadata (lowered from 70)
            base_confidence = 0.7
        elif authenticity_score >= 45:    # Questionable but possible (lowered from 60)
            base_confidence = 0.5
        elif authenticity_score >= 35:    # Poor but not necessarily fake (lowered from 50)
            base_confidence = 0.3
        elif authenticity_score >= 25:    # Very poor (lowered from 40)
            base_confidence = 0.2
        else:                             # Extremely poor
            base_confidence = 0.1
        
        # Analisis inkonsistensi dengan bobot yang disesuaikan
        inconsistency_penalty = 0.0
        critical_inconsistencies = 0
        minor_inconsistencies = 0
        
        for inconsistency in inconsistencies:
            inconsistency_lower = inconsistency.lower()
            
            # Inkonsistensi kritis (penalty besar)
            if any(critical in inconsistency_lower for critical in 
                   ['photoshop', 'gimp', 'heavily modified', 'fake']):
                critical_inconsistencies += 1
                inconsistency_penalty += 0.15
            
            # Inkonsistensi sedang (penalty sedang)
            elif any(moderate in inconsistency_lower for moderate in 
                     ['time difference', 'software', 'editing']):
                minor_inconsistencies += 1
                inconsistency_penalty += 0.05
            
            # Inkonsistensi ringan (penalty kecil)
            else:
                inconsistency_penalty += 0.02
        
        # Terapkan penalty dengan batas maksimum
        inconsistency_penalty = min(0.4, inconsistency_penalty)
        
        # Hitung confidence akhir
        final_confidence = max(0.0, base_confidence - inconsistency_penalty)
        
        # Bonus untuk metadata yang sangat lengkap
        if authenticity_score >= 80:
            final_confidence = min(1.0, final_confidence + 0.1)
        
        # Detail yang informatif
        details = (
            f"Skor autentisitas: {authenticity_score}/100, "
            f"Inkonsistensi kritis: {critical_inconsistencies}, "
            f"Inkonsistensi minor: {minor_inconsistencies}, "
            f"Total inkonsistensi: {len(inconsistencies)}"
        )
        
        return final_confidence, details
    
    def validate_feature_matching(self, analysis_results):
        """Validasi kualitas pencocokan fitur SIFT/ORB/AKAZE"""
        ransac_inliers = analysis_results.get('ransac_inliers', 0)
        sift_matches = analysis_results.get('sift_matches', 0) # Raw matches before RANSAC
        sift_keypoints = analysis_results.get('sift_keypoints', [])
        
        # Ensure ransac_inliers are not negative
        if ransac_inliers < 0: 
            ransac_inliers = 0
        
        # Check if we have keypoints but no matches (legitimate scenario)
        if len(sift_keypoints) > 0 and sift_matches == 0:
            # Still give a small score for having features, even without matches
            return 0.1, f"Terdeteksi {len(sift_keypoints)} keypoints tetapi tidak ada matches (kemungkinan tidak ada copy-move)"
            
        if sift_matches < 1: # No matches at all
             return 0.0, "Tidak ada data pencocokan fitur yang signifikan"
            
        # 1. Periksa kecocokan yang signifikan (RANSAC inliers sebagai indikator kuat)
        # Normalisasi inlier: A good amount (e.g. 20-30 inliers) is strong.
        inlier_score = min(1.0, ransac_inliers / 25.0) # Score 1.0 at 25 inliers
        
        # Raw matches count (provides context for potential matching opportunities)
        match_score = min(1.0, sift_matches / 150.0) # Score 1.0 at 150 raw matches
        
        # 2. Periksa transformasi geometris yang ditemukan oleh RANSAC
        has_transform = analysis_results.get('geometric_transform') is not None
        transform_type = None
        if has_transform: # geometric_transform format is (type_string, matrix)
            try: # Robust access for tuple/list
                transform_type = analysis_results['geometric_transform'][0]
            except (TypeError, IndexError): # In case it's not a tuple or is empty
                transform_type = "Unknown_Type"
        
        # 3. Periksa kecocokan blok (harus berkorelasi dengan kecocokan fitur untuk copy-move)
        block_matches = len(analysis_results.get('block_matches', []))
        block_score = min(1.0, block_matches / 15.0) # Score 1.0 at 15 block matches
        
        # Cross-algorithm correlation: High RANSAC and Block matches
        correlation_score = 0.0
        if ransac_inliers > 10 and block_matches > 5: # Both strong: high correlation
            correlation_score = 1.0
        elif ransac_inliers > 0 and block_matches > 0: # Both exist: some correlation
            correlation_score = 0.5
        
        # Gabungkan skor dengan bobot
        confidence = (
            0.35 * inlier_score # Highest weight for RANSAC inliers
            + 0.20 * match_score # Medium for overall matches
            + 0.20 * float(has_transform) # Medium for detecting transform type
            + 0.10 * block_score # Lower for general block matching
            + 0.15 * correlation_score # Consistency score between two detection methods
        )
        
        confidence = min(1.0, max(0.0, confidence)) # Ensure 0-1 range
        
        details = (
            f"RANSAC inliers: {ransac_inliers} (score: {inlier_score:.2f}), "
            f"Raw SIFT matches: {sift_matches} (score: {match_score:.2f}), "
            f"Tipe transformasi: {transform_type if transform_type else 'Tidak ada'}, "
            f"Kecocokan blok: {block_matches} (score: {block_score:.2f})"
        )
        
        return confidence, details
    
    def validate_cross_algorithm(self, analysis_results):
        """Validasi konsistensi silang algoritma"""
        if not analysis_results:
            return [], 0.0, "Tidak ada hasil analisis yang tersedia", []
        
        validation_results = {}
        for technique, validate_func in [
            ('clustering', self.validate_clustering),
            ('localization', self.validate_localization),
            ('ela', self.validate_ela),
            ('feature_matching', self.validate_feature_matching),
            ('metadata', self.validate_metadata)
        ]:
            confidence, details = validate_func(analysis_results)
            # Ensure confidence is a float, especially important from fallback paths
            confidence = float(confidence)
            validation_results[technique] = {
                'confidence': confidence,
                'details': details,
                'weight': self.weights[technique],
                'threshold': self.thresholds[technique],
                'passed': confidence >= self.thresholds[technique]
            }
        
        # Prepare textual results for console/logging
        process_results_list = []
        
        for technique, result in validation_results.items():
            status = "[LULUS]" if result['passed'] else "[GAGAL]"
            emoji = "✅" if result['passed'] else "❌"
            process_results_list.append(f"{emoji} {status:10} | Validasi {technique.capitalize()} - Skor: {result['confidence']:.2f}")
            
        # Calculate weighted individual technique scores
        weighted_scores = {
            technique: result['confidence'] * result['weight']
            for technique, result in validation_results.items()
        }
        
        # Calculate inter-algorithm agreement ratio
        agreement_pairs = 0
        total_pairs = 0
        techniques_list = list(validation_results.keys()) # Convert to list to iterate
        
        for i in range(len(techniques_list)):
            for j in range(i + 1, len(techniques_list)):
                t1, t2 = techniques_list[i], techniques_list[j]
                total_pairs += 1
                # If both passed or both failed, they "agree"
                if validation_results[t1]['passed'] == validation_results[t2]['passed']:
                    agreement_pairs += 1
        
        if total_pairs > 0:
            agreement_ratio = float(agreement_pairs) / total_pairs
        else: # Handle case of 0 total pairs (e.g., less than 2 techniques or specific edge cases)
            agreement_ratio = 1.0 # If nothing to compare, assume perfect agreement logically
        
        # Combine weighted score and agreement bonus
        raw_weighted_total = sum(weighted_scores.values())
        consensus_boost = agreement_ratio * 0.10 # Add max 10% bonus for perfect agreement (tuned)
        
        final_score = (raw_weighted_total * 100) + (consensus_boost * 100)
        
        # Clamp final score between 0 and 100
        final_score = min(100.0, max(0.0, final_score))
        
        # Collect failed validations for detailed reporting
        failed_validations_detail = [
            {
                'name': f"Validasi {technique.capitalize()}",
                'reason': f"Skor kepercayaan di bawah ambang batas {result['threshold']:.2f}",
                'rule': f"LULUS = (Kepercayaan >= {result['threshold']:.2f})",
                'values': f"Nilai aktual: Kepercayaan = {result['confidence']:.2f}\nDetail: {result['details']}"
            }
            for technique, result in validation_results.items()
            if not result['passed']
        ]
        
        # Determine confidence level description for summary text
        if final_score >= 95:
            confidence_level = "Sangat Tinggi (Very High)"
            summary_text = f"Validasi sistem menunjukkan tingkat kepercayaan {confidence_level} dengan skor {final_score:.1f}%. "
            summary_text += "Semua metode analisis menunjukkan konsistensi dan kualitas tinggi."
        elif final_score >= 90:
            confidence_level = "Tinggi (High)"
            summary_text = f"Validasi sistem menunjukkan tingkat kepercayaan {confidence_level} dengan skor {final_score:.1f}%. "
            summary_text += "Sebagian besar metode analisis menunjukkan konsistensi dan kualitas baik."
        elif final_score >= 85:
            confidence_level = "Sedang (Medium)"
            summary_text = f"Validasi sistem menunjukkan tingkat kepercayaan {confidence_level} dengan skor {final_score:.1f}%. "
            summary_text += "Beberapa metode analisis menunjukkan inkonsistensi minor."
        else:
            confidence_level = "Rendah (Low)"
            summary_text = f"Validasi sistem menunjukkan tingkat kepercayaan {confidence_level} dengan skor {final_score:.1f}%. "
            summary_text += "Terdapat inkonsistensi signifikan antar metode analisis yang memerlukan perhatian."
        
        return process_results_list, final_score, summary_text, failed_validations_detail

# --- END OF FILE validator.py ---