# --- START OF FILE uncertainty_classification.py ---

"""
Advanced Uncertainty-Based Classification System
Sistem klasifikasi dengan probabilitas dan ketidakpastian untuk forensik gambar
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class UncertaintyClassifier:
    """
    Klasifikasi dengan model ketidakpastian yang mempertimbangkan:
    1. Confidence intervals
    2. Probabilitas manipulasi
    3. Indikator keraguan
    4. Multiple evidence weighting
    """
    
    def __init__(self):
        # Base uncertainty represents inherent ambiguity in forensic image analysis, capped
        self.base_uncertainty = 0.05  # Lowered to reduce overall uncertainty
        # PERBAIKAN: Sesuaikan threshold untuk distribusi yang lebih realistis
        self.confidence_thresholds = {
            'high': 0.60,      # Tinggi >= 60% (diturunkan dari 70%)
            'medium': 0.30,    # Sedang 30-60% (diturunkan dari 40%)
            'low': 0.0         # Rendah < 30%
        }
        
    def calculate_manipulation_probability(self, analysis_results: Dict) -> Dict:
        """
        Hitung probabilitas manipulasi dengan mempertimbangkan ketidakpastian
        """
        # Extract scores dari hasil analisis
        copy_move_indicators = self._extract_copy_move_indicators(analysis_results)
        splicing_indicators = self._extract_splicing_indicators(analysis_results)
        authenticity_indicators = self._extract_authenticity_indicators(analysis_results)
        
        # Hitung probabilitas dasar (skor gabungan, belum dinormalisasi atau diatur untuk konflik)
        copy_move_raw_prob = self._calculate_weighted_probability(copy_move_indicators)
        splicing_raw_prob = self._calculate_weighted_probability(splicing_indicators)
        authentic_raw_prob = self._calculate_weighted_probability(authenticity_indicators)
        
        # Apply mutual exclusivity logic if conflicting high scores
        # If strong evidence for both CM and Splicing, treat as "Complex" but reduce
        # the certainty that one is *only* CM or Splicing.
        if copy_move_raw_prob > 0.6 and splicing_raw_prob > 0.6:
            # Adjust to represent complex manipulation
            # Reduce individual pure probabilities but keep high total manipulation
            temp_cm_raw = copy_move_raw_prob * 0.7
            temp_sp_raw = splicing_raw_prob * 0.7
            temp_au_raw = authentic_raw_prob * 0.3 # Less likely to be authentic if both high
            copy_move_raw_prob, splicing_raw_prob, authentic_raw_prob = temp_cm_raw, temp_sp_raw, temp_au_raw
            print("  [Uncertainty] Detected high conflicting signals, adjusting raw probabilities for 'Complex'.")

        # Now, normalize the probabilities so they sum to 1.
        # This acts as the softmax layer in a neural network.
        # PERBAIKAN: Kurangi faktor eksponensial dari 3 ke 1.5 untuk mengurangi bias
        exp_cm = np.exp(copy_move_raw_prob * 1.5) # Reduced from 3 to 1.5 for more balanced results
        exp_sp = np.exp(splicing_raw_prob * 1.5)
        exp_au = np.exp(authentic_raw_prob * 1.5)

        # Normalize to get final probabilities (using exponential for soft decision boundary)
        sum_exp = exp_cm + exp_sp + exp_au
        
        if sum_exp > 0:
            copy_move_prob_final = exp_cm / sum_exp
            splicing_prob_final = exp_sp / sum_exp
            authentic_prob_final = exp_au / sum_exp
        else: # Fallback for no data
            copy_move_prob_final = splicing_prob_final = authentic_prob_final = 1/3
            
        # Recalculate uncertainty factors based on ALL evidence, not just the "raw" scores
        # This will include how sparse or ambiguous the *actual input evidence* was.
        uncertainty_factors = self._calculate_uncertainty_factors(
            analysis_results, copy_move_indicators, splicing_indicators, authenticity_indicators
        )
        
        # Apply uncertainty *after* normalization to show confidence around the final probability
        # The probability is what the model believes; uncertainty is how sure it is of that belief.
        
        return {
            'copy_move_probability': copy_move_prob_final,
            'splicing_probability': splicing_prob_final,
            'authentic_probability': authentic_prob_final,
            'uncertainty_level': self._calculate_overall_uncertainty(uncertainty_factors),
            'confidence_intervals': self._calculate_confidence_intervals(
                copy_move_prob_final, splicing_prob_final, authentic_prob_final, uncertainty_factors
            )
        }
    
    def _extract_copy_move_indicators(self, results: Dict) -> List[Tuple[float, float]]:
        """Extract copy-move indicators dengan weights"""
        indicators = [] # Format: (score, weight)
        
        # RANSAC inliers (high weight) - higher inliers means more confident CM
        ransac_inliers = results.get('ransac_inliers', 0)
        # Using a non-linear scaling (log/sqrt) or threshold-based score for features for better discrimination
        if ransac_inliers >= 5: 
            score = min(np.sqrt(ransac_inliers) / np.sqrt(50), 1.0) # Sqrt scale for large range of inliers
            indicators.append((score, 0.30))  # High weight, as RANSAC is strong evidence
        
        # Block matches
        block_matches_count = len(results.get('block_matches', []))
        if block_matches_count >= 3:
            score = min(block_matches_count / 30.0, 1.0) # Direct ratio up to 30 matches
            indicators.append((score, 0.25))
        
        # Geometric transform existence (strong, but 0/1 indicator)
        if results.get('geometric_transform') is not None:
            indicators.append((1.0, 0.20)) # High weight for transform
        
        # SIFT raw matches (supportive)
        sift_matches_raw = results.get('sift_matches', 0)
        if sift_matches_raw > 20:
            score = min(sift_matches_raw / 200.0, 1.0)
            indicators.append((score, 0.10))
        
        # ELA regional inconsistency (indirectly, if low means consistency implying same source for CM)
        ela_regional_inconsistency = results.get('ela_regional_stats', {}).get('regional_inconsistency', 1.0)
        if ela_regional_inconsistency < 0.25: # Low inconsistency implies possible copy-move
            indicators.append((1.0 - ela_regional_inconsistency / 0.5, 0.10)) # Inverted score for consistency

        # Tampering localization percentage (medium-high percentage can indicate CM areas)
        tampering_pct = results.get('localization_analysis', {}).get('tampering_percentage', 0.0)
        if 5.0 < tampering_pct < 60.0: # Range likely for tampering (not too small, not too large)
             score = min(tampering_pct / 50.0, 1.0)
             indicators.append((score, 0.05))

        return indicators
    
    def _extract_splicing_indicators(self, results: Dict) -> List[Tuple[float, float]]:
        """Extract splicing indicators dengan weights"""
        indicators = []
        
        # ELA analysis (main indicator for splicing due to compression difference)
        ela_mean = results.get('ela_mean', 0)
        ela_std = results.get('ela_std', 0)
        # Higher ELA mean/std indicates compression inconsistencies likely from splicing
        if ela_mean > 10 or ela_std > 18:
            score = min(max(ela_mean / 25.0, ela_std / 35.0), 1.0)
            indicators.append((score, 0.25))
        
        # Noise inconsistency (non-uniform noise patterns often due to splicing)
        noise_inconsistency = results.get('noise_analysis', {}).get('overall_inconsistency', 0)
        if noise_inconsistency > 0.2:
            score = min(noise_inconsistency / 0.6, 1.0)
            indicators.append((score, 0.20))
        
        # JPEG Ghost analysis (direct evidence of double compression or pasting)
        jpeg_ghost_suspicious_ratio = results.get('jpeg_ghost_suspicious_ratio', 0)
        if jpeg_ghost_suspicious_ratio > 0.05: # Even small ratio can be indicative
            score = min(jpeg_ghost_suspicious_ratio / 0.3, 1.0)
            indicators.append((score, 0.20))
        
        # Frequency inconsistency (spectral artifacts from pasting)
        freq_inconsistency = results.get('frequency_analysis', {}).get('frequency_inconsistency', 0)
        if freq_inconsistency > 0.8: # Threshold could be lower, more common artifact
            score = min(freq_inconsistency / 2.0, 1.0)
            indicators.append((score, 0.10))
        
        # Illumination inconsistency (very strong sign of splicing)
        illum_inconsistency = results.get('illumination_analysis', {}).get('overall_illumination_inconsistency', 0)
        if illum_inconsistency > 0.2: # Significant illumination change
            score = min(illum_inconsistency / 0.5, 1.0)
            indicators.append((score, 0.15)) # High weight
        
        # Edge inconsistency (blurriness or sharp edges mismatches)
        edge_inconsistency = results.get('edge_analysis', {}).get('edge_inconsistency', 0)
        if edge_inconsistency > 0.2:
            score = min(edge_inconsistency / 0.5, 1.0)
            indicators.append((score, 0.05))

        # Metadata issues (soft indicator, often tampered in conjunction with image manipulation)
        metadata_inconsistencies = len(results.get('metadata', {}).get('Metadata_Inconsistency', []))
        if metadata_inconsistencies > 0:
            score = min(metadata_inconsistencies / 5.0, 1.0) # Up to 5 issues is max score
            indicators.append((score, 0.05))

        # Statistical anomalies (entropy/correlation changes)
        stat_analysis = results.get('statistical_analysis', {})
        # If any correlation is far from 1 (or -1), might be suspicious
        rg_corr = stat_analysis.get('rg_correlation', 1.0)
        rb_corr = stat_analysis.get('rb_correlation', 1.0)
        gb_corr = stat_analysis.get('gb_correlation', 1.0)
        # Anomaly if absolute correlation is low (0.0 to 0.5 typically for manipulated areas)
        if abs(rg_corr) < 0.7 or abs(rb_corr) < 0.7 or abs(gb_corr) < 0.7:
            # Score higher for lower correlations
            score = max(max(0, 1.0 - abs(rg_corr)), max(0, 1.0 - abs(rb_corr)), max(0, 1.0 - abs(gb_corr)))
            indicators.append((score, 0.05))

        return indicators
    
    def _extract_authenticity_indicators(self, results: Dict) -> List[Tuple[float, float]]:
        """
        Extract authenticity indicators. These are inverse to manipulation indicators.
        Absence of manipulation evidence or positive integrity checks.
        """
        indicators = [] # (score, weight)
        
        # PENINGKATAN #1: Beri bobot sangat tinggi jika TIDAK ADA bukti manipulasi sama sekali
        ransac_inliers = results.get('ransac_inliers', 0)
        block_matches_count = len(results.get('block_matches', []))
        jpeg_ghost_suspicious_ratio = results.get('jpeg_ghost_suspicious_ratio', 0)
        noise_inconsistency = results.get('noise_analysis', {}).get('overall_inconsistency', 0)
        
        # Jika tidak ada satupun bukti copy-move, ini adalah sinyal keaslian yang kuat
        if ransac_inliers == 0 and block_matches_count == 0:
            indicators.append((1.0, 0.35)) # Sedikit dikurangi untuk keseimbangan
        
        # PENINGKATAN #2: Tambahkan kondisi untuk noise dan ELA yang sangat rendah
        ela_mean = results.get('ela_mean', 0)
        if noise_inconsistency < 0.15 and ela_mean < 5:
             indicators.append((1.0 - noise_inconsistency, 0.25)) # Sinyal kuat lainnya
        
        # PERBAIKAN: Turunkan threshold metadata score untuk lebih inklusif
        metadata_auth_score = results.get('metadata', {}).get('Metadata_Authenticity_Score', 0)
        if metadata_auth_score > 60: # Diturunkan dari 80 ke 60
            indicators.append((metadata_auth_score / 100.0, 0.25)) # Bobot ditingkatkan
        elif metadata_auth_score > 40: # Tambahan untuk skor sedang
            indicators.append((metadata_auth_score / 100.0, 0.15)) # Bobot sedang untuk skor sedang
        
        # Low ELA mean/std (no compression artifacts that indicate tampering)
        ela_std = results.get('ela_std', 0)
        if ela_mean < 10 and ela_std < 18: # Diperluas threshold untuk lebih inklusif
            indicators.append((1.0 - (ela_mean / 12.0), 0.20)) # Bobot ditingkatkan
        
        # Low JPEG Ghost ratio
        if jpeg_ghost_suspicious_ratio < 0.05: # Diperluas threshold
            indicators.append((1.0 - jpeg_ghost_suspicious_ratio / 0.1, 0.15)) # Bobot ditingkatkan

        # Low Tampering percentage (localization shows clean image)
        tampering_pct = results.get('localization_analysis', {}).get('tampering_percentage', 0.0)
        if tampering_pct < 5.0: # Diperluas dari 2.0 ke 5.0
            indicators.append((1.0 - tampering_pct / 8.0, 0.15)) # Bobot ditingkatkan
        
        # TAMBAHAN: Indikator autentisitas baru
        # Noise consistency yang baik
        if noise_inconsistency < 0.20: # Noise yang konsisten
            indicators.append((1.0 - noise_inconsistency / 0.25, 0.10))
        
        # Illumination consistency
        illum_inconsistency = results.get('illumination_analysis', {}).get('overall_illumination_inconsistency', 0)
        if illum_inconsistency < 0.15: # Pencahayaan yang konsisten
            indicators.append((1.0 - illum_inconsistency / 0.20, 0.10))
        
        # Frequency domain consistency
        freq_inconsistency = results.get('frequency_analysis', {}).get('frequency_inconsistency', 0)
        if freq_inconsistency < 0.5: # Frekuensi yang konsisten
            indicators.append((1.0 - freq_inconsistency / 0.8, 0.08))
        
        # Edge consistency
        edge_inconsistency = results.get('edge_analysis', {}).get('edge_inconsistency', 0)
        if edge_inconsistency < 0.15: # Edge yang konsisten
            indicators.append((1.0 - edge_inconsistency / 0.20, 0.08))

        return indicators
    
    def _calculate_weighted_probability(self, indicators: List[Tuple[float, float]]) -> float:
        """Hitung probabilitas dengan weighted average"""
        if not indicators:
            return 0.1 # Small baseline for empty indicators, allows other categories to dominate
        
        total_weight = sum(weight for _, weight in indicators)
        if total_weight == 0:
            return 0.1 # Avoid division by zero
        
        weighted_sum = sum(score * weight for score, weight in indicators)
        # Scale sum based on maximum possible weighted sum for the given weights
        # max_possible_sum_if_all_scores_1 = total_weight
        # To get a value that ranges from ~0 to ~1 (or max possible for sum of weights)
        # Using sigmoid to squash the sum if raw weighted sum gets very high
        prob_sum = weighted_sum / total_weight # Max score could be 1.0 (if all scores are 1)
        return float(prob_sum)
    
    def _calculate_uncertainty_factors(self, results: Dict, cm_indicators: List, sp_indicators: List, au_indicators: List) -> Dict[str, float]:
        """
        Hitung faktor ketidakpastian untuk setiap kategori.
        Ketidakpastian meningkat jika:
        1. Bukti ambigu (misalnya, ELA di zona abu-abu)
        2. Bukti jarang (sedikit indikator ditemukan)
        3. Konteks gambar membuat deteksi sulit (e.g., noise tinggi alami, terlalu mulus, kecil)
        """
        factors = {}
        
        # General image context for overall uncertainty contribution
        overall_uncertainty_context = 0.0
        
        # If image is very small, more uncertainty
        dimensions = results.get('metadata', {}).get('Dimensions', (0,0))
        img_width = 0
        if isinstance(dimensions, str) and 'x' in dimensions: # Handles "WxH" string
            try:
                img_width = int(dimensions.split('x')[0])
            except ValueError:
                img_width = 0 # Failsafe
        elif isinstance(dimensions, (tuple, list)) and len(dimensions) > 0:
             img_width = int(dimensions[0])
        
        if img_width < 300: # Small image means more uncertainty for all
            overall_uncertainty_context += 0.08
        
        # If too simple (e.g., solid color) or too complex (e.g., highly textured natural images), affects some
        ela_mean = results.get('ela_mean', 0)
        ela_std = results.get('ela_std', 0)
        if ela_mean < 3 and ela_std < 5: # Very uniform/smooth image
            overall_uncertainty_context += 0.05
        elif ela_mean > 20 or ela_std > 30: # Extremely noisy/highly compressed
             overall_uncertainty_context += 0.03 # Moderate noise can confuse, but extreme noise makes everything random, hard to judge.

        # --- Copy-move uncertainty ---
        cm_uncertainty = self.base_uncertainty + overall_uncertainty_context
        # Reduced uncertainty additions for more reasonable values
        if 0 < results.get('ransac_inliers', 0) < 10: cm_uncertainty += 0.03
        if 0 < len(results.get('block_matches', [])) < 5: cm_uncertainty += 0.03
        
        # If number of CM indicators is low, it makes it less certain.
        if len(cm_indicators) < 2: cm_uncertainty += 0.05 # Not enough evidence
        factors['copy_move'] = min(cm_uncertainty, 0.25) # Cap at 25%

        # --- Splicing uncertainty ---
        sp_uncertainty = self.base_uncertainty + overall_uncertainty_context
        # Reduced uncertainty additions
        if 5 < ela_mean < 12 or 10 < ela_std < 20: sp_uncertainty += 0.04 # Ambiguous ELA zone
        noise_inconsistency = results.get('noise_analysis', {}).get('overall_inconsistency', 0)
        if 0.15 < noise_inconsistency < 0.3: sp_uncertainty += 0.03 # Ambiguous noise
        
        if len(sp_indicators) < 2: sp_uncertainty += 0.05 # Not enough evidence
        factors['splicing'] = min(sp_uncertainty, 0.30) # Cap at 30%

        # --- Authentic uncertainty ---
        au_uncertainty = self.base_uncertainty + overall_uncertainty_context
        metadata_score = results.get('metadata', {}).get('Metadata_Authenticity_Score', 0)
        if 40 < metadata_score < 70: au_uncertainty += 0.04 # Ambiguous metadata
        
        # If the authentic indicators are very few (could just be lucky)
        if len(au_indicators) < 2: au_uncertainty += 0.05
        factors['authentic'] = min(au_uncertainty, 0.20) # Lower cap for authentic

        return factors

    def _calculate_overall_uncertainty(self, uncertainty_factors: Dict[str, float]) -> float:
        """Calculate overall uncertainty level with better weighting."""
        if not uncertainty_factors: return 1.0
        # Use weighted average instead of max for more balanced results
        avg_uncertainty = np.mean(list(uncertainty_factors.values()))
        # Scale down to reduce overall uncertainty
        return float(np.clip(avg_uncertainty * 0.8, 0.0, 0.35)) # Cap at 35% max
    
    def _calculate_confidence_intervals(self, copy_move_prob: float, splicing_prob: float, 
                                      authentic_prob: float, uncertainty_factors: Dict) -> Dict:
        """Calculate confidence intervals for each probability based on their adjusted uncertainty factors."""
        intervals = {}
        
        # A heuristic based on uncertainty for the confidence interval.
        # True interval would require more complex statistical modeling (e.g., beta distribution).
        
        cm_uncertainty = uncertainty_factors['copy_move']
        intervals['copy_move'] = {
            'lower': max(0.0, copy_move_prob - cm_uncertainty * 0.8), # Make intervals a bit tighter
            'upper': min(1.0, copy_move_prob + cm_uncertainty * 0.8)
        }
        
        sp_uncertainty = uncertainty_factors['splicing']
        intervals['splicing'] = {
            'lower': max(0.0, splicing_prob - sp_uncertainty * 0.8),
            'upper': min(1.0, splicing_prob + sp_uncertainty * 0.8)
        }
        
        au_uncertainty = uncertainty_factors['authentic']
        intervals['authentic'] = {
            'lower': max(0.0, authentic_prob - au_uncertainty * 0.8),
            'upper': min(1.0, authentic_prob + au_uncertainty * 0.8)
        }
        
        return intervals
    
    def generate_uncertainty_report(self, probabilities: Dict) -> Dict:
        """Generate detailed uncertainty report dengan terminologi yang diperbarui."""
        
        # Pastikan semua nilai adalah float numerik untuk menghindari error
        for key in ['copy_move_probability', 'splicing_probability', 'authentic_probability', 'uncertainty_level']:
            probabilities[key] = float(probabilities.get(key, 0.0))

        report = {
            'primary_assessment': self._determine_primary_assessment(probabilities),
            'assessment_reliability': self._determine_assessment_reliability(probabilities),
            'indicator_coherence': self._assess_indicator_coherence(probabilities['uncertainty_level']),
            'reliability_indicators': self._generate_reliability_indicators(probabilities),
            'recommendation': self._generate_recommendation(probabilities)
        }
        return report
    
    def _determine_primary_assessment(self, probabilities: Dict) -> str:
        """Determine primary assessment with uncertainty language"""
        copy_move_prob = probabilities['copy_move_probability']
        splicing_prob = probabilities['splicing_probability']
        authentic_prob = probabilities['authentic_probability']
        
        max_prob_value = max(copy_move_prob, splicing_prob, authentic_prob)
        
        # PERBAIKAN: Turunkan threshold untuk hasil ambigu agar lebih realistis
        if max_prob_value < 0.35: # Diturunkan dari 0.4 ke 0.35
            return "Indikasi: Hasil Ambigu (membutuhkan pemeriksaan lebih lanjut)"
            
        # Tentukan jenis manipulasi atau keaslian
        is_authentic = (authentic_prob == max_prob_value) and (authentic_prob > copy_move_prob * 1.1 and authentic_prob > splicing_prob * 1.1)
        is_copy_move = (copy_move_prob == max_prob_value) and (copy_move_prob > authentic_prob * 1.1 and copy_move_prob > splicing_prob * 1.1)
        is_splicing = (splicing_prob == max_prob_value) and (splicing_prob > authentic_prob * 1.1 and splicing_prob > copy_move_prob * 1.1)

        if is_authentic:
            return "Indikasi: Gambar Asli/Autentik"
        elif is_copy_move:
            return "Indikasi: Manipulasi Copy-Move Terdeteksi"
        elif is_splicing:
            return "Indikasi: Manipulasi Splicing Terdeteksi"
        
        # Handle kasus kompleks (probabilitas tinggi untuk CM dan Splicing)
        if copy_move_prob > 0.35 and splicing_prob > 0.35:
            return "Indikasi: Manipulasi Kompleks Terdeteksi (Copy-Move & Splicing)"
        
        # Fallback untuk kasus yang tidak jelas
        if copy_move_prob > splicing_prob and copy_move_prob > authentic_prob:
             return "Indikasi: Kecenderungan Manipulasi Copy-Move"
        if splicing_prob > copy_move_prob and splicing_prob > authentic_prob:
             return "Indikasi: Kecenderungan Manipulasi Splicing"
             
        return "Indikasi: Tidak Terdeteksi Manipulasi Signifikan"

    def _determine_assessment_reliability(self, probabilities: Dict) -> str:
        """Menentukan tingkat reliabilitas asesmen utama dengan 3 kategori."""
        copy_move_prob = probabilities['copy_move_probability']
        splicing_prob = probabilities['splicing_probability']
        authentic_prob = probabilities['authentic_probability']
        uncertainty = probabilities['uncertainty_level']
        
        main_prob_value = max(copy_move_prob, splicing_prob, authentic_prob)

        # PERBAIKAN: Kurangi penalti uncertainty untuk hasil yang lebih realistis
        # Reliabilitas adalah kombinasi dari probabilitas tertinggi dan tingkat ketidakpastian
        reliability_score = main_prob_value * (1.0 - uncertainty * 0.5) # Dikurangi dari 0.7 ke 0.5

        # Only 3 categories: Rendah, Sedang, Tinggi
        if reliability_score >= self.confidence_thresholds['high']:
            return "Tinggi"
        elif reliability_score >= self.confidence_thresholds['medium']:
            return "Sedang"
        else:
            return "Rendah"
    
    def _assess_indicator_coherence(self, uncertainty_level: float) -> str:
        """Menilai koherensi (konsistensi) dari semua indikator forensik dengan 3 kategori."""
        # Only 3 categories with more reasonable thresholds
        if uncertainty_level < 0.15:
            return "Tinggi: Indikator forensik menunjukkan koherensi yang kuat. Berbagai metode deteksi memberikan hasil yang konsisten dan saling mendukung, menghasilkan analisis yang dapat diandalkan."
        elif uncertainty_level < 0.25:
            return "Sedang: Sebagian besar indikator menunjukkan koherensi yang cukup baik. Terdapat beberapa perbedaan minor antar metode deteksi namun secara umum hasil analisis dapat dipercaya."
        else:
            return "Rendah: Ditemukan perbedaan yang signifikan antar berbagai metode deteksi. Diperlukan kehati-hatian dalam interpretasi hasil dan disarankan untuk melakukan validasi tambahan."
    
    def _generate_reliability_indicators(self, probabilities: Dict) -> List[str]:
        """Generate reliability indicators."""
        indicators = []
        
        probs = [
            probabilities['copy_move_probability'],
            probabilities['splicing_probability'],
            probabilities['authentic_probability']
        ]
        
        prob_std = np.std(probs)
        
        if prob_std < 0.15:
            indicators.append("⚠️ Probabilitas antar kategori cukup dekat, yang dapat mengindikasikan hasil yang ambigu.")
        elif prob_std > 0.35:
            indicators.append("✅ Probabilitas menunjukkan perbedaan yang jelas antar kategori, memperkuat spesifisitas prediksi.")
        else:
            indicators.append("🔎 Probabilitas memiliki pemisahan sedang, menunjukkan kecenderungan yang jelas namun tidak mutlak.")
        
        intervals = probabilities['confidence_intervals']
        for category, interval in intervals.items():
            range_size = interval['upper'] - interval['lower']
            if range_size > 0.4:
                indicators.append(f"⚠️ Interval kepercayaan untuk '{category.replace('_',' ').title()}' lebar ({range_size:.1%}), menandakan prediksi pada kategori ini kurang stabil.")
        
        if probabilities['uncertainty_level'] < 0.1:
            indicators.append("✅ Tingkat ketidakpastian keseluruhan sangat rendah, menunjukkan hasil analisis yang sangat stabil.")
        elif probabilities['uncertainty_level'] > 0.35:
            indicators.append("❌ Tingkat ketidakpastian keseluruhan tinggi, hasil harus dianggap sebagai indikasi awal yang memerlukan validasi lebih lanjut.")
        
        return indicators
    
    def _generate_recommendation(self, probabilities: Dict) -> str:
        """Generate recommendation based on analysis and uncertainty."""
        uncertainty = probabilities['uncertainty_level']
        
        main_prob_value = max(probabilities['copy_move_probability'], probabilities['splicing_probability'], probabilities['authentic_probability'])
        
        reliability_score = main_prob_value * (1.0 - uncertainty)

        # Adjusted for 3 categories with more reasonable thresholds
        if reliability_score >= 0.65 and uncertainty < 0.20:
            return ("Hasil analisis menunjukkan indikasi yang kuat dengan tingkat kepercayaan tinggi. Berbagai metode forensik memberikan hasil yang konsisten. Keputusan akhir tetap memerlukan pertimbangan manusia dengan memperhatikan konteks dan bukti lainnya.")
        elif reliability_score >= 0.40:
            return ("Hasil analisis memberikan indikasi yang cukup jelas namun dengan beberapa ketidakpastian. Disarankan untuk melakukan pemeriksaan lebih lanjut pada area-area yang ditandai dan mempertimbangkan faktor-faktor eksternal dalam pengambilan keputusan.")
        else:
            return ("Hasil analisis menunjukkan indikasi awal yang perlu dikonfirmasi lebih lanjut. Terdapat ketidakpastian yang cukup signifikan dalam deteksi. Sangat disarankan untuk melakukan validasi tambahan dan tidak mengandalkan hasil ini sebagai bukti tunggal.")

def format_probability_results(probabilities: Dict, details: Dict) -> str:
    """Format probability results for display"""
    output = []
    
    output.append("="*60)
    output.append("LAPORAN ANALISIS FORENSIK PROBABILISTIK")
    output.append("="*60)
    output.append("")
    
    output.append(f"ASESMEN UTAMA: {details['primary_assessment']}")
    output.append(f"RELIABILITAS PENILAIAN: {details['assessment_reliability']}")
    output.append("")
    
    output.append("DISTRIBUSI PROBABILITAS:")
    output.append(f"  - Asli/Autentik: {probabilities['authentic_probability']:.1%} (Interval: [{probabilities['confidence_intervals']['authentic']['lower']:.1%} - {probabilities['confidence_intervals']['authentic']['upper']:.1%}])")
    output.append(f"  - Copy-Move: {probabilities['copy_move_probability']:.1%} (Interval: [{probabilities['confidence_intervals']['copy_move']['lower']:.1%} - {probabilities['confidence_intervals']['copy_move']['upper']:.1%}])")
    output.append(f"  - Splicing: {probabilities['splicing_probability']:.1%} (Interval: [{probabilities['confidence_intervals']['splicing']['lower']:.1%} - {probabilities['confidence_intervals']['splicing']['upper']:.1%}])")
    output.append("")
    
    output.append("ANALISIS STABILITAS & KETIDAKPASTIAN:")
    output.append(f"  - Tingkat Ketidakpastian: {probabilities['uncertainty_level']:.1%}")
    output.append(f"  - Koherensi Indikator: {details['indicator_coherence']}")
    output.append("")
    
    output.append("INDIKATOR KEANDALAN TEKNIS:")
    if details['reliability_indicators']:
        for indicator in details['reliability_indicators']:
            output.append(f"  {indicator}")
    else:
        output.append("  Tidak ada indikator spesifik yang perlu disorot.")
    output.append("")
    
    output.append("REKOMENDASI:")
    output.append(details['recommendation'])
    output.append("")
    output.append("="*60)
    
    return "\n".join(output)

# --- END OF FILE uncertainty_classification.py ---