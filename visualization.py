# --- START OF FILE visualization.py ---

"""
Visualization Module for Forensic Image Analysis System
Contains functions for creating comprehensive visualizations, plots, and visual reports
"""

import numpy as np
import cv2
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    gridspec = None # Ensure gridspec is also None when matplotlib fails
    class PdfPages: # Dummy class
        def __init__(self, *a, **k):
            raise RuntimeError('matplotlib not available')
from PIL import Image
from datetime import datetime
try:
    from skimage.filters import sobel
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False
    def sobel(x): # Fallback sobel that returns black image
        return np.zeros_like(x, dtype=float)
import os
import io
import warnings
try:
    from sklearn.metrics import confusion_matrix, accuracy_score
    import seaborn as sns
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
    SKLEARN_METRICS_AVAILABLE = True
except Exception:
    SKLEARN_METRICS_AVAILABLE = False
    SCIPY_AVAILABLE = False # Also for gaussian_kde which needs scipy.stats

warnings.filterwarnings('ignore')

# ======================= FORENSIC DISCLAIMER & METHODOLOGY =======================
"""
PRINSIP FORENSIK DIGITAL - SISTEM ANALISIS GAMBAR

DISCLAIMER UTAMA:
Sistem ini dirancang untuk mendeteksi kejanggalan (anomalies) dalam gambar digital berdasarkan 
prinsip-prinsip analisis forensik. Sistem TIDAK dapat:
1. Menentukan dengan pasti apakah gambar "asli" atau "palsu"
2. Memberikan skor definitif tentang keaslian gambar
3. Menggantikan analisis manual oleh ahli forensik digital

YANG DILAKUKAN SISTEM:
- Mendeteksi pola statistik yang tidak biasa
- Mengidentifikasi artefak kompresi yang mencurigakan
- Menganalisis konsistensi metadata dan struktur file
- Menghitung kekuatan evidence forensik berdasarkan temuan teknis

PRINSIP ANALISIS:
1. Evidence-based: Setiap temuan didasarkan pada evidence teknis yang dapat diverifikasi
2. Uncertainty-aware: Semua hasil dilengkapi dengan tingkat ketidakpastian
3. Non-conclusive: Tidak ada kesimpulan definitif, hanya indikasi kekuatan evidence
4. Expert-verification: Diperlukan verifikasi manual oleh ahli forensik

METODOLOGI:
- Error Level Analysis (ELA) untuk deteksi kompresi tidak merata
- Copy-Move Detection untuk identifikasi duplikasi area
- JPEG Ghost Analysis untuk deteksi kompresi berganda
- Metadata Analysis untuk inkonsistensi informasi file
- Statistical Analysis untuk anomali distribusi pixel

TINGKAT KEPERCAYAAN:
- Tinggi: Evidence kuat mendukung temuan
- Sedang: Evidence cukup, perlu verifikasi tambahan
- Rendah: Evidence lemah, kemungkinan besar normal
"""

# ======================= Main Visualization Function =======================

def visualize_results_advanced(original_pil, analysis_results, output_filename="advanced_forensic_analysis.png"):
    """Advanced visualization with comprehensive forensic analysis results"""
    if not MATPLOTLIB_AVAILABLE:
        print("❌ Matplotlib not available. Cannot generate visualization.")
        return None
    print("📊 Creating advanced forensic visualization...")
    
    fig = plt.figure(figsize=(32, 28))  # Increased from 24x20 to 32x28
    gs = fig.add_gridspec(4, 4, hspace=1.2, wspace=0.6)  # Increased spacing: hspace 0.7→1.2, wspace 0.3→0.6
    
    fig.suptitle(
        f"Analisis Forensik Digital - Deteksi Kejanggalan Gambar\nFile: {analysis_results['metadata'].get('Filename', 'N/A')} | Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=20, fontweight='bold', y=0.98
    )
    
    # Add forensic disclaimer as footer
    fig.text(0.5, 0.01, 
             "DISCLAIMER: Sistem ini mendeteksi kejanggalan berdasarkan evidence forensik. "
             "Tidak dapat menentukan dengan pasti apakah gambar asli atau dimanipulasi. "
             "Diperlukan analisis manual oleh ahli forensik untuk kesimpulan definitif.",
             ha='center', va='bottom', fontsize=8, style='italic', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    # Row 1: Core Visuals
    ax1_1 = fig.add_subplot(gs[0, 0])
    ax1_2 = fig.add_subplot(gs[0, 1])
    ax1_3 = fig.add_subplot(gs[0, 2])
    ax1_4 = fig.add_subplot(gs[0, 3])
    create_core_visuals_grid(ax1_1, ax1_2, ax1_3, ax1_4, original_pil, analysis_results)
    
    # Row 2: Advanced Analysis Visuals
    ax2_1 = fig.add_subplot(gs[1, 0])
    ax2_2 = fig.add_subplot(gs[1, 1])
    ax2_3 = fig.add_subplot(gs[1, 2])
    ax2_4 = fig.add_subplot(gs[1, 3])
    create_advanced_analysis_grid(ax2_1, ax2_2, ax2_3, ax2_4, original_pil, analysis_results)
    
    # Row 3: Statistical & Metric Visuals
    ax3_1 = fig.add_subplot(gs[2, 0])
    ax3_2 = fig.add_subplot(gs[2, 1])
    ax3_3 = fig.add_subplot(gs[2, 2])
    ax3_4 = fig.add_subplot(gs[2, 3])
    create_statistical_grid(ax3_1, ax3_2, ax3_3, ax3_4, analysis_results)

    # Row 4: Summary and Uncertainty
    ax_report = fig.add_subplot(gs[3, 0])
    ax_probability_bars = fig.add_subplot(gs[3, 1])
    ax_uncertainty_vis = fig.add_subplot(gs[3, 2])
    ax_validation_summary = fig.add_subplot(gs[3, 3])
    
    create_summary_report(ax_report, analysis_results)
    
    if 'classification' in analysis_results and 'uncertainty_analysis' in analysis_results['classification']:
        create_probability_bars(ax_probability_bars, analysis_results)
        create_uncertainty_visualization(ax_uncertainty_vis, analysis_results)
    else:
        populate_validation_visuals(ax_probability_bars, ax_uncertainty_vis)
    
    ax_validation_summary.axis('off')
    pipeline_status_summary = analysis_results.get('pipeline_status', {})
    total_stages = pipeline_status_summary.get('total_stages', 0)
    completed_stages = pipeline_status_summary.get('completed_stages', 0)
    failed_stages_count = len(pipeline_status_summary.get('failed_stages', []))
    success_rate = (completed_stages / total_stages) * 100 if total_stages > 0 else 0
    
    validation_text = f"**Ringkasan Analisis Forensik**\n\n" \
                      f"Evidence Pipeline: {completed_stages}/{total_stages} proses selesai ({success_rate:.1f}%)\n" \
                      f"Proses yang memerlukan perhatian: {failed_stages_count}"
    ax_validation_summary.text(0.5, 0.5, validation_text, transform=ax_validation_summary.transAxes,
                                ha='center', va='center', fontsize=10, 
                                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    try:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight', pad_inches=0.3)
        print(f"📊 Advanced forensic visualization saved as '{output_filename}'")
        plt.close(fig)
        return output_filename
    except Exception as e:
        print(f"❌ Error saving visualization: {e}")
        import traceback
        traceback.print_exc()
        plt.close(fig)
        return None

# ======================= Grid Helper Functions =======================

def create_metadata_table(ax, metadata):
    """Create detailed metadata table visualization for forensic analysis"""
    ax.clear()
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.98, '1. Analisis Metadata Forensik', ha='center', va='top',
            fontsize=9, fontweight='bold', transform=ax.transAxes)
    
    # Group metadata into categories
    basic_info = {}
    time_info = {}
    camera_info = {}
    image_info = {}
    forensic_info = {}
    other_info = {}
    
    # Categorize metadata
    for key, value in metadata.items():
        if key in ['Filename', 'FileSize (bytes)', 'LastModified']:
            basic_info[key] = value
        elif 'DateTime' in key or 'Date' in key:
            time_info[key] = value
        elif key in ['Image Make', 'Image Model', 'EXIF LensModel', 'EXIF FocalLength', 'EXIF ISO', 'EXIF ExposureTime', 'EXIF Flash', 'EXIF WhiteBalance']:
            camera_info[key] = value
        elif key in ['Image ImageWidth', 'Image ImageLength', 'EXIF ColorSpace', 'Image Orientation', 'EXIF ExifVersion']:
            image_info[key] = value
        elif key in ['Metadata_Authenticity_Score', 'Metadata_Inconsistency', 'Image Software']:
            forensic_info[key] = value
        else:
            other_info[key] = value
    
    # Format values for display
    def format_value(val):
        if isinstance(val, list):
            return ', '.join(str(v) for v in val[:2]) + ('...' if len(val) > 2 else '')
        return str(val)[:50] + ('...' if len(str(val)) > 50 else '')
    
    y_pos = 0.90
    line_height = 0.03
    section_gap = 0.04
    
    # Display sections (limit to most important ones to prevent crowding)
    sections = [
        ('Informasi Dasar', basic_info),
        ('Informasi Kamera', camera_info),
        ('Informasi Gambar', image_info),
        ('Analisis Forensik', forensic_info)
    ]
    
    for section_title, section_data in sections:
        if section_data and y_pos > 0.15:  # Prevent overflow
            # Section header
            ax.text(0.05, y_pos, f'{section_title}:', fontsize=9, fontweight='bold',
                   transform=ax.transAxes, color='darkblue')
            y_pos -= line_height * 1.2
            
            # Section items (limit to 2-3 items per section)
            items_shown = 0
            for key, value in list(section_data.items())[:2]:  # Limit to 2 items per section
                if y_pos < 0.1:  # Stop if running out of space
                    break
                    
                # Format key
                display_key = key.replace('_', ' ').replace('EXIF ', '').replace('Image ', '')
                
                # Special formatting for certain fields
                if key == 'FileSize (bytes)':
                    size_mb = float(value) / (1024 * 1024)
                    display_value = f"{size_mb:.1f} MB"
                elif key == 'Metadata_Authenticity_Score':
                    score = float(value) if value else 0
                    display_value = f"{score}/100"
                elif key == 'Metadata_Inconsistency':
                    if isinstance(value, list) and len(value) > 0:
                        display_value = f"{len(value)} issues"
                    else:
                        display_value = "OK"
                else:
                    display_value = format_value(value)
                
                # Display key-value pair with adjusted positions
                ax.text(0.08, y_pos, f'{display_key}:', fontsize=8, transform=ax.transAxes)
                ax.text(0.5, y_pos, display_value, fontsize=8, transform=ax.transAxes,
                       color='red' if key == 'Image Software' and value and any(s in str(value).lower() for s in ['photoshop', 'gimp']) else 'black')
                y_pos -= line_height * 0.8
                items_shown += 1
            
            y_pos -= section_gap * 0.8
    
    # Forensic summary box
    if 'Metadata_Authenticity_Score' in metadata:
        score = float(metadata.get('Metadata_Authenticity_Score', 0))
        inconsistencies = metadata.get('Metadata_Inconsistency', [])
        
        if score < 40 or len(inconsistencies) > 2:
            summary_color = 'red'
            summary_text = 'Metadata menunjukkan tanda-tanda manipulasi'
        elif score < 70 or len(inconsistencies) > 0:
            summary_color = 'orange'
            summary_text = 'Metadata memiliki beberapa anomali'
        else:
            summary_color = 'green'
            summary_text = 'Metadata tampak konsisten dan autentik'
        
        ax.text(0.5, 0.05, summary_text, ha='center', va='bottom',
               fontsize=9, fontweight='bold', color=summary_color,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor=summary_color),
               transform=ax.transAxes)

def create_core_visuals_grid(ax1, ax2, ax3, ax4, original_pil, results):
    """Create core visuals grid with detailed metadata table"""
    
    # Create detailed metadata table visualization
    create_metadata_table(ax1, results.get('metadata', {}))

    ela_img_data = results.get('ela_image')
    ela_mean = results.get('ela_mean', 0.0)
    
    ela_display_array = np.zeros(original_pil.size[::-1], dtype=np.uint8)
    if ela_img_data is not None:
        if isinstance(ela_img_data, Image.Image):
            ela_display_array = np.array(ela_img_data.convert('L'))
        elif isinstance(ela_img_data, np.ndarray):
            if np.issubdtype(ela_img_data.dtype, np.floating):
                ela_display_array = (ela_img_data / (ela_img_data.max() + 1e-9) * 255).astype(np.uint8)
            else:
                ela_display_array = ela_img_data
            
    ela_display = ax2.imshow(ela_display_array, cmap='hot')
    ax2.set_title(f"2. ELA (μ={ela_mean:.1f})", fontsize=9)
    ax2.axis('off')
    plt.colorbar(ela_display, ax=ax2, fraction=0.046, pad=0.04)

    create_feature_match_visualization(ax3, original_pil, results)
    create_block_match_visualization(ax4, original_pil, results)

def create_advanced_analysis_grid(ax1, ax2, ax3, ax4, original_pil, results):
    """Create advanced analysis grid (Edge, Illumination, JPEG Ghost, Combined Heatmap)"""
    create_edge_visualization(ax1, original_pil, results)
    create_illumination_visualization(ax2, original_pil, results)
    
    jpeg_ghost_data = results.get('jpeg_ghost')
    jpeg_ghost_display_array = np.zeros(original_pil.size[::-1], dtype=np.uint8)
    if jpeg_ghost_data is not None:
        if jpeg_ghost_data.ndim == 2:
            if np.issubdtype(jpeg_ghost_data.dtype, np.floating) and jpeg_ghost_data.max() > 0:
                jpeg_ghost_display_array = (jpeg_ghost_data / (jpeg_ghost_data.max() + 1e-9) * 255).astype(np.uint8)
            else:
                jpeg_ghost_display_array = jpeg_ghost_data.astype(np.uint8)
        else:
             print("Warning: JPEG ghost data not 2D for visualization.")
             
    ghost_display = ax3.imshow(jpeg_ghost_display_array, cmap='hot')
    ax3.set_title(f"7. JPEG Ghost ({results.get('jpeg_ghost_suspicious_ratio', 0):.1%} area)", fontsize=9)
    ax3.axis('off')
    plt.colorbar(ghost_display, ax=ax3, fraction=0.046, pad=0.04)

    original_pil_array = np.array(original_pil.convert('RGB'))
    combined_heatmap = create_advanced_combined_heatmap(results, original_pil.size)
    ax4.imshow(original_pil_array, alpha=0.4)
    ax4.imshow(combined_heatmap, cmap='hot', alpha=0.6)
    ax4.set_title("8. Peta Kecurigaan Gabungan", fontsize=9)
    ax4.axis('off')

def create_statistical_grid(ax1, ax2, ax3, ax4, results):
    """Create statistical analysis grid (Frequency, Texture, Statistical, JPEG Quality Response)"""
    create_frequency_visualization(ax1, results)
    create_texture_visualization(ax2, results)
    create_statistical_visualization(ax3, results)
    create_quality_response_plot(ax4, results)
    
# ======================= Individual Visualization Functions =======================

def create_feature_match_visualization(ax, original_pil, results):
    img_matches = np.array(original_pil.convert('RGB'))
    keypoints = results.get('sift_keypoints')
    ransac_matches = results.get('ransac_matches')
    MAX_MATCHES_DISPLAY = 50
    
    if keypoints and ransac_matches and len(ransac_matches) > 0:
        display_matches = sorted(ransac_matches, key=lambda x: x.distance)[:MAX_MATCHES_DISPLAY]
        for m in display_matches:
            if m.queryIdx < len(keypoints) and m.trainIdx < len(keypoints):
                pt1 = tuple(map(int, keypoints[m.queryIdx].pt))
                pt2 = tuple(map(int, keypoints[m.trainIdx].pt))
                if pt1 == pt2 or (abs(pt1[0]-pt2[0]) < 2 and abs(pt1[1]-pt2[1]) < 2):
                    continue
                cv2.line(img_matches, pt1, pt2, (50, 255, 50), 1, cv2.LINE_AA)
                cv2.circle(img_matches, pt1, 3, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(img_matches, pt2, 3, (0, 0, 255), -1, cv2.LINE_AA)
            
    ax.imshow(img_matches)
    ax.set_title(f"3. Feature Matches ({results.get('ransac_inliers',0)} inliers)", fontsize=9)
    ax.axis('off')

def create_block_match_visualization(ax, original_pil, results):
    img_blocks = np.array(original_pil.convert('RGB'))
    block_matches = results.get('block_matches', [])
    MAX_BLOCK_MATCHES_DISPLAY = 20
    
    if block_matches:
        display_block_matches = block_matches[:MAX_BLOCK_MATCHES_DISPLAY]
        for i, match in enumerate(display_block_matches):
            # Corregido: usar 'block1_pos' y 'block2_pos' en lugar de 'block1' y 'block2'
            x1, y1 = match['block1_pos']
            x2, y2 = match['block2_pos']
            block_size = 16
            color = (255, 165, 0) if i % 2 == 0 else (0, 165, 255)
            cv2.rectangle(img_blocks, (x1, y1), (x1 + block_size, y1 + block_size), color, 2)
            cv2.rectangle(img_blocks, (x2, y2), (x2 + block_size, y2 + block_size), color, 2)
            center1 = (x1 + block_size // 2, y1 + block_size // 2)
            center2 = (x2 + block_size // 2, y2 + block_size // 2)
            cv2.line(img_blocks, center1, center2, color, 1, cv2.LINE_AA)

    ax.imshow(img_blocks)
    ax.set_title(f"4. Block Matches ({len(block_matches or [])} found)", fontsize=9)
    ax.axis('off')

def create_localization_visualization(ax, original_pil, analysis_results):
    loc_analysis = analysis_results.get('localization_analysis', {})
    mask = loc_analysis.get('combined_tampering_mask') 
    tampering_pct = loc_analysis.get('tampering_percentage', 0)
    original_img_array = np.array(original_pil.convert('RGB'))
    ax.imshow(original_img_array)
    
    if mask is not None and mask.size > 0 and tampering_pct > 0.1:
        mask_resized_uint8 = cv2.resize(mask.astype(np.uint8), (original_img_array.shape[1], original_img_array.shape[0]), interpolation=cv2.INTER_NEAREST)
        red_overlay = np.zeros((*original_img_array.shape[:2], 4), dtype=np.uint8)
        red_overlay[mask_resized_uint8 == 1] = [255, 0, 0, 100]
        ax.imshow(red_overlay)
    
    ax.set_title(f"5. K-Means Localization ({tampering_pct:.1f}%)", fontsize=9)
    ax.axis('off')

# ======================= AWAL FUNGSI YANG DIPERBAIKI =======================
def create_uncertainty_visualization(ax, results):
    """Membuat visualisasi ketidakpastian tanpa menampilkan "Keyakinan"."""
    uncertainty_analysis_data = results.get('classification', {}).get('uncertainty_analysis', {})
    
    if not uncertainty_analysis_data:
        ax.text(0.5, 0.5, 'Uncertainty Data Not Available', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
        ax.set_title("Analisis Ketidakpastian", fontsize=12)
        ax.axis('off')
        return

    report_details = uncertainty_analysis_data.get('report', {})
    
    # Kosongkan axis
    ax.clear()
    ax.set_aspect('auto')
    ax.axis('off')

    # Posisi awal teks
    y_pos = 0.95 
    line_sep_major = 0.14 # Jarak antar item utama
    line_sep_minor = 0.08 # Jarak antar sub-item

    # Judul
    ax.text(0.5, y_pos, "Analisis Ketidakpastian Forensik", ha='center', va='top', 
            fontsize=12, fontweight='bold', transform=ax.transAxes)
    y_pos -= line_sep_major

    # Fungsi helper untuk tata letak rapi
    def add_info_pair(y, label, value):
        ax.text(0.05, y, f"{label}:", fontsize=9.5, fontweight='bold', va='top', transform=ax.transAxes)
        ax.text(0.4, y, value, fontsize=9.5, va='top', transform=ax.transAxes, wrap=True)
        return y - line_sep_major

    # Asesmen Utama
    y_pos = add_info_pair(y_pos, "Asesmen Utama", report_details.get('primary_assessment', 'N/A'))
    
    # Reliabilitas Penilaian
    y_pos = add_info_pair(y_pos, "Reliabilitas Penilaian", report_details.get('assessment_reliability', 'N/A'))
    
    # Tingkat Ketidakpastian
    uncertainty_level_pct = uncertainty_analysis_data.get('probabilities', {}).get('uncertainty_level', 0.0) * 100
    y_pos = add_info_pair(y_pos, "Tingkat Ketidakpastian", f"{uncertainty_level_pct:.1f}%")

    # Koherensi Bukti (diberi lebih banyak ruang)
    ax.text(0.05, y_pos, "Koherensi Bukti:", fontsize=9.5, fontweight='bold', va='top', transform=ax.transAxes)
    coherence_text = report_details.get('indicator_coherence', 'N/A')
    ax.text(0.05, y_pos - 0.05, f"_{coherence_text}_", style='italic',
            fontsize=9, va='top', wrap=True, transform=ax.transAxes)
    y_pos -= line_sep_major + 0.1 # Tambahan ruang karena bisa beberapa baris

    # Indikator Keandalan
    reliability_indicators = report_details.get('reliability_indicators', [])
    if reliability_indicators:
        ax.text(0.05, y_pos, "Indikator Keandalan Teknis:", fontsize=9.5, fontweight='bold', va='top', transform=ax.transAxes)
        y_pos -= line_sep_minor
        
        for indicator in reliability_indicators[:2]: # Tampilkan maks 2 indikator agar tidak penuh
            ax.text(0.1, y_pos, f"• {indicator}", fontsize=9, va='top', wrap=True, transform=ax.transAxes)
            y_pos -= line_sep_minor
# ======================= AKHIR FUNGSI YANG DIPERBAIKI =======================


def create_probability_bars(ax, results):
    """Create evidence strength distribution chart with confidence intervals."""
    probabilities_data = results.get('classification', {}).get('uncertainty_analysis', {}).get('probabilities', {})

    if not probabilities_data:
        ax.text(0.5, 0.5, 'Evidence Data Not Available', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('14. Distribusi Kekuatan Evidence Forensik', fontsize=12)
        ax.axis('off')
        return

    ax.clear()

    categories = ['Kecil Bukti Manipulasi', 'Evidence Copy-Move', 'Evidence Splicing']
    
    authentic_prob = probabilities_data.get('authentic_probability', 0.0)
    copy_move_prob = probabilities_data.get('copy_move_probability', 0.0)
    splicing_prob = probabilities_data.get('splicing_probability', 0.0)
    probabilities = np.array([authentic_prob, copy_move_prob, splicing_prob]) * 100

    intervals = probabilities_data.get('confidence_intervals', {})
    
    authentic_int = intervals.get('authentic', {'lower':authentic_prob, 'upper':authentic_prob})
    cm_int = intervals.get('copy_move', {'lower':copy_move_prob, 'upper':copy_move_prob})
    splicing_int = intervals.get('splicing', {'lower':splicing_prob, 'upper':splicing_prob})
    
    errors_lower = np.array([
        probabilities[0] - (authentic_int['lower'] * 100),
        probabilities[1] - (cm_int['lower'] * 100),
        probabilities[2] - (splicing_int['lower'] * 100)
    ])
    errors_upper = np.array([
        (authentic_int['upper'] * 100) - probabilities[0],
        (cm_int['upper'] * 100) - probabilities[1],
        (splicing_int['upper'] * 100) - probabilities[2]
    ])
    
    errors = np.vstack([errors_lower, errors_upper])
    errors = np.maximum(0, errors)

    colors = ['#28a745', '#dc3545', '#ffc107']
    
    bars = ax.bar(categories, probabilities, color=colors, alpha=0.8, yerr=errors, capsize=7, ecolor='black')
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2,
                f'{yval:.1f}%', va='bottom', ha='center', fontsize=9, fontweight='bold')
    
    ax.set_ylim(0, 110)
    ax.set_ylabel('Kekuatan Evidence (%)', fontsize=10)
    ax.set_title('14. Distribusi Kekuatan Evidence Forensik', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_xticklabels(categories, fontsize=9, rotation=15, ha='right')
    
    overall_uncertainty_level_pct = probabilities_data.get('uncertainty_level', 0.0) * 100
    ax.text(0.5, -0.25,
            f'Tingkat Ketidakpastian Keseluruhan: {overall_uncertainty_level_pct:.1f}%',
            horizontalalignment='center', verticalalignment='top', 
            transform=ax.transAxes, fontsize=8, color='gray')


def create_frequency_visualization(ax, results):
    freq_data = results.get('frequency_analysis', {}).get('dct_stats', {})
    values = [freq_data.get('low_freq_energy', 0), freq_data.get('mid_freq_energy', 0), freq_data.get('high_freq_energy', 0)]
    labels = ['Rendah', 'Menengah', 'Tinggi']
    ax.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
    ax.set_title(f"9. Analisis Frekuensi", fontsize=9)
    ax.set_ylabel('Energi DCT', fontsize=8)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_xlabel('Pita Frekuensi', fontsize=8)


def create_texture_visualization(ax, results):
    texture_data = results.get('texture_analysis', {}).get('texture_consistency', {})
    if not texture_data or all(v == 0.0 or np.isnan(v) or np.isinf(v) for v in texture_data.values()):
        ax.text(0.5, 0.5, 'Texture Data Not Available', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(f"10. Konsistensi Tekstur", fontsize=12)
        ax.axis('off')
        return

    metrics = []
    values = []
    ordered_metrics = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'lbp_uniformity']
    for m in ordered_metrics:
        key = f'{m}_consistency'
        if key in texture_data and np.isfinite(texture_data[key]):
            metrics.append(m.capitalize().replace('_', ' '))
            values.append(texture_data[key])
    
    if not values:
        ax.text(0.5, 0.5, 'Texture Data Not Available', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(f"10. Konsistensi Tekstur", fontsize=12)
        ax.axis('off')
        return

    ax.barh(metrics, values, color='#8c564b', alpha=0.8)
    ax.set_title(f"10. Konsistensi Tekstur", fontsize=9)
    ax.set_xlabel('Skor Inkonsistensi (↑ lebih buruk)', fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    ax.tick_params(axis='y', labelsize=8)


def create_edge_visualization(ax, original_pil, results):
    image_gray_data = np.array(original_pil.convert('L'))
    edges = np.zeros_like(image_gray_data, dtype=float)
    edge_inconsistency = results.get('edge_analysis', {}).get('edge_inconsistency', 0.0)

    try:
        if image_gray_data.shape[0] > 1 and image_gray_data.shape[1] > 1:
            if SKIMAGE_AVAILABLE:
                edges = sobel(image_gray_data.astype(np.float32))
            else:
                if image_gray_data.shape[0] >= 3 and image_gray_data.shape[1] >= 3:
                    grad_x = cv2.Sobel(image_gray_data, cv2.CV_64F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(image_gray_data, cv2.CV_64F, 0, 1, ksize=3)
                    edges = np.sqrt(grad_x**2 + grad_y**2)
                else:
                    edges = np.zeros_like(image_gray_data, dtype=float)
        
        if edges.max() > 0:
            edges = edges / edges.max()
        else:
            edges = np.zeros_like(edges)

    except Exception as e:
        print(f"Warning: Edge visualization failed during edge detection: {e}. Displaying black image.")
        edges = np.zeros_like(image_gray_data, dtype=float)

    ax.imshow(edges, cmap='gray')
    ax.set_title(f"6. Analisis Tepi (Incons: {edge_inconsistency:.2f})", fontsize=9)
    ax.axis('off')


def create_illumination_visualization(ax, original_pil, results):
    image_array = np.array(original_pil.convert('RGB'))
    illumination_data = np.zeros(original_pil.size[::-1], dtype=np.uint8)
    illum_inconsistency = results.get('illumination_analysis', {}).get('overall_illumination_inconsistency', 0.0)

    if image_array.size > 0:
        try:
            lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
            illumination_data = lab[:, :, 0]
        except Exception as e:
            print(f"Warning: Illumination visualization failed during LAB conversion: {e}. Displaying black image.")
    else:
        print("Warning: Input image for illumination analysis is empty. Displaying black image.")

    disp = ax.imshow(illumination_data, cmap='magma')
    ax.set_title(f"7. Peta Iluminasi (Incons: {illum_inconsistency:.2f})", fontsize=9)
    ax.axis('off')
    plt.colorbar(disp, ax=ax, fraction=0.046, pad=0.04)


def create_statistical_visualization(ax, results):
    stats = results.get('statistical_analysis', {})
    
    r_entropy = stats.get('R_entropy', 0.0)
    g_entropy = stats.get('G_entropy', 0.0)
    b_entropy = stats.get('B_entropy', 0.0)
    entropy_values = [r_entropy, g_entropy, b_entropy]

    if all(v == 0.0 for v in entropy_values):
        ax.text(0.5, 0.5, 'Statistical Data Not Available', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(f"11. Entropi Kanal", fontsize=12)
        ax.axis('off')
        return

    channels = ['Red', 'Green', 'Blue']
    colors = ['#d62728', '#2ca02c', '#1f77b4']
    
    ax.bar(channels, entropy_values, color=colors, alpha=0.8)
    ax.set_title(f"11. Entropi Kanal", fontsize=9)
    ax.set_ylabel('Entropi (bits)', fontsize=8)
    ax.set_ylim(0, 8.5)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', labelsize=8)


def create_quality_response_plot(ax, results):
    jpeg_analysis_data = results.get('jpeg_analysis', {})
    quality_responses = jpeg_analysis_data.get('basic_analysis', {}).get('quality_responses', [])
    estimated_original_quality = jpeg_analysis_data.get('basic_analysis', {}).get('estimated_original_quality', None)
    
    if not quality_responses:
        ax.text(0.5, 0.5, 'Quality Response Data Not Available', 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(f"12. Respons Kualitas JPEG", fontsize=12)
        ax.axis('off')
        return

    qualities = [r['quality'] for r in quality_responses]
    mean_responses = [r['response_mean'] for r in quality_responses]
    
    ax.plot(qualities, mean_responses, 'b-o', markersize=5, linewidth=1.5, markerfacecolor='cornflowerblue', markeredgecolor='darkblue')
    
    if estimated_original_quality is not None and estimated_original_quality > 0:
        ax.axvline(x=estimated_original_quality, color='r', linestyle='--', linewidth=1.5, label=f'Est. Q: {estimated_original_quality}')
        ax.legend(fontsize=8, loc='upper right')

    ax.set_title(f"12. Respons Kualitas JPEG", fontsize=9)
    ax.set_xlabel('Kualitas JPEG', fontsize=8)
    ax.set_ylabel('Rata-rata Error', fontsize=8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.6, linestyle=':')

def create_advanced_combined_heatmap(analysis_results, image_size):
    """Create combined heatmap with robust size handling."""
    w, h = 512, 512
    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        w, h = int(image_size[0]), int(image_size[1])
    elif hasattr(image_size, 'width') and hasattr(image_size, 'height'):
        w, h = image_size.width, image_size.height
    
    if w <= 0 or h <= 0:
        w, h = 512, 512

    heatmap = np.zeros((h, w), dtype=np.float32)

    ela_image_data = analysis_results.get('ela_image')
    if ela_image_data is not None:
        try:
            ela_array = np.array(ela_image_data.convert('L')) if isinstance(ela_image_data, Image.Image) else np.array(ela_image_data)
            if ela_array.size > 0:
                ela_resized = cv2.resize(ela_array, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                heatmap += (ela_resized / 255.0) * 0.35 
            else: print("Warning: ELA array is empty for heatmap.")
        except Exception as e:
            print(f"Warning: ELA contribution to heatmap failed: {e}")

    jpeg_ghost_data = analysis_results.get('jpeg_ghost')
    if jpeg_ghost_data is not None:
        try:
            if jpeg_ghost_data.size > 0:
                ghost_resized = cv2.resize(jpeg_ghost_data, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                heatmap += ghost_resized * 0.25 
            else: print("Warning: JPEG ghost array is empty for heatmap.")
        except Exception as e:
            print(f"Warning: JPEG Ghost contribution to heatmap failed: {e}")

    loc_analysis = analysis_results.get('localization_analysis', {})
    combined_mask_data = loc_analysis.get('combined_tampering_mask')
    if combined_mask_data is not None:
        try:
            if combined_mask_data.size > 0:
                mask_resized = cv2.resize(combined_mask_data.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
                heatmap += mask_resized * 0.40
            else: print("Warning: Localization mask is empty for heatmap.")
        except Exception as e:
            print(f"Warning: Localization mask contribution to heatmap failed: {e}")

    heatmap_max = np.max(heatmap)
    if heatmap_max > 0:
        heatmap_norm = heatmap / heatmap_max
    else:
        heatmap_norm = heatmap 

    blur_ksize = int(max(1, min(101, w // 20)))
    if blur_ksize % 2 == 0: blur_ksize += 1
    if blur_ksize > 1:
        heatmap_blurred = cv2.GaussianBlur(heatmap_norm, (blur_ksize, blur_ksize), 0)
    else:
        heatmap_blurred = heatmap_norm 

    return heatmap_blurred


def create_summary_report(ax, analysis_results):
    ax.axis('off')
    ax.clear()
    classification = analysis_results.get('classification', {})
    result_type = classification.get('type', 'N/A')
    confidence_level = classification.get('confidence', 'N/A')
    
    # Header section - lebih ringkas
    header_text = f"""LAPORAN ANALISIS FORENSIK
Temuan: {result_type}
Reliabilitas: {confidence_level}"""
    
    ax.text(0.02, 0.95, header_text, 
            transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='lightgray', alpha=0.8))

    # Evidence details - dengan wrapping otomatis
    details = classification.get('details', [])
    evidence_text = "EVIDENCE:\n"
    
    if details:
        for i, detail_item in enumerate(details[:3]):
            if len(detail_item) > 60:
                evidence_text += f"• {detail_item[:57]}...\n"
            else:
                evidence_text += f"• {detail_item}\n"
    else:
        evidence_text += "• Tidak ada evidence signifikan\n"
    
    ax.text(0.02, 0.75, evidence_text, 
            transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='azure', edgecolor='lightgray', alpha=0.7))

    # Disclaimer - dipersingkat dan diposisikan lebih baik
    disclaimer = "DISCLAIMER:\nSistem hanya mendeteksi kejanggalan. Analisis ahli diperlukan untuk kesimpulan definitif."
    
    ax.text(0.02, 0.25, disclaimer, 
            transform=ax.transAxes,
            fontsize=7, verticalalignment='top', style='italic',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='lightyellow', edgecolor='lightgray', alpha=0.6))
            
    ax.set_title("13. Ringkasan Temuan", fontsize=10, y=0.98)

def populate_validation_visuals(ax1, ax2):
    """
    Populates two subplots with system validation visuals.
    """
    ax1.clear()
    ax2.clear()

    ax1.set_title("16. Matriks Konfusi (Contoh)", fontsize=9)
    ax2.set_title("17. Distribusi Kepercayaan (Contoh)", fontsize=9)

    if SKLEARN_METRICS_AVAILABLE and SCIPY_AVAILABLE:
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.array([y_true[i] if np.random.rand() < 0.9 else 1-y_true[i] for i in range(100)])
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False,
                    xticklabels=['Normal', 'Mencurigakan'],
                    yticklabels=['Normal', 'Mencurigakan'], 
                    linewidths=.5, linecolor='gray')
        ax1.set_xlabel("Hasil Analisis", fontsize=9)
        ax1.set_ylabel("Kondisi Sebenarnya", fontsize=9)
        ax1.tick_params(axis='both', which='major', labelsize=8)
        ax1.set_title(f"16. Matriks Evaluasi Evidence (Akurasi: {accuracy:.1%})", fontsize=9)

        normal_scores = np.random.normal(loc=20, scale=10, size=50)
        suspicious_scores = np.random.normal(loc=80, scale=10, size=50)
        combined_scores = np.clip(np.concatenate((normal_scores, suspicious_scores)), 0, 100)

        sns.histplot(combined_scores, kde=True, ax=ax2, color="purple", bins=15, alpha=0.6, stat="density", linewidth=0)
        ax2.set_xlabel("Kekuatan Evidence (%)", fontsize=9)
        ax2.set_ylabel("Densitas", fontsize=9)
        ax2.axvline(x=50, color='r', linestyle=':', label='Ambang Evidence (50%)')
        ax2.legend(fontsize=8)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.set_ylim(bottom=0)
        ax2.tick_params(axis='both', which='major', labelsize=8)

    else:
        ax1.text(0.5, 0.5, 'Sklearn / Scipy Not Available', ha='center', va='center', transform=ax1.transAxes)
        ax1.axis('off')
        ax2.text(0.5, 0.5, 'Sklearn / Scipy Not Available', ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')
        
# --- END OF FILE visualization.py ---