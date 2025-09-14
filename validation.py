"""
Image validation and preprocessing functions
"""

import os
import numpy as np
from PIL import Image, ImageEnhance
import cv2
try:
    import exifread
    EXIFREAD_AVAILABLE = True
except Exception:
    EXIFREAD_AVAILABLE = False
from datetime import datetime
from config import VALID_EXTENSIONS, MIN_FILE_SIZE, TARGET_MAX_DIM

def validate_image_file(filepath):
    """Enhanced validation with more format support"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in VALID_EXTENSIONS:
        raise ValueError(f"Format file tidak didukung: {ext}")
    
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File {filepath} tidak ditemukan.")
    
    file_size = os.path.getsize(filepath)
    if file_size < MIN_FILE_SIZE:
        print(f"⚠ Warning: File sangat kecil ({file_size} bytes), hasil mungkin kurang akurat")
    
    return True

def extract_enhanced_metadata(filepath):
    """Enhanced metadata extraction dengan analisis inkonsistensi yang lebih detail"""
    metadata = {}
    try:
        with open(filepath, 'rb') as f:
            if EXIFREAD_AVAILABLE:
                tags = exifread.process_file(f, details=False, strict=False)
            else:
                tags = {}
        
        metadata['Filename'] = os.path.basename(filepath)
        metadata['FileSize (bytes)'] = os.path.getsize(filepath)
        
        try:
            metadata['LastModified'] = datetime.fromtimestamp(
                os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            metadata['LastModified'] = str(os.path.getmtime(filepath))
        
        # Extract comprehensive EXIF tags
        comprehensive_tags = [
            'Image DateTime', 'EXIF DateTimeOriginal', 'EXIF DateTimeDigitized',
            'Image Software', 'Image Make', 'Image Model', 'Image ImageWidth',
            'Image ImageLength', 'EXIF ExifVersion', 'EXIF ColorSpace',
            'Image Orientation', 'EXIF Flash', 'EXIF WhiteBalance',
            'GPS GPSLatitudeRef', 'GPS GPSLatitude', 'GPS GPSLongitudeRef',
            'EXIF LensModel', 'EXIF FocalLength', 'EXIF ISO', 'EXIF ExposureTime'
        ]
        
        for tag in comprehensive_tags:
            if tag in tags:
                metadata[tag] = str(tags[tag])
        
        metadata['Metadata_Inconsistency'] = check_enhanced_metadata_consistency(tags)
        metadata['Metadata_Authenticity_Score'] = calculate_metadata_authenticity_score(tags)
        
    except Exception as e:
        print(f"⚠ Peringatan: Gagal membaca metadata EXIF: {e}")
    
    return metadata

def check_enhanced_metadata_consistency(tags):
    """Enhanced metadata consistency check"""
    inconsistencies = []
    
    # Time consistency check
    datetime_tags = ['Image DateTime', 'EXIF DateTimeOriginal', 'EXIF DateTimeDigitized']
    datetimes = []
    
    for tag in datetime_tags:
        if tag in tags:
            try:
                dt_str = str(tags[tag])
                dt = datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
                datetimes.append((tag, dt))
            except:
                pass
    
    if len(datetimes) > 1:
        for i in range(len(datetimes)-1):
            for j in range(i+1, len(datetimes)):
                diff = abs((datetimes[i][1] - datetimes[j][1]).total_seconds())
                if diff > 60:  # 1 minute
                    inconsistencies.append(f"Time difference: {datetimes[i][0]} vs {datetimes[j][0]} ({diff:.0f}s)")
    
    # Software signature check
    if 'Image Software' in tags:
        software = str(tags['Image Software']).lower()
        suspicious_software = ['photoshop', 'gimp', 'paint', 'editor', 'modified']
        if any(sus in software for sus in suspicious_software):
            inconsistencies.append(f"Editing software detected: {software}")
    
    return inconsistencies

def calculate_metadata_authenticity_score(tags):
    """Calculate metadata authenticity score (0-100) with realistic and professional analysis"""
    # Start with a more realistic baseline score
    score = 70  # Changed from 100 to 70 for more realistic scoring
    
    # Reduced penalties for essential metadata - many legitimate photos may lack some tags
    essential_tags = {
        'Image DateTime': 4,        # Reduced from 10
        'EXIF DateTimeOriginal': 4, # Reduced from 10  
        'EXIF DateTimeDigitized': 2, # Reduced from 5
        'Image Make': 3,            # Reduced from 8
        'Image Model': 3,           # Reduced from 8
        'EXIF ExifVersion': 1,      # Reduced from 3
        'EXIF ColorSpace': 1,       # Reduced from 3
        'Image Orientation': 1      # Reduced from 3
    }
    
    for tag, penalty in essential_tags.items():
        if tag not in tags:
            score -= penalty
    
    # More nuanced software analysis - many legitimate tools are used for basic processing
    if 'Image Software' in tags:
        software = str(tags['Image Software']).lower()
        
        # Professional editing software (moderate penalty)
        if 'photoshop' in software:
            score -= 12  # Reduced from 25
        elif 'gimp' in software:
            score -= 8   # Reduced from 20
        elif 'lightroom' in software:
            score -= 5   # Adobe Lightroom is common for legitimate photo processing
        
        # Basic editing tools (minimal penalty)
        elif any(editor in software for editor in ['paint', 'editor', 'edit']):
            score -= 6   # Reduced from 15
        
        # Camera and manufacturer software (bonus)
        elif any(cam in software for cam in ['camera', 'canon', 'nikon', 'sony', 'fuji', 'olympus', 'panasonic']):
            score += 8   # Increased from 5
        
        # Phone camera software (bonus)
        elif any(phone in software for phone in ['iphone', 'samsung', 'huawei', 'xiaomi', 'oneplus']):
            score += 6
        
        # RAW processors (minimal penalty as they're legitimate)
        elif any(raw in software for raw in ['capture one', 'dxo', 'luminar', 'on1']):
            score -= 3
    
    # GPS data is now completely optional with small bonus (many cameras don't have GPS)
    gps_tags = [tag for tag in tags if str(tag).startswith('GPS')]
    if len(gps_tags) >= 4:  # Complete GPS data
        score += 5   # Reduced from 8
    elif len(gps_tags) > 0:
        score += 2   # Reduced from 3
    
    # Camera settings bonus (more generous)
    camera_settings = ['EXIF FocalLength', 'EXIF ISO', 'EXIF ExposureTime', 
                      'EXIF FNumber', 'EXIF Flash', 'EXIF WhiteBalance']
    camera_settings_count = sum(1 for tag in camera_settings if tag in tags)
    if camera_settings_count >= 5:
        score += 15  # Increased from 10
    elif camera_settings_count >= 3:
        score += 8   # Increased from 5
    elif camera_settings_count >= 1:
        score += 3   # New tier for minimal camera data
    
    # Thumbnail data bonus
    if any('Thumbnail' in str(tag) for tag in tags):
        score += 5   # Increased from 3
    
    # More sophisticated suspicious pattern detection
    time_tags = ['Image DateTime', 'EXIF DateTimeOriginal', 'EXIF DateTimeDigitized']
    time_values = [str(tags[tag]) for tag in time_tags if tag in tags]
    
    # Only penalize if ALL timestamps are identical AND there are multiple timestamps
    if len(time_values) >= 3 and len(set(time_values)) == 1:
        score -= 8   # Reduced from 10
    
    # More realistic tag count assessment
    all_tags = len([tag for tag in tags if str(tag).startswith(('Image', 'EXIF', 'GPS', 'Thumbnail'))])
    if all_tags > 40:        # Very rich metadata
        score += 15
    elif all_tags > 25:      # Good metadata
        score += 10
    elif all_tags > 15:      # Adequate metadata
        score += 5
    elif all_tags > 8:       # Minimal but acceptable
        score += 2
    elif all_tags < 4:       # Very sparse metadata (suspicious)
        score -= 10  # Reduced from 15
    
    # Additional bonus for lens information (indicates serious camera)
    if any('Lens' in str(tag) for tag in tags):
        score += 5
    
    # Bonus for maker notes (camera-specific data)
    if any('MakerNote' in str(tag) for tag in tags):
        score += 3
    
    return max(0, min(100, score))

def advanced_preprocess_image(image_pil, target_max_dim=TARGET_MAX_DIM, normalize=True):
    """Advanced preprocessing dengan enhancement, size optimization, dan normalisasi."""
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')
    
    original_width, original_height = image_pil.size
    print(f"  Original size: {original_width} × {original_height}")
    
    # Simpan salinan dari gambar asli sebelum diubah ukurannya untuk analisis tertentu
    original_pil_copy = image_pil.copy()

    # Pengubahan ukuran yang lebih agresif untuk gambar yang sangat besar
    if original_width > target_max_dim or original_height > target_max_dim:
        ratio = min(target_max_dim / original_width, target_max_dim / original_height)
        new_size = (int(original_width * ratio), int(original_height * ratio))
        image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)
        print(f"  Resized to: {new_size[0]} × {new_size[1]} (ratio: {ratio:.3f})")
    
    image_array = np.array(image_pil)

    # Normalisasi (opsional, default True)
    if normalize:
        # Terapkan CLAHE (Contrast Limited Adaptive Histogram Equalization) untuk meningkatkan kontras lokal
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        image_array = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        print("  Applied CLAHE for contrast enhancement.")

    # Denoising ringan hanya untuk gambar yang lebih kecil
    if max(image_pil.size) <= 2000:
        try:
            denoised = cv2.fastNlMeansDenoisingColored(image_array, None, 3, 3, 7, 21)
            print("  Applied light denoising.")
        except AttributeError:
            # Fallback denoising jika fungsi tidak tersedia
            denoised = cv2.bilateralFilter(image_array, 9, 75, 75)
            print("  Applied fallback bilateral filter for denoising.")
        
        # Kembalikan gambar yang telah diproses dan salinan asli yang telah diubah ukurannya
        return Image.fromarray(denoised), image_pil
    else:
        print("  Skipping denoising for large image.")
        # Kembalikan gambar yang telah diproses (hanya normalisasi) dan salinan asli yang telah diubah ukurannya
        return Image.fromarray(image_array), image_pil
