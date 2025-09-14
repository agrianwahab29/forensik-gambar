"""
Configuration file for Forensic Image Analysis System
"""

# Analysis parameters
ELA_QUALITIES = [50,70, 80, 90, 95]  #ASLINYA TIDAK ADA 50
ELA_SCALE_FACTOR = 10 #ASLINYA 20
BLOCK_SIZE = 16 #aslinya 16
NOISE_BLOCK_SIZE = 32
TEXTURE_BLOCK_SIZE = 64

# Feature detection parameters
SIFT_FEATURES = 3000
SIFT_CONTRAST_THRESHOLD = 0.02
SIFT_EDGE_THRESHOLD = 10

ORB_FEATURES = 2000
ORB_SCALE_FACTOR = 1.2
ORB_LEVELS = 8

# Copy-move detection parameters
RATIO_THRESH = 0.75 # Increased from 0.65 for more matches
MIN_DISTANCE = 30 # Reduced from 40 for closer matches
RANSAC_THRESH = 8.0 # Increased from 5.0 for more tolerant geometric verification
MIN_INLIERS = 6 # Reduced from 10 for better sensitivity

# Classification thresholds
DETECTION_THRESHOLD = 60 #Aslinya 45
CONFIDENCE_THRESHOLD = 75 #aslinya 60

# File format support
VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
MIN_FILE_SIZE = 50000  # 50KB

# Processing parameters
TARGET_MAX_DIM = 2000
MAX_SAMPLES_DBSCAN = 50000