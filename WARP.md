# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is an **Advanced Forensic Image Analysis System** designed to detect image tampering and manipulation through multiple forensic techniques including:
- Error Level Analysis (ELA)
- Copy-move detection (feature-based and block-based)
- JPEG artifact analysis
- Noise consistency analysis
- Frequency domain analysis
- Texture and edge consistency analysis

## Common Development Commands

### Installation
```bash
# Install all dependencies
pip install -r requirements.txt

# For development with specific Python version
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### Running the Application

```bash
# Run the Streamlit web interface
streamlit run app2.py

# Run command-line analysis on a single image
python main.py <image_path>

# Run with additional options
python main.py test_image.jpg --export-all
python main.py test_image.jpg --output-dir ./results
```

### Testing
```bash
# Run the validator test suite (built-in)
python validator.py

# Run main analysis in test mode (faster, simplified analysis)
python -c "from main import analyze_image_comprehensive_advanced; analyze_image_comprehensive_advanced('test_image.jpg', test_mode=True)"
```

## Architecture & Key Components

### Core Analysis Pipeline (`main.py`)
The main analysis pipeline follows a 17-stage process:
1. File validation
2. Image loading
3. Metadata extraction
4. Advanced preprocessing
5. Multi-quality ELA analysis
6. Multi-detector feature extraction
7. Feature-based copy-move detection
8. Block-based copy-move detection
9. Noise consistency analysis
10. JPEG artifact analysis
11. Frequency domain analysis
12. Texture consistency analysis
13. Edge consistency analysis
14. Illumination consistency analysis
15. Statistical analysis
16. Advanced tampering localization
17. Final classification

Each stage has error handling and fallback mechanisms. The pipeline tracks completion status and can continue even if individual stages fail.

### Module Structure

**Detection Modules:**
- `ela_analysis.py`: Error Level Analysis with multi-quality assessment
- `copy_move_detection.py`: Feature-based and block-based copy-move detection
- `jpeg_analysis.py`: Comprehensive JPEG artifact and ghost analysis
- `feature_detection.py`: Multi-detector (SIFT, ORB, AKAZE) feature extraction

**Analysis Modules:**
- `advanced_analysis.py`: Noise, frequency, texture, edge, illumination analysis
- `classification.py`: Advanced manipulation classification with confidence scoring
- `uncertainty_classification.py`: Probabilistic classification with uncertainty quantification
- `validation.py`: Image validation and metadata extraction

**UI/Export Modules:**
- `app2.py`: Main Streamlit web interface
- `visualization.py`: Generates all visualization plots and heatmaps
- `export_utils.py`: Handles export to DOCX, PDF, PNG, and ZIP packages

**Utilities:**
- `config.py`: Central configuration for all analysis parameters
- `utils.py`: Helper functions, history management, outlier detection
- `validator.py`: System validation and test suite

### Key Data Flow

1. **Image Input** → Validation → Preprocessing
2. **Feature Extraction** → Multiple detectors run in parallel
3. **Analysis Results** → Dictionary structure with all findings:
   - Each analysis module adds its results to `analysis_results` dict
   - Results include both numerical scores and visual maps/arrays
4. **Classification** → Combines all evidence to determine manipulation type
5. **Export** → Multiple format options with comprehensive reporting

### Critical Implementation Details

**Image Coordinate Systems:**
- PIL uses (width, height) for image.size
- NumPy arrays use (height, width) for shape
- Many functions handle conversion between these formats

**Error Handling:**
- Each pipeline stage has try-except blocks
- Failed stages are tracked but don't stop the pipeline
- Default/fallback values ensure the pipeline continues

**Performance Optimization:**
- Test mode available for faster analysis (reduced features)
- Block matching uses sampling for large images
- Feature extraction uses ROI masking based on ELA

**State Management (Streamlit):**
- Uses `st.session_state` for persistence across reruns
- Analysis history saved to JSON with thumbnails
- Supports multiple image analysis in single session

## Configuration Parameters (`config.py`)

Key tunable parameters:
- `ELA_QUALITIES`: Quality levels for ELA analysis
- `BLOCK_SIZE`: Size for block-based detection (default: 16)
- `SIFT_FEATURES`: Number of SIFT features to extract
- `DETECTION_THRESHOLD`: Minimum score for positive detection
- `TARGET_MAX_DIM`: Maximum dimension for image resizing

## Dependencies Management

The project requires several computer vision and machine learning libraries. Missing `copy_move_detection.py` has been created if it doesn't exist. All dependencies are listed in `requirements.txt`.

For PDF export functionality, LibreOffice must be installed separately on the system.

## Important File Paths

- Analysis history: Stored in `analysis_history.json`
- Thumbnails: Saved in `thumbnails/` directory
- Export outputs: Default to `exported_reports/` directory
- Temporary files: Created in working directory during processing

## Common Issues & Solutions

1. **Missing copy_move_detection.py**: This file has been created with the required functions
2. **Memory issues with large images**: Images are automatically resized if larger than `TARGET_MAX_DIM`
3. **Streamlit dummy import**: `streamlit.py` provides stub functions for non-Streamlit environments
4. **PDF export failures**: Ensure LibreOffice is installed or use DOCX export instead