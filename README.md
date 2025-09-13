# Sudoku OCR Solver

A Python program for extracting, solving, and validating Sudoku puzzles from images using OCR
Created by Nathan Ahmann as a way to get used to coding using AI. 

## Features

- **Automatic Grid Detection**: Detects and extracts Sudoku grids from images with perspective correction
- **Advanced OCR**: Dual-stage OCR using Tesseract and EasyOCR for maximum accuracy
- **Sudoku Solver**: Backtracking algorithm to solve extracted puzzles
- **Validation**: OCR accuracy measurement against the provided master key
- **Batch Processing**: Process multiple images at once
- **Visual Feedback**: Overlay visualizations showing extraction accuracy (when provided master keys)

## Installation

### Prerequisites

- Python 3.7+
- OpenCV
- Tesseract OCR


### Directory Structure

```
project/
├── sudoku_ocr_solver.py
├── images/
│   ├── sudoku_grid1.png
│   ├── sudoku_grid2.png
│   └── ...
└── README.md
```

### Basic Usage

```python
from sudoku_ocr_solver import SudokuOCRSolver

# Initialize the solver
solver = SudokuOCRSolver()

# Process a single image
result = solver.process_image("images/sudoku_grid1.png")
print(f"Extraction accuracy: {result['accuracy']:.1f}%")
print(f"Solved successfully: {result['is_solved']}")

# Process multiple images
results = solver.batch_process("images")
solver.display_results(results)
```

### Debug Mode

Enable detailed processing visualization for troubleshooting:

```python
# Debug a specific image
result = solver.process_image("images/sudoku_grid1.png", debug=True)

# Debug specific images in batch processing
results = solver.batch_process(
    "images", 
    debug_images=["sudoku_grid1.png", "sudoku_grid2.png"]
)
```

## Processing Pipeline

### 1. Image Preprocessing
- Convert to grayscale
- Grid detection and perspective correction
- Adaptive thresholding
- Image inversion for OCR optimization

### 2. Grid Extraction
- Divide the image into 9x9 cells
- Content detection using pixel analysis and connected components
- Dual-stage OCR:
  - Primary: Tesseract with digit-specific configuration
  - Fallback: EasyOCR for difficult cases
  - Extra detection for failure/low confidence in both cases and when Content Detection believes the cell is not empty

### 3. Sudoku Solving
- Backtracking algorithm
- Constraint validation (row, column, 3x3 box)
- Solution verification

### 4. Validation and Visualization
- OCR accuracy calculation against known master keys
- Overlay visualization showing correct/incorrect OCR

### Preprocessing Parameters

```python
processed_image = solver.preprocess_sudoku_image(
    image,
    detect_grid=True,           # Enable grid detection
    threshold_block_size=11,    # Adaptive threshold block size
    threshold_c=2              # Adaptive threshold C parameter
)
```

### OCR Confidence Thresholds

The solver uses different confidence thresholds:
- Tesseract: 60% confidence threshold
- EasyOCR: 40% confidence threshold
- Content detection: 3% pixel ratio threshold

## Troubleshooting

### Common Issues

**1. Low Extraction Accuracy**
- Ensure good image quality and lighting
- Try adjusting preprocessing parameters
- Enable debug mode to see cell-by-cell processing
- Verify Tesseract installation and language data

**2. Grid Detection Fails**
- Disable automatic grid detection: `detect_grid=False`
- Ensure the Sudoku grid is clearly visible with good contrast
- Check that the image contains a complete 9x9 grid

**3. OCR Errors**
- Install EasyOCR for improved accuracy: `pip install easyocr`
- Check Tesseract configuration and language packs
- Verify image preprocessing quality

### Debug Output

Enable debug mode to see:
- Step-by-step image preprocessing
- Individual cell extraction and OCR results
- Solving progress
- Accuracy measurements
