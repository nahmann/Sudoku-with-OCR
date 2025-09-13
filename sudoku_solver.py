# Sudoku Solver From an Image
# This program reads a Sudoku grid from an image, extracts the digits using OCR,
# solves the Sudoku puzzle, and validates the solution. 
# 9/8 - 9/12/2025
#


import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import os
from typing import List, Tuple, Optional, Dict
import warnings
import glob
from scipy import ndimage
from skimage import morphology, measure

warnings.filterwarnings('ignore')

class SudokuOCRSolver:
    """
    A comprehensive Sudoku OCR solver that can detect, extract, solve, and validate Sudoku puzzles from images.
    """
    
    def __init__(self, tesseract_path: str = None):
        """
        Initialize the Sudoku OCR Solver.
        
        Args:
            tesseract_path (str, optional): Path to tesseract executable if not in PATH
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Initialize EasyOCR reader lazily
        self._easyocr_reader = None
    
    @property
    def easyocr_reader(self):
        """Lazy initialization of EasyOCR reader."""
        if self._easyocr_reader is None:
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            except ImportError:
                print("Warning: EasyOCR not available. Install with: pip install easyocr")
                self._easyocr_reader = False
        return self._easyocr_reader if self._easyocr_reader is not False else None
    
#### IMAGE LOADING AND PREPROCESSING METHODS
    
    def read_sudoku_image(self, image_path: str, debug: bool = False) -> np.ndarray:
        """
        Read a Sudoku grid image from the specified path.
        
        Args:
            image_path (str): Path to the image file
            debug (bool): If True, displays the loaded image
            
        Returns:
            np.ndarray: The loaded image as a numpy array
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If the image cannot be loaded
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not load image from: {image_path}")
        
        # Convert BGR to RGB for matplotlib display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if debug:
            print(f"Successfully loaded image: {image_path}")
            print(f"Image shape: {image.shape}")
            print(f"Image dtype: {image.dtype}")
            
            plt.figure(figsize=(8, 8))
            plt.imshow(image_rgb)
            plt.title(f"Loaded Sudoku Image: {os.path.basename(image_path)}")
            plt.axis('off')
            plt.show()
        
        return image
    
    def preprocess_sudoku_image(self, image: np.ndarray, debug: bool = False, 
                              detect_grid: bool = True,
                              threshold_block_size: int = 11,
                              threshold_c: int = 2) -> np.ndarray:
        """
        Preprocess the Sudoku image with integrated grid detection, thresholding, and inversion.
        
        Args:
            image (np.ndarray): Input Sudoku image
            debug (bool): If True, displays intermediate processing steps
            detect_grid (bool): If True, detects and crops the Sudoku grid
            threshold_block_size (int): Adaptive threshold block size
            threshold_c (int): Adaptive threshold C parameter
            
        Returns:
            np.ndarray: Preprocessed image optimized for OCR
        """
        if debug:
            print("Starting Sudoku image preprocessing...")
            print(f"Original image shape: {image.shape}")
            print(f"Grid detection: {detect_grid}")
            plt.figure(figsize=(20, 5))
        
        current_image = image.copy()
        step_count = 0
        
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        step_count += 1
        
        if debug:
            plt.subplot(1, 6, step_count)
            plt.imshow(gray, cmap='gray')
            plt.title("1. Grayscale")
            plt.axis('off')
        
        # Step 2: Grid detection and perspective correction (if enabled)
        if detect_grid:
            step_count += 1
            
            # Apply Gaussian blur for contour detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding for contour detection
            thresh_for_contours = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                       cv2.THRESH_BINARY, 11, 2)
            
            # Invert for contour detection (grid lines should be white)
            thresh_for_contours = 255 - thresh_for_contours
            
            if debug:
                plt.subplot(1, 6, step_count)
                plt.imshow(thresh_for_contours, cmap='gray')
                plt.title("2. Thresh for Contours")
                plt.axis('off')
            
            # Find contours and apply perspective correction
            current_image, gray, grid_detected = self._detect_and_correct_grid(
                current_image, thresh_for_contours, image, debug, step_count
            )
            
            if grid_detected:
                step_count += 1
        
        # Apply adaptive thresholding to final grayscale image
        step_count += 1
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, threshold_block_size, threshold_c
        )
        
        if debug:
            plt.subplot(1, 6, step_count)
            plt.imshow(thresh, cmap='gray')
            plt.title(f"{step_count}. Adaptive Threshold")
            plt.axis('off')
        
        # Invert image for better tesseract performance
        step_count += 1
        inverted = 255 - thresh
        
        if debug:
            plt.subplot(1, 6, step_count)
            plt.imshow(inverted, cmap='gray')
            plt.title(f"{step_count}. Final (Inverted)")
            plt.axis('off')
            
            # Show before/after comparison
            step_count += 1
            if step_count <= 6:
                plt.subplot(1, 6, step_count)
                original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                plt.imshow(original_gray, cmap='gray')
                plt.title("Original for Comparison")
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            print("Preprocessing completed!")
            print(f"Final image shape: {inverted.shape}")
        
        return inverted
    
    def _detect_and_correct_grid(self, current_image: np.ndarray, thresh_for_contours: np.ndarray, 
                               original_image: np.ndarray, debug: bool, step_count: int) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Helper method for grid detection and perspective correction."""
        # Find contours
        contours, _ = cv2.findContours(thresh_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        grid_detected = False
        if contours:
            # Find the largest contour (should be the Sudoku grid)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # If we don't get exactly 4 corners, try different epsilon values
            if len(approx) != 4:
                for epsilon_factor in [0.01, 0.03, 0.04, 0.05]:
                    epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    if len(approx) == 4:
                        break
            
            # If we found 4 corners, apply perspective correction
            if len(approx) == 4:
                current_image, gray = self._apply_perspective_correction(
                    current_image, approx, original_image, debug, step_count
                )
                grid_detected = True
        
        if debug and not grid_detected:
            plt.subplot(1, 6, step_count + 1)
            plt.imshow(cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY), cmap='gray')
            plt.title("3. Grid Not Detected")
            plt.axis('off')
            print("Warning: Could not detect 4 corners, using original image")
        
        return current_image, gray, grid_detected
    
    def _apply_perspective_correction(self, current_image: np.ndarray, approx: np.ndarray,
                                    original_image: np.ndarray, debug: bool, step_count: int) -> Tuple[np.ndarray, np.ndarray]:
        """Helper method for applying perspective correction."""
        # Order the corners: top-left, top-right, bottom-right, bottom-left
        corners = approx.reshape(4, 2).astype(np.float32)
        
        # Calculate the centroid
        centroid = np.mean(corners, axis=0)
        
        # Sort corners based on their position relative to centroid
        def classify_corner(point, center):
            if point[0] < center[0] and point[1] < center[1]:
                return 0  # top-left
            elif point[0] > center[0] and point[1] < center[1]:
                return 1  # top-right
            elif point[0] > center[0] and point[1] > center[1]:
                return 2  # bottom-right
            else:
                return 3  # bottom-left
        
        ordered_corners = np.zeros((4, 2), dtype=np.float32)
        for corner in corners:
            idx = classify_corner(corner, centroid)
            ordered_corners[idx] = corner
        
        # Calculate the size of the output square
        width1 = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
        width2 = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
        height1 = np.linalg.norm(ordered_corners[3] - ordered_corners[0])
        height2 = np.linalg.norm(ordered_corners[2] - ordered_corners[1])
        
        max_dim = int(max(width1, width2, height1, height2))
        
        # Define destination points for perspective transformation
        dst_corners = np.array([
            [0, 0],
            [max_dim - 1, 0],
            [max_dim - 1, max_dim - 1],
            [0, max_dim - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_corners)
        
        # Apply perspective transformation to the original image
        corrected_image = cv2.warpPerspective(current_image, transform_matrix, (max_dim, max_dim))
        
        # Update grayscale image after perspective correction
        gray = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
        
        if debug:
            # Show corner detection
            contour_img = original_image.copy()
            cv2.drawContours(contour_img, [approx], -1, (255, 0, 0), 5)
            for i, point in enumerate(approx):
                cv2.circle(contour_img, tuple(point[0]), 10, (0, 0, 255), -1)
            
            plt.subplot(1, 6, step_count + 1)
            plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
            plt.title(f"3. Grid Detected")
            plt.axis('off')
        
        return corrected_image, gray
    
#### GRID EXTRACTION METHODS
    
    def extract_sudoku_grid(self, processed_image: np.ndarray, debug: bool = False) -> np.ndarray:
        """
        Extract the Sudoku grid from the processed image using advanced OCR techniques.
        
        Args:
            processed_image (np.ndarray): Preprocessed Sudoku image (should be inverted)
            debug (bool): If True, displays the grid extraction and OCR results
            
        Returns:
            np.ndarray: 9x9 array representing the Sudoku board (0 for empty cells)
        """
        if debug:
            print("Starting Sudoku grid extraction...")
        
        # Initialize the 9x9 Sudoku board
        sudoku_board = np.zeros((9, 9), dtype=int)
        
        # Calculate cell dimensions
        height, width = processed_image.shape
        cell_height = height // 9
        cell_width = width // 9
        
        if debug:
            print(f"Image dimensions: {width}x{height}")
            print(f"Cell dimensions: {cell_width}x{cell_height}")
            
            # Create a figure to show all cells
            fig, axes = plt.subplots(9, 9, figsize=(12, 12))
            fig.suptitle("Sudoku Cells with Two-Stage OCR", fontsize=16)
        
        # Process each cell
        for row in range(9):
            for col in range(9):
                if debug:
                    print(f"Processing cell ({row+1}, {col+1})...")
                
                # Calculate cell boundaries with margin to avoid grid lines
                margin = 3
                y1 = row * cell_height + margin
                y2 = (row + 1) * cell_height - margin
                x1 = col * cell_width + margin
                x2 = (col + 1) * cell_width - margin
                
                # Extract cell image
                cell = processed_image[y1:y2, x1:x2]
                
                # Check if cell has content and perform OCR
                digit = self._process_cell(cell, debug)
                sudoku_board[row, col] = digit if digit is not None else 0
                
                if debug:
                    # Display the cell with result
                    axes[row, col].imshow(cell, cmap='gray')
                    result_val = sudoku_board[row, col]
                    title = f"R{row+1}C{col+1}: {result_val if result_val != 0 else 'Empty'}"
                    axes[row, col].set_title(title, fontsize=8)
                    axes[row, col].axis('off')
        
        if debug:
            plt.tight_layout()
            plt.show()
            
            print("\nExtracted Sudoku Board:")
            self.print_sudoku_board(sudoku_board)
            
            # Count filled cells
            filled_cells = np.count_nonzero(sudoku_board)
            print(f"\nFilled cells: {filled_cells}/81")
            print(f"Empty cells: {81 - filled_cells}/81")
            
        return sudoku_board
    
    def _process_cell(self, cell: np.ndarray, debug: bool) -> Optional[int]:
        """Process a single cell to extract digit."""
        # Step 1: Quick content check
        has_content = self._has_content_pixels(cell)
        
        if debug:
            print(f"  Initial content check: {has_content}")
        
        if not has_content:
            # Step 2: Additional content check before declaring empty
            has_content = self._additional_content_check(cell)
            if debug:
                print(f"  Additional content check: {has_content}")
        
        if not has_content:
            if debug:
                print("  Cell marked as empty")
            return None
        
        # Cell has content - perform OCR
        if debug:
            print("  Cell has content, performing OCR...")
        
        # Preprocess cell for OCR
        processed_cell = self._preprocess_cell_for_ocr(cell)
        
        # Basic Tesseract OCR
        digit_tesseract, confidence = self._basic_tesseract_ocr(processed_cell, debug)
        
        if debug:
            print(f"  Tesseract result: digit={digit_tesseract}, confidence={confidence:.1f}")
        
        # Check if Tesseract result is confident enough
        if digit_tesseract is not None and confidence > 60:
            if debug:
                print(f"  Accepted Tesseract result: {digit_tesseract}")
            return digit_tesseract
        
        # Fallback to EasyOCR
        if debug:
            print("  Tesseract confidence low, trying EasyOCR...")
        
        digit_easy, confidence_easy = self._easyocr_fallback(processed_cell, debug)
        
        if debug:
            print(f"  EasyOCR result: digit={digit_easy}, confidence={confidence_easy:.1f}")
        
        # Check if EasyOCR result is confident enough
        if digit_easy is not None and confidence_easy > 40:
            if debug:
                print(f"  Accepted EasyOCR result: {digit_easy}")
            return digit_easy
        elif digit_tesseract is not None:
            # EasyOCR failed but Tesseract found something - use Tesseract despite low confidence
            if debug:
                print(f"  EasyOCR failed, falling back to Tesseract result: {digit_tesseract}")
            return digit_tesseract
        
        return None
    
    def _has_content_pixels(self, cell_image: np.ndarray) -> bool:
        """Check if a cell contains any non-background pixels that might indicate a digit."""
        border_margin = 3
        h, w = cell_image.shape
        if h <= 2 * border_margin or w <= 2 * border_margin:
            return False
            
        inner_region = cell_image[border_margin:-border_margin, border_margin:-border_margin]
        
        # Count white pixels (in inverted image, digits are white)
        white_pixels = np.sum(inner_region > 100)
        total_inner_pixels = inner_region.size
        
        # If more than 3% of inner pixels are white, consider it has content
        content_ratio = white_pixels / total_inner_pixels
        return content_ratio > 0.03
    
    def _additional_content_check(self, cell_image: np.ndarray) -> bool:
        """More thorough content check using connected components."""
        # Apply threshold to get binary image
        _, binary = cv2.threshold(cell_image, 100, 255, cv2.THRESH_BINARY)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Look for components that could be digits
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            # Check if component looks like it could be a digit
            if (area > 50 and  # Minimum area for a digit
                w > 5 and h > 5 and  # Minimum dimensions
                x > 2 and y > 2 and  # Not touching edges
                x + w < cell_image.shape[1] - 2 and 
                y + h < cell_image.shape[0] - 2):
                return True
        
        return False
    
    def _preprocess_cell_for_ocr(self, cell_image: np.ndarray) -> np.ndarray:
        """Preprocess cell for optimal OCR.

        Args:
            cell_image (np.ndarray): Raw cell image
            
        Returns:
            np.ndarray: Preprocessed cell image
        """
        # Add padding
        padding = 15
        padded = cv2.copyMakeBorder(
            cell_image, padding, padding, padding, padding, 
            cv2.BORDER_CONSTANT, value=0
        )
        
        # Resize to a standard size
        resized = cv2.resize(padded, (100, 100), interpolation=cv2.INTER_CUBIC)
        
        # Apply slight blur to smooth edges
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        
        return blurred
    
    def _basic_tesseract_ocr(self, cell_image: np.ndarray, debug: bool = False) -> Tuple[Optional[int], float]:
        """
        Perform basic Tesseract OCR on cell.

        Args:
            cell_image (np.ndarray): Preprocessed cell image
            
        Returns:
            tuple: (digit, confidence) where digit is int or None
        """
        try:
            # Basic Tesseract config for single digits
            config = '--oem 3 --psm 10 -c tessedit_char_whitelist=123456789'
            
            # Get confidence data
            data = pytesseract.image_to_data(cell_image, config=config, output_type=pytesseract.Output.DICT)
            
            # Extract text
            text = pytesseract.image_to_string(cell_image, config=config).strip()
            
            # Find digit in text
            digit = None
            for char in text:
                if char.isdigit() and '1' <= char <= '9':
                    digit = int(char)
                    break
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return digit, avg_confidence
            
        except Exception as e:
            if debug:
                print(f"    Tesseract OCR error: {e}")
            return None, 0
    
    def _easyocr_fallback(self, cell_image: np.ndarray, debug: bool = False) -> Tuple[Optional[int], float]:
        """
        Fallback OCR using EasyOCR for difficult cases.

        Args:
            cell_image (np.ndarray): Preprocessed cell image
            
        Returns:
            tuple: (digit, confidence) where digit is int or None
        """
        if not self.easyocr_reader:
            if debug:
                print("    EasyOCR not available")
            return None, 0
        
        try:
            # Run EasyOCR
            results = self.easyocr_reader.readtext(cell_image, allowlist='123456789', width_ths=0.1)
            
            # Extract digit with highest confidence
            best_digit = None
            best_confidence = 0
            
            for (bbox, text, confidence) in results:
                text = text.strip()
                
                # Only accept EXACTLY one digit between 1-9
                if (len(text) == 1 and 
                    text.isdigit() and 
                    '1' <= text <= '9'):
                    confidence_percent = confidence * 100  # Convert to 0-100 scale
                    if confidence_percent > best_confidence:
                        best_digit = int(text)
                        best_confidence = confidence_percent
                        if debug:
                            print(f"    EasyOCR found single digit: '{text}' with confidence {confidence_percent:.1f}")
                elif len(text) == 2 and text.isdigit():
                    # Two digit result - likely a false positive
                    # Check if it's a repeated digit (like "77") or starts with "1" (like "17")
                    if text[0] == text[1]:
                        # Repeated digit like "77" -> use the digit
                        digit = int(text[0])
                        if '1' <= str(digit) <= '9':
                            confidence_percent = confidence * 100 * 0.8  # Reduce confidence slightly
                            if confidence_percent > best_confidence:
                                best_digit = digit
                                best_confidence = confidence_percent
                                if debug:
                                    print(f"    EasyOCR extracted {digit} from repeated '{text}'")
                    elif text.startswith('1') and len(text) == 2:
                        # Cases like "17" where "7" is likely the real digit
                        digit = int(text[1])
                        if '1' <= str(digit) <= '9':
                            confidence_percent = confidence * 100 * 0.7  # Reduce confidence more
                            if confidence_percent > best_confidence:
                                best_digit = digit
                                best_confidence = confidence_percent
                                if debug:
                                    print(f"    EasyOCR extracted {digit} from '1{digit}' pattern")
                elif debug and text:
                    # Log rejected results for debugging
                    print(f"    EasyOCR rejected result: '{text}' (not single digit 1-9)")
            
            return best_digit, best_confidence
            
        except Exception as e:
            if debug:
                print(f"    EasyOCR error: {e}")
            return None, 0
    
    # SUDOKU SOLVING METHODS
    
    def solve_sudoku(self, board: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, bool]:
        """
        Solve a Sudoku puzzle using backtracking algorithm.
        
        Args:
            board (np.ndarray): 9x9 Sudoku board (0 for empty cells)
            debug (bool): If True, shows solving progress
            
        Returns:
            Tuple[np.ndarray, bool]: (solved_board, is_solved)
        """
        if debug:
            print("Starting Sudoku solving process...")
            print("Initial board:")
            self.print_sudoku_board(board)
        
        # Make a copy to avoid modifying the original
        solved_board = board.copy()
        
        # Attempt to solve
        is_solved = self._solve_recursive(solved_board, debug)
        
        if debug:
            if is_solved:
                print("\n✓ Sudoku solved successfully!")
                print("Final solved board:")
                self.print_sudoku_board(solved_board)
            else:
                print("\n✗ Could not solve the Sudoku puzzle")
                print("This might indicate an invalid initial board")
        
        return solved_board, is_solved
    
    def _solve_recursive(self, board: np.ndarray, debug: bool = False) -> bool:
        """Recursive backtracking solver."""
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:  # Empty cell
                    for num in range(1, 10):
                        if self._is_valid_placement(board, row, col, num):
                            board[row][col] = num
                            
                            if debug:
                                print(f"Placing {num} at ({row}, {col})")
                            
                            if self._solve_recursive(board, debug):
                                return True
                            
                            board[row][col] = 0  # Backtrack
                            
                            if debug:
                                print(f"Backtracking from ({row}, {col})")
                    
                    return False  # No valid number found
        return True  # All cells filled
    
    def _is_valid_placement(self, board: np.ndarray, row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid."""
        # Check row
        for x in range(9):
            if board[row][x] == num:
                return False
        
        # Check column
        for x in range(9):
            if board[x][col] == num:
                return False
        
        # Check 3x3 box
        start_row = row - row % 3
        start_col = col - col % 3
        for i in range(3):
            for j in range(3):
                if board[i + start_row][j + start_col] == num:
                    return False
        
        return True
    
##### VALIDATION METHODS
    
    def validate_sudoku_board(self, board: np.ndarray) -> bool:
        """
        Validate if a Sudoku board is correctly solved.
        
        Args:
            board (np.ndarray): 9x9 Sudoku board
            
        Returns:
            bool: True if the board is valid, False otherwise
        """
        # Check rows
        for row in board:
            if len(set(row)) != 9 or 0 in row:
                return False
        
        # Check columns
        for col in range(9):
            column = [board[row][col] for row in range(9)]
            if len(set(column)) != 9 or 0 in column:
                return False
        
        # Check 3x3 boxes
        for box_row in range(3):
            for box_col in range(3):
                box = []
                for i in range(3):
                    for j in range(3):
                        box.append(board[box_row * 3 + i][box_col * 3 + j])
                if len(set(box)) != 9 or 0 in box:
                    return False
        
        return True
    
    
##### UTILITY METHODS

    def calculate_accuracy(self, extracted_board: np.ndarray, master_key: np.ndarray) -> float:
        """Calculate accuracy compared to master key."""
        if extracted_board.shape != master_key.shape:
            return 0.0
        
        correct = 0
        total = 0
        
        for i in range(9):
            for j in range(9):
                total += 1
                if extracted_board[i, j] == master_key[i, j]:
                    correct += 1
        
        return (correct / total * 100) if total > 0 else 0.0
    
    def print_sudoku_board(self, board: np.ndarray) -> None:
        """Print a formatted Sudoku board."""
        print("=" * 25)
        for i, row in enumerate(board):
            if i % 3 == 0 and i != 0:
                print("-" * 25)
            row_str = " | ".join([str(x) if x != 0 else "." for x in row])
            print(f"{row_str}")
        print("=" * 25)
    
    def create_overlay_visualization(self, original_img: np.ndarray, extracted_board: np.ndarray, 
                                   master_key: np.ndarray) -> np.ndarray:
        """Create an overlay visualization showing extracted digits on the original image."""
        # Convert original image to RGB
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        height, width = original_rgb.shape[:2]
        
        # Calculate cell dimensions
        cell_height = height // 9
        cell_width = width // 9
        
        # Create a copy for overlay
        overlay_img = original_rgb.copy()
        
        # Draw grid lines
        for i in range(10):
            # Vertical lines
            cv2.line(overlay_img, (i * cell_width, 0), (i * cell_width, height), (255, 0, 0), 2)
            # Horizontal lines
            cv2.line(overlay_img, (0, i * cell_height), (width, i * cell_height), (255, 0, 0), 2)
        
        # Overlay extracted digits
        for row in range(9):
            for col in range(9):
                # Calculate cell boundaries
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = (col + 1) * cell_width
                y2 = (row + 1) * cell_height
                
                # Check if this digit matches the master key
                is_correct = (master_key[row, col] == extracted_board[row, col])
                # Use RGB color values
                color = (0, 255, 0) if is_correct else (255, 0, 0)  # Green if correct, red if incorrect
                
                # Add semi-transparent overlay
                alpha = 0.3
                overlay_color = np.full((y2-y1, x2-x1, 3), color, dtype=np.uint8)
                if (y2-y1) > 0 and (x2-x1) > 0:  # Ensure valid dimensions
                    overlay_img[y1:y2, x1:x2] = cv2.addWeighted(
                        overlay_img[y1:y2, x1:x2], 1 - alpha, 
                        overlay_color, alpha, 0
                    )
        
        return overlay_img
    
##### HIGH-LEVEL PROCESSING METHODS
    
    def process_image(self, image_path: str, master_keys=None, debug: bool = False, 
                     detect_grid: bool = True) -> Dict:
        """
        Complete processing pipeline for a single Sudoku image.
        
        Args:
            image_path (str): Path to the Sudoku image
            debug (bool): Enable debug output
            detect_grid (bool): Enable grid detection and perspective correction
            
        Returns:
            Dict: Processing results including extracted board, solved board, etc.
        """
        filename = os.path.basename(image_path)
        result = {
            'filename': filename,
            'success': False,
            'error': None,
            'original_image': None,
            'processed_image': None,
            'extracted_board': None,
            'solved_board': None,
            'is_solved': False,
            'accuracy': None,
            'master_key': None
        }
        
        try:
            # Load and preprocess the image
            result['original_image'] = self.read_sudoku_image(image_path, debug=debug)
            result['processed_image'] = self.preprocess_sudoku_image(
                result['original_image'], debug=debug, detect_grid=detect_grid
            )
            
            # Extract the Sudoku grid
            result['extracted_board'] = self.extract_sudoku_grid(
                result['processed_image'], debug=debug
            )
            
            # Attempt to solve
            result['solved_board'], result['is_solved'] = self.solve_sudoku(
                result['extracted_board'], debug=debug
            )
            
            # Calculate accuracy if master key is available
            if master_keys is None:
                master_keys = {}
            if filename in master_keys:
                result['master_key'] = master_keys[filename]
                result['accuracy'] = self.calculate_accuracy(
                    result['extracted_board'], result['master_key']
                )
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            if debug:
                print(f"Error processing {filename}: {e}")
        
        return result
    
    def batch_process(self, images_directory: str = "images", debug_images: List[str] = None,
                     image_extensions: List[str] = None) -> List[Dict]:
        """
        Process multiple Sudoku images in batch.
        
        Args:
            images_directory (str): Directory containing Sudoku images
            debug_images (List[str]): List of filenames to process in debug mode
            image_extensions (List[str]): File extensions to look for
            
        Returns:
            List[Dict]: List of processing results for each image
        """
        if debug_images is None:
            debug_images = []
        
        if image_extensions is None:
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        
        # Get all image files
        image_files = []
        for ext in image_extensions:
            pattern = os.path.join(images_directory, ext)
            image_files.extend(glob.glob(pattern))
        
        image_files = sorted(image_files)
        
        if not image_files:
            print(f"No image files found in '{images_directory}' folder.")
            return []
        
        print(f"Found {len(image_files)} images to process")
        print(f"Debug mode enabled for: {debug_images if debug_images else 'None'}")
        
        results = []
        
        for image_path in image_files:
            filename = os.path.basename(image_path)
            
            # Check if debug mode should be enabled for this image
            debug_mode = filename in debug_images
            debug_status = " (DEBUG MODE)" if debug_mode else ""
            print(f"Processing: {filename}{debug_status}")
            
            result = self.process_image(image_path, debug=debug_mode)
            results.append(result)
            
            if result['success']:
                if result['accuracy'] is not None:
                    print(f"✓ Completed. Accuracy: {result['accuracy']:.1f}%")
                else:
                    print("✓ Completed. No master key available.")
            else:
                print(f"✗ Error: {result['error']}")
        
        return results
    
    def display_results(self, results: List[Dict]) -> None:
        """
        Display visualization of processing results.
        
        Args:
            results (List[Dict]): Results from batch_process or individual process_image calls
        """
        successful_results = [r for r in results if r['success'] and r['master_key'] is not None]
        
        if not successful_results:
            print("No successful extractions with master keys to display.")
            return
        
        print(f"\nDisplaying overlay visualizations for {len(successful_results)} images...")
        
        # Calculate subplot layout
        n_images = len(successful_results)
        n_cols = min(3, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
        
        # Handle single image case
        if n_images == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        # Generate and display overlays
        for idx, result in enumerate(successful_results):
            overlay_img = self.create_overlay_visualization(
                result['processed_image'], 
                result['extracted_board'], 
                result['master_key']
            )
            
            axes[idx].imshow(overlay_img)
            axes[idx].set_title(
                f"{result['filename']}\nAccuracy: {result['accuracy']:.1f}%\n"
                f"(Green=Correct, Red=Incorrect)",
                fontsize=10
            )
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(successful_results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        avg_accuracy = np.mean([r['accuracy'] for r in successful_results])
        print(f"\nSummary: {len(successful_results)} images processed")
        print(f"Average accuracy: {avg_accuracy:.1f}%")


# Example usage:
if __name__ == "__main__":
    # Initialize the solver
    solver = SudokuOCRSolver()
    
    # # Process images in batch
    # results = solver.batch_process(
    #     images_directory="images",
    #     debug_images=[]  # Add filenames here to enable debug mode
    # )
    
    # # Display results
    # solver.display_results(results)
    
    # # Or process a single image
    result = solver.process_image("images/sudoku_grid1.png", debug=True)
    if result['accuracy'] is not None:
        print(f"Extraction accuracy: {result['accuracy']:.1f}%")
    else:
        print("Extraction accuracy: N/A (no master key)")
    print(f"Solved successfully: {result['is_solved']}")

