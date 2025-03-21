# LSB Watermarking with Huffman Encoding

## Author
Christina Bartsch  

## Overview
This project implements a **Least Significant Bit (LSB) watermarking technique** combined with **Huffman encoding** for digital image processing. The program embeds an **invisible watermark** into an image by encoding it within the least significant bits of the image pixels. The watermark can later be extracted and compared to the original.

## Features
- **Embeds a watermark image** within a host image using LSB modification.
- **Applies Huffman encoding** to compress the watermark before embedding.
- **Extracts and reconstructs the watermark** from the modified image.
- **Computes Peak Signal-to-Noise Ratio (PSNR)** to measure distortion after watermarking.
- Uses **Python with NumPy, PIL (Pillow), and Matplotlib** for image processing.

## Dependencies
This project requires the following Python libraries:
- `numpy`
- `Pillow` (PIL)
- `matplotlib`

To install dependencies:
```bash
pip install numpy pillow matplotlib
