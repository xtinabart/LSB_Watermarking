import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Huffman Node Creation
class Node:
    def __init__(self, symbol, frequency):
        self.symbol = symbol
        self.frequency = frequency
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.frequency < other.frequency  # Compare frequencies for priority queue

# Priority Queue Implementation
class PriorityQueue:
    def __init__(self):
        self.queue = []

    def put(self, node):
        self.queue.append(node)
        self.queue.sort(key=lambda x: x.frequency)

    def get(self):
        return self.queue.pop(0)

    def qsize(self):
        return len(self.queue)

# Frequency Count
def freq_count(data):
    frequency = {}
    for symbol in data:
        if symbol in frequency:
            frequency[symbol] += 1
        else:
            frequency[symbol] = 1
    return frequency

# Huffman Tree Construction
def huffman_tree(data):
    freq = freq_count(data)
    pq = PriorityQueue()

    for symbol, count in freq.items():
        pq.put(Node(symbol, count))

    while pq.qsize() > 1:
        left = pq.get()
        right = pq.get()
        merged = Node(None, left.frequency + right.frequency)
        merged.left = left
        merged.right = right
        pq.put(merged)

    return pq.get()

# Generate Huffman Codes
def generate_codes(root, prefix="", codes=None):
    if codes is None:
        codes = {}
    if root is None:
        return codes

    if root.symbol is not None:  # Leaf node
        codes[root.symbol] = prefix
    else:
        generate_codes(root.left, prefix + "0", codes)
        generate_codes(root.right, prefix + "1", codes)

    return codes

# Encode Data Using Huffman Codes
def encode(data):
    root = huffman_tree(data)
    codes = generate_codes(root)
    encoded_data = ""
    for symbol in data:
        encoded_data += codes[symbol]
    return encoded_data, codes, root

# Decode Encoded Data
def decode(encoded_data, root):
    decoded = []
    curr = root

    for bit in encoded_data:
        curr = curr.left if bit == "0" else curr.right
        if curr.symbol is not None:
            decoded.append(curr.symbol)
            curr = root

    return decoded


# Calculate PSNR
def calculate_psnr(original, reconstructed):
    original_array = np.array(original, dtype=np.float64) # Convert to array for calculation
    reconstructed_array = np.array(reconstructed, dtype=np.float64)
    mse = np.mean((original_array - reconstructed_array) ** 2) # Mean squared error
    if mse == 0:  # Case if there is no difference
        return "Perfect Match! (infinity)"
    max_pixel_value = 255.0
    psnr = 10 * math.log10((max_pixel_value ** 2) / mse)
    return psnr

# Main Program
if __name__ == "__main__":
    # Load watermark image
    watermark_image = Image.open("DIP Icon.jpg").convert("L")
    watermark_data = np.array(watermark_image).flatten().tolist()  # Flatten to 1D

    # Encode watermark
    encoded_binary, huffman_codes, huffman_root = encode(watermark_data)
    watermark_size = len(encoded_binary)

    # Load Baboon image as grayscale
    baboon_image = Image.open("baboon.png").convert("L")
    baboon_data = np.array(baboon_image)
    height, width = baboon_data.shape
    flat_baboon = baboon_data.flatten()
    flat_baboon = flat_baboon.astype(np.uint8)

    # Embed watermark size in the first 64 bits
    watermark_size_binary = f"{watermark_size:064b}"
    for i in range(64):
        flat_baboon[i] = (flat_baboon[i] & 254) | int(watermark_size_binary[i])

    # Embed encoded watermark starting from the last pixel
    for i, bit in enumerate(encoded_binary):
        index = -1 - i
        flat_baboon[index] = (flat_baboon[index] & 254) | int(bit)

    # Reshape into original dimensions
    watermarked_baboon = flat_baboon.reshape((height, width))
    watermarked_image = Image.fromarray(watermarked_baboon.astype(np.uint8))
    watermarked_image.save("Watermarked_Baboon.png")

    # Extract watermark size from first 64 bits
    extracted_size_binary = "".join(str(flat_baboon[i] & 1) for i in range(64))
    extracted_size = int(extracted_size_binary, 2)

    # Extract encoded watermark
    extracted_binary = "".join(str(flat_baboon[-1 - i] & 1) for i in range(extracted_size))

    # Decode extracted watermark
    decoded_data = decode(extracted_binary, huffman_root)

    # Ensure correct size and reshape
    expected_size = watermark_image.size[0] * watermark_image.size[1]
    decoded_data = (
        [0] * expected_size if len(decoded_data) < expected_size else decoded_data[:expected_size]
    )
    decoded_watermark = np.array(decoded_data).reshape(watermark_image.size[::-1])

    # Display images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(baboon_image, cmap="gray")  # Grayscale for original baboon
    axes[0].set_title("Original Baboon Image")
    axes[0].axis("off")

    axes[1].imshow(watermarked_baboon, cmap="gray")  # Grayscale for watermarked image
    axes[1].set_title("Watermarked Baboon Image")
    axes[1].axis("off")

    axes[2].imshow(decoded_watermark, cmap="gray")  # Grayscale for extracted watermark
    axes[2].set_title("Extracted Watermark")
    axes[2].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

    # Display PSNR value
    original_watermark = np.array(watermark_image)
    psnr_value = calculate_psnr(original_watermark, decoded_watermark)
    if isinstance(psnr_value, str):
        print(f"PSNR between original and extracted watermark: {psnr_value}")
    else:
        print(f"PSNR between original and extracted watermark: {psnr_value:.2f} dB")

