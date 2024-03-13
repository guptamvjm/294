import random
import sys
import lzma, zlib

lzma_scores = []
zlib_scores = []
for i in range(1000):
    length = 4096
    bitstring = b''
    hexstring = ""
    for i in range(length):
        hexstring += random.choice("0123456789abcdef")
    bitstring = bytes.fromhex(hexstring)
    # print(bitstring.hex())
    # print(f"\n\nSize of original bitstring: {sys.getsizeof(bitstring)}")
    compressed = lzma.compress(bitstring)
    # print(type(compressed), type(bitstring))
    ratio = sys.getsizeof(bitstring) / sys.getsizeof(compressed)
    # print(f"Size of LZMA compressed bitstring: {sys.getsizeof(compressed)}; \
    #    Compression Ratio: {ratio}")
    lzma_scores.append(ratio)
    compressed = zlib.compress(bitstring)
    ratio = sys.getsizeof(bitstring) / sys.getsizeof(compressed)

    # print(f"Size of ZLIB compressed bitstring: {sys.getsizeof(compressed)}; \
    #     Compression Ratio: {ratio}")
    zlib_scores.append(ratio)

print(f"Average LZMA compression ratio: {sum(lzma_scores) / len(lzma_scores)}")
print(f"Average ZLIB compression ratio: {sum(zlib_scores) / len(zlib_scores)}")