import random
import sys
import lzma, zlib

length = 4096

bitstring = b''
hexstring = ""
for i in range(length):
    hexstring += random.choice("0123456789abcdef")
bitstring = bytes.fromhex(hexstring)
print(bitstring.hex())
print(f"\n\nSize of original bitstring: {sys.getsizeof(bitstring)}")
compressed = lzma.compress(bitstring)
print(f"Size of LZMA compressed bitstring: {sys.getsizeof(compressed)}; \
    Compression Ratio: {sys.getsizeof(bitstring) / sys.getsizeof(compressed)}")
compressed = zlib.compress(bitstring)
print(f"Size of ZLIB compressed bitstring: {sys.getsizeof(compressed)}; \
    Compression Ratio: {sys.getsizeof(bitstring) / sys.getsizeof(compressed)}")