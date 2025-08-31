test_string = "hello! こんにちは!"

utf8_encoded = test_string.encode("utf-8")
utf16_encoded = test_string.encode("utf-16")
utf32_encoded = test_string.encode("utf-32")

print("UTF-8 encoded string:", utf8_encoded, len(utf8_encoded), "bytes")
print("UTF-16 encoded string:", utf16_encoded, len(utf16_encoded), "bytes")
print("UTF-32 encoded string:", utf32_encoded, len(utf32_encoded), "bytes")

# UTF-8 takes 1 to 4 bytes per character
# UTF-16 takes 2 or 4 bytes per character
# UTF-32 takes 4 bytes per character


def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])


def decode_utf8_bytes_to_str(bytestring: bytes):
    return bytestring.decode("utf-8")

print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))

print(decode_utf8_bytes_to_str(test_string.encode("utf-8")))