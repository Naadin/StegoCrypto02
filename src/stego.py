import io
import os
import struct
import hashlib
import numpy as np
from PIL import Image
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    return kdf.derive(password.encode("utf-8"))

def encrypt_message(plaintext: str, password: str) -> bytes:
    salt = os.urandom(16)
    nonce = os.urandom(12)
    key = derive_key(password, salt)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    return salt + nonce + ciphertext

def decrypt_message(blob: bytes, password: str) -> str:
    if len(blob) < 44: 
        raise ValueError("Ошибка")
    salt = blob[:16]
    nonce = blob[16:28]
    ciphertext = blob[28:]
    key = derive_key(password, salt)
    aesgcm = AESGCM(key)
    try:
        plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext_bytes.decode("utf-8")
    except (InvalidTag, UnicodeDecodeError) as e:
        raise ValueError("Ошибка") from e

def pack_plain_message(plaintext: str) -> bytes:
    raw = plaintext.encode("utf-8")
    checksum = hashlib.sha256(raw).digest()[:4]
    return checksum + raw

def unpack_plain_message(blob: bytes) -> str:
    if len(blob) < 4:
        raise ValueError("Ошибка")
    expected_checksum = blob[:4]
    raw = blob[4:]
    actual_checksum = hashlib.sha256(raw).digest()[:4]
    if expected_checksum != actual_checksum:
        raise ValueError("Ошибка")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError("Ошибка") from e

def bytes_to_bits(data: bytes) -> np.ndarray:
    byte_arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(byte_arr)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits).tobytes()

def get_positions(total_components: int, password: str = None) -> np.ndarray:
    if password is None:
        return np.arange(total_components, dtype=np.int64)
    seed = int.from_bytes(hashlib.sha256(password.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.permutation(total_components)

def embed(cover_path: str, stego_path, message: str, mode: str = "plain", password: str = "Naadin") -> int:
    img = Image.open(cover_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    flat = arr.flatten()
    capacity_bits = flat.size

    if mode == "plain":
        payload = pack_plain_message(message)
        positions = get_positions(capacity_bits, password=None)
    elif mode == "encrypted":
        payload = encrypt_message(message, password)
        positions = get_positions(capacity_bits, password=None)
    elif mode == "random":
        payload = encrypt_message(message, password)
        positions = get_positions(capacity_bits, password=password)
    else:
        raise ValueError("Ошибка")

    packet = struct.pack(">I", len(payload)) + payload
    bits = bytes_to_bits(packet)

    if bits.size > capacity_bits:
        raise ValueError(f"Ошибка")

    target_indices = positions[: bits.size]
    flat[target_indices] = (flat[target_indices] & 0xFE) | bits

    stego_arr = flat.reshape(arr.shape)
    Image.fromarray(stego_arr, mode="RGB").save(stego_path, format="PNG")
    return bits.size

def extract(stego_path, mode: str = "plain", password: str = "Naadin") -> str:
    try:
        img = Image.open(stego_path).convert("RGB")
    except Exception as e:
        raise ValueError("Ошибка") from e

    flat = np.array(img, dtype=np.uint8).flatten()
    capacity_bits = flat.size

    if capacity_bits < 32:
        raise ValueError("Ошибка")

    positions = get_positions(capacity_bits, password=password if mode == "random" else None)

    try:
        len_bits = flat[positions[:32]] & 1
        payload_len = struct.unpack(">I", bits_to_bytes(len_bits))[0]
    except struct.error as e:
        raise ValueError("Ошибка") from e

    total_bits_needed = 32 + payload_len * 8
    if payload_len == 0 or total_bits_needed > capacity_bits:
        raise ValueError("Ошибка")

    payload_bits = flat[positions[32:total_bits_needed]] & 1
    payload = bits_to_bytes(payload_bits)

    if mode == "plain":
        return unpack_plain_message(payload)
    elif mode in ("encrypted", "random"):
        return decrypt_message(payload, password)
    else:
        raise ValueError("Ошибка")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    imgs_dir = os.path.join(base_dir, "imgs")

    candidates = [os.path.join(imgs_dir, "cover.png")] + [
        os.path.join(imgs_dir, f) for f in sorted(os.listdir(imgs_dir)) if f.startswith("cover_") and f.endswith(".png")
    ]
    test_cover = next((p for p in candidates if os.path.exists(p)), None)

    if test_cover:
        msg = "Сообщение"
        for m in ["plain", "encrypted", "random"]:
            buf = io.BytesIO()
            embed(test_cover, buf, msg, mode=m, password="Naadin")
            buf.seek(0)
            restored = extract(buf, mode=m, password="Naadin")
            assert msg == restored, f"Ошибка: {m}"