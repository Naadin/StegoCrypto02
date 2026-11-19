# StegoCrypto02

# 1. Зависимости. Команды запуска

Python 3.10+

```python
cd src
# установка зависимостей
pip install pillow numpy scipy scikit-image cryptography matplotlib
# запуск
python analyze.py
```

# 2. Параметры

```
base = "Криптография"         # встраиваемый текст
password = "Naadin_password"  # пароль для шифрования
fill_percent = 1 / 100.0      # уровень заполнения 1%
fill_percent = 5 / 100.0      # уровень заполнения 5%
```

# 3. Примеры
## Пример обычного LSB

```python
from stego import embed, extract

# встраивание
embed(
    cover_path="cover_road.png",
    stego_path="stego_plain.png",
    base="Криптография",
    mode="plain"
)

# извлечение
restored_text = extract(
    stego_path="stego_plain.png",
    mode="plain" # режим извлечения должен совпадать с режимом встраивания
)
print(restored_text)
```

## Пример гибридного режима

```python
from stego import embed

# встраивание
embed(
    cover_path="cover_road.png",
    stego_path="stego_random.png",
    base="Криптография",
    mode="random",
    password="Naadin_password"
)
```

## Пример извлечения и расшифрования

```python
from stego import extract

# извлечение и расшифровывание
decrypted_text = extract(
    stego_path="stego_random.png",
    mode="random",
    password="Naadin_password"
)
print(decrypted_text)
```
