import io
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import chisquare
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from stego import embed, extract


def calc_chi_square(flat_arr: np.ndarray):
    hist = np.bincount(flat_arr, minlength=256)
    even = hist[0::2].astype(np.float64)
    odd = hist[1::2].astype(np.float64)
    expected = (even + odd) / 2.0

    mask = expected > 5
    if np.sum(mask) < 2:
        return 0.0, 1.0

    obs = np.concatenate([even[mask], odd[mask]])
    exp = np.concatenate([expected[mask], expected[mask]])

    ddof = int(np.sum(mask))
    stat, p_val = chisquare(f_obs=obs, f_exp=exp, ddof=ddof)
    return float(stat), float(p_val)


def evaluate_image(cover_arr: np.ndarray, stego_arr: np.ndarray):
    cover_flat = cover_arr.flatten()
    stego_flat = stego_arr.flatten()

    if np.array_equal(cover_arr, stego_arr):
        psnr_val = float("inf")
        ssim_val = 1.0
    else:
        psnr_val = float(peak_signal_noise_ratio(cover_arr, stego_arr, data_range=255))
        ssim_val = float(structural_similarity(cover_arr, stego_arr, channel_axis=2, data_range=255))

    changed_count = int(np.sum(cover_flat != stego_flat))
    total_components = int(cover_flat.size)
    changed_ratio = changed_count / total_components

    lsb = stego_flat & 1
    p_lsb_1 = float(np.mean(lsb))
    p_lsb_0 = 1.0 - p_lsb_1

    chi2_stat, p_value = calc_chi_square(stego_flat)

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "changed_count": changed_count,
        "total_components": total_components,
        "changed_ratio": changed_ratio,
        "p_lsb_0": p_lsb_0,
        "p_lsb_1": p_lsb_1,
        "chi2": chi2_stat,
        "p_value": p_value,
    }


def generate_payload_text(target_bytes: int) -> str:
    base = "Криптография"
    encoded = base.encode("utf-8")
    repeats = (target_bytes // len(encoded)) + 2
    full_bytes = (encoded * repeats)[:target_bytes]
    return full_bytes.decode("utf-8", errors="ignore")


def plot_visualizations(cover_arr: np.ndarray, stego_arr: np.ndarray, cover_label: str, out_dir: str):
    diff = np.abs(cover_arr.astype(np.int16) - stego_arr.astype(np.int16)).astype(np.uint8)
    diff_amplified = diff * 255

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cover_arr)
    plt.title(f"Исходное изображение ({cover_label})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(diff_amplified)
    plt.title("Карта разности")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "difference.png"), dpi=150)
    plt.close()

    cover_flat = cover_arr.flatten()
    stego_flat = stego_arr.flatten()

    hist_cover = np.bincount(cover_flat, minlength=256).astype(np.float64)
    hist_stego = np.bincount(stego_flat, minlength=256).astype(np.float64)

    max_val = max(hist_cover.max(), hist_stego.max())
    hist_cover_pct = (hist_cover / max_val) * 100.0
    hist_stego_pct = (hist_stego / max_val) * 100.0

    x = np.arange(256)
    plt.figure(figsize=(10, 5))
    plt.fill_between(x, hist_cover_pct, color="royalblue", alpha=0.45, label="Cover")
    plt.plot(x, hist_cover_pct, color="royalblue", linewidth=1)

    plt.fill_between(x, hist_stego_pct, color="crimson", alpha=0.35, label="Stego Plain LSB")
    plt.plot(x, hist_stego_pct, color="crimson", linewidth=1)

    plt.title("Сравнительная гистограмма распределения яркости (0–255)")
    plt.xlabel("Яркость пикселя (0–255)")
    plt.ylabel("Относительная частота (%)")
    plt.xlim(0, 255)
    plt.ylim(0, 105)
    plt.xticks((0, 64, 128, 192, 255))
    plt.yticks((0, 50, 100), ("0%", "50%", "100%"))
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "histogram.png"), dpi=150)
    plt.close()


def plot_three_methods_diff(cover_arr: np.ndarray, imgs_dir: str, out_dir: str):
    modes = [
        ("stego_plain.png", "Plain LSB"),
        ("stego_encrypted.png", "Encrypted LSB"),
        ("stego_random.png", "Encrypted Random LSB")
    ]

    plt.figure(figsize=(15, 5))
    for i, (fname, title) in enumerate(modes, 1):
        stego_path = os.path.join(imgs_dir, fname)
        stego_arr = np.array(Image.open(stego_path).convert("RGB"), dtype=np.uint8)
        diff_amp = np.abs(cover_arr.astype(np.int16) - stego_arr.astype(np.int16)).astype(np.uint8) * 255

        plt.subplot(1, 3, i)
        plt.imshow(diff_amp)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_diff.png"), dpi=150)
    plt.close()


def discover_images(imgs_dir: str) -> list:
    if os.path.exists(os.path.join(imgs_dir, "cover_road.png")):
        return ["cover_road.png"]
    if os.path.exists(os.path.join(imgs_dir, "cover.png")):
        return ["cover.png"]

    all_pngs = sorted(os.listdir(imgs_dir))
    covers = [
        f for f in all_pngs
        if f.endswith(".png") and not f.startswith("stego_") and not f.startswith("test_")
    ]
    if not covers:
        raise FileNotFoundError("Ошибка")
    return [covers[0]]


def run_all_experiments():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    imgs_dir = os.path.join(base_dir, "imgs")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    password = "Naadin_password"
    images_to_test = discover_images(imgs_dir)
    primary_image = images_to_test[0]
    csv_rows = []

    for img_name in images_to_test:
        cover_path = os.path.join(imgs_dir, img_name)
        cover_arr = np.array(Image.open(cover_path).convert("RGB"), dtype=np.uint8)
        h, w, _ = cover_arr.shape
        total_capacity_bytes = cover_arr.size // 8

        print(f"Изображение: {img_name} ({w}x{h})")
        print(f"Всего компонент: {cover_arr.size}, вместимость: {total_capacity_bytes} байт")

        for fill_percent in (1, 5):
            payload_bytes = int(total_capacity_bytes * (fill_percent / 100.0))
            secret_text = generate_payload_text(payload_bytes)

            print(f"\n--- Уровень заполнения: {fill_percent}% (~{payload_bytes} байт) ---")
            print("| Вариант | PSNR (дБ) | SSIM | Изменено | Changed ratio | P(LSB=0) | P(LSB=1) | Chi2 | p-value |")
            print("|---|---|---|---|---|---|---|---|---|")

            m_cover = evaluate_image(cover_arr, cover_arr)
            print(
                f"| Cover | inf | 1.000000 | 0 | 0.000000 | "
                f"{m_cover['p_lsb_0']:.4f} | {m_cover['p_lsb_1']:.4f} | "
                f"{m_cover['chi2']:.2f} | {m_cover['p_value']:.6f} |"
            )
            csv_rows.append([
                img_name, f"{fill_percent}%", "Cover", "inf", 1.0, 0, 0.0,
                m_cover['p_lsb_0'], m_cover['p_lsb_1'], m_cover['chi2'], m_cover['p_value']
            ])

            for mode, label in [("plain", "Plain LSB"), ("encrypted", "Encrypted LSB"), ("random", "Random LSB")]:
                save_to_disk = (img_name == primary_image and fill_percent == 1)
                target = os.path.join(imgs_dir, f"stego_{mode}.png") if save_to_disk else io.BytesIO()

                embed(cover_path, target, secret_text, mode=mode, password=password)
                if not save_to_disk:
                    target.seek(0)

                extracted = extract(target, mode=mode, password=password)
                assert extracted == secret_text, f"Ошибка {mode}!"

                if not save_to_disk:
                    target.seek(0)
                stego_arr = np.array(Image.open(target).convert("RGB"), dtype=np.uint8)
                m = evaluate_image(cover_arr, stego_arr)

                print(
                    f"| {label} | {m['psnr']:.2f} | {m['ssim']:.6f} | {m['changed_count']} | "
                    f"{m['changed_ratio']:.6f} | {m['p_lsb_0']:.4f} | {m['p_lsb_1']:.4f} | "
                    f"{m['chi2']:.2f} | {m['p_value']:.6f} |"
                )
                csv_rows.append([
                    img_name, f"{fill_percent}%", label, round(m['psnr'], 4), round(m['ssim'], 6),
                    m['changed_count'], round(m['changed_ratio'], 6), round(m['p_lsb_0'], 4),
                    round(m['p_lsb_1'], 4), round(m['chi2'], 2), round(m['p_value'], 6)
                ])

                if save_to_disk and mode == "plain":
                    plot_visualizations(cover_arr, stego_arr, primary_image, results_dir)

    primary_arr = np.array(Image.open(os.path.join(imgs_dir, primary_image)).convert("RGB"), dtype=np.uint8)
    plot_three_methods_diff(primary_arr, imgs_dir, results_dir)

    csv_path = os.path.join(results_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Image", "Payload_Level", "Method", "PSNR", "SSIM",
            "Changed_Count", "Changed_Ratio", "P_LSB_0", "P_LSB_1", "Chi2", "P_Value"
        ])
        writer.writerows(csv_rows)

if __name__ == "__main__":
    run_all_experiments()