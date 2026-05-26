# BPSK, QPSK, 8PSK, 16QAM, 64QAM synthetic I/Q 데이터를 생성하는 코드.

import argparse
import os

import numpy as np


MODULATION_PRESETS = {
    4: ["BPSK", "QPSK", "8PSK", "16QAM"],
    5: ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"],
}
DEFAULT_MODULATIONS = MODULATION_PRESETS[4]


def generate_distorted_symbols(modulation_type="QPSK", num_symbols=1024, snr_db=20):
    if modulation_type == "BPSK":
        symbols = np.sign(np.random.randn(num_symbols)).astype(complex)
    elif modulation_type == "QPSK":
        symbols = (
            np.sign(np.random.randn(num_symbols))
            + 1j * np.sign(np.random.randn(num_symbols))
        ) / np.sqrt(2)
    elif modulation_type == "8PSK":
        bits = np.random.randint(0, 8, num_symbols)
        symbols = np.exp(1j * 2 * np.pi * bits / 8)
    elif modulation_type == "16QAM":
        m = np.array([-3, -1, 1, 3])
        i_comp = np.random.choice(m, num_symbols)
        q_comp = np.random.choice(m, num_symbols)
        symbols = (i_comp + 1j * q_comp) / np.sqrt(10)
    elif modulation_type == "64QAM":
        m = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        i_comp = np.random.choice(m, num_symbols)
        q_comp = np.random.choice(m, num_symbols)
        symbols = (i_comp + 1j * q_comp) / np.sqrt(42)
    elif modulation_type == "256QAM":
        m = np.array([-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15])
        i_comp = np.random.choice(m, num_symbols)
        q_comp = np.random.choice(m, num_symbols)
        symbols = (i_comp + 1j * q_comp) / np.sqrt(170)
    else:
        raise ValueError(f"Unsupported modulation: {modulation_type}")

    snr_linear = 10 ** (snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr_linear))
    noise = noise_std * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))
    rx_symbols = symbols + noise

    t = np.arange(num_symbols)
    freq_offset = 0.0
    rx_symbols *= np.exp(1j * 2 * np.pi * freq_offset * t)

    phase_noise_std = 0.0
    rx_symbols *= np.exp(1j * np.random.normal(0, phase_noise_std, num_symbols))

    return rx_symbols


def generate_dataset(modulations, snrs, samples_per_mod_snr, seq_length):
    data_list = []
    label_list = []
    snr_label_list = []

    for m_idx, mod in enumerate(modulations):
        for snr in snrs:
            for _ in range(samples_per_mod_snr):
                signal = generate_distorted_symbols(mod, seq_length, snr)
                iq_data = np.stack([signal.real, signal.imag], axis=0)
                data_list.append(iq_data)
                label_list.append(m_idx)
                snr_label_list.append(snr)

    x_data = np.array(data_list, dtype=np.float32)
    y_label = np.array(label_list, dtype=np.int64)
    snr_label = np.array(snr_label_list, dtype=np.int64)
    return x_data, y_label, snr_label


def parse_csv_list(value, item_type=str):
    return [item_type(item.strip()) for item in value.split(",") if item.strip()]


def save_array(path, array):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.save(path, array)


def stratified_train_val_split(x_data, y_label, snr_label=None, val_ratio=0.2):
    train_indices = []
    val_indices = []

    if snr_label is None:
        group_keys = [(label, None) for label in np.unique(y_label)]
    else:
        group_keys = [
            (label, snr)
            for label in np.unique(y_label)
            for snr in np.unique(snr_label)
        ]

    for label, snr in group_keys:
        if snr is None:
            group_indices = np.where(y_label == label)[0]
        else:
            group_indices = np.where((y_label == label) & (snr_label == snr))[0]
        if len(group_indices) == 0:
            continue
        np.random.shuffle(group_indices)
        val_count = max(1, int(len(group_indices) * val_ratio))
        val_indices.extend(group_indices[:val_count])
        train_indices.extend(group_indices[val_count:])

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    return (
        x_data[train_indices],
        y_label[train_indices],
        None if snr_label is None else snr_label[train_indices],
        x_data[val_indices],
        y_label[val_indices],
        None if snr_label is None else snr_label[val_indices],
    )


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic I/Q modulation datasets.")
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--samples-per-mod-snr", type=int, default=3000)
    parser.add_argument("--modulations", default=",".join(DEFAULT_MODULATIONS))
    parser.add_argument(
        "--num-modulations",
        type=int,
        choices=sorted(MODULATION_PRESETS),
        default=None,
        help="Use 4 for BPSK/QPSK/8PSK/16QAM, or 5 to add 64QAM.",
    )
    parser.add_argument("--snrs", default="20")
    parser.add_argument("--x-out", default="X_data.npy")
    parser.add_argument("--y-out", default="Y_label.npy")
    parser.add_argument("--snr-label-out", default=None)
    parser.add_argument("--save-full", action="store_true")
    parser.add_argument("--make-train-val", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--train-x-out", default=None)
    parser.add_argument("--train-y-out", default=None)
    parser.add_argument("--train-snr-out", default=None)
    parser.add_argument("--val-x-out", default=None)
    parser.add_argument("--val-y-out", default=None)
    parser.add_argument("--val-snr-out", default=None)
    parser.add_argument("--make-snr-tests", action="store_true")
    parser.add_argument("--test-snr-dir", default="test_data_snr")
    parser.add_argument("--test-snrs", default="-10,-5,0,5,10,15,20")
    parser.add_argument("--test-samples-per-mod-snr", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--preset",
        choices=["default", "ssl4096"],
        default="default",
        help="Use ssl4096 for 4096-length 5-class data saved as X_data_4096/Y_label_4096.",
    )
    args = parser.parse_args()

    if args.preset == "ssl4096":
        args.seq_length = 4096
        args.num_modulations = 5
        if args.snrs == "20":
            args.snrs = "0,5,10,15,20"
        if args.samples_per_mod_snr == 3000:
            args.samples_per_mod_snr = 2000
        if args.x_out == "X_data.npy":
            args.x_out = "X_data_4096.npy"
        if args.y_out == "Y_label.npy":
            args.y_out = "Y_label_4096.npy"
        args.snr_label_out = args.snr_label_out or "SNR_label_4096.npy"
        args.train_x_out = args.train_x_out or "X_train_4096.npy"
        args.train_y_out = args.train_y_out or "Y_train_4096.npy"
        args.train_snr_out = args.train_snr_out or "SNR_train_4096.npy"
        args.val_x_out = args.val_x_out or "X_val_4096.npy"
        args.val_y_out = args.val_y_out or "Y_val_4096.npy"
        args.val_snr_out = args.val_snr_out or "SNR_val_4096.npy"

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.num_modulations is not None:
        modulations = MODULATION_PRESETS[args.num_modulations]
    else:
        modulations = parse_csv_list(args.modulations, str)
    snrs = parse_csv_list(args.snrs, int)

    print("Generating dataset...")
    print(f"  modulations: {modulations}")
    print(f"  snrs: {snrs}")
    print(f"  samples/mod/snr: {args.samples_per_mod_snr}")
    print(f"  seq_length: {args.seq_length}")

    x_data, y_label, snr_label = generate_dataset(
        modulations=modulations,
        snrs=snrs,
        samples_per_mod_snr=args.samples_per_mod_snr,
        seq_length=args.seq_length,
    )

    print(f"Done. Total samples: {len(x_data)}")

    save_full = args.save_full or not args.make_train_val
    if save_full:
        save_array(args.x_out, x_data)
        save_array(args.y_out, y_label)
        if args.snr_label_out:
            save_array(args.snr_label_out, snr_label)
        print(f"X shape: {x_data.shape} -> {args.x_out}")
        print(f"Y shape: {y_label.shape} -> {args.y_out}")
        if args.snr_label_out:
            print(f"SNR label shape: {snr_label.shape} -> {args.snr_label_out}")
    else:
        print("Full dataset file saving skipped. Use --save-full to save X/Y/SNR full arrays.")

    if args.make_train_val:
        if not all([args.train_x_out, args.train_y_out, args.val_x_out, args.val_y_out]):
            base = os.path.splitext(args.x_out)[0]
            label_base = os.path.splitext(args.y_out)[0]
            args.train_x_out = args.train_x_out or f"{base}_train.npy"
            args.train_y_out = args.train_y_out or f"{label_base}_train.npy"
            args.val_x_out = args.val_x_out or f"{base}_val.npy"
            args.val_y_out = args.val_y_out or f"{label_base}_val.npy"

        x_train, y_train, snr_train, x_val, y_val, snr_val = stratified_train_val_split(
            x_data,
            y_label,
            snr_label=snr_label,
            val_ratio=args.val_ratio,
        )
        save_array(args.train_x_out, x_train)
        save_array(args.train_y_out, y_train)
        save_array(args.val_x_out, x_val)
        save_array(args.val_y_out, y_val)
        if args.train_snr_out and args.val_snr_out:
            save_array(args.train_snr_out, snr_train)
            save_array(args.val_snr_out, snr_val)
        print(f"Train shape: {x_train.shape}, {y_train.shape} -> {args.train_x_out}, {args.train_y_out}")
        print(f"Val shape: {x_val.shape}, {y_val.shape} -> {args.val_x_out}, {args.val_y_out}")
        if args.train_snr_out and args.val_snr_out:
            print(f"Train/Val SNR labels -> {args.train_snr_out}, {args.val_snr_out}")

    if args.make_snr_tests:
        test_snrs = parse_csv_list(args.test_snrs, int)
        print(f"Generating SNR test sets in {args.test_snr_dir}...")
        for snr in test_snrs:
            x_test, y_test, snr_test = generate_dataset(
                modulations=modulations,
                snrs=[snr],
                samples_per_mod_snr=args.test_samples_per_mod_snr,
                seq_length=args.seq_length,
            )
            x_path = os.path.join(args.test_snr_dir, f"X_test_snr_{args.seq_length}_{snr}.npy")
            y_path = os.path.join(args.test_snr_dir, f"Y_test_snr_{args.seq_length}_{snr}.npy")
            s_path = os.path.join(args.test_snr_dir, f"SNR_test_snr_{args.seq_length}_{snr}.npy")
            save_array(x_path, x_test)
            save_array(y_path, y_test)
            save_array(s_path, snr_test)
            print(f"  SNR {snr:3d} dB: {x_test.shape} -> {x_path}")


if __name__ == "__main__":
    main()
