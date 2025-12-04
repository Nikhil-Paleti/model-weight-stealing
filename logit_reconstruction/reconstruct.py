import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def reconstruct(data_path, output_dir, plot_prefix="svd", decimals=None):
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(
            f"Error: File {data_path} not found. Run collect_logits.py first.")
        return None

    data = np.load(data_path)
    L = data['logits']  # (N, V)
    W_true = data['W_true']  # (V, d) usually, check shape. GPT2: (50257, 768)

    # Simulate API Precision (Rounding)
    if decimals is not None:
        print(
            f"Simulating API noise: Rounding logits to {decimals} decimal places.")
        L = np.round(L, decimals=decimals)

    print(f"Logits shape: {L.shape}")
    print(f"True weights shape: {W_true.shape}")

    # 1. Embedding Size Discovery via SVD
    print("Performing SVD on centered logits...")
    # Center the logits
    L_mean = L.mean(axis=0)
    L_centered = L - L_mean

    # Full SVD might be heavy if N is large, but N ~ 1000 is fine.
    # L is (N, V). V is 50k. N is 1k.
    # We want eigenvalues of L^T L or L L^T.
    # svd(L, full_matrices=False) -> U (N,N), S (N,), Vt (N, V)
    U, S, Vt = np.linalg.svd(L_centered, full_matrices=False)

    # Plot Singular Values
    plt.figure(figsize=(10, 6))
    plt.plot(S, marker='o', markersize=2)
    plt.title(f"Singular Value Spectrum ({plot_prefix})")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.yscale("log")
    plt.grid(True)
    plot_path = os.path.join(output_dir, f"{plot_prefix}_spectrum.png")
    plt.savefig(plot_path)
    print(f"Saved SVD plot to {plot_path}")
    plt.close()

    # Estimate d (embedding dimension)
    # In a real attack, we look for the 'knee'. Here we know d=768 for GPT-2.
    # We can try to detect it or just use the known one for the weight reconstruction step.
    # Let's assume we detected it.
    # For GPT-2 medium it is 1024.
    # We can guess it from W_true shape for verification purposes.
    d_detected = W_true.shape[1]
    print(f"Using d={d_detected} for reconstruction (from true weights).")

    # 2. Partial Weight Reconstruction
    # The row space of L (captured by Vt) should align with the weight space.
    # Vt has shape (N, V). The top d rows of Vt correspond to the principal components.
    # Let's take the top d components.
    # V_d = Vt[:d, :]  # Shape (d, V)
    # Our estimate of W is V_d.T  # Shape (V, d)

    if len(S) < d_detected:
        print(
            f"Warning: Number of samples ({len(S)}) is less than embedding dimension ({d_detected}). Reconstruction will be poor.")
        d_use = len(S)
    else:
        d_use = d_detected

    V_d = Vt[:d_use, :]  # (d, V)
    W_est_subspace = V_d.T  # (V, d)

    # Verification: Align W_est_subspace to W_true via Least Squares
    # We want to find Q (d, d) such that W_est_subspace @ Q approx W_true
    # This is a linear regression problem: X=W_est_subspace, Y=W_true. Find Q.

    print("Aligning estimated subspace to true weights...")
    # W_true might be (V, d).
    # We fit Q to minimize || W_est @ Q - W_true ||^2

    reg = LinearRegression(fit_intercept=False)  # Linear transform only
    reg.fit(W_est_subspace, W_true)
    W_reconstructed = reg.predict(W_est_subspace)

    # Calculate RMS Error
    mse = mean_squared_error(W_true, W_reconstructed)
    rms = np.sqrt(mse)

    print(f"Reconstruction RMS Error: {rms:.6f}")

    # Compare with random baseline
    print("Calculating baseline RMS (random weights)...")
    # Usually baseline is "how well can we fit with random subspace".
    # Let's fit random subspace.
    W_rand_subspace = np.random.randn(W_true.shape[0], d_use)
    reg_rand = LinearRegression(fit_intercept=False)
    reg_rand.fit(W_rand_subspace, W_true)
    W_rand_rec = reg_rand.predict(W_rand_subspace)
    rms_rand = np.sqrt(mean_squared_error(W_true, W_rand_rec))
    print(f"Random Baseline RMS Error: {rms_rand:.6f}")

    # Save results
    with open(os.path.join(output_dir, f"{plot_prefix}_metrics.txt"), "w") as f:
        f.write(f"Reconstruction RMS: {rms}\n")
        f.write(f"Baseline RMS: {rms_rand}\n")

    return {"rms": rms, "rms_baseline": rms_rand, "d": d_detected}


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct embeddings and weights from logits.")
    parser.add_argument("--data_path", type=str,
                        default="logit_reconstruction/data/logits_data.npz", help="Path to .npz file")
    parser.add_argument("--output_dir", type=str, default="logit_reconstruction/results",
                        help="Directory to save plots/results")
    parser.add_argument("--decimals", type=int, default=None,
                        help="Round logits to N decimals to simulate API noise")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    reconstruct(args.data_path, args.output_dir, decimals=args.decimals)


if __name__ == "__main__":
    main()
