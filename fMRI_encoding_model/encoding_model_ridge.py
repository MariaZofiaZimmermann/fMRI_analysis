import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr
from himalaya.backend import set_backend
from himalaya.kernel_ridge import MultipleKernelRidgeCV, Kernelizer, ColumnKernelizer
from himalaya.scoring import r2_score_split

# === Setup ===
backend = set_backend("torch_cuda", on_error="warn")

# Base path and subject ID
base_dir = os.path.join(os.environ["STUDY"], "encoding_pjm")
features_dir = os.path.join(base_dir, "features")
bold_dir = os.path.join(base_dir, "bold")
output_dir = os.path.join(base_dir, "results")
os.makedirs(output_dir, exist_ok=True)

# SLURM subject index (or default)
sub_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))
subject = f"sub-deaf{sub_index:02d}"
print(f"\n🎯 Processing {subject}")

# === Settings ===
chunk_size = 20
n_splits = 10
test_chunks_per_split = 6
random_seed = 42
feature_names = ["VERB", "depVERB", "VERB_DIR", "VERB_CL", "NOUN", "depNOUN", "MOTION"]
delays = [1, 2, 3, 4, 5]
max_delay = max(delays)
kernel_feature_names = [f.lower() for f in feature_names]
VERB_GROUP = ["verb", "depverb", "verb_dir", "verb_cl"]
NOUN_GROUP = ["noun", "depnoun"]

# === Load and stack features ===
X_list = []
slices = []
start = 0
for name in feature_names:
    x_path = os.path.join(features_dir, f"{name}.npy")
    X = np.load(x_path)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    X_list.append(X)
    end = start + X.shape[1]
    slices.append(slice(start, end))
    start = end

X_full = np.concatenate(X_list, axis=1)
X_full = X_full[:-max_delay]  # trim for delay

# === Load BOLD ===
y_path = os.path.join(bold_dir, f"{subject}_bold_trimmed.npy")
Y_full = np.load(y_path)
Y_full = Y_full[max_delay:]  # align with delay

# === Cross-validation helper ===
def generate_chunk_splits(n_chunks, test_chunks_per_split=6, n_splits=10, random_seed=42):
    rng = np.random.RandomState(random_seed)
    all_indices = np.arange(n_chunks)
    return [sorted(rng.choice(all_indices, size=test_chunks_per_split, replace=False)) for _ in range(n_splits)]

def chunked_train_test_split(X, Y, chunk_size, test_indices):
    n_chunks = X.shape[0] // chunk_size
    chunks_X = [X[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]
    chunks_Y = [Y[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]
    X_train = np.concatenate([chunks_X[i] for i in range(n_chunks) if i not in test_indices])
    X_test  = np.concatenate([chunks_X[i] for i in test_chunks])
    Y_train = np.concatenate([chunks_Y[i] for i in range(n_chunks) if i not in test_indices])
    Y_test  = np.concatenate([chunks_Y[i] for i in test_chunks])
    return X_train, X_test, Y_train, Y_test

# === Containers ===
r_vals = {name: [] for name in kernel_feature_names}
r2_vals = {name: [] for name in kernel_feature_names}
uv_vals = {name: [] for name in kernel_feature_names}
full_r_all = []
full_r2_all = []
verb_r2_all = []
noun_r2_all = []

# === Splits ===
n_chunks = X_full.shape[0] // chunk_size
split_indices = generate_chunk_splits(n_chunks, test_chunks_per_split, n_splits, random_seed)

for split_i, test_chunks in enumerate(split_indices):
    print(f"  - Split {split_i+1}/{n_splits} using chunks: {test_chunks}")

    X_train, X_test, Y_train, Y_test = chunked_train_test_split(X_full, Y_full, chunk_size, test_chunks)

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_Y = StandardScaler()
    Y_train = scaler_Y.fit_transform(Y_train)
    Y_test = scaler_Y.transform(Y_test)

    # === Full model ===
    kernelizers = [
        (name, make_pipeline(StandardScaler(), Kernelizer(kernel="linear")), slc)
        for name, slc in zip(kernel_feature_names, slices)
    ]
    column_kernelizer = ColumnKernelizer(kernelizers)
    model = MultipleKernelRidgeCV(
        kernels="precomputed",
        solver="random_search",
        solver_params=dict(
            n_iter=80,
            alphas=np.logspace(-2, 6, 30),
            n_targets_batch=200,
            n_alphas_batch=5,
            n_targets_batch_refit=200
        ),
        cv=3
    )
    pipeline = make_pipeline(column_kernelizer, model)
    pipeline.fit(X_train, Y_train)

    Y_pred = backend.to_numpy(pipeline.predict(X_test))
    Y_test_np = backend.to_numpy(Y_test)
    full_r_all.append([pearsonr(Y_pred[:, v], Y_test_np[:, v])[0] for v in range(Y_test_np.shape[1])])
    r2_full = backend.to_numpy(pipeline.score(X_test, Y_test))
    full_r2_all.append(r2_full)

    Y_pred_split = backend.to_numpy(pipeline.predict(X_test, split=True))
    for i, name in enumerate(kernel_feature_names):
        r = np.array([pearsonr(Y_pred_split[i, :, v], Y_test_np[:, v])[0] for v in range(Y_test_np.shape[1])])
        r2 = r2_score_split(Y_test_np, Y_pred_split[i])
        r_vals[name].append(r)
        r2_vals[name].append(r2)

    for i, name in enumerate(kernel_feature_names):
        reduced_kernelizers = [
            (nm, make_pipeline(StandardScaler(), Kernelizer(kernel="linear")), slc)
            for j, (nm, slc) in enumerate(zip(kernel_feature_names, slices))
            if j != i
        ]
        reduced_pipeline = make_pipeline(ColumnKernelizer(reduced_kernelizers), model)
        reduced_pipeline.fit(X_train, Y_train)
        r2_reduced = backend.to_numpy(reduced_pipeline.score(X_test, Y_test))
        uv_vals[name].append(r2_full - r2_reduced)

    # === Group models ===
    def run_group_model(group_names):
        group_kernelizers = [
            (name, make_pipeline(StandardScaler(), Kernelizer(kernel="linear")), slc)
            for name, slc in zip(kernel_feature_names, slices)
            if name in group_names
        ]
        group_pipeline = make_pipeline(ColumnKernelizer(group_kernelizers), model)
        group_pipeline.fit(X_train, Y_train)
        return backend.to_numpy(group_pipeline.score(X_test, Y_test))

    verb_r2_all.append(run_group_model(VERB_GROUP))
    noun_r2_all.append(run_group_model(NOUN_GROUP))

# === Average and Save ===
r_avg = {name: np.mean(r_list, axis=0) for name, r_list in r_vals.items()}
r2_avg = {name: np.mean(r2_list, axis=0) for name, r2_list in r2_vals.items()}
uv_avg = {name: np.mean(uv_list, axis=0) for name, uv_list in uv_vals.items()}
full_r_mean = np.mean(full_r_all, axis=0)
full_r2_mean = np.mean(full_r2_all, axis=0)
verb_r2_mean = np.mean(verb_r2_all, axis=0)
noun_r2_mean = np.mean(noun_r2_all, axis=0)

np.savez(os.path.join(output_dir, f"VERBNOUN6_split_r_{subject}.npz"), **r_avg)
np.savez(os.path.join(output_dir, f"VERBNOUN6_split_r2_{subject}.npz"), **r2_avg)
np.savez(os.path.join(output_dir, f"VERBNOUN6_split_uv_{subject}.npz"), **uv_avg)
np.savez(os.path.join(output_dir, f"VERBNOUN6_full_model_{subject}.npz"), r=full_r_mean, r2=full_r2_mean)
np.savez(os.path.join(output_dir, f"VERBNOUN6_group_r2_{subject}.npz"), verb=verb_r2_mean, noun=noun_r2_mean)

print(f"✅ Saved all results for {subject}")

