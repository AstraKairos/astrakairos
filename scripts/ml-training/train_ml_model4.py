"""
    Script for training a LightGBM model to predict log(R_chance_align) from El-Badry's catalog.
    Based on El-Badry et al. 2021.

    A couple of notes (post-training):
    - The sample weighting scheme is AGGRESSIVE to improve mid-range R accuracy.
    - The model hyperparameters were tuned via manual experimentation to balance accuracy
      across the entire R range, with emphasis on mid-range R values.
    - The objective is regression of log10(R_chance_align), with a floor at -10. It's NOT
      a classification model.
    - We do regression of log(R) as this allows us to capture the wide dynamic range of R 
    (from ~0 to ~a lot more than 1). - As explained in the paper, R is not strictly a probability.
    - The higher cross-contamination rates between R=~0.2 and R=~0.8 are most likely due to the
    catalog not having enough systems in this range. Artificially decreasing the sample of systems
    left out of training (for validation) in this range may decrease the cross-contamination by a few % points.
    - We prioritized precision (within 0.01, 0.05, and 0.1 units of R, and R > 0.5 for "true" R values greater than 1) 
    over overall accuracy.
    - We attempted using the original 7D features as well as the scaled ones. Both results ended up being very similar,
    so we opted for the scaled features as they match the original paper.

    Some stats from training:
    - Precision (pred bound -> "true" bound (R < 0.5)): 0.997
    - Recall (true bound -> pred bound): 0.997
    - F1 Score: 0.997
    - Confusion: TP=261085, FP=698, TN=101050, FN=686
    - The accuracy for the range R > 2 is very low, however, the model still correctly predicts
    an R value greater than 0.5 for a vast majority of these systems.

    For the homies, we call this, "model 4" (although it took a lot more than 4 attempts).
    Models 2 and 3 showed slightly better accuracy overall, but had noticeably worse precision.
    We may consider open-sourcing these other models in the future, for the sake of transparency.
"""

import numpy as np
from astropy.io import fits
from scipy.special import erf
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "el-badry.fits"
MODEL_PATH = BASE_DIR / "models" / "r_chance_align_model.joblib"


def compute_scaled_features(data):
    """
    Compute 7 scaled features matching El-Badry et al. 2021, Table A1.
    
    Features:
        1. log(θ) - log angular separation
        2. 4/ϖ₁ - scaled inverse parallax
        3. |Δϖ|/σ_Δϖ - normalized parallax difference
        4. 4·log(Σ₁₈) - scaled log density
        5. v⊥,1/50 - scaled tangential velocity
        6. 2·erf[(Δμ - Δμ_orbit)/σ_Δμ] - scaled PM difference
        7. σ_Δϖ - parallax error (raw, for context)
    """
    # Extract raw values from the fits file
    theta = data['pairdistance'] * 3600  # degrees → arcsec
    parallax1 = data['parallax1']  # mas
    parallax2 = data['parallax2']  # mas
    parallax_error1 = data['parallax_error1']  # mas
    parallax_error2 = data['parallax_error2']  # mas
    sigma18 = data['Sigma18']  # deg^-2
    pmra1 = data['pmra1']  # mas/yr
    pmra2 = data['pmra2']  # mas/yr
    pmdec1 = data['pmdec1']  # mas/yr
    pmdec2 = data['pmdec2']  # mas/yr
    pmra_error1 = data['pmra_error1']  # mas/yr
    pmra_error2 = data['pmra_error2']  # mas/yr
    pmdec_error1 = data['pmdec_error1']  # mas/yr
    pmdec_error2 = data['pmdec_error2']  # mas/yr


    # Input features

    # Feature 1: log(θ) (Angular separation)
    f1 = np.log10(theta)

    # Feature 2: 4/ϖ₁ (Parallax (primary) ~ distance in kpc)
    f2 = 4.0 / parallax1

    # Feature 3: |Δϖ|/σ_Δϖ
    delta_parallax = np.abs(parallax1 - parallax2)
    sigma_delta_parallax = np.sqrt(parallax_error1**2 + parallax_error2**2)
    f3 = delta_parallax / sigma_delta_parallax

    # Feature 4: 4·log(Σ₁₈)
    f4 = 4.0 * np.log10(sigma18)

    # Feature 5: v⊥,1/50 (tangential velocity scaled)
    mu1 = np.sqrt(pmra1**2 + pmdec1**2)
    v_tan1 = 4.74047 * mu1 / parallax1
    f5 = v_tan1 / 50.0

    # Feature 6: 2·erf[(Δμ - Δμ_orbit)/σ_Δμ] (Scaled proper motion difference)
    delta_pmra = pmra1 - pmra2
    delta_pmdec = pmdec1 - pmdec2
    delta_mu = np.sqrt(delta_pmra**2 + delta_pmdec**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_delta_mu = (1.0 / delta_mu) * np.sqrt(
            (pmra_error1**2 + pmra_error2**2) * delta_pmra**2 +
            (pmdec_error1**2 + pmdec_error2**2) * delta_pmdec**2
        )
        sigma_delta_mu = np.where(delta_mu > 0, sigma_delta_mu, 1.0)
    
    delta_mu_orbit = 0.44 * (parallax1**1.5) * (theta**-0.5)
    pm_ratio = (delta_mu - delta_mu_orbit) / sigma_delta_mu
    f6 = 2.0 * erf(pm_ratio)

    # Feature 7: σ_Δϖ (Parallax difference error)
    f7 = sigma_delta_parallax

    features = np.column_stack([f1, f2, f3, f4, f5, f6, f7])
    
    return features


def compute_sample_weights(R):
    """
    Compute sample weights using inverse frequency (N_total / N_bin) with
    additional boost for bins with lowest accuracy.
    """
    # Granular bins for balanced learning
    # Note that R can be >1, so we need bins that go beyond 1. The last bin is [2, inf) to capture all high-R systems.
    # This will results in poor accuracy and precision for R > 2, but our results show that
    # although the accuracy is poor for this range, the model still correctly predicts an R value
    # greater than 0.5 (and even 0.9), which is the most important thing for our application.
    bin_edges = [0, 1e-6, 1e-4, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 
                 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0, 1.01, 1.1, 1.2, 1.5, 2.0, np.inf]
    
    # Additional boost for bins with worst accuracy (optimized via Optuna)
    # These are on top of inverse frequency weighting
    accuracy_boost = {
        (0.3, 0.4): 6.98,   # Optuna-optimized
        (0.4, 0.5): 1.87,   # Optuna-optimized
        (0.5, 0.6): 5.61,   # Optuna-optimized
        (0.6, 0.7): 12.63,  # Optuna-optimized (highest boost)
        (0.7, 0.8): 7.44,   # Optuna-optimized
    }
    
    N_total = len(R)
    weights = np.ones(len(R))
    
    print("\nBin distribution and weights (inverse frequency + accuracy boost):")
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (R >= lo) & (R < hi)
        count = np.sum(mask)
        if count > 0:
            # Inverse frequency: N_total / N_bin
            inv_freq = N_total / count
            boost = accuracy_boost.get((lo, hi), 1.0)
            weights[mask] = inv_freq * boost
            boost_str = f" (boost={boost}x)" if boost > 1 else ""
            print(f"  [{lo:.2e}, {hi:.2e}): {count:>10,} samples, weight={inv_freq * boost:.1f}{boost_str}")
    
    # Normalize weights to have mean = 1
    weights = weights / np.mean(weights)
    
    return weights


def main():
    print("=" * 60)
    print("Training LightGBM model for R_chance_align prediction")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    with fits.open(DATA_PATH) as hdul:
        data = hdul[1].data
        R = data['R_chance_align'].copy()
        print(f"Loaded {len(R):,} binary systems")
        
        # Compute features
        print("\nComputing SCALED features (Table A1)...")
        X = compute_scaled_features(data)
    
    # Target: log(R_chance_align) with floor at -10
    y = np.log10(R)
    y = np.clip(y, -10, None)
    print(f"Target range: log(R) = [{y.min():.2f}, {y.max():.2f}] (floored at -10)")
    
    # Remove invalid samples
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid_mask]
    y = y[valid_mask]
    R = R[valid_mask]
    print(f"Valid samples after filtering: {len(y):,}")
    
    # First split: hold out 20% as a completely separate TEST set
    print("\nSplitting data...")
    print("  Step 1: Hold out 20% as TEST set (completely unseen)")
    X_trainval, X_test, y_trainval, y_test, R_trainval, R_test = train_test_split(
        X, y, R, test_size=0.2, random_state=42
    )
    print(f"  Test set (holdout): {len(y_test):,} samples")
    
    # Save the holdout test set
    HOLDOUT_PATH = Path('models/holdout_test_set.npz')
    np.savez(HOLDOUT_PATH, X=X_test, y=y_test, R=R_test)
    print(f"  Holdout test set saved to {HOLDOUT_PATH}")
    
    # Compute weights on trainval set only
    print("\nComputing sample weights...")
    weights = compute_sample_weights(R_trainval)
    
    # Second split: 80/20 of trainval for train/validation (for early stopping)
    print("\n  Step 2: Split remaining 80% into train (80%) / validation (20%)")
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_trainval, y_trainval, weights, test_size=0.2, random_state=42
    )
    print(f"  Training samples: {len(y_train):,}")
    print(f"  Validation samples (for early stopping): {len(y_val):,}")
    
    # LightGBM model
    print("\nTraining LightGBM model...")
    feature_names = [
        'log_theta', 'inv_parallax_scaled', 'norm_parallax_diff',
        'log_sigma18_scaled', 'v_tan_scaled', 'pm_diff_erf',
        'sigma_parallax'
    ]
    
    model = lgb.LGBMRegressor(
        # We've found that a Huber loss (with default delta) works best to balance outliers 
        # and inliers across the wide R range, especially with the aggressive weighting scheme.
        # We don't expect that further tuning these parameters will result in a significant
        # improvement in precision/accuracy. You're always welcome to try though!
        objective='huber',
        metric='rmse',
        boosting_type='gbdt',
        n_estimators=10000,
        learning_rate=0.02,
        num_leaves=128,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_alpha=5.0,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Evaluate
    print("\nEvaluation:")
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    rmse_train = np.sqrt(np.mean((y_train - y_pred_train)**2))
    rmse_val = np.sqrt(np.mean((y_val - y_pred_val)**2))
    
    print(f"  Train RMSE: {rmse_train:.4f}")
    print(f"  Validation RMSE: {rmse_val:.4f}")
    
    # Feature importance
    print("\nFeature importance:")
    for name, importance in sorted(zip(feature_names, model.feature_importances_), 
                                    key=lambda x: -x[1]):
        print(f"  {name}: {importance}")
    
    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'feature_names': feature_names}, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
