# Quick Start Guide - Advanced ML Models

## 🚀 What's New

You now have 3 new notebooks:
1. **Notebook 13**: Time Series Cross-Validation (CV framework)
2. **Notebook 12a**: CatBoost model (better than XGBoost)
3. **Notebook 12b**: LightGBM model (fast and accurate)

## ⚡ Quick Start (5 minutes)

### Step 1: Install Dependencies
```bash
./install_advanced_ml.sh
```

Or manually:
```bash
pip install catboost lightgbm
```

### Step 2: Run New Models

**Option A - Jupyter Interface**:
```bash
jupyter notebook
# Open and run: notebooks/12a_catboost_model.ipynb
# Open and run: notebooks/12b_lightgbm_model.ipynb
```

**Option B - Command Line**:
```bash
jupyter nbconvert --execute --to notebook --inplace notebooks/12a_catboost_model.ipynb
jupyter nbconvert --execute --to notebook --inplace notebooks/12b_lightgbm_model.ipynb
```

### Step 3: Re-run CV with All Models
```bash
# First, manually add CatBoost/LightGBM wrappers to Notebook 13 (see section below)
jupyter notebook notebooks/13_time_series_cross_validation.ipynb
# Run all cells
```

---

## 📊 Expected Timeline

| Task | Time | Status |
|------|------|--------|
| Install dependencies | 2 min | ⏳ Pending |
| Run Notebook 12a (CatBoost) | 5-10 min | ⏳ Pending |
| Run Notebook 12b (LightGBM) | 5-10 min | ⏳ Pending |
| Update Notebook 13 CV | 5 min | ⏳ Pending |
| Re-run Notebook 13 | 15-20 min | ⏳ Pending |
| **Total** | **~45 min** | |

---

## 📝 Adding CatBoost/LightGBM to Notebook 13 CV

To integrate new models into the CV framework, add these wrapper functions to **Notebook 13, Section 2** (after the `wrap_xgboost` function):

```python
def wrap_catboost(train_df, test_df, target_col):
    """
    CatBoost model wrapper.
    """
    from catboost import CatBoostRegressor
    
    def create_features(df, target_col):
        df_feat = df.copy()
        df_feat['year'] = df_feat['date'].dt.year
        df_feat['month'] = df_feat['date'].dt.month
        df_feat['quarter'] = df_feat['date'].dt.quarter
        df_feat['week'] = df_feat['date'].dt.isocalendar().week
        df_feat['lag_1'] = df_feat[target_col].shift(1)
        df_feat['lag_3'] = df_feat[target_col].shift(3)
        df_feat['lag_6'] = df_feat[target_col].shift(6)
        df_feat['rolling_mean_3'] = df_feat[target_col].rolling(window=3, min_periods=1).mean()
        df_feat['rolling_std_3'] = df_feat[target_col].rolling(window=3, min_periods=1).std()
        return df_feat
    
    train_feat = create_features(train_df, target_col)
    test_feat = create_features(test_df, target_col)
    
    feature_cols = ['year', 'month', 'quarter', 'week', 
                   'lag_1', 'lag_3', 'lag_6',
                   'rolling_mean_3', 'rolling_std_3']
    
    train_feat = train_feat.dropna(subset=[target_col] + feature_cols)
    
    if len(train_feat) < 10:
        return np.full(len(test_df), train_df[target_col].mean())
    
    X_train = train_feat[feature_cols].copy()
    y_train = train_feat[target_col]
    
    # Convert to categorical
    X_train['month'] = X_train['month'].astype(str)
    X_train['quarter'] = X_train['quarter'].astype(str)
    cat_features = ['month', 'quarter']
    
    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.03,
        loss_function='MAPE',
        random_state=42,
        verbose=False
    )
    
    model.fit(X_train, y_train, cat_features=cat_features, verbose=False)
    
    X_test = test_feat[feature_cols].fillna(X_train.mean())
    X_test['month'] = X_test['month'].astype(str)
    X_test['quarter'] = X_test['quarter'].astype(str)
    
    predictions = model.predict(X_test)
    return predictions


def wrap_lightgbm(train_df, test_df, target_col):
    """
    LightGBM model wrapper.
    """
    import lightgbm as lgb
    
    def create_features(df, target_col):
        df_feat = df.copy()
        df_feat['year'] = df_feat['date'].dt.year
        df_feat['month'] = df_feat['date'].dt.month
        df_feat['quarter'] = df_feat['date'].dt.quarter
        df_feat['week'] = df_feat['date'].dt.isocalendar().week
        df_feat['lag_1'] = df_feat[target_col].shift(1)
        df_feat['lag_3'] = df_feat[target_col].shift(3)
        df_feat['lag_6'] = df_feat[target_col].shift(6)
        df_feat['rolling_mean_3'] = df_feat[target_col].rolling(window=3, min_periods=1).mean()
        df_feat['rolling_std_3'] = df_feat[target_col].rolling(window=3, min_periods=1).std()
        return df_feat
    
    train_feat = create_features(train_df, target_col)
    test_feat = create_features(test_df, target_col)
    
    feature_cols = ['year', 'month', 'quarter', 'week', 
                   'lag_1', 'lag_3', 'lag_6',
                   'rolling_mean_3', 'rolling_std_3']
    
    train_feat = train_feat.dropna(subset=[target_col] + feature_cols)
    
    if len(train_feat) < 10:
        return np.full(len(test_df), train_df[target_col].mean())
    
    X_train = train_feat[feature_cols]
    y_train = train_feat[target_col]
    
    model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        metric='mape',
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    X_test = test_feat[feature_cols].fillna(X_train.mean())
    predictions = model.predict(X_test)
    
    return predictions
```

Then update the `models` dictionary in **Section 3**:

```python
models = {
    'Seasonal Naive': lambda train, test, col: wrap_seasonal_naive(train, test, col),
    'MA-3': lambda train, test, col: wrap_moving_average(train, test, col, window=3),
    'MA-6': lambda train, test, col: wrap_moving_average(train, test, col, window=6),
    'Prophet': lambda train, test, col: wrap_prophet(train, test, col),
    'SARIMAX': lambda train, test, col: wrap_sarimax(train, test, col),
    'XGBoost': lambda train, test, col: wrap_xgboost(train, test, col),
    'CatBoost': lambda train, test, col: wrap_catboost(train, test, col),  # NEW
    'LightGBM': lambda train, test, col: wrap_lightgbm(train, test, col),  # NEW
}
```

---

## 📈 What to Expect

After running everything, you'll have:

### Files Created:
- `data/processed/catboost_metrics.csv`
- `data/processed/catboost_forecast_validation.csv`
- `data/processed/lightgbm_metrics.csv`
- `data/processed/lightgbm_forecast_validation.csv`
- `data/processed/cv_results_all_models.csv` (updated with 8 models)
- `data/processed/cv_best_models.csv` (updated)

### Visualizations:
- `results/catboost_feature_importance_*.html`
- `results/catboost_forecast_*.html` (5 metrics)
- `results/lightgbm_feature_importance_*.html`
- `results/lightgbm_forecast_*.html` (5 metrics)
- `results/cv_results_heatmap.html` (updated)
- `results/cv_stability_boxplot.html` (updated)

### Performance Comparison:
You'll be able to see which model performs best for each metric based on robust CV evaluation!

---

## 🎯 Success Checklist

- [ ] Dependencies installed (`catboost`, `lightgbm`)
- [ ] Notebook 12a runs without errors
- [ ] Notebook 12b runs without errors
- [ ] CatBoost MAPE < 2.6% average
- [ ] LightGBM MAPE < 2.7% average
- [ ] Notebook 13 updated with new model wrappers
- [ ] CV shows rankings for all 8 models
- [ ] Best model identified per metric

---

## 🆘 Troubleshooting

**Error: "No module named 'catboost'"**
```bash
pip install catboost
```

**Error: "No module named 'lightgbm'"**
```bash
pip install lightgbm
```

**Error: Kernel crashes during training**
- Your dataset is small (30 months), so this shouldn't happen
- Try reducing `iterations` to 100 in both models

**Warning: "NaN values in features"**
- This is expected for early predictions (first 6-12 months)
- The code handles this with `.fillna()`

---

**Questions?** Check `IMPLEMENTATION_PROGRESS.md` for detailed information.
