# Dataset format for sign-to-text

Each CSV row should contain 89 columns:

`label, feature_1, feature_2, ..., feature_88`

Accepted alternatives:
- one CSV per label, where the filename becomes the label and each row contains 88 numeric features
- one combined CSV where the label is the first column
- one combined CSV where the label is the last column

Recommended commands:

```bash
python backend/holistic_data_extract.py --label สวัสดี --output backend/data/raw/sign_samples.csv
python backend/train_model.py --data backend/data/raw/sign_samples.csv
```
