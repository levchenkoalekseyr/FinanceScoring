import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import warnings
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone  # ← Добавить в начало файла
from sklearn.preprocessing import MinMaxScaler
import catboost
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# ============================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================
print("=" * 60)
print("ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

bureau = pd.read_csv("bureau.csv")
previous_loans = pd.read_csv("previous_loans.csv")
# sample_submission = pd.read_csv("sample_submission.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
transactions = pd.read_csv("transactions.csv")

print(f"Train: {train.shape}")
print(f"Test: {test.shape}")
print(f"Bureau: {bureau.shape}")
print(f"Transactions: {transactions.shape}")
print(f"Previous Loans: {previous_loans.shape}")

# ============================================
# 2. ОБРАБОТКА BUREAU
# ============================================
print("\n" + "=" * 60)
print("ОБРАБОТКА BUREAU")
print("=" * 60)

# Удаление дубликатов
bureau = bureau.drop_duplicates(keep='first')
print(f"После удаления дубликатов: {bureau.shape}")

# Заполнение пропусков в категориальных колонках
print(f"Пропусков в account_type: {bureau['account_type'].isnull().sum()}")
bureau['account_type'] = bureau['account_type'].fillna('unknown')

print(f"Пропусков в bureau_status: {bureau['bureau_status'].isnull().sum()}")
bureau['bureau_status'] = bureau['bureau_status'].fillna('unknown')

# Заполнение пропусков в числовых колонках
bureau['max_dpd_last_12m'] = bureau['max_dpd_last_12m'].fillna(0)
bureau['current_balance'] = bureau['current_balance'].fillna(0)

# Заполнение opened_days_ago по группам
print("\nЗаполнение opened_days_ago...")
medians = bureau.groupby('account_type')['opened_days_ago'].median()
bureau['opened_days_ago'] = bureau['opened_days_ago'].fillna(
    bureau['account_type'].map(medians)
)
bureau['opened_days_ago'] = bureau['opened_days_ago'].fillna(bureau['opened_days_ago'].median())

# Заполнение credit_limit по группам
print("\nЗаполнение credit_limit...")
medians = bureau.groupby('account_type')['credit_limit'].median()
bureau['credit_limit'] = bureau['credit_limit'].fillna(
    bureau['account_type'].map(medians)
)
bureau['credit_limit'] = bureau['credit_limit'].fillna(bureau['credit_limit'].median())

print(f"\nПропусков после обработки bureau: {bureau.isnull().sum().sum()}")

# ============================================
# 3. ОБРАБОТКА PREVIOUS_LOANS
# ============================================
print("\n" + "=" * 60)
print("ОБРАБОТКА PREVIOUS_LOANS")
print("=" * 60)

previous_loans = previous_loans.drop_duplicates(keep='first')
print(f"После удаления дубликатов: {previous_loans.shape}")

previous_loans['was_overdue'] = previous_loans['was_overdue'].fillna(0)
previous_loans['max_overdue_days'] = previous_loans['max_overdue_days'].fillna(0)

for col in ['previous_amount', 'previous_term_months', 'closed_days_ago']:
    previous_loans[col] = previous_loans[col].fillna(previous_loans[col].median())

print(f"Пропусков после обработки previous_loans: {previous_loans.isnull().sum().sum()}")

# ============================================
# 4. ОБРАБОТКА TRANSACTIONS
# ============================================
print("\n" + "=" * 60)
print("ОБРАБОТКА TRANSACTIONS")
print("=" * 60)

transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])

if transactions['amount'].isnull().sum() > 0:
    transactions['amount'] = transactions['amount'].fillna(transactions['amount'].median())

if 'transaction_category' in transactions.columns:
    transactions['transaction_category'] = transactions['transaction_category'].fillna('unknown')

print(f"Пропусков после обработки transactions: {transactions.isnull().sum().sum()}")

# ============================================
# 5. ГРУППИРОВКА BUREAU
# ============================================

print("\n" + "=" * 60)
print("ГРУППИРОВКА BUREAU")
print("=" * 60)

# Общие признаки (добавили opened_days_ago)
general = bureau.groupby('client_id').agg({
    'bureau_account_id': 'count',
    'credit_limit': ['sum', 'mean', 'max'],
    'current_balance': ['sum', 'mean'],
    'max_dpd_last_12m': ['max', 'mean'],
    'opened_days_ago': ['mean', 'min', 'max']  # ДОБАВИЛИ!
}).reset_index()

general.columns = [
    'client_id',
    'num_accounts',
    'total_limit', 'avg_limit', 'max_limit',
    'total_balance', 'avg_balance',
    'max_dpd', 'avg_dpd',
    'opened_days_ago_mean', 'opened_days_ago_min', 'opened_days_ago_max'  # ДОБАВИЛИ!
]

# Признаки по типам (через get_dummies)
dummies = pd.get_dummies(bureau['account_type'], prefix='type')
df_with_dummies = pd.concat([bureau[['client_id']], dummies], axis=1)

type_counts = df_with_dummies.groupby('client_id').sum().reset_index()
type_counts.columns = ['client_id'] + [f'count_{col}' for col in type_counts.columns[1:]]

# Объединяем
bureau_grouped = general.merge(type_counts, on='client_id', how='left')
bureau_grouped = bureau_grouped.fillna(0)

print(f"Bureau группировка: {bureau_grouped.shape}")
print(f"Колонки bureau_grouped: {bureau_grouped.columns.tolist()}")

# ============================================
# 6. ГРУППИРОВКА PREVIOUS_LOANS
# ============================================
print("\n" + "=" * 60)
print("ГРУППИРОВКА PREVIOUS_LOANS")
print("=" * 60)

prev_loans_grouped = previous_loans.groupby('client_id').agg({
    'previous_loan_id': 'count',
    'previous_amount': ['sum', 'mean', 'max'],
    'previous_term_months': ['mean', 'max'],
    'closed_days_ago': ['mean', 'min', 'max'],
    'was_overdue': 'sum',
    'max_overdue_days': ['max', 'mean']
}).reset_index()

prev_loans_grouped.columns = [
    'client_id',
    'prev_loans_count',
    'prev_amount_sum', 'prev_amount_mean', 'prev_amount_max',
    'prev_term_mean', 'prev_term_max',
    'prev_closed_days_mean', 'prev_closed_days_min', 'prev_closed_days_max',
    'prev_overdue_count',
    'prev_max_overdue_max', 'prev_max_overdue_mean'
]

prev_loans_grouped['prev_overdue_rate'] = prev_loans_grouped['prev_overdue_count'] / (
        prev_loans_grouped['prev_loans_count'] + 1)
prev_loans_grouped['prev_has_overdue'] = (prev_loans_grouped['prev_overdue_count'] > 0).astype(int)

print(f"Previous Loans группировка: {prev_loans_grouped.shape}")

# ============================================
# 7. ГРУППИРОВКА TRANSACTIONS
# ============================================
print("\n" + "=" * 60)
print("ГРУППИРОВКА TRANSACTIONS")
print("=" * 60)

grouped = transactions.groupby('client_id').agg({
    'amount': ['count', 'sum', 'mean', 'max', 'min', 'std'],
    'transaction_date': ['min', 'max']
}).reset_index()

grouped.columns = [
    'client_id',
    'transactions_count',
    'amount_sum', 'amount_mean', 'amount_max', 'amount_min', 'amount_std',
    'first_transaction', 'last_transaction'
]

grouped['transactions_span_days'] = (grouped['last_transaction'] - grouped['first_transaction']).dt.days
grouped['transactions_per_day'] = grouped['transactions_count'] / (grouped['transactions_span_days'] + 1)

transactions_grouped = grouped.drop(['first_transaction', 'last_transaction'], axis=1)
transactions_grouped = transactions_grouped.fillna(0)

print(f"Transactions группировка: {transactions_grouped.shape}")

# ============================================
# 8. ОБЪЕДИНЕНИЕ ВСЕХ ДАННЫХ
# ============================================
print("\n" + "=" * 60)
print("ОБЪЕДИНЕНИЕ ДАННЫХ")
print("=" * 60)

merged = bureau_grouped.merge(transactions_grouped, on='client_id', how='outer')
merged = merged.merge(prev_loans_grouped, on='client_id', how='outer')
merged = merged.fillna(0)

print(f"Объединенные данные: {merged.shape}")

train_merged = train.merge(bureau_grouped, on='client_id', how='left')
train_merged = train_merged.merge(transactions_grouped, on='client_id', how='left')
train_merged = train_merged.merge(prev_loans_grouped, on='client_id', how='left')
train_merged = train_merged.fillna(0)

test_merged = test.merge(bureau_grouped, on='client_id', how='left')
test_merged = test_merged.merge(transactions_grouped, on='client_id', how='left')
test_merged = test_merged.merge(prev_loans_grouped, on='client_id', how='left')
test_merged = test_merged.fillna(0)

print(f"Train после объединения: {train_merged.shape}")
print(f"Test после объединения: {test_merged.shape}")

# ============================================
# 8.5 СОЗДАНИЕ ДОПОЛНИТЕЛЬНЫХ ПРИЗНАКОВ
# ============================================
print("\n" + "=" * 60)
print("СОЗДАНИЕ ДОПОЛНИТЕЛЬНЫХ ПРИЗНАКОВ")
print("=" * 60)

# === ДЛЯ TRAIN ===
print("Добавление признаков для train...")

################################################# ПРИЗНАКИ ##########################################################

# 1. Кредитные отношения
train_merged['debt_to_income'] = train_merged['total_balance'] / (train_merged['amount_sum'] + 1)
train_merged['credit_utilization'] = train_merged['total_balance'] / (train_merged['total_limit'] + 1)
train_merged['limit_to_income'] = train_merged['total_limit'] / (train_merged['amount_sum'] + 1)

# 2. Активность клиента
train_merged['trans_per_account'] = train_merged['transactions_count'] / (train_merged['num_accounts'] + 1)
train_merged['trans_per_day'] = train_merged['transactions_count'] / (train_merged['transactions_span_days'] + 1)
train_merged['avg_trans_per_limit'] = train_merged['amount_mean'] / (train_merged['avg_limit'] + 1)

# 3. Просрочки и риски
train_merged['overdue_prev_ratio'] = train_merged['prev_overdue_count'] / (train_merged['prev_loans_count'] + 1)
train_merged['dpd_per_account'] = train_merged['max_dpd'] / (train_merged['num_accounts'] + 1)
train_merged['avg_dpd_per_account'] = train_merged['avg_dpd'] / (train_merged['num_accounts'] + 1)
train_merged['has_serious_overdue'] = (train_merged['max_dpd'] > 30).astype(int)
train_merged['overdue_severity'] = train_merged['max_dpd'] * train_merged['prev_overdue_count']

# 4. Временные характеристики
train_merged['closed_accounts_ratio'] = train_merged['prev_closed_days_min'] / (train_merged['prev_closed_days_max'] + 1)
train_merged['avg_loan_term'] = train_merged['prev_term_mean'] / (train_merged['prev_closed_days_mean'] + 1)
train_merged['loan_age_ratio'] = train_merged['prev_closed_days_mean'] / (train_merged['prev_term_mean'] + 1)

# 5. Стабильность
train_merged['balance_to_limit_ratio'] = train_merged['total_balance'] / (train_merged['total_limit'] + 1)
train_merged['transactions_std_mean_ratio'] = train_merged['amount_std'] / (train_merged['amount_mean'] + 1)
train_merged['credit_limit_variation'] = train_merged['max_limit'] - train_merged['avg_limit']

# 6. Процентные соотношения
train_merged['overdue_percent'] = (train_merged['prev_overdue_count'] / (train_merged['prev_loans_count'] + 1)) * 100
train_merged['utilization_percent'] = (train_merged['total_balance'] / (train_merged['total_limit'] + 1)) * 100
# ИСПРАВЛЕНО: используем opened_days_ago_mean
train_merged['dpd_percent'] = (train_merged['max_dpd'] / (train_merged['opened_days_ago_mean'] + 1)) * 100

# 7. Композитные риски
train_merged['risk_score'] = (
    train_merged['overdue_prev_ratio'] * 0.4 +
    train_merged['credit_utilization'] * 0.3 +
    (train_merged['max_dpd'] / 100) * 0.3
)
train_merged['payment_behavior'] = (
    train_merged['prev_overdue_count'] / (train_merged['prev_loans_count'] + 1) +
    train_merged['max_dpd'] / 100
)

print(f"  Добавлено 19 новых признаков для train")

# === ДЛЯ TEST ===
print("Добавление признаков для test...")

# 1. Кредитные отношения
test_merged['debt_to_income'] = test_merged['total_balance'] / (test_merged['amount_sum'] + 1)
test_merged['credit_utilization'] = test_merged['total_balance'] / (test_merged['total_limit'] + 1)
test_merged['limit_to_income'] = test_merged['total_limit'] / (test_merged['amount_sum'] + 1)

# 2. Активность клиента
test_merged['trans_per_account'] = test_merged['transactions_count'] / (test_merged['num_accounts'] + 1)
test_merged['trans_per_day'] = test_merged['transactions_count'] / (test_merged['transactions_span_days'] + 1)
test_merged['avg_trans_per_limit'] = test_merged['amount_mean'] / (test_merged['avg_limit'] + 1)

# 3. Просрочки и риски
test_merged['overdue_prev_ratio'] = test_merged['prev_overdue_count'] / (test_merged['prev_loans_count'] + 1)
test_merged['dpd_per_account'] = test_merged['max_dpd'] / (test_merged['num_accounts'] + 1)
test_merged['avg_dpd_per_account'] = test_merged['avg_dpd'] / (test_merged['num_accounts'] + 1)
test_merged['has_serious_overdue'] = (test_merged['max_dpd'] > 30).astype(int)
test_merged['overdue_severity'] = test_merged['max_dpd'] * test_merged['prev_overdue_count']

# 4. Временные характеристики
test_merged['closed_accounts_ratio'] = test_merged['prev_closed_days_min'] / (test_merged['prev_closed_days_max'] + 1)
test_merged['avg_loan_term'] = test_merged['prev_term_mean'] / (test_merged['prev_closed_days_mean'] + 1)
test_merged['loan_age_ratio'] = test_merged['prev_closed_days_mean'] / (test_merged['prev_term_mean'] + 1)

# 5. Стабильность
test_merged['balance_to_limit_ratio'] = test_merged['total_balance'] / (test_merged['total_limit'] + 1)
test_merged['transactions_std_mean_ratio'] = test_merged['amount_std'] / (test_merged['amount_mean'] + 1)
test_merged['credit_limit_variation'] = test_merged['max_limit'] - test_merged['avg_limit']

# 6. Процентные соотношения
test_merged['overdue_percent'] = (test_merged['prev_overdue_count'] / (test_merged['prev_loans_count'] + 1)) * 100
test_merged['utilization_percent'] = (test_merged['total_balance'] / (test_merged['total_limit'] + 1)) * 100
# ИСПРАВЛЕНО: используем opened_days_ago_mean
test_merged['dpd_percent'] = (test_merged['max_dpd'] / (test_merged['opened_days_ago_mean'] + 1)) * 100


# 7. Композитные риски
test_merged['risk_score'] = (
    test_merged['overdue_prev_ratio'] * 0.4 +
    test_merged['credit_utilization'] * 0.3 +
    (test_merged['max_dpd'] / 100) * 0.3
)
test_merged['payment_behavior'] = (
    test_merged['prev_overdue_count'] / (test_merged['prev_loans_count'] + 1) +
    test_merged['max_dpd'] / 100
)

print(f"  Добавлено 19 новых признаков для test")
####################################################### ПРИЗНАКИ #####################################################

# === ЗАМЕНА INF НА 0 ===
print("\nЗамена INF значений на 0...")
train_merged = train_merged.replace([np.inf, -np.inf], 0)
test_merged = test_merged.replace([np.inf, -np.inf], 0)

# === ЗАПОЛНЕНИЕ ПРОПУСКОВ ===
print("Заполнение пропусков...")
train_merged = train_merged.fillna(0)
test_merged = test_merged.fillna(0)

print(f"\n✅ Добавлено 19 новых признаков!")
print(f"  Train shape: {train_merged.shape}")
print(f"  Test shape: {test_merged.shape}")

# ============================================
# 8. МАТРИЦА КОРРЕЛЯЦИИ
# ============================================

train_test = pd.concat([train_merged.drop(columns=['target']), test_merged], axis=0, ignore_index=True)

corr = train_test.select_dtypes(include=np.number).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(80, 80))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
corr = corr.where(np.abs(corr) > 0.7, np.nan)  # выводить только признаки с корреляцией больше 0.7
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True)
plt.show()

# ============================================
# 9. ПОДГОТОВКА ДАННЫХ ДЛЯ МОДЕЛИ
# ============================================

drop_cols = [
    #
    'hash_id',
    'application_date',
    'client_id',
    'application_id',
    # большая корреляции
    #'incoming_amount', # 'monthly_income'
    #'trans_per_day',
    #'max_dpd',
    #'avg_dpd',
    #'prev_overdue_count',
    #'prev_overdue_rate',
    #'credit_limit_variation',
    #'credit_utilization',
    #'num_accounts',
    #'overdue_percent',
    #'balance_to_limit_ratio',
    #'amount_mean',
    # дубликаты
    'region_coefficient',  # дубликат region_coefficient_extended
    'prev_amount_mean',  # есть prev_amount_sum
    'prev_amount_max',  # есть prev_amount_sum
    'prev_term_max',  # есть prev_term_mean
    'prev_closed_days_min',  # есть prev_closed_days_mean
    'prev_closed_days_max',  # есть prev_closed_days_mean
    'prev_max_overdue_max',  # дубликат max_dpd
    'prev_max_overdue_mean',  # дубликат avg_dpd
    #'transactions_per_day'  # можно вычислить
    # проверить
    'amount_max',  # если есть mean и std
    'amount_min',  # если есть mean и std
    'internal_decision_code',  # может быть закодированным результатом
    'marketing_segment',  # если слабо влияет
    'post_loan_collection_score',  # пост-кредитный признак
    'siberia_northern_score'  # почти все значения одинаковые
]

X_train = train_merged.drop(columns=drop_cols + ['target'])
y_train = train_merged['target']
X_test = test_merged.drop(columns=drop_cols)


# ============================================
# 8. МАТРИЦА КОРРЕЛЯЦИИ
# ============================================

train_test = pd.concat([X_train, X_test], axis=0, ignore_index=True)

corr = train_test.select_dtypes(include=np.number).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(80, 80))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
corr = corr.where(np.abs(corr) > 0.7, np.nan)  # выводить только признаки с корреляцией больше 0.7
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True)
plt.show()

# ============================================
# 10. КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
# ============================================
print("\n" + "=" * 60)
print("КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
print("=" * 60)

object_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print(f"Найдено {len(object_cols)} категориальных колонок:")

X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

for col in object_cols:
    print(f"\n  {col}:")

    # Объединяем все значения
    all_values = pd.concat([X_train[col], X_test[col]]).astype(str)
    unique_values = all_values.unique()
    print(f"    Уникальных: {len(unique_values)}")

    if len(unique_values) <= 15:
        # One-Hot
        print(f"    → One-Hot Encoding")
        dummies_train = pd.get_dummies(X_train[col].astype(str), prefix=col)
        dummies_test = pd.get_dummies(X_test[col].astype(str), prefix=col)

        # Добавляем недостающие колонки
        for val in unique_values:
            col_name = f"{col}_{val}"
            if col_name not in dummies_train.columns:
                dummies_train[col_name] = 0
            if col_name not in dummies_test.columns:
                dummies_test[col_name] = 0

        X_train_encoded = pd.concat([X_train_encoded, dummies_train], axis=1)
        X_test_encoded = pd.concat([X_test_encoded, dummies_test], axis=1)
        X_train_encoded = X_train_encoded.drop(col, axis=1)
        X_test_encoded = X_test_encoded.drop(col, axis=1)
    else:
        # Label Encoding
        print(f"    → Label Encoding")
        le = LabelEncoder()
        le.fit(all_values)
        X_train_encoded[col] = le.transform(X_train[col].astype(str))
        X_test_encoded[col] = le.transform(X_test[col].astype(str))

# Синхронизация колонок
print("\nСинхронизация колонок...")
common_cols = list(set(X_train_encoded.columns) & set(X_test_encoded.columns))

# Добавляем недостающие колонки
for col in set(X_train_encoded.columns) - set(X_test_encoded.columns):
    X_test_encoded[col] = 0
for col in set(X_test_encoded.columns) - set(X_train_encoded.columns):
    X_train_encoded[col] = 0

# Приводим к одинаковому порядку
common_cols = sorted(list(set(X_train_encoded.columns) & set(X_test_encoded.columns)))
X_train_encoded = X_train_encoded[common_cols]
X_test_encoded = X_test_encoded[common_cols]

print(f"\n✅ Готово!")
print(f"  X_train: {X_train_encoded.shape}")
print(f"  X_test: {X_test_encoded.shape}")
print(X_train.columns)
print(X_test.columns)
print(f"  Колонки совпадают: {set(X_train_encoded.columns) == set(X_test_encoded.columns)}")


# ============================================
# 10. СТАНДАРТИЗАЦИЯ
# ============================================

# Обучаем скейлер (находим min и max) на ОБЪЕДИНЕННЫХ данных!
scaler = MinMaxScaler()
scaler.fit(pd.concat([X_train_encoded, X_test_encoded], axis=0))

# Применяем трансформацию к каждому набору отдельно
X_train_encoded_Scaler = scaler.transform(X_train_encoded)
X_test_encoded_Scaler = scaler.transform(X_test_encoded)

X_train_encoded = pd.DataFrame(X_train_encoded_Scaler, columns=X_train_encoded.columns)
X_test_encoded = pd.DataFrame(X_test_encoded_Scaler, columns=X_test_encoded.columns)

print(X_train_encoded.head())
print(X_test_encoded.head())

# ============================================
# ПРОСМОТР ОБРАБОТАНЫХ ДАННЫХ
# ============================================
print("\n" + "=" * 60)
print("СОЗДАНИЕ ФАЙЛОВ X_train_encoded и X_test_encoded для просмотра")
print("=" * 60)

X_train_encoded.to_csv('X_train_encoded.csv', index=False)
X_test_encoded.to_csv('X_test_encoded.csv', index=False)
print("\n✅  X_train_encoded и X_test_encoded для просмотра сохранены!")


# ============================================
# 11. КРОСС-ВАЛИДАЦИЯ ДЛЯ ВСЕХ 4 МОДЕЛЕЙ
# ============================================
print("\n" + "=" * 60)
print("КРОСС-ВАЛИДАЦИЯ ВСЕХ МОДЕЛЕЙ")
print("=" * 60)

# 1. Подготовка
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale_pos_weight: {scale_pos_weight:.2f}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Словари для хранения результатов
cv_results = {
    'XGBoost': [],
    'LightGBM': [],
    'RandomForest': [],
    'CatBoost': []
}

# Параметры для каждой модели (из вашего кода)
xgb_params = {
    'reg_lambda': 2.5,
    'n_estimators': 1000,
    'max_depth': 4,
    'learning_rate': 0.15,
    'gamma': 0.11,
    'alpha': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': scale_pos_weight,
    'colsample_bylevel': 0.8,
    'min_child_weight': 3,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50,   ##################################################################
    'verbosity': 0
}

lgb_params = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.07,
    'max_depth': 3,
    'min_child_samples': 50,
    'n_estimators': 1000,
    'num_leaves': 70,
    'reg_alpha': 0.5,
    'reg_lambda': 3.0,
    'subsample': 0.9,
    'min_split_gain': 0.01,
    'min_child_weight': 0.001,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

rf_params = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

cat_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'border_count': 128,
    'random_seed': 42,
    'verbose': False
}

print("\nКросс-валидация (5 folds):")

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_encoded, y_train)):
    print(f"\n  Fold {fold + 1}:")

    X_tr = X_train_encoded.iloc[train_idx]
    X_val = X_train_encoded.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    # ============================================
    # 1. XGBoost
    # ============================================
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_tr, y_tr,
        eval_set=[(X_val, y_val)],  #################################################
        verbose=False)###############################################################
    xgb_preds = xgb.predict_proba(X_val)[:, 1]
    xgb_score = roc_auc_score(y_val, xgb_preds)
    cv_results['XGBoost'].append(xgb_score)

    # ============================================
    # 2. LightGBM
    # ============================================
    import lightgbm as lgbm

    lgb = LGBMClassifier(**lgb_params)
    lgb.fit(X_tr, y_tr,
        eval_set=[(X_val, y_val)], #######################################################
        eval_metric='auc',#############################################################
        callbacks=[lgbm.early_stopping(50)])############################################
    lgb_preds = lgb.predict_proba(X_val)[:, 1]
    lgb_score = roc_auc_score(y_val, lgb_preds)
    cv_results['LightGBM'].append(lgb_score)

    # ============================================
    # 3. Random Forest
    # ============================================
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_tr, y_tr)
    rf_preds = rf.predict_proba(X_val)[:, 1]
    rf_score = roc_auc_score(y_val, rf_preds)
    cv_results['RandomForest'].append(rf_score)

    # ============================================
    # 4. CatBoost
    # ============================================
    from catboost import CatBoostClassifier

    cat = CatBoostClassifier(**cat_params)
    cat.fit(X_tr, y_tr)
    cat_preds = cat.predict_proba(X_val)[:, 1]
    cat_score = roc_auc_score(y_val, cat_preds)
    cv_results['CatBoost'].append(cat_score)

    # Вывод результатов фолда
    print(f"    XGBoost:    {xgb_score:.4f}")
    print(f"    LightGBM:   {lgb_score:.4f}")
    print(f"    RandomForest: {rf_score:.4f}")
    print(f"    CatBoost:   {cat_score:.4f}")

# ============================================
# ИТОГИ КРОСС-ВАЛИДАЦИИ
# ============================================
print("\n" + "=" * 60)
print("ИТОГИ КРОСС-ВАЛИДАЦИИ")
print("=" * 60)

cv_means = {}
cv_stds = {}

for model, scores in cv_results.items():
    if len(scores) > 0 and np.mean(scores) > 0:
        cv_means[model] = np.mean(scores)
        cv_stds[model] = np.std(scores)
        print(f"\n{model}:")
        print(f"  Средний CV AUC: {cv_means[model]:.4f} (+/- {cv_stds[model]:.4f})")
        print(f"  Минимум: {min(scores):.4f}")
        print(f"  Максимум: {max(scores):.4f}")

# ============================================
# ОПРЕДЕЛЕНИЕ ВЕСОВ ДЛЯ АНСАМБЛЯ
# ============================================
print("\n" + "=" * 60)
print("ВЕСА ДЛЯ АНСАМБЛЯ")
print("=" * 60)

# Веса пропорционально CV AUC
total = sum(cv_means.values())
weights = {model: score / total for model, score in cv_means.items()}

print("\nВеса (на основе кросс-валидации):")
for model, weight in weights.items():
    print(f"  {model}: {weight:.3f}")

# ============================================
# 12. ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ
# ============================================
print("\n" + "=" * 60)
print("ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ")
print("=" * 60)

import lightgbm as lgbm

# Разделяем данные для early stopping
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train_encoded, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# ============================================
# 1. XGBoost с early stopping
# ============================================
print("\n1. XGBoost...")
xgb = XGBClassifier(
    # Лучшие параметры из поиска
    reg_lambda=2.5,
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.15,
    gamma=0.11,
    alpha=0.1,

    # Дополнительные настройки
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
#    early_stopping_rounds=50,  # ✅ ЗДЕСЬ!  ############################################################
    colsample_bylevel=0.8,  # ← Добавить!
    min_child_weight=3  # ← Добавить!
)

xgb.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# print(f"  ✅ Оптимальное число деревьев: {xgb.best_iteration}")

# ============================================
# 2. LightGBM с early stopping
# ============================================
print("\n2. LightGBM...")
lgb = LGBMClassifier(
    # Лучшие параметры из поиска
    colsample_bytree=0.8,
    learning_rate=0.07,
    max_depth=3,
    min_child_samples=50,
    n_estimators=1000,  # ✅ Большое число для early stopping
    num_leaves=70,
    reg_alpha=0.5,
    reg_lambda=3.0,
    subsample=0.9,
    min_split_gain=0.01,  # ← Добавить!
    min_child_weight=0.001,  # ← Добавить!

    # Базовые настройки
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    scale_pos_weight=scale_pos_weight
)

lgb.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgbm.early_stopping(50)]  # ✅ Добавили!
)

print(f"  ✅ Оптимальное число деревьев: {lgb.best_iteration_}")

# ============================================
# 3. Random Forest (БЕЗ early stopping)
# ============================================
print("\n3. Random Forest...")
rf = RandomForestClassifier(
    n_estimators=500,  # Random Forest не нужен early stopping
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_encoded, y_train)  # ✅ Обучаем на ВСЕХ данных
print("  ✅ Random Forest обучен")

print("\n4. CatBoost...")
cat = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    border_count=128,
    random_seed=42,
    verbose=False,
    early_stopping_rounds=50
)

cat.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    verbose=False
)

from sklearn.model_selection import StratifiedKFold


def get_oof_predictions(model, X, y, X_test, n_folds=5):
    """Получить out-of-fold предсказания для стекинга"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_train = np.zeros((X.shape[0],))
    oof_test = np.zeros((X_test.shape[0],))
    oof_test_skf = np.zeros((n_folds, X_test.shape[0]))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y.iloc[train_idx]

        model_clone = clone(model)
        model_clone.fit(X_tr, y_tr)

        oof_train[val_idx] = model_clone.predict_proba(X_val)[:, 1]
        oof_test_skf[fold, :] = model_clone.predict_proba(X_test)[:, 1]

    oof_test = oof_test_skf.mean(axis=0)
    return oof_train, oof_test


# Использование
xgb_oof_train, xgb_oof_test = get_oof_predictions(xgb, X_train_encoded, y_train, X_test_encoded)
lgb_oof_train, lgb_oof_test = get_oof_predictions(lgb, X_train_encoded, y_train, X_test_encoded)
rf_oof_train, rf_oof_test = get_oof_predictions(rf, X_train_encoded, y_train, X_test_encoded)
cat_oof_train, cat_oof_test = get_oof_predictions(cat, X_train_encoded, y_train, X_test_encoded)

# Обучаем мета-модель на OOF предсказаниях
X_meta_train = np.column_stack([xgb_oof_train, lgb_oof_train, rf_oof_train, cat_oof_train])
meta_model = LogisticRegression(class_weight='balanced', max_iter=1000)
meta_model.fit(X_meta_train, y_train)

# Предсказания
X_meta_test = np.column_stack([xgb_oof_train, lgb_oof_train, rf_oof_train, cat_oof_train])
ensemble_preds = meta_model.predict_proba(X_meta_test)[:, 1]

#model = cat.fit(X_train_encoded, y_train)
#ensemble_preds = model.predict_proba(X_test_encoded)[:, 1]

# ============================================
# 14. СОЗДАНИЕ SUBMISSION
# ============================================
print("\n" + "=" * 60)
print("СОЗДАНИЕ SUBMISSION")
print("=" * 60)

submission = pd.DataFrame({
    'application_id': test_merged['application_id'],
    'target': ensemble_preds
})

submission.to_csv('submission.csv', index=False)
print("\n✅ submission.csv сохранен!")

X_train_encoded.to_csv('X_train_encoded.csv', index=False)
print("\n✅ X_train.csv сохранен!")

# ============================================
# ПРОВЕРКА МЕТАМОДЕЛИ НА ОБУЧАЮЩЕЙ ВЫБОРКЕ
# ============================================
print("\n" + "=" * 60)
print("ПРОВЕРКА МЕТАМОДЕЛИ (Ансамбль)")
print("=" * 60)

# Получаем OOF-предсказания для всей обучающей выборки
# Используем функцию из вашего кода
xgb_oof_train, _ = get_oof_predictions(xgb, X_train_encoded, y_train, X_test_encoded)
lgb_oof_train, _ = get_oof_predictions(lgb, X_train_encoded, y_train, X_test_encoded)
rf_oof_train, _ = get_oof_predictions(rf, X_train_encoded, y_train, X_test_encoded)
cat_oof_train, _ = get_oof_predictions(cat, X_train_encoded, y_train, X_test_encoded)

# Формируем матрицу OOF-предсказаний для метамодели
X_meta_train_oof = np.column_stack([
    xgb_oof_train,
    lgb_oof_train,
    rf_oof_train,
    cat_oof_train
])

# Обучаем метамодель на OOF-предсказаниях
meta_model = LogisticRegression(class_weight='balanced', max_iter=1000)
meta_model.fit(X_meta_train_oof, y_train)

# Получаем предсказания метамодели на OOF-данных
meta_oof_preds = meta_model.predict_proba(X_meta_train_oof)[:, 1]

# Вычисляем AUC
meta_train_auc = roc_auc_score(y_train, meta_oof_preds)
print(f"\n✅ AUC метамодели на OOF-предсказаниях: {meta_train_auc:.4f}")

# Для сравнения — отдельные модели
print("\nСравнение с отдельными моделями (OOF AUC):")
print(f"  XGBoost OOF AUC:    {roc_auc_score(y_train, xgb_oof_train):.4f}")
print(f"  LightGBM OOF AUC:   {roc_auc_score(y_train, lgb_oof_train):.4f}")
print(f"  RandomForest OOF AUC: {roc_auc_score(y_train, rf_oof_train):.4f}")
print(f"  CatBoost OOF AUC:   {roc_auc_score(y_train, cat_oof_train):.4f}")
print(f"  Meta-model OOF AUC: {meta_train_auc:.4f}")