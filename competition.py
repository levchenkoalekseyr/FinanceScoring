#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Импорт необходимых модулей

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
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
warnings.filterwarnings('ignore')


# In[16]:


#загрузка данных

bureau = pd.read_csv("bureau.csv")
previous_loans = pd.read_csv("previous_loans.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
transactions = pd.read_csv("transactions.csv")

#просмотр размеров файлов
print(f"Train: {train.shape}")
print(f"Test: {test.shape}")
print(f"Bureau: {bureau.shape}")
print(f"Transactions: {transactions.shape}")
print(f"Previous Loans: {previous_loans.shape}")


# In[17]:


#обработка первого файла bureau

# Удаление дубликатов. 
#Хотя возможно это и ошибка, т.к. один клиент может брать два одинаковых кредита, с другой стороны будет разная дата
bureau = bureau.drop_duplicates(keep='first')

# Заполнение пропусков в категориальных колонках
bureau['account_type'] = bureau['account_type'].fillna('unknown')
bureau['bureau_status'] = bureau['bureau_status'].fillna('unknown')

# Заполнение пропусков в числовых колонках
bureau['max_dpd_last_12m'] = bureau['max_dpd_last_12m'].fillna(0)
bureau['current_balance'] = bureau['current_balance'].fillna(0)

# Заполнение opened_days_ago по группам медианой
medians = bureau.groupby('account_type')['opened_days_ago'].median()
bureau['opened_days_ago'] = bureau['opened_days_ago'].fillna(bureau['account_type'].map(medians))
bureau['opened_days_ago'] = bureau['opened_days_ago'].fillna(bureau['opened_days_ago'].median())

# Заполнение credit_limit по группам медиано
#print("\nЗаполнение credit_limit...")
medians = bureau.groupby('account_type')['credit_limit'].median()
bureau['credit_limit'] = bureau['credit_limit'].fillna(bureau['account_type'].map(medians))
bureau['credit_limit'] = bureau['credit_limit'].fillna(bureau['credit_limit'].median())


# In[18]:


# обработка файла previous_loans

# Удаление дубликатов
previous_loans = previous_loans.drop_duplicates(keep='first')

#заполнение пропусков
previous_loans['was_overdue'] = previous_loans['was_overdue'].fillna(0) #заполнение нулевым значением
previous_loans['max_overdue_days'] = previous_loans['max_overdue_days'].fillna(0) #заполнение нулевым значением


for col in ['previous_amount', 'previous_term_months', 'closed_days_ago']:
    previous_loans[col] = previous_loans[col].fillna(previous_loans[col].median()) #заполненение медианой


# In[19]:


# обработка файла transactions

#приведение к типу даты и времени
transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])


if transactions['amount'].isnull().sum() > 0:
    transactions['amount'] = transactions['amount'].fillna(transactions['amount'].median()) #заполненение медианой

#категорийные столбцы заполняем значением "неизвестно"
if 'transaction_category' in transactions.columns:
    transactions['transaction_category'] = transactions['transaction_category'].fillna('unknown') 


# In[20]:


#наодного клиента может быть разное число записей, поэтому нужна группировка
#плюс добавим новые признакие, из текущих (среднее, максимум, минимум)


# In[21]:


# bureau
general = bureau.groupby('client_id').agg({
    'bureau_account_id': 'count',
    'credit_limit': ['sum', 'mean', 'max'],
    'current_balance': ['sum', 'mean'],
    'max_dpd_last_12m': ['max', 'mean'],
    'opened_days_ago': ['mean', 'min', 'max']
}).reset_index()

general.columns = [
    'client_id',
    'num_accounts',
    'total_limit', 'avg_limit', 'max_limit',
    'total_balance', 'avg_balance',
    'max_dpd', 'avg_dpd',
    'opened_days_ago_mean', 'opened_days_ago_min', 'opened_days_ago_max'
]

# Рассщипление категориальных колонок и замена категориальных значений на числовые
dummies = pd.get_dummies(bureau['account_type'], prefix='type')
df_with_dummies = pd.concat([bureau[['client_id']], dummies], axis=1)

type_counts = df_with_dummies.groupby('client_id').sum().reset_index()
type_counts.columns = ['client_id'] + [f'count_{col}' for col in type_counts.columns[1:]]

# Объединяем
bureau_grouped = general.merge(type_counts, on='client_id', how='left')
bureau_grouped = bureau_grouped.fillna(0)


# In[22]:


# previous loans

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


# In[23]:


# группировка TRANSACTIONS

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


# In[24]:


# объединениие всех данных после групировки

merged = bureau_grouped.merge(transactions_grouped, on='client_id', how='outer')
merged = merged.merge(prev_loans_grouped, on='client_id', how='outer')
merged = merged.fillna(0)

train_merged = train.merge(bureau_grouped, on='client_id', how='left')
train_merged = train_merged.merge(transactions_grouped, on='client_id', how='left')
train_merged = train_merged.merge(prev_loans_grouped, on='client_id', how='left')
train_merged = train_merged.fillna(0)

test_merged = test.merge(bureau_grouped, on='client_id', how='left')
test_merged = test_merged.merge(transactions_grouped, on='client_id', how='left')
test_merged = test_merged.merge(prev_loans_grouped, on='client_id', how='left')
test_merged = test_merged.fillna(0)


# In[ ]:





# In[25]:


# матрица корреляции

train_test = pd.concat([X_train, X_test], axis=0, ignore_index=True)

corr = train_test.select_dtypes(include=np.number).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(80, 80))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
corr = corr.where(np.abs(corr) > 0.7, np.nan)  # выводить только признаки с корреляцией больше 0.7
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True)
plt.show()


# In[281]:


# подготовка данных для модели (отделение результата и удаление лишних колонок, не влияющих на результат)
drop_cols = ['client_id', 'application_id']

X_train = train_merged.drop(columns=drop_cols + ['target'])
y_train = train_merged['target']
X_test = test_merged.drop(columns=drop_cols)

print(X_train.columns)


# In[282]:


# повышаем значение при увеличении уровня

education_mapping = {
    'school': 0.25,
    'college': 0.5,
    'bachelor': 0.75,
    'master': 1,
    '0': 0,        # если есть пропуски или неизвестно
    'unknown': 0,  # если есть другие значения
    np.nan: 0      # если есть пропуски
}

# Применяем замену
X_train['education'] = X_train['education'].map(education_mapping).fillna(0).astype(int)
X_test['education'] = X_test['education'].map(education_mapping).fillna(0).astype(int)

print(X_train.columns)


# In[283]:


# имеет значение трудоустроен или нет

employment_map = {
    'unemployed': 1,
    'self_employed': 1,
    'employee': 1,
    'business': 1,
    'contractor': 1,
    0: 0,           # если 0 в данных
    '0': 0,         # если 0 как строка
    np.nan: 0       # если пропуски
}

X_train['employment_type'] = X_train['employment_type'].map(employment_map).fillna(0).astype(int)
X_test['employment_type'] = X_test['employment_type'].map(employment_map).fillna(0).astype(int)


# In[284]:


#опыт работы чаще всего делят на "до года", 'от 1 до 3', 'более 3'

def categorize_months(months):
    if months < 12:
        return 0
    elif months <= 36:
        return 1
    else:
        return 2

X_train['months_at_job'] = X_train['months_at_job'].apply(categorize_months)
X_test['months_at_job'] = X_test['months_at_job'].apply(categorize_months)

print(X_train.columns)


# In[285]:


# считаем, что много счетов также плохо как и мало, среднее значение хорошо

def num_accounts(num):
    if (num < 3 or num > 6):
        return 0
    else:
        return 1

X_train['num_accounts'] = X_train['num_accounts'].apply(num_accounts)
X_test['num_accounts'] = X_test['num_accounts'].apply(num_accounts)


# In[286]:


# приводим к интервалу 0-1

X_train['dependents'] = X_train['dependents']/X_train['dependents'].max()
X_test['dependents'] = X_test['dependents']/X_test['dependents'].max()


# In[287]:


# requested_product считаем, что кредиты разные по сложности

requested_product = {
    'refinance': 0.2,
    'cash': 0.8,
    'card': 0.6,
    'pos': 1.0,
    'auto': 0.4,
    0: 0,           # если 0 в данных
    '0': 0,         # если 0 как строка
    np.nan: 0       # если пропуски
}

X_train['requested_product'] = X_train['requested_product'].map(requested_product).fillna(0)
X_test['requested_product'] = X_test['requested_product'].map(requested_product).fillna(0)


# In[288]:


print(X_train['channel'].unique())
print(X_test['channel'].unique())

channel = {
    'partner': 1,
    'office': 0.8,
    'call_center': 0.6,
    'mobile': 0.4,
    'auto': 0.3,
    'web': 0.1,
    0: 0,           # если 0 в данных
    '0': 0,         # если 0 как строка
    np.nan: 0       # если пропуски
}

X_train['channel'] = X_train['channel'].map(channel).fillna(0)
X_test['channel'] = X_test['channel'].map(channel).fillna(0)

print(X_train['channel'].unique())
print(X_test['channel'].unique())


# In[289]:


X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()


# In[ ]:





# In[291]:


# 1. Удаляем все не-числовые колонки
print(non_numeric_train)
print(non_numeric_test)

non_numeric_train = X_train_encoded.select_dtypes(exclude=['number']).columns.tolist()
non_numeric_test = X_test_encoded.select_dtypes(exclude=['number']).columns.tolist()

print(non_numeric_train)
print(non_numeric_test)

all_non_numeric = set(non_numeric_train) | set(non_numeric_test)
if all_non_numeric:
    print(f"Удаляем не-числовые колонки: {list(all_non_numeric)}")
    X_train_encoded = X_train_encoded.drop(columns=list(all_non_numeric), errors='ignore')
    X_test_encoded = X_test_encoded.drop(columns=list(all_non_numeric), errors='ignore')

# 2. Выравниваем колонки (ОБЯЗАТЕЛЬНЫЙ ШАГ!)
X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded, 
    join='inner',      # оставляем только ОБЩИЕ колонки
    axis=1, 
    fill_value=0
)


# In[ ]:





# In[292]:


#X_train_encoded_new = X_train_encoded.columns[X_train_encoded.columns.duplicated()]
# Проверяем дубликаты в test
#X_test_encoded_new = X_test_encoded.columns[X_test_encoded.columns.duplicated()]

# стандартизация
# Обучаем скейлер (находим min и max) на ОБЪЕДИНЕННЫХ данных!
# проверяем разницу между средним и медианой, приводим к нормальному распределению
scaler = MinMaxScaler()
scaler.fit(pd.concat([X_train_encoded, X_test_encoded], axis=0))

# Применяем трансформацию к каждому набору отдельно
X_train_encoded_Scaler = scaler.transform(X_train_encoded)
X_test_encoded_Scaler = scaler.transform(X_test_encoded)

X_train_encoded = pd.DataFrame(X_train_encoded_Scaler, columns=X_train_encoded.columns)
X_test_encoded = pd.DataFrame(X_test_encoded_Scaler, columns=X_test_encoded.columns)


# In[ ]:





# In[293]:


# удаление признаков (это поле использовалось для просмотра влияния признаков на результат путем их удаления или добавления)

drop_columns = [
    #
    'hash_id',
    'client_id',
    'application_date',
    # дубликаты
    'opened_days_ago_min', #корреляция со средним сроком действия счетов
    'opened_days_ago_max',  #корреляция со средним сроком действия счетов
    'region_coefficient',  # дубликат region_coefficient_extended более точный коэффициент
    'region', # есть региональный коэффициент
    'prev_amount_mean',  # есть prev_amount_sum
    'prev_amount_max',  # есть prev_amount_sum
    'prev_term_max',  # есть prev_term_mean
    'prev_closed_days_min',  # есть prev_closed_days_mean
    'prev_closed_days_max',  # есть prev_closed_days_mean
    'prev_max_overdue_max',  # дубликат max_dpd
    'prev_max_overdue_mean',  # дубликат avg_dpd
    'transactions_per_day'  # можно вычислить
    # проверить
    'amount_max',  # если есть mean и std
    'amount_min',  # если есть mean и std
    'internal_decision_code',  # может быть закодированным результатом
    'marketing_segment',  # если слабо влияет
    'post_loan_collection_score',  # пост-кредитный признак
    'siberia_northern_score'  # почти все значения одинаковые
]

X_train_encoded = X_train_encoded.drop(columns=drop_columns, errors='ignore')
X_test_encoded = X_test_encoded.drop(columns=drop_columns, errors='ignore')


# In[294]:


# просмотр обработаных данных

X_train_encoded.to_csv('X_train_encoded.csv', index=False)
X_test_encoded.to_csv('X_test_encoded.csv', index=False)
print(X_train_encoded.columns)
print(X_test_encoded.head())


# In[295]:


# ПРОСТРОЕНИЕ ГРАФИКОВ
#Смотрим результат стандартизации, проверяем выбросы и корреляцию признаков финальной модели

def plot_boxplots_with_stats(df):
    """
    Строит boxplot для каждого столбца с указанием статистик
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        print("Нет числовых столбцов")
        return

    # Настройка
    sns.set_style("whitegrid")

    # Количество строк и колонок
    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            ax = axes[i]

            data = df[col].dropna()

            if len(data) > 0:
                # Boxplot
                bp = ax.boxplot(data, vert=True, patch_artist=True,
                                boxprops=dict(facecolor='lightblue', alpha=0.7),
                                medianprops=dict(color='red', linewidth=2))

                # Вычисляем статистики
                stats = data.describe()

                # Добавляем информацию
                info_text = f'n = {stats["count"]:.0f}\n'
                info_text += f'Mean: {stats["mean"]:.2f}\n'
                info_text += f'Median: {stats["50%"]:.2f}\n'
                info_text += f'Q1: {stats["25%"]:.2f}\n'
                info_text += f'Q3: {stats["75%"]:.2f}\n'
                info_text += f'Min: {stats["min"]:.2f}\n'
                info_text += f'Max: {stats["max"]:.2f}'

                # Добавляем текст сбоку
                ax.text(1.15, data.quantile(0.5), info_text,
                        fontsize=8, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax.set_title(f'{col}', fontsize=11, fontweight='bold')
                ax.set_ylabel('Значение')
                ax.grid(True, alpha=0.3)

    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Boxplot числовых признаков со статистикой', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_column_distributions(df, cols_per_row=3, figsize=(15, 4)):
    """
    Строит графики распределения для каждого столбца с указанием статистик
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        print("Нет числовых столбцов для визуализации")
        return

    sns.set_style("whitegrid")

    n_cols = cols_per_row
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            ax = axes[i]
            data = df[col].dropna()

            if len(data) > 0:
                sns.histplot(data, bins=50, kde=True, ax=ax, color='skyblue', alpha=0.7)

                stats = data.describe()

                info_text = f'n = {stats["count"]:.0f}\n'
                info_text += f'Mean: {stats["mean"]:.2f}\n'
                info_text += f'Median: {stats["50%"]:.2f}\n'
                info_text += f'Q1: {stats["25%"]:.2f}\n'
                info_text += f'Q3: {stats["75%"]:.2f}'

                ax.text(data.quantile(0.95), ax.get_ylim()[1] * 0.85, info_text,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                ax.axvline(stats["50%"], color='red', linestyle='-', linewidth=2, label=f'Median: {stats["50%"]:.2f}')
                ax.axvline(stats["mean"], color='green', linestyle='--', linewidth=2,
                           label=f'Mean: {stats["mean"]:.2f}')

                ax.set_title(f'{col}', fontsize=11, fontweight='bold')
                ax.legend(fontsize=7, loc='upper left')
                ax.grid(True, alpha=0.3)

    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Распределение числовых признаков', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


print("\nГрафики для X_TRAIN_ENCODED:")
plot_column_distributions(X_train_encoded, cols_per_row=3, figsize=(15, 4))

print("\nГрафики для X_TEST_ENCODED:")
plot_column_distributions(X_test_encoded, cols_per_row=3, figsize=(15, 4))

# ИЛИ если хотите boxplot
print("\nBoxplot для X_TRAIN_ENCODED:")
plot_boxplots_with_stats(X_train_encoded)

# 3. Для X_train_encoded
print("\nГрафики для X_TRAIN_ENCODED:")
plot_boxplots_with_stats(X_train_encoded)

# 4. Для X_test_encoded
print("\nГрафики для X_TEST_ENCODED:")
plot_boxplots_with_stats(X_test_encoded)


# In[317]:


#пробовал сделать как по одной модели, так и ансамбль с кросс валидацией

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Словари для хранения результатов
cv_results = {
    'XGBoost': [],
    'LightGBM': [],
    'RandomForest': [],
    'CatBoost': []
}


# In[323]:


# Параметры для каждой модели
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
    'early_stopping_rounds': 50,
    'verbosity': 0,
    
    'objective': 'binary:logistic',          # для бинарной классификации
    'eval_metric': ['logloss', 'auc'],       # метрики для оценки
    'tree_method': 'hist',                   # быстрый метод для больших данных
    'max_bin': 256,                          # для hist метода
    'predictor': 'cpu_predictor',            # или 'gpu_predictor' если есть GPU
    #'sampling_method': 'gradient_based',     # улучшенный метод сэмплинга
    'max_leaves': 31,                        # максимальное количество листьев
}


# In[324]:


lgb_params = {
    'objective': 'binary',
    'metric': ['binary_logloss', 'auc'],
    'boosting_type': 'gbdt',
    
    'num_leaves': 31,
    'max_depth': 5,
    'min_child_samples': 20,
    
    'reg_alpha': 0.5,
    'reg_lambda': 3.0,
    'min_split_gain': 0.01,
    'min_child_weight': 0.01,
    
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,
    'colsample_bynode': 0.8,
    
    'learning_rate': 0.07,
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
    'scale_pos_weight': scale_pos_weight,
    
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
    
    'max_bin': 255,
    'min_data_in_bin': 3,
}


# In[328]:


rf_params = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    
    # ДОПОЛНИТЕЛЬНЫЕ РЕКОМЕНДУЕМЫЕ ПАРАМЕТРЫ
    'criterion': 'gini',                    # или 'entropy' (обычно gini лучше)
    'max_leaf_nodes': None,                 # ограничение на количество листьев
    'min_impurity_decrease': 0.0,           # минимальное уменьшение нечистоты
    'min_weight_fraction_leaf': 0.0,        # минимальная доля веса в листе
    'max_samples': None,                    # размер выборки для каждого дерева
    'ccp_alpha': 0.0,                       # минимальное сокращение сложности
    'warm_start': False,                    # продолжать обучение с предыдущего состояния
}


# In[329]:


cat_params = {
    # Основные
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'border_count': 128,
    'random_seed': 42,
    
    # Балансировка классов
    'auto_class_weights': 'Balanced',
    # или, если у вас есть scale_pos_weight:
    # 'class_weights': [1.0, scale_pos_weight],
    
    # Early stopping
    'early_stopping_rounds': 50,
    'od_type': 'Iter',
    #'od_wait': 50,
    
    # Метрики
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'use_best_model': True,
    'best_model_min_trees': 100,
    
    # Регуляризация
    'leaf_estimation_method': 'Newton',
    'leaf_estimation_iterations': 10,
    'leaf_estimation_backtracking': 'AnyImprovement',
    'rsm': 0.8,
    'subsample': 0.8,
    
    # Производительность
    'task_type': 'CPU',
    'thread_count': -1,
    
    # Логирование
    'verbose': False,
    #'logging_level': 'Silent'
}


# In[332]:


for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_encoded, y_train)):
    print(f"\n  Fold {fold + 1}:")

    X_tr = X_train_encoded.iloc[train_idx]
    X_val = X_train_encoded.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False)
    xgb_preds = xgb.predict_proba(X_val)[:, 1]
    xgb_score = roc_auc_score(y_val, xgb_preds)
    cv_results['XGBoost'].append(xgb_score)


    lgb = LGBMClassifier(**lgb_params)
    lgb.fit(X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgbm.early_stopping(50)])
    lgb_preds = lgb.predict_proba(X_val)[:, 1]
    lgb_score = roc_auc_score(y_val, lgb_preds)
    cv_results['LightGBM'].append(lgb_score)

 
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_tr, y_tr)
    rf_preds = rf.predict_proba(X_val)[:, 1]
    rf_score = roc_auc_score(y_val, rf_preds)
    cv_results['RandomForest'].append(rf_score)


    cat = CatBoostClassifier(**cat_params)
    cat.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    cat_preds = cat.predict_proba(X_val)[:, 1]
    cat_score = roc_auc_score(y_val, cat_preds)
    cv_results['CatBoost'].append(cat_score)

cv_means = {}
cv_stds = {}

for model, scores in cv_results.items():
    if len(scores) > 0 and np.mean(scores) > 0:
        cv_means[model] = np.mean(scores)
        cv_stds[model] = np.std(scores)
    


# In[333]:


# определение весов

total = sum(cv_means.values())
weights = {model: score / total for model, score in cv_means.items()}

for model, weight in weights.items():
    print(f"  {model}: {weight:.3f}")


# In[334]:


# обучение

# Разделяем данные для early stopping
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train_encoded, y_train, test_size=0.2, random_state=42, stratify=y_train
)


# In[335]:


# XGBoost

xgb = XGBClassifier(
    #параметры из поиска
    reg_lambda=2.5,
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.15,
    gamma=0.11,
    alpha=0.1,

    # остальное
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=0,

    colsample_bylevel=0.8,
    min_child_weight=3 )

xgb.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    verbose=False)


# In[336]:


# LightGBM

lgb = LGBMClassifier(
    # параметры из поиска
    colsample_bytree=0.8,
    learning_rate=0.07,
    max_depth=3,
    min_child_samples=50,
    n_estimators=1000,
    num_leaves=70,
    reg_alpha=0.5,
    reg_lambda=3.0,
    subsample=0.9,
    min_split_gain=0.01,
    min_child_weight=0.001,

    # остальное
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    scale_pos_weight=scale_pos_weight
)

lgb.fit(
    X_train_sub, y_train_sub,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgbm.early_stopping(50)]
)


# In[337]:


# 3. Random Forest
rf = RandomForestClassifier(
    n_estimators=500, 
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_encoded, y_train)


# In[338]:


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


# In[339]:


#пробовал стекинг
def get_oof_predictions(model, X, y, X_test, n_folds=5):
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


xgb_oof_train, xgb_oof_test = get_oof_predictions(xgb, X_train_encoded, y_train, X_test_encoded)
lgb_oof_train, lgb_oof_test = get_oof_predictions(lgb, X_train_encoded, y_train, X_test_encoded)
rf_oof_train, rf_oof_test = get_oof_predictions(rf, X_train_encoded, y_train, X_test_encoded)
cat_oof_train, cat_oof_test = get_oof_predictions(cat, X_train_encoded, y_train, X_test_encoded)


# In[340]:


# Обучение
X_meta_train = np.column_stack([xgb_oof_train, lgb_oof_train, rf_oof_train, cat_oof_train])
meta_model = LogisticRegression(class_weight='balanced', max_iter=1000)
meta_model.fit(X_meta_train, y_train)

# Результат
X_meta_test = np.column_stack([xgb_oof_test, lgb_oof_test, rf_oof_test, cat_oof_test])
ensemble_preds = meta_model.predict_proba(X_meta_test)[:, 1]


# In[341]:


#создание фала с ответом
submission = pd.DataFrame({
    'application_id': test_merged['application_id'],
    'target': ensemble_preds
})

submission.to_csv('submission.csv', index=False)

X_train_encoded.to_csv('X_train_encoded.csv', index=False)


# In[342]:


#оценка результата

cv_scores = cross_val_score(meta_model, X_meta_train, y_train, cv=5, scoring='roc_auc')

train_preds = meta_model.predict_proba(X_meta_train)[:, 1]
train_auc = roc_auc_score(y_train, train_preds)
print(f"Train AUC: {train_auc:.4f}")

# результат хороший 0.98 Может гдето есть переобучение
# Пробовал по матрице корреляции добалять новый признак и удалять все из которых он состоит, но это тоже улучшения не дало
# Лучший результат в ансамбле появляется при добавлении catboost, но он отдельно все равно не лучше ансамбля.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




