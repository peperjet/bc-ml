# breast_cancer_boosting_anomaly.py

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

# 이상탐지
from sklearn.ensemble import IsolationForest

# 부스팅
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# 1. 데이터 불러오기
df = pd.read_csv('data.csv')

print("원본 shape:", df.shape)
print(df.head())
print(df.columns.tolist())


# 2. 데이터 전처리
# 보통 이 데이터는 id, unnamed: 32 컬럼이 같이 들어있음
drop_cols = [col for col in ["id", "Unnamed: 32"] if col in df.columns]
df = df.drop(columns=drop_cols)

# diagnosis : M=1, B=0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

print("\n전처리 후 shape:", df.shape)
print(df["diagnosis"])

x = df.drop(columns=["diagnosis"], axis=1)
y = df["diagnosis"]


# 3. 학습/테스트 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)


# 4. 스케일링
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)




# ===================
# A. 이상탐지 시스템
# ===================
print("\n" + "="*50)
print("A. 이상탐지 결과")
print("="*50)


# contamination은 "이상치 비율" 추정값
iso = IsolationForest(
    n_estimators=200,
    contamination=0.1, 
    random_state=42
)
iso.fit(x_train_scaled)


# predict 결과 : 정상=1, 이상치=-1
anomaly_pred = iso.predict(x_test_scaled)

# 보기 쉽게 변환 : 이상치면 1, 정상이면 0
anomaly_label = np.where(anomaly_pred == -1, 1, 0)

anomaly_count = anomaly_label.sum()
print(f"테스트 데이터 이상치 개수 : {anomaly_count} / {len(anomaly_label)}")


# 이상 점수도 확인 가능
anomaly_score = iso.decision_function(x_test_scaled)

anomaly_df = pd.DataFrame({
    "real_diagnosis": y_test.values,
    "anomaly_label": anomaly_label,
    "anomaly_score": anomaly_score 
})

print("\n이상치로 잡힌 샘플 상위 10개")
print(anomaly_df[anomaly_df["anomaly_label"] == 1].head(10))


# 참고용 간단 시각화 
plt.figure(figsize=(8, 4))
plt.hist(anomaly_score, bins=30)
plt.title("Isolation Forest Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Count")
plt.tight_layout()
plt.show()



# ===================
# B. 부스팅 모델
# ===================
print("\n" + "="*50)
print("B. 부스팅 분석 결과")
print("="*50)

models = {
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100, random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100, 
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=42
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=100, 
        random_state=42
    )
}

results = []


for name, model in models.items():
    # 트리 기반 모델이라 스케일링 데이터 사용해도 되고 원본 써도 됨
    model.fit(x_train_scaled, y_train)
    pred = model.predict(x_test_scaled)
    proba = model.predict_proba(x_test_scaled)[:, 1]
    acc = accuracy_score(y_test, pred)

    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    })

    print(f"\n[{name}]")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("Classification Report:")
    print(classification_report(y_test, pred, digits=4))

result_df = pd.DataFrame(results).sort_values(by="ROC_AUC", ascending=False)
print("\n모델 비교표")
print(result_df)

plt.close('all')

fig, ax = plt.subplots(figsize=(10, 5))

x_axis = np.arange(len(result_df))
ax.bar(x_axis - 0.2, result_df["Accuracy"].values, 0.4, label="Accuracy")
ax.bar(x_axis + 0.2, result_df["ROC_AUC"].values, 0.4, label="ROC_AUC")

ax.set_xticks(x_axis)
ax.set_xticklabels(result_df["Model"].values)
ax.set_ylim(0.8, 1.0)
ax.set_title("Boosting Models - Accuracy vs ROC_AUC")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)

fig.tight_layout()
print("B 그래프 띄우기 직전")
plt.show()




# ==================
# C. 중요 변수 보기
# ==================

print("\n" + "="*50)
print("C. Feature Importance")
print("="*50)

best_model_name = result_df.iloc[0]["Model"]
best_model = models[best_model_name]

if hasattr(best_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "feature": x.columns,
        "importance": best_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print(f"\n가장 성능 좋은 모델: {best_model_name}")
    print(importance_df.head(10))

    plt.figure(figsize=(8, 5))
    top10 = importance_df.head(10).sort_values(by="importance")
    plt.barh(top10["feature"], top10["importance"])      
    plt.title(f"Top 10 Feature Importances - {best_model_name}")
    plt.tight_layout()
    plt.show()

else:
    print(f"이 모델은 feature_importances_를 지원하지 않음")