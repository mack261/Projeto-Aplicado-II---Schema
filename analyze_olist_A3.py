from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
OUT.mkdir(exist_ok=True)


def load(name):
    return pd.read_csv(BASE / name)


orders = load("olist_orders_dataset.csv")
payments = load("olist_order_payments_dataset.csv")
reviews = load("olist_order_reviews_dataset.csv")
customers = load("olist_customers_dataset.csv")
items = load("olist_order_items_dataset.csv")
products = load("olist_products_dataset.csv")
translation = load("product_category_name_translation.csv")


for col in [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")





pay_summary = (
    payments.groupby("payment_type", dropna=False)
    .agg(
        payment_records=("order_id", "size"),
        unique_orders=("order_id", "nunique"),
        mean_payment_value=("payment_value", "mean"),
        median_payment_value=("payment_value", "median"),
        std_payment_value=("payment_value", "std"),
        mean_installments=("payment_installments", "mean"),
    )
    .reset_index()
    .sort_values("mean_payment_value", ascending=False)
)
pay_summary.to_csv(OUT / "objective1_payment_summary.csv", index=False)

installment_summary = (
    payments.groupby(["payment_type", "payment_installments"], dropna=False)
    .agg(
        payment_records=("order_id", "size"),
        unique_orders=("order_id", "nunique"),
        mean_payment_value=("payment_value", "mean"),
        median_payment_value=("payment_value", "median"),
    )
    .reset_index()
    .sort_values("mean_payment_value", ascending=False)
)
installment_summary.to_csv(OUT / "objective1_payment_installments.csv", index=False)




order_ticket = (
    payments.groupby("order_id", as_index=False)
    .agg(order_ticket=("payment_value", "sum"))
    .merge(payments[["order_id", "payment_type"]].drop_duplicates(), on="order_id")
)
order_ticket_summary = (
    order_ticket.groupby("payment_type")
    .agg(
        unique_orders=("order_id", "nunique"),
        mean_order_ticket=("order_ticket", "mean"),
        median_order_ticket=("order_ticket", "median"),
        std_order_ticket=("order_ticket", "std"),
    )
    .reset_index()
    .sort_values("mean_order_ticket", ascending=False)
)
order_ticket_summary.to_csv(OUT / "objective1_order_ticket_summary.csv", index=False)




reference_date = orders["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
order_base = (
    orders[["order_id", "customer_id", "order_purchase_timestamp", "order_status"]]
    .merge(payments.groupby("order_id", as_index=False)["payment_value"].sum(), on="order_id", how="left")
)


rfm = (
    order_base.dropna(subset=["customer_id", "order_purchase_timestamp"])
    .groupby("customer_id")
    .agg(
        recency=("order_purchase_timestamp", lambda s: (reference_date - s.max()).days),
        frequency=("order_id", "nunique"),
        monetary=("payment_value", "sum"),
    )
    .reset_index()
)
rfm["monetary"] = rfm["monetary"].fillna(0)
rfm_features = rfm[["recency", "frequency", "monetary"]].copy()

rfm_model = np.log1p(rfm_features)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_model)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
rfm["cluster"] = kmeans.fit_predict(rfm_scaled)



sample_n = min(10000, len(rfm))
sample = rfm.sample(sample_n, random_state=42)
sample_scaled = rfm_scaled[sample.index]
labels_sample = sample["cluster"].to_numpy()
sil = silhouette_score(sample_scaled, labels_sample)
db = davies_bouldin_score(sample_scaled, labels_sample)
rfm_summary = (
    rfm.groupby("cluster")
    .agg(
        customers=("customer_id", "nunique"),
        mean_recency=("recency", "mean"),
        median_recency=("recency", "median"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean"),
        median_monetary=("monetary", "median"),
    )
    .reset_index()
    .sort_values("cluster")
)
rfm.to_csv(OUT / "objective2_rfm_customers.csv", index=False)
rfm_summary.to_csv(OUT / "objective2_rfm_cluster_summary.csv", index=False)




cat_map = translation.rename(columns={"product_category_name_english": "category"})
items_cat = items.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
items_cat = items_cat.merge(translation, on="product_category_name", how="left")
items_cat["category"] = items_cat["product_category_name_english"].fillna(items_cat["product_category_name"])
category_order = (
    items_cat.groupby("order_id", as_index=False)
    .agg(category=("category", lambda s: s.dropna().iloc[0] if s.dropna().size else "unknown"))
)
review_one = reviews.dropna(subset=["review_score"]).copy()
review_one["review_score"] = pd.to_numeric(review_one["review_score"], errors="coerce")
review_one = review_one.dropna(subset=["review_score"]).drop_duplicates("order_id", keep="first")
category_reviews = review_one[["order_id", "review_score"]].merge(category_order, on="order_id", how="left")
category_summary = (
    category_reviews.dropna(subset=["category"])
    .groupby("category")
    .agg(
        reviews=("review_score", "size"),
        mean_review_score=("review_score", "mean"),
        median_review_score=("review_score", "median"),
        std_review_score=("review_score", "std"),
    )
    .reset_index()
    .sort_values(["mean_review_score", "reviews"], ascending=[True, False])
)
category_summary.to_csv(OUT / "objective3_category_review_summary.csv", index=False)




item_features = (
    items_cat.groupby("order_id")
    .agg(
        item_count=("product_id", "size"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        category=("category", lambda s: s.dropna().iloc[0] if s.dropna().size else "unknown"),
    )
    .reset_index()
)
payment_features = (
    payments.groupby("order_id")
    .agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        payment_type=("payment_type", lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown"),
    )
    .reset_index()
)
customer_features = customers[["customer_id", "customer_state"]]
model_df = (
    review_one[["order_id", "review_score"]]
    .merge(orders[["order_id", "customer_id", "order_purchase_timestamp"]], on="order_id", how="inner")
    .merge(customer_features, on="customer_id", how="left")
    .merge(item_features, on="order_id", how="left")
    .merge(payment_features, on="order_id", how="left")
)
model_df["purchase_month"] = model_df["order_purchase_timestamp"].dt.month.astype("Int64").astype(str)
model_df["satisfaction_class"] = pd.cut(
    model_df["review_score"], bins=[0, 2, 3, 5], labels=["insatisfatoria", "neutra", "satisfatoria"], include_lowest=True
).astype(str)
model_df = model_df.dropna(subset=["satisfaction_class"]).copy()

features = [
    "payment_type", "payment_installments", "payment_value", "item_count",
    "total_price", "total_freight", "category", "customer_state", "purchase_month",
]
X = model_df[features].copy()
y = model_df["satisfaction_class"]
cat_cols = ["payment_type", "category", "customer_state", "purchase_month"]
num_cols = [c for c in features if c not in cat_cols]
preprocess = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), num_cols),
])
clf = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="saga", random_state=42)),
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
labels = ["insatisfatoria", "neutra", "satisfatoria"]
cm = confusion_matrix(y_test, pred, labels=labels)
report = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
classifier_metrics = {
    "rows_total": int(len(model_df)),
    "rows_train": int(len(X_train)),
    "rows_test": int(len(X_test)),
    "class_distribution": y.value_counts().to_dict(),
    "accuracy": float(accuracy_score(y_test, pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
    "precision_macro": float(precision_score(y_test, pred, labels=labels, average="macro", zero_division=0)),
    "recall_macro": float(recall_score(y_test, pred, labels=labels, average="macro", zero_division=0)),
    "f1_macro": float(f1_score(y_test, pred, labels=labels, average="macro", zero_division=0)),
    "confusion_matrix_labels": labels,
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
}
with open(OUT / "objective3_classifier_metrics.json", "w", encoding="utf-8") as f:
    json.dump(classifier_metrics, f, ensure_ascii=False, indent=2)

summary = {
    "objective1": {
        "highest_mean_payment_value_type": str(pay_summary.iloc[0]["payment_type"]),
        "highest_mean_payment_value": float(pay_summary.iloc[0]["mean_payment_value"]),
        "highest_mean_order_ticket_type": str(order_ticket_summary.iloc[0]["payment_type"]),
        "highest_mean_order_ticket": float(order_ticket_summary.iloc[0]["mean_order_ticket"]),
    },
    "objective2": {
        "customers": int(len(rfm)),
        "clusters": 4,
        "silhouette_sample": float(sil),
        "davies_bouldin_sample": float(db),
        "sample_n": int(sample_n),
    },
    "objective3": {
        "lowest_mean_category": str(category_summary.iloc[0]["category"]),
        "lowest_mean_review_score": float(category_summary.iloc[0]["mean_review_score"]),
        "categories_below_4_2": int((category_summary["mean_review_score"] < 4.2).sum()),
        "classifier": classifier_metrics,
    },
}
with open(OUT / "analysis_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
