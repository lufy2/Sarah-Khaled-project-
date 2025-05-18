# استدعاء المكتبات المطلوبة
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import gdown  # مكتبة لتحميل الملفات من Google Drive

# تحميل البيانات من Google Drive
url = 'https://drive.google.com/uc?id=1lBrQwn-j_wyoIjBNDlkSYHuSkLMcfStA'
output = 'ai_ghibli_trend_dataset.csv'
gdown.download(url, output, quiet=False)

# قراءة ملف البيانات
AISG = pd.read_csv('ai_ghibli_trend_dataset.csv')

# استكشاف البيانات
print(AISG.head(10))
print(AISG.info())
print(AISG.describe())

# تحويل الأعمدة النصية إلى رقمية
le = LabelEncoder()
categorical_columns = ['resolution', 'creation_date', 'is_hand_edited', 'ethical_concerns_flag']
for col in categorical_columns:
    AISG[col] = le.fit_transform(AISG[col])

# معالجة عمود 'platform'
AISG["platform"] = AISG["platform"].str.strip().str.replace(" ", "_", regex=True)
AISG['platform_encoded'] = le.fit_transform(AISG['platform'])

# استخراج معلومات التاريخ
AISG['creation_date'] = pd.to_datetime(AISG['creation_date'])
AISG['year'] = AISG['creation_date'].dt.year
AISG['month'] = AISG['creation_date'].dt.month
AISG['day'] = AISG['creation_date'].dt.day
AISG['weekday'] = AISG['creation_date'].dt.weekday

# تحويل الأعمدة المنطقية
AISG['is_hand_edited'] = AISG['is_hand_edited'].map({'Yes': 1, 'No': 0})

# حذف الأعمدة غير الضرورية
AISG.drop(['resolution', 'creation_date'], axis=1, inplace=True)

# تصور البيانات
sns.pairplot(AISG, hue='platform', markers='+')
plt.show()

sns.violinplot(y='platform', x='likes', data=AISG, inner='quartile')
plt.show()
sns.violinplot(y='platform', x='shares', data=AISG, inner='quartile')
plt.show()
sns.violinplot(y='platform', x='comments', data=AISG, inner='quartile')
plt.show()

# إعداد البيانات للنمذجة
AISG_cleaned = AISG.drop(['image_id', 'user_id', 'prompt', 'top_comment'], axis=1)
X = AISG_cleaned.drop(['likes'], axis=1)
y = AISG_cleaned['likes']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# تدريب النموذج
clf = svm.SVC()
clf.fit(X_train, y_train)

# تقييم النموذج
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))