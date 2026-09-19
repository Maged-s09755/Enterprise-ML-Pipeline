import pandas as pd
import numpy as np
import joblib
import logging
import warnings

# تجاهل التحذيرات غير الضرورية
warnings.filterwarnings('ignore')

# استيراد أدوات Scikit-Learn المتقدمة
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, TargetEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, roc_auc_score

# استخدام خط أنابيب Imbalanced-Learn
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ==========================================
# 1. إعداد نظام مراقبة احترافي (Logging)
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AutoML_Enterprise_Pipeline")

# ==========================================
# 2. المحولات المخصصة فائقة الأداء
# ==========================================
class MemoryOptimizer(BaseEstimator, TransformerMixin):
    """محول آمن لتقليل استهلاك الذاكرة دون كسر مكتبات الحسابات الخطية"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            col_type = X_df[col].dtype
            if np.issubdtype(col_type, np.integer):
                X_df[col] = pd.to_numeric(X_df[col], downcast='integer')
            elif np.issubdtype(col_type, np.floating):
                X_df[col] = X_df[col].astype(np.float32) 
        return X_df

class OutlierClipper(BaseEstimator, TransformerMixin):
    """اقتطاع القيم الشاذة رياضياً لمنع انهيار النموذج"""
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.lower_bounds_ = X_df.quantile(self.lower_quantile)
        self.upper_bounds_ = X_df.quantile(self.upper_quantile)
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        X_df = X_df.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        return X_df.values 

# ==========================================
# 3. الهيكل الأساسي للنموذج (ML Pipeline Manager)
# ==========================================
class AdvancedEnterpriseModel:
    def __init__(self):
        self.pipeline = None
        self.best_model = None
        
        self.num_features = ['Age', 'Income', 'Account_Balance']
        self.cat_low = ['Gender', 'Marital_Status']
        self.cat_high = ['City', 'Job_Title']
        
        self._build_pipeline()

    def _build_pipeline(self):
        # 1. مسار الأرقام
        num_transformer = ImbPipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('clipper', OutlierClipper()),
            ('power_transform', PowerTransformer(method='yeo-johnson')), 
            ('scaler', StandardScaler())
        ])

        # 2. مسار النصوص (تعدد قليل)
        cat_low_transformer = ImbPipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('encoder', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=0.05, sparse_output=False))
        ])

        # 3. مسار النصوص (تعدد عالي)
        cat_high_transformer = ImbPipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('target_encoder', TargetEncoder(cv=5, smooth='auto'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.num_features),
                ('cat_low', cat_low_transformer, self.cat_low),
                ('cat_high', cat_high_transformer, self.cat_high)
            ], remainder='drop')

        self.pipeline = ImbPipeline(steps=[
            ('memory_opt', MemoryOptimizer()),      
            ('preprocessor', preprocessor),         
            ('variance_filter', VarianceThreshold(threshold=0.01)), 
            ('smote', SMOTE(k_neighbors=5, random_state=42)), 
            ('classifier', xgb.XGBClassifier(
                eval_metric='logloss',
                n_jobs=-1,
                random_state=42
            ))
        ])

    def train_and_optimize(self, X, y):
        logger.info("بدء التدريب وتحسين المعلمات على 1000 سجل...")

        param_dist = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [3, 5, 7],
            'classifier__subsample': [0.8, 1.0]
        }

        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        search = RandomizedSearchCV(
            self.pipeline, 
            param_distributions=param_dist, 
            n_iter=5, 
            scoring='roc_auc', 
            cv=cv_strategy,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )

        search.fit(X, y)
        self.best_model = search.best_estimator_
        
        logger.info(f"أفضل المعلمات: {search.best_params_}")
        logger.info(f"أفضل تقييم (ROC-AUC) أثناء التدريب: {search.best_score_:.4f}")

    def evaluate(self, X_test, y_test):
        preds = self.best_model.predict(X_test)
        probs = self.best_model.predict_proba(X_test)[:, 1]
        
        logger.info("\nتقرير التصنيف النهائي (Classification Report):")
        print(classification_report(y_test, preds))
        
        roc_auc = roc_auc_score(y_test, probs)
        logger.info(f"مؤشر ROC-AUC النهائي: {roc_auc:.4f}")
        return roc_auc

    def save_model(self, filepath='ultimate_production_model.pkl'):
        joblib.dump(self.best_model, filepath)
        logger.info(f"تم حفظ النموذج بنجاح في: {filepath}")

# ==========================================
# 4. دورة التشغيل والمحاكاة (1000 صف بيانات واقعية)
# ==========================================
if __name__ == "__main__":
    
    logger.info("توليد 1000 سجل بيانات محاكاة متطورة...")
    np.random.seed(42)
    n_samples = 1000
    
    # الأعمدة الرقمية (تم تركها كأرقام عشرية لقبول القيم المفقودة NaN دون أخطاء)
    ages = np.random.normal(35, 10, n_samples)
    ages[np.random.choice(n_samples, 50, replace=False)] = np.nan
    ages[np.random.choice(n_samples, 10, replace=False)] = 150.0
    
    incomes = np.random.exponential(60000, n_samples)
    incomes[np.random.choice(n_samples, 60, replace=False)] = np.nan
    incomes[np.random.choice(n_samples, 15, replace=False)] = 2500000.0
    
    balances = np.random.uniform(500, 15000, n_samples)
    balances[np.random.choice(n_samples, 40, replace=False)] = np.nan
    
    # الأعمدة النصية مع تفاوت في التعدد والتصنيف
    genders = np.random.choice(['M', 'F', 'O', np.nan], n_samples, p=[0.48, 0.48, 0.02, 0.02])
    marital = np.random.choice(['S', 'M', np.nan], n_samples, p=[0.45, 0.50, 0.05])
    cities = np.random.choice(['Cairo', 'Alex', 'Giza', 'Aswan', 'Luxor'], n_samples, p=[0.4, 0.2, 0.2, 0.1, 0.1])
    jobs = np.random.choice(['Eng', 'Doc', 'Teacher', 'Dev', 'Manager'], n_samples, p=[0.3, 0.2, 0.2, 0.2, 0.1])
    
    # فئة مستهدفة غير متوازنة (حوالي 15% فئة إيجابية) لتختبر كفاءة SMOTE
    target = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    
    df = pd.DataFrame({
        'Age': ages,
        'Income': incomes,
        'Account_Balance': balances,
        'Gender': genders,
        'Marital_Status': marital,
        'City': cities,
        'Job_Title': jobs,
        'Target_Column': target
    })
    
    X = df.drop('Target_Column', axis=1)
    y = df['Target_Column']
    
    # تقسيم البيانات بنسبة 80/20 مع الحفاظ على التوازن الطبقي
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    ml_system = AdvancedEnterpriseModel()
    ml_system.train_and_optimize(X_train, y_train)
    ml_system.evaluate(X_test, y_test)
    ml_system.save_model('v1_enterprise_model.pkl')
