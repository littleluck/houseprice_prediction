import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

from scipy import stats
from scipy.stats import norm, skew


train = pd.read_csv("train.csv")
print("***********train:", type(train))
test = pd.read_csv("test.csv")
train_id = train["Id"]
test_id = test["Id"]

train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

# print(train.shape)
# print(test.shape)

# print(train.dtypes)
# print(train.info())
# print(train.describe())
# print(train["MSSubClass"].isnull().sum())

# Deleting outliers删除异常记录
train = train.drop(train[(train["GrLivArea"]>4000) & (train["SalePrice"]<300000)].index)

#Check the graphic again
# fig, ax = plt.subplots()
# ax.scatter(train['GrLivArea'], train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)
# plt.show()

# 分析预测值SalePrice的值
# sns.distplot(train["SalePrice"], fit=norm)
# plt.show()

# SalePrice不服从正态分布，所以我们对其做对数变换
# We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])
# sns.distplot(train["SalePrice"], fit=norm)
# plt.show()

# 特征工程
y_train = train["SalePrice"]
# print(y_train)
# print(y_train.values)
train_len = len(train)
dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
dataset.drop(["SalePrice"], axis=1, inplace=True)
dataset = dataset.fillna(np.nan)
# print(dataset.shape)
# print(dataset.count())

# 处理缺失数据
# train_na = train.isnull().sum()/train_len
# train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)
# missing_data = pd.DataFrame({"Missing Ratio":train_na})
# print(missing_data.head(20))

# Correlation map to see how features are correlated with SalePrice
# plt.subplots()
# sns.heatmap(train.corr(), annot=True, fmt=".2f", cmap="coolwarm")
# plt.show()

# 填补缺失值
# 当缺失值比率超过95%以上时，可以考虑将此特征删除，
# 当缺失值比率一般时，可以填充None，或者根据其他特征将其拟合上
# 当缺失值很少时，可以用该特征取值次数最多的那个值填充,或者填充None(此时需根据特征实际意义来决定)
# print(dataset["MiscFeature"].isnull().sum())
dataset["PoolQC"] = dataset["PoolQC"].fillna("None")
dataset["MiscFeature"] = dataset["MiscFeature"].fillna("None")
dataset["Alley"] = dataset["Alley"].fillna("None")
dataset["Fence"] = dataset["Fence"].fillna("None")
dataset["FireplaceQu"] = dataset["FireplaceQu"].fillna("None")

# print(dataset["LotFrontage"].isnull().sum())
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
# 用与此房子相邻的房子的LotFrontage中位数填充缺失值
dataset["LotFrontage"] = dataset.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

for col in ("GarageType", "GarageFinish", "GarageQual", "GarageCond"):
    dataset[col] = dataset[col].fillna("None")

for col in ("GarageYrBlt", "GarageArea", "GarageCars"):
    dataset[col] = dataset[col].fillna(0)

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    dataset[col] = dataset[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    dataset[col] = dataset[col].fillna("None")

dataset["MasVnrArea"] = dataset["MasVnrArea"].fillna(0)
dataset["MasVnrType"] = dataset["MasVnrType"].fillna("None")

# MSZoning特征中RL最常见，所以缺失值填充RL
dataset["MSZoning"] = dataset["MSZoning"].fillna(dataset["MSZoning"].mode()[0])

# Utilities特征有1个取值NoSeWa,2个缺失值，其余全为AllPub，对建模作用不大，删除
dataset = dataset.drop(["Utilities"], axis=1)   # 如果加上关键字参数inplace=True后面会出现NoneType问题

# Functional: data description says NA means typical
# print(dataset["Functional"].isnull().sum())
dataset["Functional"] = dataset["Functional"].fillna("Typ")

# Electrical有一个缺失值，用取值最多的那个填充
# print(dataset["Electrical"].isnull().sum())
# print(dataset["Electrical"].mode()[0])
dataset["Electrical"] = dataset["Electrical"].fillna(dataset["Electrical"].mode()[0])

# KitchenQual跟Electrical一样处理
# print(dataset["KitchenQual"].isnull().sum())
dataset["KitchenQual"] = dataset["KitchenQual"].fillna(dataset["KitchenQual"].mode()[0])

# Exterior1st and Exterior2nd 都只有一个缺失值，用出现最多的那个值填充
dataset["Exterior1st"] = dataset["Exterior1st"].fillna(dataset["Exterior1st"].mode()[0])
dataset["Exterior2nd"] = dataset["Exterior2nd"].fillna(dataset["Exterior2nd"].mode()[0])

# SaleType有一个缺失值
# print(dataset["SaleType"].isnull().sum())
dataset["SaleType"] = dataset["SaleType"].fillna(dataset["SaleType"].mode()[0])

# 查看是否还有缺失值
# dataset_na = dataset.isnull().sum()/len(dataset)
# dataset_na = dataset_na.drop(dataset_na[dataset_na == 0].index).sort_values(ascending=False)
# print(dataset_na)

# 至此缺失值已全部填充完毕
# 接下来将与预测值SalePrice无线性关系的数值特征转化为分类特征
dataset["MSSubClass"] = dataset["MSSubClass"].apply(str)    # 由int64型转化为object
dataset["OverallCond"] = dataset["OverallCond"].astype(str)    # 由int64型转化为object
dataset["YrSold"] = dataset["YrSold"].astype(str)    # 由int64型转化为object
dataset["MoSold"] = dataset["MoSold"].astype(str)    # 由int64型转化为object
# 这些object类型的特征经过下面的编码之后都会变为int64型
# print(dataset.info())
# print(dataset.dtypes)
# 将一些分类特征的取值编码，以体现出他们之间的顺序
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features
for c in cols:
    # print(dataset[c].dtypes)
    lbl = LabelEncoder()
    lbl.fit(list(dataset[c].values))
    dataset[c] = lbl.transform(list(dataset[c].values))
# print(dataset.dtypes)
# 经上述操作之后，上面的分类特征的取值都用数字代替，数字为排序之后的序号，且上述特征都变为int64型了
# for c in cols:
#     print(dataset[c].dtypes)


# 添加一个新特征
# 因为面积与房价息息相关，所以一个新特征表示地下室、一楼和二楼的总面积
dataset["TotalSF"] = dataset["TotalBsmtSF"] + dataset["1stFlrSF"] + dataset["2ndFlrSF"]

# Skewed features
numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index
# Check the skew of all numerical features，这一步作用是啥还不知道
skewed_feats = dataset[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({"Skew":skewed_feats})
# print(skewed_feats)

# Box Cox Transformation of (highly) skewed features
skewness = skewness[abs(skewness.Skew) > 0.75]

from scipy import special
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    dataset[feat] = boxcox1p(dataset[feat], lam)

# Getting dummy categorical features,get_dummies只对object特征起作用？
dataset = pd.get_dummies(dataset)
# print(dataset.shape)
# print(dataset.dtypes)

train = dataset[:train_len]
test = dataset[train_len:]

# 训练模型
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb 
import lightgbm as lgb 

# 交叉验证
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    print("rmsle中y_train:", type(y_train))
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)

def rmsle_cv2(model):
    model.fit(train, y_train)
    y_train_pred = model.predict(train)
    rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    return (rmse)


# 单模型
# 1.LASSO回归：该模型对异常值比较敏感，所以在pipeline上使用RobustScaler()方法
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))

# 2.ElasticNet回归:对异常值也敏感
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3))

# 3.Kernel Ridge回归
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

# 4.Gradient Boosting回归
# huber损失使其对异常值具有鲁棒性，Huber loss是为了增强平方误差损失函数对噪声（或叫离群点，outliers）的鲁棒性提出的。
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=5)

# 5.xgboost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
learning_rate=0.05, max_depth=3, min_child_weight=1.7817, 
n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571, 
subsample=0.5213, silent=1, random_state=7, nthread=-1)

# 6.LightGBM
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5, 
learning_rate=0.05, n_estimators=720, 
max_bin=55, bagging_fraction=0.8, 
bagging_freq=5, feature_fraction=0.2319, 
feature_fraction_seed=9, bagging_seed=9, 
min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

# 通过交叉验证误差看看每个模型的表现
# score = rmsle_cv(lasso)
# print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(ENet)
# print("\nElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(KRR)
# print("\nKernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(GBoost)
# print("\nGradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(model_xgb)
# print("\nxgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# score = rmsle_cv(model_lgb)
# print("\nLightGBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# stacking models
# 用最简单的堆叠方法：平均模型法
# 先实现一个类，方便模型融合
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
    
    # 定义原始模型的克隆以拟合数据
    def fit(self, X, y):
        self.models_ = [clone(m) for m in self.models]

        # 训练克隆的模型
        for model in self.models_:
            model.fit(X, y)

        return self
    
    # 接下来做模型预测并将其平均
    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1) # 在列上做平均

# 平均基模型分数，这里只平均ENet,GBoost,KRR和lasso
averaged_models = AveragingModels(models=(ENet, GBoost, model_xgb, lasso))
# score = rmsle_cv(averaged_models)   # 把集成模型当做单模型一样来计算分数
# print("Averaged base models score: {:.4f}({:.4f})\n".format(score.mean(), score.std()))


# Less simple Stacking : Adding a Meta-model
# 在平均基础模型上添加元模型，并使用这些基础模型的折叠预测来训练我们的元模型
# 步骤如下:
# 1.Split the total training set into two disjoint sets (here train and .holdout )
# 2.Train several base models on the first part (train)
# 3.Test these base models on the second part (holdout)
# 4.Use the predictions from 3) (called out-of-folds predictions) as the inputs, 
# and the correct responses (target variable) as the outputs to train a higher level learner 
# called meta-model.
# 将训练集划分为k部分，一部分作为验证集，将每个单学习模型在此验证集上的预测结果作为一个新的特征，
# 此过程迭代的进行k次，最终训练集中的每条记录都会对应一个预测值，
# 这样一来n个单模型对应产生n个特征，由这n个新特征组成的数据集来训练元模型
# Stacking averaged Models Class
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        print("X=", X.shape)
        print("y=", len(y))
        # 下面这句代码不太理解
        self.base_models_ = [clone(m) for m in self.base_models]
        # print("base_models_:", self.base_models_)
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        # out_of_fold_predictions为len(X)*len(self.base_models)的二维数组？
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        print("out_of_fold_predictions:", out_of_fold_predictions.shape)
        print(out_of_fold_predictions)
        for i, model in enumerate(self.base_models):    # 对每个模型
            for train_index, holdout_index in kfold.split(X, y):    # 每个模型做n_splits次迭代
                instance = clone(model)
                # self.base_models_[i].append(instance)
                # print("train_index:", train_index)
                # print("train_index:", holdout_index)
                # print("***********X:", type(X[train_index]))
                # print(X[train_index])
                # print("***********y:", type(y[train_index]))
                # print(y[train_index])
                # print(X.loc[train_index])
                # print(y.loc[train_index])
                instance.fit(X.loc[train_index], y.loc[train_index])    # 有bug
                print("********************")
                y_pred = instance.predict(X.loc[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # 接下来使用out_of_fold_predictions作为新特征训练clone meta_model
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # 让所有的基模型对测试集做预测，把基模型的预测结果给meta_model,然后再预测最终结果
    def predict(self, X):
        # print("**************")
        mata_features = np.column_stack([
            # 每个模型会对测试集预测n_folds次，然后将这n_folds次的结果平均
            np.column_stack([ model.predict(X) for model in base_models ]).mean(axis=1) 
            for base_models in self.base_models_
        ])
        print("meta_features:", mata_features)
        return self.meta_model_.predict(mata_features)

# Stacking Averaged models Score
# 为了比较上面两种stacking方法，我们使用相同的基模型Enet,model_xgb和Gboost,
# 并使用lasso作为meta_model
# stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, model_xgb), 
# meta_model=lasso)
# score = rmsle_cv2(stacked_averaged_models)
# print("Stacking Averaged models score: {:.4f}({:.4f})".format(score.mean(), score.std()))


# Ensembling StackedRegressor, XGBoost and LightGBM
# 将xgb和LightGBM加入到StackedRegressor
# 先定义一个评估函数
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# 进行最终的训练与预测
# StackedRegressor的训练与预测
# stacked_averaged_models.fit(train.values, y_train)
# stacked_train_pred = stacked_averaged_models.predict(train.values)
# stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
# print(rmsle(y_train, stacked_train_pred))

# AveragingModels的训练与预测
averaged_models.fit(train.values, y_train)
averaged_train_pred = averaged_models.predict(train.values)
averaged_pred = np.expm1(averaged_models.predict(test.values))
print("averaged models:", rmsle(y_train, averaged_train_pred))

# XGBoost的训练与预测
# model_xgb.fit(train, y_train)
# xgb_train_pred = model_xgb.predict(train)
# xgb_pred = np.expm1(model_xgb.predict(test))
# print("xgb:", rmsle(y_train, xgb_train_pred))

# # LightGBM的训练与预测
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print("lgb", rmsle(y_train, lgb_train_pred))

# # 平均这几个模型
# print("rmsle score on train data:")
# print(rmsle(y_train, averaged_train_pred*0.7 + xgb_train_pred*0.15 + lgb_train_pred*0.15))

# 集成预测结果
ensemble = averaged_pred*0.7 + lgb_pred*0.3


# 提交预测结果
sub = pd.DataFrame()
sub["Id"] = test_id
sub["SalePrice"] = ensemble
# sub["SalePrice"] = averaged_pred
sub.to_csv("ensemble_submission.csv", index=False)




# dataset.isnull().sum().to_csv("missing_count.csv")
# print(dataset.head())
# print(dataset.columns)

# for feature in dataset.columns:
#     dataset[feature].value_counts().to_csv("value_counts.csv", mode='a')

# print(train["Utilities"].describe())
# print(train["PoolQC"].value_counts())

# # 对每个特征，统计其每种取值的记录数
# g = sns.countplot(train["Functional"])
# # g.set_xticklabels(["Master", "Miss/Ms/Mme/Mlle/Mrs", "Mr", "Rare"])
# plt.show()

# 将特征分为数值特征和类别特征
# 1.对于类别特征与预测值的关系一般用箱线图boxplot来观测，这里是训练集train中的特征与SalePrice的关系
# g = sns.boxplot(x="RoofStyle", y="SalePrice", data=train)
# plt.show()

# 直方图也可以观察类别特征的分布
# g = sns.barplot(x="MSZoning", y="SalePrice", data=train)
# plt.show()


# 2.对于数值特征与预测值的关系一般用回归图lmplot和regplot来观测
# 线性回归图
# g = sns.lmplot(x="YearRemodAdd", y="SalePrice", data=train)
# # g = sns.regplot(x="LotArea", y="SalePrice", data=train)
# plt.show()

