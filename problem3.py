import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文字体和图表风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


class YChromosomeAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self._load_data()
        self.processed_data = None
        self.label_encoders = {}

        # 模型结果缓存
        self.logreg_model = None
        self.rf_model = None
        self.predict_pass_rate = None
        self.feature_importance = None
        self.logreg_report = None

    def _load_data(self):
        """加载并预处理数据"""
        try:
            excel_file = pd.ExcelFile(self.file_path)
            df = excel_file.parse('男胎检测数据')

            # 提取检测孕周的数字部分
            df['检测孕周数'] = df['检测孕周'].str.extract('(\d+)').astype(int)

            # 只保留孕期在 10 周 - 25 周之间的数据
            df = df[(df['检测孕周数'] >= 10) & (df['检测孕周数'] <= 25)]

            # 处理数值类型字段
            if '怀孕次数' in df.columns:
                df['怀孕次数'] = pd.to_numeric(df['怀孕次数'], errors='coerce').fillna(0).astype(int)
            if '生产次数' in df.columns:
                df['生产次数'] = pd.to_numeric(df['生产次数'], errors='coerce').fillna(0).astype(int)

            # 计算 Y 染色体浓度是否达标
            df['Y染色体浓度达标'] = df['Y染色体浓度'] >= 0.04

            # 模拟检测误差（标准差为实际值的5%）
            np.random.seed(42)
            df['Y染色体浓度误差'] = df['Y染色体浓度'] + np.random.normal(0, 0.05 * df['Y染色体浓度'], len(df))
            df['Y染色体浓度误差达标'] = df['Y染色体浓度误差'] >= 0.04

            # 处理日期字段
            if '末次月经' in df.columns:
                df['末次月经'] = pd.to_datetime(df['末次月经'], errors='coerce')
            if '检测日期' in df.columns:
                df['检测日期'] = pd.to_datetime(df['检测日期'], errors='coerce')
                if '末次月经' in df.columns and '检测日期' in df.columns:
                    df['怀孕天数'] = (df['检测日期'] - df['末次月经']).dt.days

            return df
        except Exception as e:
            print(f"数据加载错误: {e}")
            return pd.DataFrame()

    def feature_engineering(self):
        """特征工程"""
        if self.df.empty:
            return

        self.processed_data = self.df.copy()

        # 年龄分组
        age_bins = [0, 25, 30, 35, 40, float('inf')]
        age_labels = ['<25', '25-30', '30-35', '35-40', '≥40']
        self.processed_data['年龄分组'] = pd.cut(self.processed_data['年龄'], bins=age_bins, labels=age_labels)

        # BMI分组
        bmi_bins = [0, 18.5, 24, 28, 32, 36, float('inf')]
        bmi_labels = ['偏轻', '正常', '超重', '肥胖I', '肥胖II', '肥胖III']
        self.processed_data['BMI分组'] = pd.cut(self.processed_data['孕妇BMI'], bins=bmi_bins, labels=bmi_labels)

        # 编码分类变量
        categorical_vars = ['IVF妊娠', '染色体的非整倍体', '胎儿是否健康']
        for var in categorical_vars:
            if var in self.processed_data.columns:
                le = LabelEncoder()
                self.processed_data[var] = self.processed_data[var].fillna('未知')
                self.processed_data[var + '_编码'] = le.fit_transform(self.processed_data[var])
                self.label_encoders[var] = le

        # 创建复合特征
        if '身高' in self.processed_data.columns and '体重' in self.processed_data.columns:
            self.processed_data['身高体重比'] = self.processed_data['身高'] / self.processed_data['体重']

        if '怀孕次数' in self.processed_data.columns and '生产次数' in self.processed_data.columns:
            self.processed_data['流产次数'] = self.processed_data['怀孕次数'] - self.processed_data['生产次数']

        # 创建综合风险指数
        features_to_standardize = ['年龄', '孕妇BMI', '检测孕周数']

        additional_features = ['检测抽血次数', '怀孕次数', '生产次数']
        for feature in additional_features:
            if feature in self.processed_data.columns:
                features_to_standardize.append(feature)

        chromosome_features = ['Y染色体的Z值', 'X染色体的Z值', 'GC含量']
        for feature in chromosome_features:
            if feature in self.processed_data.columns and not pd.api.types.is_categorical_dtype(
                    self.processed_data[feature]):
                features_to_standardize.append(feature)

        scaler = StandardScaler()
        self.processed_data[[f'标准化{feature}' for feature in features_to_standardize]] = scaler.fit_transform(
            self.processed_data[features_to_standardize]
        )

        # 构建综合风险指数
        risk_components = []
        if '标准化年龄' in self.processed_data.columns:
            risk_components.append(self.processed_data['标准化年龄'] * 0.3)
        if '标准化孕妇BMI' in self.processed_data.columns:
            risk_components.append(self.processed_data['标准化孕妇BMI'] * 0.3)
        if '标准化检测抽血次数' in self.processed_data.columns:
            risk_components.append(self.processed_data['标准化检测抽血次数'] * 0.2)
        if '标准化检测孕周数' in self.processed_data.columns:
            risk_components.append(-self.processed_data['标准化检测孕周数'] * 0.4)

        if risk_components:
            self.processed_data['综合风险指数'] = sum(risk_components)

    def descriptive_analysis(self):
        """描述性统计分析"""
        if self.processed_data is None:
            self.feature_engineering()

        print("=== 数据基本统计信息 ===")
        key_features = ['年龄', '孕妇BMI', '检测孕周数', 'Y染色体浓度', '检测抽血次数', '怀孕次数', '生产次数']
        available_features = [f for f in key_features if f in self.processed_data.columns]
        print(self.processed_data[available_features].describe())

        # 计算总体达标率
        overall_rate = self.processed_data['Y染色体浓度达标'].mean()
        print(f"\n总体Y染色体浓度达标率: {overall_rate:.2%}")

        # 计算误差影响
        error_impact = abs(self.processed_data['Y染色体浓度达标'].mean() -
                           self.processed_data['Y染色体浓度误差达标'].mean())
        print(f"检测误差对达标率的影响: {error_impact:.2%}")

        # IVF妊娠与非IVF妊娠的达标率比较
        if 'IVF妊娠' in self.processed_data.columns:
            ivf_comparison = self.processed_data.groupby('IVF妊娠')['Y染色体浓度达标'].agg(
                样本数='count', 达标率='mean').reset_index()
            print("\nIVF妊娠与非IVF妊娠的达标率比较:")
            print(ivf_comparison)

        # 胎儿健康状况与达标率的关系
        if '胎儿是否健康' in self.processed_data.columns:
            health_comparison = self.processed_data.groupby('胎儿是否健康')['Y染色体浓度达标'].agg(
                样本数='count', 达标率='mean').reset_index()
            print("\n胎儿健康状况与达标率的关系:")
            print(health_comparison)

    def multi_factor_analysis(self):
        """多因素分析"""
        if self.processed_data is None:
            self.feature_engineering()

        # 按年龄分组统计达标率
        age_stats = self.processed_data.groupby('年龄分组')['Y染色体浓度达标'].agg(
            样本数='count', 达标数='sum', 达标率='mean').reset_index()

        # 按BMI分组统计达标率
        bmi_stats = self.processed_data.groupby('BMI分组')['Y染色体浓度达标'].agg(
            样本数='count', 达标数='sum', 达标率='mean').reset_index()

        # 按孕周分组统计达标率
        week_stats = self.processed_data.groupby('检测孕周数')['Y染色体浓度达标'].agg(
            样本数='count', 达标数='sum', 达标率='mean').reset_index()

        # 按检测抽血次数统计达标率
        blood_stats = None
        if '检测抽血次数' in self.processed_data.columns:
            blood_stats = self.processed_data.groupby('检测抽血次数')['Y染色体浓度达标'].agg(
                样本数='count', 达标数='sum', 达标率='mean').reset_index()

        # 按IVF妊娠状态统计达标率
        ivf_stats = None
        if 'IVF妊娠' in self.processed_data.columns:
            ivf_stats = self.processed_data.groupby('IVF妊娠')['Y染色体浓度达标'].agg(
                样本数='count', 达标数='sum', 达标率='mean').reset_index()

        return age_stats, bmi_stats, week_stats, blood_stats, ivf_stats

    def predict_optimal_time(self):
        """使用机器学习模型预测最佳检测时间"""
        if self.processed_data is None:
            self.feature_engineering()

        # 准备特征和目标变量
        base_features = ['年龄', '孕妇BMI', '检测孕周数']

        additional_features = ['检测抽血次数', '怀孕次数', '生产次数']
        for feature in additional_features:
            if feature in self.processed_data.columns:
                base_features.append(feature)

        # 添加编码后的分类特征
        encoded_features = [col for col in self.processed_data.columns if col.endswith('_编码')]
        base_features.extend(encoded_features)

        # 添加染色体相关指标
        chromosome_features = ['Y染色体的Z值', 'X染色体的Z值', 'GC含量']
        for feature in chromosome_features:
            if feature in self.processed_data.columns and not pd.api.types.is_categorical_dtype(
                    self.processed_data[feature]):
                base_features.append(feature)

        # 过滤掉不存在的特征
        available_features = [f for f in base_features if f in self.processed_data.columns]
        X = self.processed_data[available_features]
        y = self.processed_data['Y染色体浓度达标']

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 训练逻辑回归模型
        self.logreg_model = LogisticRegression(random_state=42, max_iter=1000)
        self.logreg_model.fit(X_train, y_train)

        # 训练随机森林模型
        self.rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
        self.rf_model.fit(X_train, y_train)

        # 模型评估
        logreg_pred = self.logreg_model.predict(X_test)
        self.logreg_report = classification_report(y_test, logreg_pred)
        print("\n=== 逻辑回归模型评估 ===")
        print(self.logreg_report)

        # 获取随机森林的特征重要性
        self.feature_importance = pd.DataFrame({
            '特征': available_features,
            '重要性': self.rf_model.feature_importances_
        }).sort_values('重要性', ascending=False)

        print("\n特征重要性排序 (基于随机森林):")
        print(self.feature_importance)

        # 创建预测函数
        def predict_pass_rate(input_features):
            weeks = np.arange(10, 26)
            predictions = []

            feature_dict = input_features.copy()

            for week in weeks:
                feature_dict['检测孕周数'] = week
                feature_vector = [feature_dict.get(f, 0) for f in available_features]

                while len(feature_vector) < len(available_features):
                    feature_vector.append(0)
                feature_vector = feature_vector[:len(available_features)]

                prob = self.logreg_model.predict_proba([feature_vector])[0][1]
                predictions.append((week, prob))

            return predictions

        self.predict_pass_rate = predict_pass_rate

        return self.logreg_model, self.rf_model, self.predict_pass_rate, self.feature_importance, self.logreg_report

    def visualize_results(self, folder_path):
        """可视化分析结果"""
        if self.processed_data is None:
            self.feature_engineering()

        # 确保模型已训练
        if not self.predict_pass_rate or self.feature_importance is None:
            self.predict_optimal_time()

        # 获取分析数据
        age_stats, bmi_stats, week_stats, blood_stats, ivf_stats = self.multi_factor_analysis()
        predict_pass_rate = self.predict_pass_rate
        feature_importance = self.feature_importance

        # 创建可视化图表
        fig = plt.figure(figsize=(22, 20))

        # 1. 孕周与Y染色体浓度的关系散点图
        ax1 = plt.subplot(4, 2, 1)
        sns.scatterplot(data=self.processed_data, x='检测孕周数', y='Y染色体浓度', alpha=0.6)
        sns.regplot(data=self.processed_data, x='检测孕周数', y='Y染色体浓度', scatter=False, color='red')
        plt.axhline(y=0.04, color='g', linestyle='--', label='达标阈值(4%)')
        plt.title('孕周与Y染色体浓度的关系')
        plt.xlabel('检测孕周')
        plt.ylabel('Y染色体浓度')
        plt.legend()

        # 2. 年龄分组达标率柱状图
        ax2 = plt.subplot(4, 2, 2)
        sns.barplot(data=age_stats, x='年龄分组', y='达标率', palette='Blues')
        for i, row in age_stats.iterrows():
            ax2.text(i, row['达标率'] + 0.01, f"{row['达标率']:.1%}", ha='center')
        plt.title('不同年龄组的Y染色体浓度达标率')
        plt.xlabel('年龄分组')
        plt.ylabel('达标率')
        plt.ylim(0, 1.1)

        # 3. BMI分组达标率柱状图
        ax3 = plt.subplot(4, 2, 3)
        sns.barplot(data=bmi_stats, x='BMI分组', y='达标率', palette='Greens')
        for i, row in bmi_stats.iterrows():
            ax3.text(i, row['达标率'] + 0.01, f"{row['达标率']:.1%}", ha='center')
        plt.title('不同BMI组的Y染色体浓度达标率')
        plt.xlabel('BMI分组')
        plt.ylabel('达标率')
        plt.ylim(0, 1.1)

        # 4. 孕周达标率趋势图
        ax4 = plt.subplot(4, 2, 4)
        sns.lineplot(data=week_stats, x='检测孕周数', y='达标率', marker='o', linewidth=2, label='达标率趋势')
        plt.fill_between(week_stats['检测孕周数'],
                         week_stats['达标率'] - stats.sem(week_stats['达标率']),
                         week_stats['达标率'] + stats.sem(week_stats['达标率']),
                         alpha=0.2, label='95%置信区间')
        plt.title('不同孕周的Y染色体浓度达标率趋势')
        plt.xlabel('检测孕周')
        plt.ylabel('达标率')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # 5. 检测抽血次数与达标率的关系
        ax5 = plt.subplot(4, 2, 5)
        if blood_stats is not None:
            sns.barplot(data=blood_stats, x='检测抽血次数', y='达标率', palette='Purples')
            for i, row in blood_stats.iterrows():
                ax5.text(i, row['达标率'] + 0.01, f"{row['达标率']:.1%}", ha='center')
            plt.title('检测抽血次数与Y染色体浓度达标率的关系')
            plt.xlabel('检测抽血次数')
            plt.ylabel('达标率')
            plt.ylim(0, 1.1)
        else:
            ax5.text(0.5, 0.5, '无检测抽血次数数据', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('检测抽血次数与Y染色体浓度达标率的关系')

        # 6. 误差影响可视化
        ax6 = plt.subplot(4, 2, 6)
        error_comparison = self.processed_data.groupby('检测孕周数')[
            ['Y染色体浓度达标', 'Y染色体浓度误差达标']].mean().reset_index()
        error_comparison_melt = error_comparison.melt(id_vars='检测孕周数',
                                                      value_vars=['Y染色体浓度达标', 'Y染色体浓度误差达标'],
                                                      var_name='类型', value_name='达标率')

        sns.lineplot(data=error_comparison_melt, x='检测孕周数', y='达标率', hue='类型',
                     marker='o', linewidth=2,
                     palette={'Y染色体浓度达标': 'blue', 'Y染色体浓度误差达标': 'orange'})

        plt.title('检测误差对达标率的影响')
        plt.xlabel('检测孕周')
        plt.ylabel('达标率')

        handles, labels = ax6.get_legend_handles_labels()
        plt.legend(handles=handles[:2], labels=['实际达标率', '考虑5%误差后达标率'], loc='best')

        # 7. 特征重要性图
        ax7 = plt.subplot(4, 2, 7)
        top_n = min(10, len(feature_importance))
        sns.barplot(data=feature_importance.head(top_n), y='特征', x='重要性', palette='Oranges')
        plt.title('影响Y染色体浓度达标的重要因素')
        plt.xlabel('重要性得分')
        plt.ylabel('特征')
        plt.yticks(rotation=15)

        # 8. 不同人群的最佳检测时间预测
        ax8 = plt.subplot(4, 2, 8)

        # 模拟不同人群的预测结果
        populations = [
            {'年龄': 25, '孕妇BMI': 22, '检测抽血次数': 1, '怀孕次数': 1, '生产次数': 0, '标签': '年轻正常BMI初产妇'},
            {'年龄': 35, '孕妇BMI': 22, '检测抽血次数': 1, '怀孕次数': 2, '生产次数': 1, '标签': '高龄正常BMI经产妇'},
            {'年龄': 25, '孕妇BMI': 30, '检测抽血次数': 1, '怀孕次数': 1, '生产次数': 0, '标签': '年轻肥胖初产妇'},
            {'年龄': 35, '孕妇BMI': 30, '检测抽血次数': 2, '怀孕次数': 3, '生产次数': 1, '标签': '高龄肥胖多产妇'}
        ]

        # 为每个群体预测达标概率
        for i, pop in enumerate(populations):
            predictions = predict_pass_rate(pop)
            if predictions:
                weeks, rates = zip(*predictions)
                markers = ['o', 's', '^', 'D']
                ax8.plot(weeks, rates, marker=markers[i], label=pop['标签'])

                optimal_week = weeks[np.argmax(rates)]
                optimal_rate = max(rates)

                offset_x = 0.5 if i % 2 == 0 else -2.5
                offset_y = 0.05 if i < 2 else -0.15

                ax8.annotate(f'最佳: {optimal_week}周',
                             xy=(optimal_week, optimal_rate),
                             xytext=(optimal_week + offset_x, optimal_rate + offset_y),
                             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

        plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80%信心阈值')
        plt.title('不同人群的最佳检测时间预测')
        plt.xlabel('检测孕周')
        plt.ylabel('预测达标概率')
        plt.legend(loc='lower left', bbox_to_anchor=(0, 0))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle('男胎Y染色体浓度达标时间多因素综合分析', fontsize=16, y=1.02)

        # 保存图表
        img_path = os.path.join(folder_path, "Y染色体浓度达标时间综合分析结果.png")
        plt.savefig(img_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ 可视化图表已保存至: {os.path.abspath(img_path)}")

    def save_processed_data(self, folder_path):
        """保存处理后的完整数据集"""
        if self.processed_data is None:
            self.feature_engineering()

        if not self.processed_data.empty:
            csv_path = os.path.join(folder_path, "processed_data.csv")
            self.processed_data.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ 处理后数据已保存至: {os.path.abspath(csv_path)}")
        else:
            print("❌ 无处理后数据可保存")

    def generate_conclusion(self, folder_path, age_stats, bmi_stats, week_stats, blood_stats, ivf_stats):
        """生成结果报告（仅包含数据结果和运行日志）"""
        conclusion_path = os.path.join(folder_path, "conclusion.txt")

        # 基础数据收集
        total_samples = len(self.df) if not self.df.empty else 0
        processed_samples = len(self.processed_data) if not self.processed_data.empty else 0
        week_range = f"{self.df['检测孕周数'].min()} - {self.df['检测孕周数'].max()}周" if not self.df.empty else "无数据"
        overall_rate = self.processed_data['Y染色体浓度达标'].mean() if not self.processed_data.empty else 0.0
        error_impact = abs(self.processed_data['Y染色体浓度达标'].mean() - self.processed_data[
            'Y染色体浓度误差达标'].mean()) if not self.processed_data.empty else 0.0

        # 构建报告内容
        conclusion_content = f"""# Y染色体浓度分析结果报告
{'=' * 60}

## 一、数据概况
- 原始数据文件: {os.path.abspath(self.file_path)}
- 原始样本总量: {total_samples} 例
- 筛选后样本量: {processed_samples} 例（10-25周数据）
- 检测孕周范围: {week_range}
- 数据时间范围: {self.df['检测日期'].min().strftime('%Y-%m-%d') if '检测日期' in self.df.columns and not pd.isna(self.df['检测日期'].min()) else '未知'} 至 {self.df['检测日期'].max().strftime('%Y-%m-%d') if '检测日期' in self.df.columns and not pd.isna(self.df['检测日期'].max()) else '未知'}

## 二、描述性统计结果
### 核心指标统计
"""

        # 添加关键指标描述性统计
        key_features = ['年龄', '孕妇BMI', '检测孕周数', 'Y染色体浓度', '检测抽血次数', '怀孕次数', '生产次数']
        available_features = [f for f in key_features if f in self.processed_data.columns]
        if available_features:
            desc_stats = self.processed_data[available_features].describe().round(3)
            conclusion_content += f"\n{desc_stats.to_string()}\n"
        else:
            conclusion_content += "\n无可用关键指标统计数据\n"

        conclusion_content += f"""
### 达标情况统计
- Y染色体浓度达标标准: 浓度 ≥ 0.04
- 总体达标率: {overall_rate:.2%}（{int(overall_rate * processed_samples)} 例达标 / {processed_samples} 例总样本）
- 检测误差影响: {error_impact:.2%}

### 分组达标率数据
#### IVF妊娠与非IVF妊娠对比
"""

        if ivf_stats is not None and not ivf_stats.empty:
            conclusion_content += f"\n{ivf_stats.to_string(index=False, float_format=lambda x: f'{x:.2%}')}\n"
        else:
            conclusion_content += "\n无IVF妊娠相关数据\n"

        conclusion_content += f"""
#### 胎儿健康状况对比
"""

        if '胎儿是否健康' in self.processed_data.columns:
            health_stats = self.processed_data.groupby('胎儿是否健康')['Y染色体浓度达标'].agg(
                样本数='count', 达标率='mean').reset_index()
            conclusion_content += f"\n{health_stats.to_string(index=False, float_format=lambda x: f'{x:.2%}')}\n"
        else:
            conclusion_content += "\n无胎儿健康状况相关数据\n"

        conclusion_content += f"""
## 三、多因素分析数据
### 年龄分组达标率
"""

        if not age_stats.empty:
            conclusion_content += f"\n{age_stats.to_string(index=False, float_format=lambda x: f'{x:.2%}')}\n"
        else:
            conclusion_content += "\n无年龄分组达标率数据\n"

        conclusion_content += f"""
### BMI分组达标率
"""

        if not bmi_stats.empty:
            conclusion_content += f"\n{bmi_stats.to_string(index=False, float_format=lambda x: f'{x:.2%}')}\n"
        else:
            conclusion_content += "\n无BMI分组达标率数据\n"

        conclusion_content += f"""
### 孕周趋势达标率
"""

        if not week_stats.empty:
            conclusion_content += f"\n{week_stats.to_string(index=False, float_format=lambda x: f'{x:.2%}')}\n"
        else:
            conclusion_content += "\n无孕周达标率趋势数据\n"

        conclusion_content += f"""
### 检测抽血次数影响
"""

        if blood_stats is not None and not blood_stats.empty:
            conclusion_content += f"\n{blood_stats.to_string(index=False, float_format=lambda x: f'{x:.2%}')}\n"
        else:
            conclusion_content += "\n无检测抽血次数相关数据\n"

        conclusion_content += f"""
## 四、机器学习模型结果
### 逻辑回归模型评估报告
{self.logreg_report if self.logreg_report else '模型未训练，无评估报告'}

### 特征重要性排序（随机森林模型）
"""

        if self.feature_importance is not None and not self.feature_importance.empty:
            top_features = self.feature_importance.head(10).round(4)
            conclusion_content += f"\n{top_features.to_string(index=False)}\n"
        else:
            conclusion_content += "\n无特征重要性数据\n"

        conclusion_content += f"""
## 五、典型人群预测结果
1. 年轻正常BMI初产妇: 最佳检测孕周 {self._get_optimal_week({'年龄': 25, '孕妇BMI': 22, '检测抽血次数': 1, '怀孕次数': 1, '生产次数': 0})} 周（预测达标率 {self._get_optimal_rate({'年龄': 25, '孕妇BMI': 22, '检测抽血次数': 1, '怀孕次数': 1, '生产次数': 0}):.2%}）
2. 高龄正常BMI经产妇: 最佳检测孕周 {self._get_optimal_week({'年龄': 35, '孕妇BMI': 22, '检测抽血次数': 1, '怀孕次数': 2, '生产次数': 1})} 周（预测达标率 {self._get_optimal_rate({'年龄': 35, '孕妇BMI': 22, '检测抽血次数': 1, '怀孕次数': 2, '生产次数': 1}):.2%}）
3. 年轻肥胖初产妇: 最佳检测孕周 {self._get_optimal_week({'年龄': 25, '孕妇BMI': 30, '检测抽血次数': 1, '怀孕次数': 1, '生产次数': 0})} 周（预测达标率 {self._get_optimal_rate({'年龄': 25, '孕妇BMI': 30, '检测抽血次数': 1, '怀孕次数': 1, '生产次数': 0}):.2%}）
4. 高龄肥胖多产妇: 最佳检测孕周 {self._get_optimal_week({'年龄': 35, '孕妇BMI': 30, '检测抽血次数': 2, '怀孕次数': 3, '生产次数': 1})} 周（预测达标率 {self._get_optimal_rate({'年龄': 35, '孕妇BMI': 30, '检测抽血次数': 2, '怀孕次数': 3, '生产次数': 1}):.2%}）

## 六、输出文件说明
分析结果文件存储于 {os.path.abspath(folder_path)}，包含：
1. conclusion.txt - 本报告文件
2. processed_data.csv - 处理后的完整数据集
3. Y染色体浓度达标时间综合分析结果.png - 可视化图表

---
报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        # 写入报告文件
        with open(conclusion_path, 'w', encoding='utf-8') as f:
            f.write(conclusion_content)
        print(f"✅ 结果报告已保存至: {os.path.abspath(conclusion_path)}")

    def _get_optimal_week(self, input_features):
        """获取特定人群的最佳检测孕周"""
        if not self.predict_pass_rate:
            self.predict_optimal_time()
        predictions = self.predict_pass_rate(input_features)
        return int(max(predictions, key=lambda x: x[1])[0]) if predictions else "未知"

    def _get_optimal_rate(self, input_features):
        """获取特定人群的最佳达标率"""
        if not self.predict_pass_rate:
            self.predict_optimal_time()
        predictions = self.predict_pass_rate(input_features)
        return round(max(predictions, key=lambda x: x[1])[1], 4) if predictions else 0.0


# 主函数：执行完整分析流程
if __name__ == "__main__":
    # 配置文件路径
    INPUT_FILE = r"附件.xlsx"
    RESULT_FOLDER = "Problem3 results"

    # 初始化环境
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    print(f"📂 已创建/确认结果文件夹: {os.path.abspath(RESULT_FOLDER)}")

    # 数据加载与初始化
    print("\n🔍 开始加载数据...")
    analyzer = YChromosomeAnalysis(INPUT_FILE)
    if analyzer.df.empty:
        print("❌ 数据加载失败，程序终止")
        exit()
    print(f"✅ 数据加载成功（原始样本数：{len(analyzer.df)}）")

    # 执行核心分析流程
    print("\n📊 开始特征工程与描述性分析...")
    analyzer.feature_engineering()
    analyzer.descriptive_analysis()

    print("\n💾 保存处理后数据...")
    analyzer.save_processed_data(RESULT_FOLDER)

    print("\n🤖 开始机器学习模型训练...")
    analyzer.predict_optimal_time()

    print("\n📈 开始可视化分析...")
    analyzer.visualize_results(RESULT_FOLDER)

    print("\n📝 开始生成结果报告...")
    age_stats, bmi_stats, week_stats, blood_stats, ivf_stats = analyzer.multi_factor_analysis()
    analyzer.generate_conclusion(RESULT_FOLDER, age_stats, bmi_stats, week_stats, blood_stats, ivf_stats)

    # 分析完成提示
    print(f"\n{'=' * 60}")
    print("🎉 所有分析流程完成！")
    print(f"📁 所有结果文件已集中保存至：{os.path.abspath(RESULT_FOLDER)}")
    print("📋 输出文件清单：")
    print("  1. conclusion.txt - 结果报告")
    print("  2. processed_data.csv - 处理后的数据集")
    print("  3. Y染色体浓度达标时间综合分析结果.png - 可视化图表")
    print(f"{'=' * 60}")
