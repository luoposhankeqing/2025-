import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from pygam import GAM, s, l, LinearGAM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.font_manager as fm
import warnings
from io import StringIO
from datetime import datetime

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams["font.family"] = ["Times New Roman", "SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号
plt.rcParams["figure.dpi"] = 300  # 高分辨率输出
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

# 科研级配色方案
model_colors = {
    'Linear': '#3498db',  # 亮蓝
    'Polynomial': '#2ecc71',  # 亮绿
    'GAM': '#9b59b6',  # 亮紫
    'RandomForest': '#e67e22',  # 亮橙
    'NeuralNet': '#e74c3c'  # 亮红
}


class FetalYAnalysis:
    """胎儿Y染色体浓度分析工具，整合多种模型比较和GAM深入分析"""

    def __init__(self, output_dir="Problem1 results"):
        # 初始化字体设置
        self.simhei_font = self._setup_font()
        # 初始化数据和模型变量
        self.data = None
        self.models = {}
        self.results = []
        self.y_preds = {}
        # 设置输出目录
        self.base_dir = output_dir
        self.compare_dir = os.path.join(self.base_dir, "Model_Comparison")
        self.gam_dir = os.path.join(self.base_dir, "GAM_Detailed_Analysis")
        self._setup_directories()

        # 添加日志捕获和结论相关变量
        self.log_capture_string = StringIO()
        self.original_stdout = sys.stdout
        self.conclusion_path = os.path.join(self.base_dir, "conclusion.txt")

    def _setup_font(self):
        """设置中文字体"""
        try:
            # 尝试加载Windows系统中的SimHei字体
            font_path = os.path.join(os.environ.get('WINDIR', 'C:\Windows'), 'Fonts', 'simhei.ttf')
            if os.path.exists(font_path):
                simhei_font = fm.FontProperties(fname=font_path)
            else:
                simhei_font = fm.FontProperties(family='SimHei')
            return simhei_font
        except Exception as e:
            print(f"使用系统默认中文字体配置，错误: {e}")
            return fm.FontProperties(family='SimHei')

    def _setup_directories(self):
        """创建输出目录"""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.compare_dir, exist_ok=True)
        os.makedirs(self.gam_dir, exist_ok=True)
        print(f"所有结果将保存至: {os.path.abspath(self.base_dir)}")

    @staticmethod
    def convert_ga(ga_str):
        """将孕周格式从'11w+6'或'16W+1'转换为数值型，支持大小写W"""
        if pd.isna(ga_str):
            return np.nan

        ga_str = str(ga_str).lower()

        if 'w' in ga_str:
            parts = ga_str.split('w')
            try:
                weeks = int(parts[0])
            except ValueError:
                return np.nan

            days = 0
            if len(parts) > 1 and '+' in parts[1]:
                try:
                    days = int(parts[1].split('+')[1])
                except (ValueError, IndexError):
                    days = 0

            return weeks + days / 7

        try:
            return float(ga_str)
        except ValueError:
            return np.nan

    def load_and_preprocess_data(self, file_path, sheet_name="男胎检测数据"):
        """加载并预处理数据"""
        try:
            # 读取Excel数据
            print(f"正在读取数据: {file_path}")
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # 确保关键列存在
            required_cols = ['检测孕周', '孕妇BMI', 'IVF妊娠', '原始读段数', 'Y染色体浓度']
            if not all(col in df.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"数据缺少必要列: {missing_cols}")

            # 数据预处理
            print("开始数据预处理...")

            # 转换孕周格式
            df['GA'] = df['检测孕周'].apply(self.convert_ga)
            df['gestational_age'] = df['GA']  # 为兼容不同模型命名

            # 重命名列名便于统一处理
            df = df.rename(columns={
                '孕妇BMI': 'BMI',
                'Y染色体浓度': 'fetal_Y_concentration'
            })

            # 过滤异常值
            df = df[(df['GA'] >= 10) & (df['GA'] <= 40) &
                    (df['BMI'] >= 18) & (df['BMI'] <= 40) &
                    (df['fetal_Y_concentration'] > 0)]

            # 生成关键变量
            df['Yc'] = df['fetal_Y_concentration']
            df['log_total_reads'] = np.log1p(df['原始读段数'])
            df['IVF'] = (df['IVF妊娠'] == 'IVF妊娠').astype(int)  # 将IVF妊娠转换为数值型

            # BMI分组
            df['BMI分组'] = pd.cut(
                df['BMI'],
                bins=[20, 28, 32, 36, 40, float('inf')],
                labels=['20-28', '28-32', '32-36', '36-40', '40以上']
            )

            # 孕周分组
            df['孕周分组'] = pd.cut(
                df['GA'],
                bins=[0, 12, 27, float('inf')],
                labels=['早期发现(≤12周)', '中期发现(13-27周)', '晚期发现(≥28周)']
            )

            # 检查并处理缺失值
            df = df.dropna(subset=['GA', 'BMI', 'IVF', 'log_total_reads', 'Yc', 'BMI分组', '孕周分组',
                                   'gestational_age', 'fetal_Y_concentration'])

            print(f"数据预处理完成，保留了 {len(df)} 条有效数据")
            self.data = df
            return df
        except Exception as e:
            print(f"数据加载和预处理失败: {e}")
            raise

    def train_comparison_models(self):
        """训练多种模型进行比较"""
        if self.data is None:
            raise ValueError("请先加载数据")

        try:
            print("开始训练比较模型...")

            # 定义用于比较的特征和目标变量
            X = self.data[['gestational_age', 'BMI']].values
            y = self.data['fetal_Y_concentration'].values

            # 定义比较模型
            self.models = {
                'Linear': LinearRegression(),
                'Polynomial': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
                'GAM': LinearGAM(s(0) + s(1), fit_intercept=True),
                'RandomForest': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_split=5,
                    random_state=42
                ),
                'NeuralNet': make_pipeline(
                    StandardScaler(),
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64, 32),
                        activation='relu',
                        solver='adam',
                        learning_rate='adaptive',
                        learning_rate_init=0.001,
                        max_iter=3000,
                        alpha=0.001,
                        early_stopping=True,
                        validation_fraction=0.2,
                        random_state=42,
                        verbose=False
                    )
                )
            }

            # 训练模型并收集结果
            self.results = []
            self.y_preds = {}

            for name, model in self.models.items():
                print(f"正在训练 {name} 模型...")
                model.fit(X, y)
                y_pred = model.predict(X)
                residuals = y - y_pred

                # 计算评估指标
                mse = mean_squared_error(y, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)

                self.results.append({
                    "Model": name,
                    "MSE": mse,
                    "RMSE": rmse,
                    "MAE": mae,
                    "R2": r2
                })
                self.y_preds[name] = y_pred

                # 保存单个模型可视化结果
                self._save_model_plots(name, y, y_pred, residuals)

            # 保存汇总结果
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(os.path.join(self.compare_dir, "metrics.csv"), index=False, encoding='utf-8-sig')
            print("模型比较训练完成!")

            # 生成模型对比可视化
            self._generate_comparison_plots(y)

        except Exception as e:
            print(f"模型训练失败: {e}")
            raise

    def _save_model_plots(self, name, y_true, y_pred, residuals):
        """保存单个模型的可视化结果"""
        model_dir = os.path.join(self.compare_dir, name)
        os.makedirs(model_dir, exist_ok=True)
        color = model_colors[name]

        # 1. 真实 vs 预测（带完美预测线）
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, color=color, edgecolor='white', s=50)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=1.5)
        plt.xlabel("实际值")
        plt.ylabel("预测值")
        plt.title(f"{name}模型: 实际值 vs 预测值")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "actual_vs_pred.png"), bbox_inches='tight')
        plt.close()

        # 2. 残差图（检查误差分布）
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.7, color=color, edgecolor='white', s=50)
        plt.axhline(0, color='r', linestyle='--', linewidth=1.5)
        plt.xlabel("预测值")
        plt.ylabel("残差")
        plt.title(f"{name}模型: 残差分布")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "residuals.png"), bbox_inches='tight')
        plt.close()

        # 3. 预测分布对比（核密度图）
        plt.figure(figsize=(6, 5))
        sns.kdeplot(y_true, label="实际值", linewidth=2, linestyle='--', color='gray')
        sns.kdeplot(y_pred, label="预测值", linewidth=2, color=color)
        plt.xlabel("胎儿Y染色体浓度")
        plt.ylabel("密度")
        plt.title(f"{name}模型: 分布对比")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "distribution.png"), bbox_inches='tight')
        plt.close()

    def _generate_comparison_plots(self, y_true):
        """生成模型对比可视化"""
        # (1) 实际 vs 预测散点对比
        plt.figure(figsize=(8, 6))
        for name, y_pred in self.y_preds.items():
            plt.scatter(y_true, y_pred, alpha=0.5, label=name, color=model_colors[name], s=40)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', linewidth=1.5)
        plt.xlabel("实际值")
        plt.ylabel("预测值")
        plt.title("模型对比: 实际值 vs 预测值")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.compare_dir, "compare_actual_vs_pred.png"), bbox_inches='tight')
        plt.close()

        # (2) 残差分布对比
        plt.figure(figsize=(8, 6))
        for name, y_pred in self.y_preds.items():
            residuals = y_true - y_pred
            sns.kdeplot(residuals, label=name, linewidth=2, color=model_colors[name])
        plt.axvline(0, color='k', linestyle='--', linewidth=1.5)
        plt.xlabel("残差")
        plt.ylabel("密度")
        plt.title("模型对比: 残差分布")
        plt.legend()
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.compare_dir, "compare_residuals.png"), bbox_inches='tight')
        plt.close()

        # (3) 预测分布对比
        plt.figure(figsize=(8, 6))
        sns.kdeplot(y_true, label="实际值", linewidth=3, linestyle='--', color='gray')
        for name, y_pred in self.y_preds.items():
            sns.kdeplot(y_pred, label=name, linewidth=2, color=model_colors[name])
        plt.xlabel("胎儿Y染色体浓度")
        plt.ylabel("密度")
        plt.title("模型对比: 预测分布")
        plt.legend()
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.compare_dir, "compare_distributions.png"), bbox_inches='tight')
        plt.close()

        # (4) 性能指标柱状图（每个指标单独绘图）
        results_df = pd.DataFrame(self.results)
        metrics = ["MSE", "RMSE", "MAE", "R2"]
        for metric in metrics:
            plt.figure(figsize=(8, 5))
            sns.barplot(
                x="Model",
                y=metric,
                data=results_df,
                palette=[model_colors[name] for name in results_df["Model"]]
            )
            plt.title(f"模型性能对比: {metric}")
            plt.xticks(rotation=15)
            plt.grid(alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(self.compare_dir, f"bar_{metric}.png"), bbox_inches='tight')
            plt.close()

        # (5) 性能雷达图（标准化后）
        labels = metrics
        num_vars = len(labels)

        # 计算角度
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        for i, row in results_df.iterrows():
            values = row[metrics].tolist()
            # 标准化指标（R2越大越好，其他越小越好）
            norm_values = []
            for j, m in enumerate(metrics):
                if m == "R2":
                    # R2标准化：除以最大值
                    val = values[j] / results_df[m].max() if results_df[m].max() != 0 else 0
                else:
                    # 误差指标标准化：最大值除以当前值（值越大性能越好）
                    val = results_df[m].max() / values[j] if values[j] != 0 else 0
                norm_values.append(min(val, 1.0))  # 限制最大值为1.0

            norm_values += norm_values[:1]
            ax.plot(angles, norm_values, label=row["Model"],
                    linewidth=2, color=model_colors[row["Model"]])
            ax.fill(angles, norm_values, alpha=0.2, color=model_colors[row["Model"]])

        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_ylim(0, 1.1)  # 标准化后范围在0-1之间
        plt.title("模型性能雷达图（标准化）")
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.compare_dir, "radar_performance.png"), bbox_inches='tight')
        plt.close()

    def train_detailed_gam_model(self, features=['GA', 'BMI', 'IVF', 'log_total_reads'], target='Yc'):
        """训练详细的GAM模型用于深入分析"""
        if self.data is None:
            raise ValueError("请先加载数据")

        try:
            print("开始训练详细GAM模型...")
            self.X_train = self.data[features].values
            self.y_train = self.data[target].values

            # 定义GAM模型结构，s表示平滑项，l表示线性项
            self.detailed_gam = GAM(s(0) + s(1) + l(2) + l(3))
            self.detailed_gam.fit(self.X_train, self.y_train)

            # 计算模型评估指标
            y_pred = self.detailed_gam.predict(self.X_train)
            r2 = r2_score(self.y_train, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_train, y_pred))

            # 保存GAM模型的评估指标
            self.gam_metrics = {
                "R2": r2,
                "RMSE": rmse
            }

            print(f"详细GAM模型训练完成!")
            print(f"模型评估指标 - R²: {r2:.4f}, RMSE: {rmse:.4f}")
            return self.detailed_gam
        except Exception as e:
            print(f"详细GAM模型训练失败: {e}")
            raise

    def plot_detailed_gam_results(self, save_name="GAM平滑曲线_分组优化版.png", show_confidence=True):
        """绘制详细GAM平滑曲线结果"""
        if not hasattr(self, 'detailed_gam') or self.detailed_gam is None or self.data is None:
            raise ValueError("请先加载数据并训练详细GAM模型")

        try:
            # 定义颜色方案
            line_color = '#e74c3c'  # 红色曲线

            # BMI分组颜色（使用渐变色便于区分）
            bmi_group_colors = {
                '20-28': '#1f77b4',  # 蓝色
                '28-32': '#ff7f0e',  # 橙色
                '32-36': '#2ca02c',  # 绿色
                '36-40': '#FF79BC',  # 粉红色
                '40以上': '#9467bd'  # 紫色
            }

            # 孕周分组颜色
            ga_group_colors = {
                '早期发现(≤12周)': '#1f77b4',  # 蓝色
                '中期发现(13-27周)': '#ff7f0e',  # 橙色
                '晚期发现(≥28周)': '#2ca02c'  # 绿色
            }

            # 创建一个更大的图，包含更多子图
            fig = plt.figure(figsize=(20, 12))
            plt.subplots_adjust(wspace=0.3, hspace=0.4)

            # 子图 1：孕周（GA）对 Yc 的平滑作用
            plt.subplot(2, 2, 1)
            ax1 = plt.gca()
            ga_range = np.linspace(self.data['GA'].min(), self.data['GA'].max(), 100)

            # 固定其他特征为均值，仅变化 GA
            ga_pred_data = np.array([
                [ga, self.data['BMI'].mean(), self.data['IVF'].mean(), self.data['log_total_reads'].mean()]
                for ga in ga_range
            ])

            # 获取预测值
            ga_pred = self.detailed_gam.predict(ga_pred_data)

            # 绘制主曲线
            ax1.plot(ga_range, ga_pred, color=line_color, linewidth=3, label='预测值')

            # 添加95%置信区间（如果需要）
            if show_confidence:
                try:
                    # 获取预测的置信区间
                    ga_lower, ga_upper = self.detailed_gam.confidence_intervals(ga_pred_data, width=0.95)
                    ax1.fill_between(ga_range, ga_lower, ga_upper, color=line_color, alpha=0.2, label='95%置信区间')
                except Exception as e:
                    print(f"计算置信区间时出错: {e}")

            # 按孕周分组绘制不同颜色的散点
            for group in self.data['孕周分组'].unique():
                subset = self.data[self.data['孕周分组'] == group]
                ax1.scatter(subset['GA'], subset['Yc'],
                            color=ga_group_colors[group], alpha=0.6,
                            s=15, label=group)

            # 设置图表属性
            ax1.set_title('GAM：孕周对 Yc 的平滑作用', pad=15, fontsize=14)
            ax1.set_xlabel('孕周（Weeks）', fontsize=12)
            ax1.set_ylabel('Yc 预测值（Predicted Yc）', fontsize=12)
            ax1.grid(alpha=0.3, linestyle='--')
            ax1.legend(prop=self.simhei_font, loc='best')

            # 子图 2：BMI 对 Yc 的平滑作用
            plt.subplot(2, 2, 2)
            ax2 = plt.gca()
            bmi_range = np.linspace(self.data['BMI'].min(), self.data['BMI'].max(), 100)

            # 固定其他特征为均值，仅变化 BMI
            bmi_pred_data = np.array([
                [self.data['GA'].mean(), bmi, self.data['IVF'].mean(), self.data['log_total_reads'].mean()]
                for bmi in bmi_range
            ])

            # 获取预测值
            bmi_pred = self.detailed_gam.predict(bmi_pred_data)

            # 绘制主曲线
            ax2.plot(bmi_range, bmi_pred, color=line_color, linewidth=3, label='预测值')

            # 添加95%置信区间（如果需要）
            if show_confidence:
                try:
                    # 获取预测的置信区间
                    bmi_lower, bmi_upper = self.detailed_gam.confidence_intervals(bmi_pred_data, width=0.95)
                    ax2.fill_between(bmi_range, bmi_lower, bmi_upper, color=line_color, alpha=0.2, label='95%置信区间')
                except Exception as e:
                    print(f"计算置信区间时出错: {e}")

            # 按BMI分组绘制不同颜色的散点
            for group in self.data['BMI分组'].unique():
                subset = self.data[self.data['BMI分组'] == group]
                ax2.scatter(subset['BMI'], subset['Yc'],
                            color=bmi_group_colors[group], alpha=0.6,
                            s=15, label=group)

            # 设置图表属性
            ax2.set_title('GAM：BMI 对 Yc 的平滑作用', pad=15, fontsize=14)
            ax2.set_xlabel('BMI', fontsize=12)
            ax2.set_ylabel('Yc 预测值（Predicted Yc）', fontsize=12)
            ax2.grid(alpha=0.3, linestyle='--')
            ax2.legend(prop=self.simhei_font, loc='best')

            # 子图 3：原始读段数对数对 Yc 的线性作用
            plt.subplot(2, 2, 3)
            ax3 = plt.gca()
            reads_range = np.linspace(self.data['log_total_reads'].min(), self.data['log_total_reads'].max(), 100)

            # 固定其他特征为均值，仅变化 log_total_reads
            reads_pred_data = np.array([
                [self.data['GA'].mean(), self.data['BMI'].mean(), self.data['IVF'].mean(), reads]
                for reads in reads_range
            ])

            # 获取预测值
            reads_pred = self.detailed_gam.predict(reads_pred_data)

            # 绘制主曲线
            ax3.plot(reads_range, reads_pred, color=line_color, linewidth=3, label='预测值')

            # 绘制散点图
            ax3.scatter(self.data['log_total_reads'], self.data['Yc'],
                        color='#95a5a6', alpha=0.6, s=15)

            # 设置图表属性
            ax3.set_title('GAM：log(原始读段数) 对 Yc 的线性作用', pad=15, fontsize=14)
            ax3.set_xlabel('log(原始读段数)', fontsize=12)
            ax3.set_ylabel('Yc 预测值（Predicted Yc）', fontsize=12)
            ax3.grid(alpha=0.3, linestyle='--')
            ax3.legend(prop=self.simhei_font)

            # 子图 4：IVF妊娠对 Yc 的线性作用
            plt.subplot(2, 2, 4)
            ax4 = plt.gca()

            # 固定其他特征为均值，仅变化 IVF
            ivf_pred_data = np.array([
                [self.data['GA'].mean(), self.data['BMI'].mean(), ivf, self.data['log_total_reads'].mean()]
                for ivf in [0, 1]
            ])

            # 获取预测值
            ivf_pred = self.detailed_gam.predict(ivf_pred_data)

            # 绘制条形图
            ax4.bar(['非IVF妊娠', 'IVF妊娠'], ivf_pred, color=['#3498db', '#e74c3c'], alpha=0.8)

            # 添加数值标签
            for i, v in enumerate(ivf_pred):
                ax4.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

            # 设置图表属性
            ax4.set_title('GAM：IVF妊娠对 Yc 的线性作用', pad=15, fontsize=14)
            ax4.set_xlabel('妊娠类型', fontsize=12)
            ax4.set_ylabel('Yc 预测值（Predicted Yc）', fontsize=12)
            ax4.grid(alpha=0.3, linestyle='--', axis='y')

            # 整体标题
            fig.suptitle('GAM模型特征影响分析（男胎检测数据）', fontsize=18, y=0.98)

            # 保存图片
            save_path = os.path.join(self.gam_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"GAM详细分析图已保存至: {save_path}")

        except Exception as e:
            print(f"绘制GAM结果时出错: {e}")
            raise

    def start_log_capture(self):
        """开始捕获标准输出日志"""
        sys.stdout = self.log_capture_string

    def stop_log_capture(self):
        """停止捕获标准输出日志并恢复默认输出"""
        sys.stdout = self.original_stdout

    def generate_conclusion(self):
        """生成结论文件，包含模型结果和日志"""
        try:
            # 获取捕获的日志
            log_content = self.log_capture_string.getvalue()

            # 创建结论内容
            conclusion = []
            conclusion.append(f"胎儿Y染色体浓度分析结论报告")
            conclusion.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            conclusion.append(f"输出目录: {os.path.abspath(self.base_dir)}\n")

            # 添加数据基本信息
            if self.data is not None:
                conclusion.append("1. 数据基本信息")
                conclusion.append(f"   - 有效样本数量: {len(self.data)}")
                conclusion.append(f"   - 孕周范围: {self.data['GA'].min():.2f} - {self.data['GA'].max():.2f} 周")
                conclusion.append(f"   - BMI范围: {self.data['BMI'].min():.2f} - {self.data['BMI'].max():.2f}")
                conclusion.append(f"   - Y染色体浓度范围: {self.data['Yc'].min():.4f} - {self.data['Yc'].max():.4f}\n")

            # 添加模型比较结果
            if self.results:
                conclusion.append("2. 模型性能比较")
                results_df = pd.DataFrame(self.results)
                # 按R2降序排序
                results_df = results_df.sort_values('R2', ascending=False)
                for _, row in results_df.iterrows():
                    conclusion.append(f"   - {row['Model']}:")
                    conclusion.append(f"       R²: {row['R2']:.4f}")
                    conclusion.append(f"       RMSE: {row['RMSE']:.4f}")
                    conclusion.append(f"       MAE: {row['MAE']:.4f}")
                    conclusion.append(f"       MSE: {row['MSE']:.4f}")
                best_model = results_df.iloc[0]['Model']
                conclusion.append(f"   - 最佳模型: {best_model} (R²: {results_df.iloc[0]['R2']:.4f})\n")

            # 添加GAM模型详细结果
            if hasattr(self, 'gam_metrics'):
                conclusion.append("3. GAM模型详细分析")
                conclusion.append(f"   - 模型性能:")
                conclusion.append(f"       R²: {self.gam_metrics['R2']:.4f}")
                conclusion.append(f"       RMSE: {self.gam_metrics['RMSE']:.4f}")
                conclusion.append(f"   - 特征影响分析图表已保存至: {os.path.abspath(self.gam_dir)}\n")

            # 添加日志内容
            conclusion.append("4. 分析过程日志")
            conclusion.append("=" * 50)
            conclusion.append(log_content)
            conclusion.append("=" * 50)

            # 写入结论文件
            with open(self.conclusion_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(conclusion))

            print(f"结论报告已生成: {self.conclusion_path}")

        except Exception as e:
            print(f"生成结论文件时出错: {e}")

    def run_complete_analysis(self, file_path):
        """运行完整的分析流程"""
        try:
            # 开始捕获日志
            self.start_log_capture()

            # 加载和预处理数据
            self.load_and_preprocess_data(file_path)

            # 训练比较模型并生成对比结果
            self.train_comparison_models()

            # 训练详细GAM模型
            self.train_detailed_gam_model()

            # 绘制和保存GAM详细结果
            self.plot_detailed_gam_results()

            # 停止捕获日志
            self.stop_log_capture()

            # 生成结论文件
            self.generate_conclusion()

            print("完整分析已完成!")
            print(f"所有结果已保存至: {os.path.abspath(self.base_dir)}")
        except Exception as e:
            # 确保在出错时恢复标准输出
            self.stop_log_capture()
            print(f"完整分析过程出错: {e}")
            raise


# 主程序入口
if __name__ == "__main__":
    # 文件路径 - 请根据实际情况修改
    file_path = r"附件.xlsx"

    # 创建分析器实例
    analyzer = FetalYAnalysis()

    # 运行完整分析
    analyzer.run_complete_analysis(file_path)
