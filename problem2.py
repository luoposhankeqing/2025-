# problem2.py —— 自动分组 + 完整阈值线热力图
import os, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.optimize import minimize_scalar
from matplotlib.patches import Ellipse
from lifelines import KaplanMeierFitter

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 工具函数 ----------
def ensure_dirs():
    os.makedirs("Problem2 results/problem2/data", exist_ok=True)
    os.makedirs("Problem2 results/problem2/images", exist_ok=True)

def convert_gestational_week(s):
    if pd.isna(s): return np.nan
    s = str(s).lower().strip()
    try:
        if "+" in s:
            w,d=s.split("+"); return float(w.replace("w",""))+float(d)/7
        if "w" in s: return float(s.replace("w",""))
        return float(s)
    except: return np.nan

def calculate_bmi_if_missing(df):
    df2=df.copy()
    if "孕妇BMI" not in df2.columns:
        if "身高" in df2.columns and "体重" in df2.columns:
            df2["孕妇BMI"]=df2["体重"]/((df2["身高"]/100)**2)
        else: raise ValueError("缺少孕妇BMI或身高体重")
    return df2

def judge_fetal_gender(df):
    df2=df.copy()
    if "胎儿性别" not in df2.columns:
        df2["胎儿性别"]=df2.apply(lambda r:"男" if (pd.notna(r.get("Y染色体的Z值")) and pd.notna(r.get("Y染色体浓度")) and r["Y染色体浓度"]>0) else "女",axis=1)
    return df2

def filter_male_fetus_data(df):
    df2=df.copy()
    male=df2[df2["胎儿性别"]=="男"].copy()
    male=male.dropna(subset=["Y染色体浓度","检测孕周数值","孕妇BMI"])
    return male

# ---------- 模型 ----------
def fit_prediction_model(male_df,model_type="nonlinear"):
    X=male_df[["检测孕周数值","孕妇BMI"]].values
    y=male_df["Y染色体浓度"].values
    if model_type=="nonlinear":
        poly=PolynomialFeatures(2,include_bias=False)
        X_poly=poly.fit_transform(X)
        model=LinearRegression().fit(X_poly,y)
        return model,poly
    else:
        model=LinearRegression().fit(X,y)
        return model,None

def calculate_reaching_time(df,model_type="nonlinear"):
    ensure_dirs()
    male=filter_male_fetus_data(df)
    model,poly=fit_prediction_model(male,model_type)
    times=[]
    def pred(w,bmi):
        X=np.array([[w,bmi]])
        if poly: X=poly.transform(X)
        return model.predict(X)[0]
    for _,row in male.iterrows():
        f=lambda w:(pred(w,row["孕妇BMI"])-0.04)**2
        res=minimize_scalar(f,bounds=(10,30),method="bounded")
        times.append(res.x if res.success else np.nan)
    male["达标时间"]=times
    male=male.dropna(subset=["达标时间"])
    male.to_excel("Problem2 results/problem2/data/male_with_rt.xlsx",index=False)
    return male,model,poly

# ---------- BMI聚类 ----------
def plot_confidence_ellipse(x,y,ax,color):
    cov=np.cov(x,y)
    vals,vecs=np.linalg.eigh(cov)
    order=vals.argsort()[::-1]
    vals,vecs=vals[order],vecs[:,order]
    theta=np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width,height=2*np.sqrt(vals*5.991) # 95%CI
    ell=Ellipse(
        (np.mean(x),np.mean(y)),
        width,
        height,
        angle=theta,
        edgecolor=color,
        facecolor='none',
        lw=2,
        alpha=0.7
    )
    ax.add_patch(ell)

def bmi_clustering(df,k_min=2,k_max=8):
    X=df[["孕妇BMI","达标时间"]].values
    best_k,best_labels,best_score=None,None,-1
    metrics_log=[]
    for k in range(k_min,min(k_max,len(df))+1):
        km=KMeans(n_clusters=k,n_init=20,random_state=42).fit(X)
        labels=km.labels_
        try:
            sil=silhouette_score(X,labels)
            ch=calinski_harabasz_score(X,labels)
            db=davies_bouldin_score(X,labels)
        except: continue
        score=sil*0.5+(ch/max(ch,1))*0.3+(1/(1+db))*0.2
        metrics_log.append((k,sil,ch,db,score))
        if score>best_score:
            best_k,best_labels,best_score=k,labels,score
    if best_k is None:
        best_k=2
        best_labels=KMeans(n_clusters=2,n_init=20,random_state=42).fit_predict(X)
    df["BMI分组"]=best_labels
    df["BMI分组名称"]=["BMI分组"+str(l+1) for l in df["BMI分组"]]
    km=KMeans(n_clusters=best_k,n_init=20,random_state=42).fit(X)
    centers=km.cluster_centers_
    # 保存指标
    metrics_df=pd.DataFrame(metrics_log,columns=["K","Silhouette","Calinski-H","Davies-B","综合得分"])
    metrics_df.to_excel("Problem2 results/problem2/data/cluster_metrics.xlsx",index=False)
    # 可视化 - 优化版本
    plt.figure(figsize=(10,7))
    ax=plt.gca()
    # 使用更美观的颜色方案
    palette=sns.color_palette("viridis",best_k)
    # 增加网格线
    ax.grid(True,linestyle='--',alpha=0.7)
    # 绘制散点图
    scatter=sns.scatterplot(x="孕妇BMI",y="达标时间",hue="BMI分组名称",data=df,palette=palette,
                            s=80,edgecolor="white",linewidth=1.5,ax=ax,alpha=0.8)
    # 绘制置信椭圆，使用与数据点相同的颜色
    for g in np.unique(best_labels):
        sub=df[df["BMI分组"]==g]
        plot_confidence_ellipse(sub["孕妇BMI"],sub["达标时间"],ax,palette[g])
        ax.text(sub["孕妇BMI"].mean(),sub["达标时间"].mean()+0.3,f"n={len(sub)}",
                ha="center",fontsize=9,color="black",bbox=dict(facecolor="white",alpha=0.6,edgecolor="none"))
    ax.scatter(centers[:,0],centers[:,1],c="red",marker="X",s=150,label="中心")
    ax.legend(bbox_to_anchor=(1.05,1),loc="upper left")
    ax.set_title(f"BMI聚类 自动选择K={best_k} (综合得分={best_score:.3f})")
    plt.tight_layout()
    plt.savefig("Problem2 results/problem2/images/bmi_clustering_auto.png",dpi=300)
    plt.close()
    return df,best_k,best_score

# ---------- Kaplan-Meier ----------
def plot_km(df):
    kmf=KaplanMeierFitter()
    plt.figure(figsize=(8,6))
    for name,sub in df.groupby("BMI分组名称"):
        label=f"{name} (n={len(sub)})"
        kmf.fit(sub["达标时间"],event_observed=np.ones(len(sub)))
        kmf.plot_survival_function(ci_show=True,label=label)
    plt.title("Kaplan–Meier 曲线：不同BMI分组达标率")
    plt.xlabel("孕周")
    plt.ylabel("未达标概率")
    plt.legend(bbox_to_anchor=(1.05,1),loc="upper left")
    plt.tight_layout()
    plt.savefig("Problem2 results/problem2/images/km_curve.png",dpi=300)
    plt.close()

# ---------- 热力图 ----------
def plot_heatmap(df,model,poly):
    # 限制范围
    weeks=np.linspace(10,25,100)
    bmis=np.linspace(df["孕妇BMI"].min()-2,df["孕妇BMI"].max()+2,100)
    W,B=np.meshgrid(weeks,bmis)
    Z=np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            X=np.array([[W[i,j],B[i,j]]])
            if poly: X=poly.transform(X)
            Z[i,j]=model.predict(X)[0]

    plt.figure(figsize=(9,7))
    # 绘制热力图
    im=plt.imshow(Z,extent=[weeks.min(),weeks.max(),bmis.min(),bmis.max()],
                  origin="lower",aspect="auto",cmap="coolwarm")
    cbar=plt.colorbar(im)
    cbar.set_label("预测Y浓度")

    # 绘制完整阈值线
    CS=plt.contour(W,B,Z,levels=[0.04],colors="black",linewidths=2,linestyles="--")
    plt.clabel(CS,inline=True,fmt="阈值0.04",fontsize=9)

    plt.title("孕周-BMI 达标概率热力图（含阈值曲线）")
    plt.xlabel("孕周")
    plt.ylabel("BMI")
    plt.xlim(10,25)
    plt.ylim(df["孕妇BMI"].min()-2,df["孕妇BMI"].max()+2)
    plt.tight_layout()
    plt.savefig("Problem2 results/problem2/images/ga_bmi_heatmap.png",dpi=300)
    plt.close()

# ---------- 主流程 ----------
def process(df,model_type="nonlinear"):
    ensure_dirs()
    df["检测孕周数值"]=df["检测孕周"].apply(convert_gestational_week)
    df=calculate_bmi_if_missing(df)
    df=judge_fetal_gender(df)
    male,model,poly=calculate_reaching_time(df,model_type)
    clustered,k,score=bmi_clustering(male)
    plot_km(clustered)
    plot_heatmap(male,model,poly)
    with open("Problem2 results/problem2/conclusion.txt","w",encoding="utf-8") as f:
        f.write("问题2：男胎孕妇BMI分组与最佳NIPT时点分析\n")
        f.write("="*60+"\n")
        f.write(f"自动选择的分组数: K={k}, 综合得分={score:.3f}\n")
        for name,sub in clustered.groupby("BMI分组名称"):
            f.write(f"{name}: 样本数={len(sub)}, 平均BMI={sub['孕妇BMI'].mean():.2f}, 平均达标时间={sub['达标时间'].mean():.2f}周\n")
        f.write("\nKaplan-Meier 曲线与孕周-BMI 热力图已生成。\n")
    return clustered

if __name__=="__main__":
    # 修改文件路径为当前目录下的附件.xlsx
    file_path="附件.xlsx"
    df=pd.read_excel(file_path)
    process(df,"nonlinear")
    print("Problem2 完成，结果在 Problem2 results/problem2/ 下。")
