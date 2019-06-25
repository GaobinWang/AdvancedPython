# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 21:07:41 2019

@author: Lesile
"""
#%%
import os
import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime 
import seaborn as sns
#%%
path = "E:\\Github\\AdvancedPython\\Seaborn"
os.chdir(path)


"""
Seaborn:统计数据可视化
变量间的统计关系 - relplot
分类变量间的关系 -catplot
数据集的分布 - kdeplot distplot jointplot pairplot
线性关系 - regplot lmplot 
"""
#%%变量间的统计关系 - relplot
sns.set(style="darkgrid")

tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", data=tips);


sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips);

sns.relplot(x="total_bill", y="tip", size="size", data=tips);

df = pd.DataFrame(dict(time=np.arange(500),
                       value=np.random.randn(500).cumsum()))
g = sns.relplot(x="time", y="value", kind="line", data=df)
g.fig.autofmt_xdate()

#%%分类变量间的关系 -catplot

tips = sns.load_dataset("tips")
sns.catplot(x="day", y="total_bill", data=tips);

sns.catplot(x="day", y="total_bill", kind="box", data=tips);

sns.catplot(x="day", y="total_bill", hue="smoker", kind="box", data=tips);

titanic = sns.load_dataset("titanic")
sns.catplot(x="sex", y="survived", hue="class", kind="bar", data=titanic);

#%%数据集的分布 - kdeplot distplot jointplot pairplot
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

###单变量的分布
x = np.random.normal(size=100)
sns.distplot(x) 
sns.distplot(x, kde=False, rug=True);
sns.distplot(x, bins=20, kde=False, rug=True);



sns.kdeplot(x)
sns.kdeplot(x, bw=.2, label="bw: 0.2")
sns.kdeplot(x, bw=2, label="bw: 2")
plt.legend();

###双变量的分布
mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])

sns.jointplot(x="x", y="y", data=df);

sns.jointplot(x="x", y="y", data=df, kind="kde");

f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(df.x, df.y, ax=ax)
sns.rugplot(df.x, color="g", ax=ax)
sns.rugplot(df.y, vertical=True, ax=ax);


###多变量分布
iris = sns.load_dataset("iris")
sns.pairplot(iris);


g = sns.PairGrid(iris)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6);


#%% 线性关系 - regplot lmplot
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
tips = sns.load_dataset("tips")


sns.regplot(x="total_bill", y="tip", data=tips);

sns.lmplot(x="total_bill", y="tip", data=tips);

tips["big_tip"] = (tips.tip / tips.total_bill) > .15
sns.lmplot(x="total_bill", y="big_tip", data=tips,logistic=True, y_jitter=.03);


sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips);
sns.lmplot(x="total_bill", y="tip", hue="smoker", col="time", data=tips);
sns.jointplot(x="total_bill", y="tip", data=tips, kind="reg");





