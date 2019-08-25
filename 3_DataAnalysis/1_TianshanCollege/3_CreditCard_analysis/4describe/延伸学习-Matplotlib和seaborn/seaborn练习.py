
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


# In[2]:


x=np.linspace(0,14,100)
for i in range(7):
    plt.plot(x,np.sin(x+i*1.5)*(5-i))


# In[3]:


sns.set()
x=np.linspace(0,14,100)
for i in range(7):
    plt.plot(x,np.sin(x+i*1.5)*(5-i))


# In[4]:


sns.set_style("whitegrid")
data=np.random.normal(size=(20,6))+np.arange(6)/2
sns.boxplot(data=data)


# In[5]:


sns.set_style("ticks")
sns.despine() #去掉右侧和上面的刻度
sns.boxplot(data=data)


# In[6]:


sns.set_style("whitegrid")
#去掉右侧和上面的刻度
sns.boxplot(data=data)


# In[7]:


with sns.axes_style("dark"):
    plt.subplot(211)
    x=np.linspace(0,14,100)
    for i in range(7):
        plt.plot(x,np.sin(x+i*1.5)*(5-i))
plt.subplot(212)
x=np.linspace(0,14,100)
for i in range(7):
    plt.plot(x,np.sin(x+i*1.5)*(5-i))


# In[8]:


sns.set_context(context="talk",font_scale=1,rc={"linewidth":14})
x=np.linspace(0,14,100)
for i in range(7):
    plt.plot(x,np.sin(x+i*1.5)*(5-i))


# In[9]:


sns.palplot(sns.color_palette())


# In[10]:


sns.palplot(sns.color_palette("hls",20))


# In[11]:


data=np.random.normal(size=(200,8))+np.arange(8)/2
sns.boxplot(data=data,palette=sns.color_palette("hls",8))


# In[12]:


sns.palplot(sns.color_palette("Paired",10))


# In[13]:


plt.plot([0,1],[0,1],sns.xkcd_rgb["pale red"],linewidth=3)
plt.plot([0,1],[0,2],sns.xkcd_rgb["medium green"],linewidth=3)
plt.plot([0,1],[0,3],sns.xkcd_rgb["denim blue"],linewidth=3)


# #### 设置颜色

# In[14]:


sns.palplot(sns.color_palette("Blues"))


# In[15]:


sns.palplot(sns.color_palette("Blues_r"))


# In[16]:


data=np.random.normal(size=(200,8))+np.arange(8)/2
sns.boxplot(data=data,palette=sns.light_palette("red",reverse=True))


# In[17]:


sns.palplot(sns.color_palette("cubehelix",8))


# In[18]:


sns.palplot(sns.cubehelix_palette(8,start=5,rot=-150))


# In[19]:


sns.palplot(sns.light_palette("blue",reverse=True))


# In[20]:


sns.palplot(sns.dark_palette("red",reverse=True))


# In[21]:


sns.palplot(sns.light_palette("r",reverse=False))


# #### 单变量直方图

# In[22]:


sns.set_style("white")
x=np.random.normal(size=10000)
sns.distplot(x,kde=True,bins=100)


# #### 双变量散点图

# In[23]:


mean,cov=[0,1],[(1,0.5),(0.5,1)]
data=np.random.multivariate_normal(mean,cov,1000)
df=pd.DataFrame(data=data,columns=["x","y"])
df.head()


# In[24]:


sns.jointplot(x="x",y="y",data=df)


# #### 两个变量的相关性

# In[25]:


mean,cov=[1,0],[(1,0.5),(4,1)]
data=np.random.multivariate_normal(mean,cov,1000)
df=pd.DataFrame(data=data,columns=["z","c"])
sns.set_style("white")
sns.jointplot(x="z",y="c",data=df,kind="hex",color="r")


# #### 两个变量的相关性

# In[26]:


iris=sns.load_dataset("iris")
sns.pairplot(iris)


# #### 回归分析图

# In[27]:


tips=sns.load_dataset("tips")
tips.head()


# In[28]:


sns.regplot(x="total_bill",y="tip",data=tips)


# In[29]:


sns.lmplot(x="total_bill",y="tip",data=tips)


# In[30]:


sns.regplot(x="tip",y="size",data=tips)


# In[31]:


sns.regplot(x="size",y="tip",data=tips,x_jitter=0.9)


# In[32]:


sns.stripplot(x="day",y="tip",data=tips)


# In[33]:


sns.stripplot(x="day",y="total_bill",data=tips,jitter=True,hue="sex")


# In[34]:


sns.swarmplot(x="day",y="total_bill",data=tips)


# In[35]:


sns.swarmplot(x="day",y="total_bill",data=tips,hue="sex")


# In[36]:


sns.boxplot(x="day",y="total_bill",data=tips)


# In[37]:


sns.violinplot(x="day",y="total_bill",data=tips,hue="sex",split=True)


# In[38]:


sns.violinplot(x="day",y="total_bill",data=tips,alpha=0.9)
sns.swarmplot(x="day",y="total_bill",data=tips,color="white",alpha=0.5)


# In[39]:


titanic=sns.load_dataset("titanic")
sns.barplot(x="sex",y="survived",data=titanic,hue="class")


# #### 点图，描述差异性

# In[40]:


sns.pointplot(x="class",y="survived",data=titanic,hue="sex")


# In[41]:


sns.pointplot(x="class",y="survived",data=titanic,
             hue="sex",palette={"male":"r","female":"g"},
             markers=["^","o"],linestyles=["-","--"])


# In[42]:


sns.boxplot(data=iris,orient="h")


# In[43]:


sns.factorplot(x="day",y="total_bill",data=tips,hue="smoker",kind="bar")


# In[44]:


sns.factorplot(x="day",y="total_bill",data=tips,hue="sex",col="time",kind="violin",split=True,size=10,aspect=1
              )


# In[45]:


g=sns.FacetGrid(tips,col="time",size=6)


# In[46]:


g=sns.FacetGrid(tips,col="day",size=6)
g.map(plt.hist,"tip")


# In[47]:


g=sns.FacetGrid(tips,col="time",hue="smoker",size=6)
g.map(plt.scatter,"total_bill","tip",alpha=0.7)
g.add_legend()


# In[48]:


g=sns.FacetGrid(tips,row="smoker",col="time",size=6,margin_titles=True)
g.map(sns.regplot,"size","total_bill",fit_reg=True,x_jitter=0.5,color="0.1")


# #### 更改顺序

# In[49]:


from pandas import Categorical
index=tips.day.value_counts().index
index1=Categorical([ 'Sun','Sat', 'Thur', 'Fri'])
g=sns.FacetGrid(tips,row="day",size=1.5,aspect=4,margin_titles=True)
g.map(sns.boxplot,"total_bill",orient="h")


# In[50]:


from pandas import Categorical
index=tips.day.value_counts().index
index1=Categorical([ 'Sun','Sat', 'Thur', 'Fri'])
g=sns.FacetGrid(tips,row="day",size=1.5,aspect=4,row_order=index1,margin_titles=True)
g.map(sns.boxplot,"total_bill",orient="h")


# In[51]:


g=sns.FacetGrid(tips,hue="time")


# In[61]:


pal={"Lunch":"g","Dinner":"r"}
g=sns.FacetGrid(tips,hue="time",palette=pal,size=7)
g.map(plt.scatter,"total_bill","tip",s=100,linewidth=1,alpha=0.5,edgecolor="white")  #s表示圈圈的大小
g.add_legend()


# In[70]:


g=sns.FacetGrid(tips,hue="time",size=9,palette=pal,hue_kws={"marker":["^","v"]})
g.map(plt.scatter,"total_bill","tip",s=150,alpha=0.5,edgecolor="white")
g.add_legend()


# In[81]:


sns.axes_style("white")
g=sns.FacetGrid(tips,row="sex",col="smoker",size=7,margin_titles=True) #构建图纸
g.map(plt.scatter,"total_bill","tip",color="r",edgecolor="white",linewidth=1,s=150,alpha=0.7) #画图
g.set_axis_labels("total bill","tip") #设置标签
g.set(xticks=[10,30,50],yticks=[2,4,6,10]) #设置刻度
g.fig.subplots_adjust(wspace=0.2,hspace=0.2)  #调整四个图的间距


# In[83]:


g=sns.PairGrid(iris)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)


# In[85]:


g=sns.PairGrid(iris,hue="species",size=5)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)


# In[88]:


g=sns.PairGrid(iris,vars=["sepal_length","sepal_width"],hue="species",size=5)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)


# In[90]:


g=sns.PairGrid(tips,hue="size",palette="GnBu_d")
g.map(plt.scatter,s=50,edgecolor="r",linewidth=1)
g.add_legend()


# # 热度图

# In[98]:


data=np.random.rand(3,3)
print(data)
sns.heatmap(data)


# In[105]:


data1=np.random.randint(3,10,size=(3,3))
print(data1)
sns.heatmap(data1,vmin=2,vmax=5,)


# In[110]:


data2=np.random.randn(3,3)
print(data2)
sns.heatmap(data2,center=0)


# In[115]:


flights=sns.load_dataset("flights")
flights.head()


# In[112]:


flights=flights.pivot(index="month",columns="year",values="passengers")
flights.head()


# In[116]:


flights=flights.pivot_table(index="month",columns="year",values="passengers")
flights.head()


# In[117]:


sns.set 
sns.heatmap(flights)


# In[124]:


fig,ax=plt.subplots(figsize=(10,10))
sns.heatmap(flights,annot=True,fmt="d",ax=ax)#annot=True显示数值在热力图中，fmt="d"表示显示字体，ax=ax调整图的带下，seabourn结合matplotlib使用


# In[126]:


fig,ax=plt.subplots(figsize=(10,10))
sns.heatmap(flights,annot=True,fmt="d",ax=ax,linewidth=0.5)#格间距


# In[129]:


sns.heatmap(flights,cmap="YlGnBu")  #更换颜色，调色板


# In[130]:


sns.heatmap(flights,cmap="YlGnBu",cbar=False) 


