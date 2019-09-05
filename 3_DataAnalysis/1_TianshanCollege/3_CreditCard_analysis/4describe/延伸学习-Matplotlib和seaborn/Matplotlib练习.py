
# coding: utf-8

# #### 导入相关包

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# #### 解决汉字不能显示问题

# In[2]:


from pylab import mpl
mpl.rcParams["font.sans-serif"]=["SimHei"]  #设定默认字体
mpl.rcParams["axes.unicode_minus"]=False   #解决保存图像是负号“-”显示为方块的问题


# #### 做一个基本图形

# In[3]:


plt.plot([1,2,3,4,5],[1,16,27,64,125])
plt.xlabel("横轴",fontsize=16)
plt.ylabel("纵轴",fontsize=16)


# #### 不同的线条

# 字符|类型 | 字符|类型
# ---|--- | --- | ---
# `  '-'	`| 实线 | `'--'`|	虚线
# `'-.'`|	虚点线 | `':'`|	点线
# `'.'`|	点 | `','`| 像素点
# `'o'`	|圆点 | `'v'`|	下三角点
# `'^'`|	上三角点 | `'<'`|	左三角点
# `'>'`|	右三角点 | `'1'`|	下三叉点
# `'2'`|	上三叉点 | `'3'`|	左三叉点
# `'4'`|	右三叉点 | `'s'`|	正方点
# `'p'`	| 五角点 | `'*'`|	星形点
# `'h'`|	六边形点1 | `'H'`|	六边形点2 
# `'+'`|	加号点 | `'x'`|	乘号点
# `'D'`|	实心菱形点 | `'d'`|	瘦菱形点 
# `'_'`|	横线点 | |

# #### 不同的颜色

# 颜色
# 表示颜色的字符参数有：
# 
# 字符 | 颜色
# -- | -- 
# `‘b’`|	蓝色，blue
# `‘g’`|	绿色，green
# `‘r’`|	红色，red
# `‘c’`|	青色，cyan
# `‘m’`|	品红，magenta
# `‘y’`|	黄色，yellow
# `‘k’`|	黑色，black
# `‘w’`|	白色，white

# #### 折线图

# In[4]:


plt.plot([1,2,3,4,5],[1,16,27,64,125],color="m",linestyle="-",linewidth="2.5")
plt.xlabel("横轴",fontsize=14)
plt.ylabel("纵轴",fontsize=14)


# In[5]:


leon=np.arange(0,10,0.5)
plt.plot(leon,leon,"r-",
        leon,leon**2,"m:",
        leon,leon**3,"bv")


# In[6]:


x=np.linspace(-10,10)
y=np.sin(x)
plt.plot(x,y,"b:",linewidth=3)


# In[7]:


x=np.linspace(-10,10)
y=np.sin(x)
plt.plot(x,y,"b:",linewidth=3,marker="o",markerfacecolor="r",markersize=10,alpha=0.6)


# In[8]:


line=plt.plot(x,y)
plt.setp(line,linestyle="-",linewidth=2.0,marker="D",color="r",markerfacecolor="b",markersize=8)


# In[9]:


plt.subplot(221)
plt.plot(x,y,"r-")

plt.subplot(222)
plt.plot(x,y,"g:")

plt.subplot(223)
plt.plot(x,y,"b--")

plt.subplot(224)
plt.plot(x,y,"k-")


# In[10]:


plt.subplot(421)
plt.plot(x,y,"r-")

plt.subplot(424)
plt.plot(x,y,"g:")

plt.subplot(425)
plt.plot(x,y,"b--")

plt.subplot(428)
plt.plot(x,y,"k-")


# #### 加备注和箭头

# In[11]:


x=np.linspace(-10,10)
y=np.sin(x)
plt.plot(x,y,"b:",linewidth=3,marker="o",markerfacecolor="r",markersize=10,alpha=0.99)
plt.xlabel("X",fontsize=14)
plt.ylabel("Y",fontsize=14)
plt.grid(True)
plt.title("正弦图")
plt.text(-10,0.6,"起点")
plt.annotate("终点",fontsize=13,xy=(10,-0.6),xytext=(7.2,-0.30),arrowprops=dict(facecolor="r",shrink=0.02,headwidth=10,headlength=10))


# #### 添加线

# In[12]:


x=np.arange(5)
np.random.seed(0)
y=np.random.randint(-5,5,5)
print(y)


fig,axes=plt.subplots(2,1)
v_bar=axes[0].bar(x,y,color="r")
h_bar=axes[1].barh(x,y,color="b")

axes[0].axhline(0,color="grey",linewidth=2)
axes[1].axvline(0,color="grey",linewidth=2)


# In[13]:


x=np.arange(5)
np.random.seed(0)
y=np.random.randint(-5,5,5)

fig,ax=plt.subplots()
v_bar=ax.bar(x,y,color="lightblue")
ax.axhline(0,color="grey",linewidth=2) #添加线

for bar,height in zip(v_bar,y):
    if height<0:
        bar.set(color="green",linewidth=2)


# In[14]:


x=np.arange(5)
np.random.seed(1)
y=np.random.randint(-10,10,5)

fig,axes=plt.subplots()
axes.axhline(0,color="grey",linewidth=3)
line=axes.bar(x,y,color="r")
plt.setp(line,color="green")
plt.ylim(-7,4)
plt.xlim(-1,6)
for bar,height in zip(line,y):
    if height<0:
        bar.set(color="r")


# #### 填充颜色

# In[15]:


x=np.random.randn(100).cumsum()
y=np.linspace(0,10,100)

fig,axes=plt.subplots()
axes.fill_between(x,y,color="lightblue")


# In[16]:


plt.plot(x,y)


# In[17]:


x=np.linspace(0,10,200)
y1=2*x+1
y2=3*x+1.2
y_mean=0.5*x*np.cos(2*x)+2.5*x+1.1

fig,axes=plt.subplots()
axes.fill_between(x,y1,y2,color="r")
axes.plot(x,y_mean,linewidth=2)


# #### 条形图

# In[18]:


mean_value=[1,2,3]
variance=[0.1,0.2,0.3]
bar_label=["bar1","bar2","bar3"]
max_y=max(zip(mean_value,variance))
x_pos=list(range(len(bar_label)))

plt.bar(x_pos,mean_value,yerr=variance,color="lightblue",alpha=0.3)
plt.ylim(0,(max_y[0]+max_y[1])*1.2)
plt.ylabel("variable y")
plt.xticks(x_pos,bar_label)


# In[19]:


y=[1,2,3]
var=[0.1,0.2,0.3]
label=["bar1","bar2","bar3"]
max_y=max(zip(y,var))
x_pos=list(range(len(label)))

plt.bar(x_pos,y,yerr=var,color="lightblue",alpha=0.3)
plt.ylim(0,(max_y[0]+max_y[1])*1.2)
plt.xticks(x_pos,label)
plt.ylabel("var y")


# In[20]:


green_data=[1,2,3]
blue_data=[3,2,1]
red_data=[2,3,3]
labels=["group1","group2","group3"]

pos=list(range(len(labels)))
fig,axes=plt.subplots(figsize=(8,6))

plt.bar(pos,green_data,width=0.2,color="green",alpha=0.5,label=labels[0])
plt.bar([p+0.2 for p in pos],blue_data,width=0.2,color="blue",alpha=0.5,label=labels[1])
plt.bar([p+0.4 for p in pos],red_data,width=0.2,color="red",alpha=0.4,label=labels[2])
plt.xlim(-0.5,3)
plt.ylim(0,4)


# In[21]:


data=range(200,220,5)
bar_labels=["a","b","c","d","e"]

fig=plt.subplots(figsize=(16,9))

y_pos=np.arange(len(data))

plt.yticks(y_pos,bar_labels,fontsize=16)
barhs=plt.barh(y_pos,data,alpha=0.5,color="g")

plt.vlines(min(data),-1,len(data),colors="r",alpha=0.3,linestyle="--")
for b,d in zip(barhs,data):
    plt.text(b.get_width()+b.get_width()*0.01,b.get_y()+b.get_height()*0.5,"{0:.4}".format(d/min(data)))
    


# #### 颜色渐变

# In[22]:


mean_value=range(8,16)
pos=range(len(mean_value))

import matplotlib.colors as col
import matplotlib.cm as cm

camp1=cm.ScalarMappable(col.Normalize(min(mean_value),max(mean_value),cm.hot))
camp2=cm.ScalarMappable(col.Normalize(0,20,cm.hot))

plt.subplot(2,1,1)
plt.bar(pos,mean_value,color=camp1.to_rgba(mean_value))

plt.subplot(2,1,2)
plt.bar(pos,mean_value,color=camp2.to_rgba(mean_value))


# #### 填充

# In[23]:


patterns=["-","+","X","\\","*"]
data=range(1,len(patterns)+1)
pos=range(len(data))

bars=plt.bar(pos,data,color="lightblue")

for bar, pattern in zip(bars,patterns):
    bar.set_hatch(pattern)


# #### 箱线图

# In[24]:


leon_data=[np.random.normal(0,std,100) for std in range(1,4)]
plt.subplots(figsize=(8,5))
plt.boxplot(leon_data,notch=False,sym="*",vert=True) #vert 控制方向

plt.xticks([1,2,3],["x1","x2","x3"])
plt.xlabel("数据")
plt.title("三个箱线图")


# In[25]:


leon_data=[np.random.normal(0,std,100) for std in range(1,4)]
plt.subplots(figsize=(8,5))
bplot=plt.boxplot(leon_data,notch=False,sym="*",vert=True) #vert 控制方向

plt.xticks([1,2,3],["x1","x2","x3"])
plt.xlabel("数据")
plt.title("三个箱线图")

for components in bplot.keys():
    for line in bplot[components]:
        line.set_color("black")
        


# In[26]:


bplot.keys()


# In[27]:


leon_data=[np.random.normal(0,std,100) for std in range(1,4)]
plt.subplots(figsize=(8,5))
bplot=plt.boxplot(leon_data,notch=False,sym="*",vert=True) #vert 控制方向

plt.xticks([1,2,3],["x1","x2","x3"])
plt.xlabel("数据")
plt.title("三个箱线图")

bplot["boxes"][0].set_color("red")
bplot["boxes"][1].set_color("green")
bplot["boxes"][2].set_color("lightblue")

#一个箱线图有两个 whiskers 竖线
bplot["whiskers"][0].set_color("red")
bplot["whiskers"][1].set_color("green")
bplot["whiskers"][2].set_color("lightblue")
bplot["whiskers"][3].set_color("red")
bplot["whiskers"][4].set_color("green")
bplot["whiskers"][5].set_color("lightblue")

#横线
bplot["caps"][0].set_color("red")
bplot["caps"][1].set_color("green")
bplot["caps"][2].set_color("lightblue")

bplot["medians"][0].set_color("red")
bplot["medians"][1].set_color("green")
bplot["medians"][2].set_color("lightblue")


bplot["fliers"][0].set_color("red")
bplot["fliers"][1].set_color("green")
bplot["fliers"][2].set_color("lightblue")      


# In[28]:


leon_data=[np.random.normal(0,std,100) for std in range(1,4)]
plt.subplots(figsize=(8,5))
bplot=plt.boxplot(leon_data,notch=False,sym="*",vert=True,patch_artist=True) #vert 控制方向  

plt.xticks([1,2,3],["x1","x2","x3"])
plt.xlabel("数据")
plt.title("三个箱线图")

bplot["boxes"][0].set_facecolor("pink")
bplot["boxes"][1].set_facecolor("lightblue")
bplot["boxes"][2].set_facecolor("lightgreen")


# In[29]:


fig,axes=plt.subplots(1,2,figsize=(10,6))
leon_data=[np.random.normal(0,std,100) for std in range(10,14)]

axes[0].violinplot(leon_data)
axes[0].set_title("violin plot")
axes[1].boxplot(leon_data)
axes[1].set_title("box plot")

axes[0].yaxis.grid(True)
axes[1].yaxis.grid(True)

plt.setp(axes[0],xticks=[y+1 for y in range(len(leon_data))],xticklabels=["x1","x2","x3","x4"])
plt.setp(axes[1],xticks=[y+1 for y in range(len(leon_data))],xticklabels=["x1","x2","x3","x4"])


# In[30]:


x=range(10)
y=range(10)

fig=plt.gca()
plt.plot(x,y)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)


# In[31]:


import math
x=np.random.normal(loc=-2,scale=1.0,size=1000)
width=0.5
bins=np.arange(math.floor(x.mean())-width,math.ceil(x.max())+width,width)

ax=plt.subplot()
ax.hist(x,bins=bins)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tick_params(bottom="off",left="off",right="off",top="off")
plt.xlim(-5,5)


# In[32]:


x=range(20)
y=range(20)

labels=["caoliang" for i in range(20)]

ax=plt.subplot()
ax.plot(x,y)
ax.set_xticks(list(range(20)))
ax.set_xticklabels(labels=labels,rotation=90,horizontalalignment="right")


# In[33]:


x=np.arange(10)
for i in range(1,4):
    plt.plot(x,i*x**2,label="group %d"%i)
plt.legend(loc="best")
plt.legend(loc="upper center",bbox_to_anchor=(0.5,1.15),ncol=3,fontsize=12)


# In[34]:


x=np.arange(10)
for i in range(1,4):
    plt.plot(x,i*x**2,label="group %d"%i,marker="o")
plt.legend(loc="upper right",framealpha=0.05,fontsize=12)


# In[35]:


data=np.random.normal(0,20,1000)
bins=np.arange(-100,100,5)
plt.hist(data,bins=bins,normed=0,facecolor="red",edgecolor="white")
plt.xlim([min(data)-5,max(data)+5])


# In[36]:


import random

data1=[random.gauss(15,10) for i in range(500)]
data2=[random.gauss(5,5) for i in range(500)]

plt.hist(data1,bins=30,alpha=0.4,facecolor="r",label="class1",edgecolor="white")
plt.hist(data2,bins=30,alpha=0.4,facecolor="blue",label="class2",edgecolor="white")
plt.legend(loc="best")


# In[37]:


plt.hist


# In[38]:


data1


# In[39]:


mu_vecl=np.array([0,0])
cov_matl=np.array([[2,0],[0,2]])

x1_samples=np.random.multivariate_normal(mu_vecl,cov_matl,100)
x2_samples=np.random.multivariate_normal(mu_vecl+0.2,cov_matl+0.2,100)
x3_samples=np.random.multivariate_normal(mu_vecl+0.4,cov_matl+0.4,100)

plt.scatter(x1_samples[:,0],x1_samples[:,1],marker="x",color="blue",alpha=0.6,label="x1")
plt.scatter(x2_samples[:,0],x2_samples[:,1],marker="o",color="red",alpha=0.6,label="x2")
plt.scatter(x3_samples[:,0],x3_samples[:,1],marker="^",s=40,color="green",alpha=0.6,label="x3")  #s 控制点的大小
plt.legend(loc="best")


# In[40]:


x_data=[1,2,3,4,1,2,3,4]
y_data=[2,1,5,6,7,2,1,7]

plt.figure(figsize=(16,9))
plt.scatter(x_data,y_data,marker="^",s=50)
plt.xlim(min(x_data)-3,max(x_data)+3)
plt.ylim(min(y_data)-3,max(y_data)+3)


for x,y in zip(x_data,y_data):
    plt.annotate("(%s,%s)"%(x,y),fontsize=13,xy=(x,y),xytext=(0,-15),textcoords="offset points",ha="center")
    #textcoords="offset points" 显示坐标，ha=center 居中显示


# In[41]:


mu_vecl=np.array([0,0])
cov_matl=np.array([[2,0],[0,2]])

x=np.random.multivariate_normal(mu_vecl,cov_matl,500)
R=x**2
R_sum=R.sum(axis=1)

plt.figure(figsize=(8,6))
plt.scatter(x[:,0],x[:,1],marker="o",s=20*R_sum,alpha=0.5,color="r")


# In[68]:


m=1234122
f=2131212

m_per=m/(m+f)
f_per=f/(m+f)

plt.figure(figsize=(8,8))
paches,texts,autotexts=plt.pie([m_per,f_per],labels=["Male","Female"],autopct="%1.1f%%",explode=(0,0.1),colors=["r","b"])

for text in texts+autotexts:
    text.set_fontsize(20)
for text in autotexts:
    text.set_color("white")


# #### 子图

# In[72]:


ax1=plt.subplot2grid((3,3),(0,0))
ax2=plt.subplot2grid((3,3),(1,0))
ax3=plt.subplot2grid((3,3),(0,2),rowspan=3)
ax4=plt.subplot2grid((3,3),(0,1),rowspan=2)
ax5=plt.subplot2grid((3,3),(2,0),colspan=2)


# #### 嵌套图

# In[74]:


x=np.linspace(0,10,1000)
y1=x**2
y2=np.sin(x**2)
fig,ax1=plt.subplots()
left,bottom,width,height=[0.22,0.45,0.3,0.2]
ax2=fig.add_axes([left,bottom,width,height])

ax1.plot(x,y1)
ax2.plot(x,y2)


