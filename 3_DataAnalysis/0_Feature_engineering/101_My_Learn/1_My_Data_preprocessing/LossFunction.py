import numpy as np
import matplotlib.pyplot as plt

# 改自http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_loss_functions.html
'''
不同的损失函数有不同的优缺点：
1、**0-1损失函数(zero-one loss)**非常好理解，直接对应分类问题中判断错的个数。但是比较尴尬的是它是一个非凸函数，这意味着其实不是那么实用。
2、hinge loss(SVM中使用到的)的健壮性相对较高(对于异常点/噪声不敏感)。但是它没有那么好的概率解释。
3、**log损失函数(log-loss)**的结果能非常好地表征概率分布。因此在很多场景，尤其是多分类场景下，如果我们需要知道结果属于每个类别的置信度，那这个损失函数很适合。缺点是它的健壮性没有那么强，相对hinge loss会对噪声敏感一些。
4、多项式损失函数(exponential loss)(AdaBoost中用到的)对离群点/噪声非常非常敏感。但是它的形式对于boosting算法简单而有效。
5、**感知损失(perceptron loss)**可以看做是hinge loss的一个变种。hinge loss对于判定边界附近的点(正确端)惩罚力度很高。而perceptron loss，只要样本的判定类别结果是正确的，它就是满意的，而不管其离判定边界的距离。优点是比hinge loss简单，缺点是因为不是max-margin boundary，所以得到模型的泛化能力没有hinge loss强。
'''
xmin, xmax = -4, 4
xx = np.linspace(xmin, xmax, 100)

plt.plot([xmin, 0, 0, xmax], [1, 1, 0, 0], 'k-',
         label="Zero-one loss")

plt.plot(xx, np.where(xx < 1, 1 - xx, 0), 'g-',
         label="Hinge loss")

plt.plot(xx, np.log2(1 + np.exp(-xx)), 'r-',
         label="Log loss")

plt.plot(xx, np.exp(-xx), 'c-',
         label="Exponential loss")

plt.plot(xx, -np.minimum(xx, 0), 'm-',
         label="Perceptron loss")

plt.ylim((0, 8))
plt.legend(loc="upper right")
plt.xlabel(r"Decision function $f(x)$")
plt.ylabel("$L(y, f(x))$")
plt.show()