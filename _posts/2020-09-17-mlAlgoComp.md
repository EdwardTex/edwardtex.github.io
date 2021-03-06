---
layout:     post
title:      知识树：机器学习经典算法总结
subtitle:   MSBD5001 & MSBD5012 前置知识总结
date:       2020-09-17
author:     Tex
header-img: img/post-bg-20ml.jpg
catalog: true
tags:
    - 机器学习 (Machine Learning)

---

> 本文对机器学习经典算法进行了总结，包括每类算法模型的经典实现及优缺点；[原文](https://static.coggle.it/diagram/WHeBqDIrJRk-kDDY/t/categories-of-algorithms-non-exhaustive)在此。 

### 回归 (Regression)

![](https://coggle-attachments-production-eu-west-1.s3.eu-west-1.amazonaws.com/diagram/587781a8322b25193e9030d8/f831eed196b670a7fda7daa752ec675f?AWSAccessKeyId=ASIA4YTCGXFHGSJUIQ5W&Expires=1601628643&Signature=SHE2v4eb%2Bnq61%2FS%2BGbP3OhZ5%2Bno%3D&response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27image.png&x-amz-security-token=IQoJb3JpZ2luX2VjEFsaCWV1LXdlc3QtMSJGMEQCICVHVqwyllKdNqlg76ballDX2LRFC5tHjfw2pXlLkySRAiAvjtBpRlWxCnEhvpWa%2BwZRbiSDykc2tEQ2jQoDTgsY2iq3AwiE%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDg3NzQ1MzAzMTc1OCIMq34uO7kolJcu24CtKosDiiEhxGS2VSjLKNEqiDIX3YGITBQ8b8m%2FcI%2Bzqj5K5%2FqMuU3TgRZMtTyz3YOrK3ubmIF7xNhLSAuOYbRl%2B5Ym4juTQOgodUkNnPdPr3weYISvSH5QwlGx%2FQhPfYBevFE9H6XHdPApfDLYKtFfmwrPK6DGKZp9JrvU76TGYg1A02Z1feiVIbsQfH5Y%2Fz47V%2Bo6Jjx8nfJ1RbZ%2Bsz0Utu4oq4yexH5hl5Nps3Tn43SbragBo5ISRwSsG6HmsKTeK68oJWRN7vfkvr3QHjSjoWzFWvBwVZmj6DOrjh7HQ%2FxYfDJwFpUi9cKtZG8H0hETYqBV0GJAb0MmE%2F61wk7a6zpvHfLA5A9A2HCs%2FY4xUI5TbjOIRxdUGSHjXb7pFfDBfl8YrNye8iwT5L8Y2dQcyzqzZhjq6V7uq1fUETOrGp3nLXyNs2FTn6DS7lImjUm%2FhroJNult6sjYJKejTEh%2BvKLnTIrpP5aG3IXDxRtUhVNZqyZdglQmY2%2BtKbSxuDk7VBf9uD%2B8DzTe2%2Bt0b8Qw8LPa%2BwU67AE9hWzLAEFN8p9Y6swX6faAtisriIR3lJ1wRQ9Xfsf8hVXJ4VRhdNSKIzMakJ186DbyHl8Jfl1MDTZ1rBke%2Fw9Gr2ggzZn9392HxfE9Gl%2FDqqpQ3oby82x2MRV2%2F7Em0SmySvTDUtWsfbYFsOGf9vwjNWoXemOjhQ9vHgTXx1YmrBi7czo4ikk6yFkdfg5GdiT%2BSZV26%2BWvWCXXa3tAHdOHSPIeu74mDLDAYInndrb1U8GRUEY7xQSVtWoZ68Tz%2Ffi61KjoJ%2BEw4WMeQU5k%2BjtVQifvtDrU3o2ek3weGa8GO%2BZJdM1fbNyT8qvdIg%3D%3D)

回归是用于估计两种变量之间关系的统计过程。当用于分析因变量和一个/多个自变量之间的关系时,该算法能提供很多建模和分析的技巧。一般地,回归分析能在给定自变量的条件下估计出因变量的条件期望。

- Classical Examples
  - 普通最小二乘回归(Ordinary Least Squares Regression,OLSR)
  - 线性回归(Linear Regression)
  - 逻辑回归(Logistic Regression)
  - 逐步回归(Stepwise Regression)
  - 多元自适应回归样条(Multivariate Adaptive Regression Splines,MARS)
  - 局部散点平滑估计(Locally Estimated Scatterplot Smoothing,LOESS)

- Pros:{Straightforward, Fast}
- Cons:{Strict assumptions, Bad handling of outliers}

---

### 正则化 (Regularization)

![](https://coggle-attachments-production-eu-west-1.s3.eu-west-1.amazonaws.com/diagram/587781a8322b25193e9030d8/9adaa815a25025a3885a8a69145a1b79?AWSAccessKeyId=ASIA4YTCGXFHGSJUIQ5W&Expires=1601628643&Signature=2rlpqLLaWK3gSGA6OfINYwP40Ck%3D&response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27image.png&x-amz-security-token=IQoJb3JpZ2luX2VjEFsaCWV1LXdlc3QtMSJGMEQCICVHVqwyllKdNqlg76ballDX2LRFC5tHjfw2pXlLkySRAiAvjtBpRlWxCnEhvpWa%2BwZRbiSDykc2tEQ2jQoDTgsY2iq3AwiE%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDg3NzQ1MzAzMTc1OCIMq34uO7kolJcu24CtKosDiiEhxGS2VSjLKNEqiDIX3YGITBQ8b8m%2FcI%2Bzqj5K5%2FqMuU3TgRZMtTyz3YOrK3ubmIF7xNhLSAuOYbRl%2B5Ym4juTQOgodUkNnPdPr3weYISvSH5QwlGx%2FQhPfYBevFE9H6XHdPApfDLYKtFfmwrPK6DGKZp9JrvU76TGYg1A02Z1feiVIbsQfH5Y%2Fz47V%2Bo6Jjx8nfJ1RbZ%2Bsz0Utu4oq4yexH5hl5Nps3Tn43SbragBo5ISRwSsG6HmsKTeK68oJWRN7vfkvr3QHjSjoWzFWvBwVZmj6DOrjh7HQ%2FxYfDJwFpUi9cKtZG8H0hETYqBV0GJAb0MmE%2F61wk7a6zpvHfLA5A9A2HCs%2FY4xUI5TbjOIRxdUGSHjXb7pFfDBfl8YrNye8iwT5L8Y2dQcyzqzZhjq6V7uq1fUETOrGp3nLXyNs2FTn6DS7lImjUm%2FhroJNult6sjYJKejTEh%2BvKLnTIrpP5aG3IXDxRtUhVNZqyZdglQmY2%2BtKbSxuDk7VBf9uD%2B8DzTe2%2Bt0b8Qw8LPa%2BwU67AE9hWzLAEFN8p9Y6swX6faAtisriIR3lJ1wRQ9Xfsf8hVXJ4VRhdNSKIzMakJ186DbyHl8Jfl1MDTZ1rBke%2Fw9Gr2ggzZn9392HxfE9Gl%2FDqqpQ3oby82x2MRV2%2F7Em0SmySvTDUtWsfbYFsOGf9vwjNWoXemOjhQ9vHgTXx1YmrBi7czo4ikk6yFkdfg5GdiT%2BSZV26%2BWvWCXXa3tAHdOHSPIeu74mDLDAYInndrb1U8GRUEY7xQSVtWoZ68Tz%2Ffi61KjoJ%2BEw4WMeQU5k%2BjtVQifvtDrU3o2ek3weGa8GO%2BZJdM1fbNyT8qvdIg%3D%3D)

正则化是其他方法的扩展，基于其他模型的复杂度对其进行惩罚，通过偏好更简单的模型，进而得到更好的泛化性能。

- Classical Examples:
  - 岭回归(Ridge Regression)
  - 最小绝对收缩与选择算子(Least Absolute Shrinkage and Selection Operator, LASSO)
  - 基于图的最小绝对收缩与选择算子(Graphical Least Absolute Shrinkage and Selection Operator, GLASSO)
  - 弹性网络(Elastic Net)
  - 最小角回归(Least-Angle Regression)

- Pros:{Penalties reduce overfitting, Solution always exists}
- Cons:{Penalties may cause underfitting, Difficult to calibrate}

---

### 决策树 (Decision Tree)

![](https://coggle-attachments-production-eu-west-1.s3.eu-west-1.amazonaws.com/diagram/587781a8322b25193e9030d8/34f58fb6facca7f08b6638b4496b5c19?AWSAccessKeyId=ASIA4YTCGXFHGSJUIQ5W&Expires=1601628643&Signature=fLC9ZLLEnW9V9NijtnrydXv8xWM%3D&response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27image.png&x-amz-security-token=IQoJb3JpZ2luX2VjEFsaCWV1LXdlc3QtMSJGMEQCICVHVqwyllKdNqlg76ballDX2LRFC5tHjfw2pXlLkySRAiAvjtBpRlWxCnEhvpWa%2BwZRbiSDykc2tEQ2jQoDTgsY2iq3AwiE%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDg3NzQ1MzAzMTc1OCIMq34uO7kolJcu24CtKosDiiEhxGS2VSjLKNEqiDIX3YGITBQ8b8m%2FcI%2Bzqj5K5%2FqMuU3TgRZMtTyz3YOrK3ubmIF7xNhLSAuOYbRl%2B5Ym4juTQOgodUkNnPdPr3weYISvSH5QwlGx%2FQhPfYBevFE9H6XHdPApfDLYKtFfmwrPK6DGKZp9JrvU76TGYg1A02Z1feiVIbsQfH5Y%2Fz47V%2Bo6Jjx8nfJ1RbZ%2Bsz0Utu4oq4yexH5hl5Nps3Tn43SbragBo5ISRwSsG6HmsKTeK68oJWRN7vfkvr3QHjSjoWzFWvBwVZmj6DOrjh7HQ%2FxYfDJwFpUi9cKtZG8H0hETYqBV0GJAb0MmE%2F61wk7a6zpvHfLA5A9A2HCs%2FY4xUI5TbjOIRxdUGSHjXb7pFfDBfl8YrNye8iwT5L8Y2dQcyzqzZhjq6V7uq1fUETOrGp3nLXyNs2FTn6DS7lImjUm%2FhroJNult6sjYJKejTEh%2BvKLnTIrpP5aG3IXDxRtUhVNZqyZdglQmY2%2BtKbSxuDk7VBf9uD%2B8DzTe2%2Bt0b8Qw8LPa%2BwU67AE9hWzLAEFN8p9Y6swX6faAtisriIR3lJ1wRQ9Xfsf8hVXJ4VRhdNSKIzMakJ186DbyHl8Jfl1MDTZ1rBke%2Fw9Gr2ggzZn9392HxfE9Gl%2FDqqpQ3oby82x2MRV2%2F7Em0SmySvTDUtWsfbYFsOGf9vwjNWoXemOjhQ9vHgTXx1YmrBi7czo4ikk6yFkdfg5GdiT%2BSZV26%2BWvWCXXa3tAHdOHSPIeu74mDLDAYInndrb1U8GRUEY7xQSVtWoZ68Tz%2Ffi61KjoJ%2BEw4WMeQU5k%2BjtVQifvtDrU3o2ek3weGa8GO%2BZJdM1fbNyT8qvdIg%3D%3D)

决策树通过树形结构作为模型,通过对某项(表征在分支上)观察所得映射成关于该项的目标值的结论(表征在叶子中)。根据目标值的性质可以分为回归树和分类树。

- Classical Examples
  - 分类和回归树(Classification and Regression Tree,CART)
  - 迭代二分器(Iterative Dichotomiser 3, ID3)
  - C4.5 和 C5.0(一种强大方法的两个不同版本)
  - 条件决策树

- Pros:{Easy to interpret, Non-parametric}
- Cons:{Tend to overfit, May stuck in local optimal, No online learning}

---

### 集成 (Ensemble)

![](https://coggle-attachments-production-eu-west-1.s3.eu-west-1.amazonaws.com/diagram/587781a8322b25193e9030d8/91971fe2b390d2af1661184286d91f3c?AWSAccessKeyId=ASIA4YTCGXFHGSJUIQ5W&Expires=1601628643&Signature=OUaeVFp2MeuMBXfFIFvG21saqWM%3D&response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27image.png&x-amz-security-token=IQoJb3JpZ2luX2VjEFsaCWV1LXdlc3QtMSJGMEQCICVHVqwyllKdNqlg76ballDX2LRFC5tHjfw2pXlLkySRAiAvjtBpRlWxCnEhvpWa%2BwZRbiSDykc2tEQ2jQoDTgsY2iq3AwiE%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDg3NzQ1MzAzMTc1OCIMq34uO7kolJcu24CtKosDiiEhxGS2VSjLKNEqiDIX3YGITBQ8b8m%2FcI%2Bzqj5K5%2FqMuU3TgRZMtTyz3YOrK3ubmIF7xNhLSAuOYbRl%2B5Ym4juTQOgodUkNnPdPr3weYISvSH5QwlGx%2FQhPfYBevFE9H6XHdPApfDLYKtFfmwrPK6DGKZp9JrvU76TGYg1A02Z1feiVIbsQfH5Y%2Fz47V%2Bo6Jjx8nfJ1RbZ%2Bsz0Utu4oq4yexH5hl5Nps3Tn43SbragBo5ISRwSsG6HmsKTeK68oJWRN7vfkvr3QHjSjoWzFWvBwVZmj6DOrjh7HQ%2FxYfDJwFpUi9cKtZG8H0hETYqBV0GJAb0MmE%2F61wk7a6zpvHfLA5A9A2HCs%2FY4xUI5TbjOIRxdUGSHjXb7pFfDBfl8YrNye8iwT5L8Y2dQcyzqzZhjq6V7uq1fUETOrGp3nLXyNs2FTn6DS7lImjUm%2FhroJNult6sjYJKejTEh%2BvKLnTIrpP5aG3IXDxRtUhVNZqyZdglQmY2%2BtKbSxuDk7VBf9uD%2B8DzTe2%2Bt0b8Qw8LPa%2BwU67AE9hWzLAEFN8p9Y6swX6faAtisriIR3lJ1wRQ9Xfsf8hVXJ4VRhdNSKIzMakJ186DbyHl8Jfl1MDTZ1rBke%2Fw9Gr2ggzZn9392HxfE9Gl%2FDqqpQ3oby82x2MRV2%2F7Em0SmySvTDUtWsfbYFsOGf9vwjNWoXemOjhQ9vHgTXx1YmrBi7czo4ikk6yFkdfg5GdiT%2BSZV26%2BWvWCXXa3tAHdOHSPIeu74mDLDAYInndrb1U8GRUEY7xQSVtWoZ68Tz%2Ffi61KjoJ%2BEw4WMeQU5k%2BjtVQifvtDrU3o2ek3weGa8GO%2BZJdM1fbNyT8qvdIg%3D%3D)

集成方法是由多个较弱的模型集成模型组,其中的模型可以单独进行训练,并且它们的预测能以某种方式结合起来去做出一个总体预测。该算法主要的问题是要找出哪些较弱的模型可以结合起来,以及结合的方法。

- Classical Examples
  - Boosting
  - Bagging(Bootstrapped Aggregation)
  - AdaBoost
  - 层叠泛化(Stacked Generalization)(blending)
  - 梯度推进机(Gradient Boosting Machines,GBM)
  - 梯度提升回归树(Gradient Boosted Regression Trees,GBRT)
  - 随机森林(Random Forest)

- Pros:{More accurate than single models}
- Cons:{Lots of work and maintenance required}

---

### 支持向量机 (Support Vector Machine)

![](https://coggle-attachments-production-eu-west-1.s3.eu-west-1.amazonaws.com/diagram/587781a8322b25193e9030d8/d739a4dd7060a1ab72543c705d48af11?AWSAccessKeyId=ASIA4YTCGXFHGSJUIQ5W&Expires=1601628643&Signature=nH4L3m9TcJnUmDjgJxXMaqR%2BHmU%3D&response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27image.png&x-amz-security-token=IQoJb3JpZ2luX2VjEFsaCWV1LXdlc3QtMSJGMEQCICVHVqwyllKdNqlg76ballDX2LRFC5tHjfw2pXlLkySRAiAvjtBpRlWxCnEhvpWa%2BwZRbiSDykc2tEQ2jQoDTgsY2iq3AwiE%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDg3NzQ1MzAzMTc1OCIMq34uO7kolJcu24CtKosDiiEhxGS2VSjLKNEqiDIX3YGITBQ8b8m%2FcI%2Bzqj5K5%2FqMuU3TgRZMtTyz3YOrK3ubmIF7xNhLSAuOYbRl%2B5Ym4juTQOgodUkNnPdPr3weYISvSH5QwlGx%2FQhPfYBevFE9H6XHdPApfDLYKtFfmwrPK6DGKZp9JrvU76TGYg1A02Z1feiVIbsQfH5Y%2Fz47V%2Bo6Jjx8nfJ1RbZ%2Bsz0Utu4oq4yexH5hl5Nps3Tn43SbragBo5ISRwSsG6HmsKTeK68oJWRN7vfkvr3QHjSjoWzFWvBwVZmj6DOrjh7HQ%2FxYfDJwFpUi9cKtZG8H0hETYqBV0GJAb0MmE%2F61wk7a6zpvHfLA5A9A2HCs%2FY4xUI5TbjOIRxdUGSHjXb7pFfDBfl8YrNye8iwT5L8Y2dQcyzqzZhjq6V7uq1fUETOrGp3nLXyNs2FTn6DS7lImjUm%2FhroJNult6sjYJKejTEh%2BvKLnTIrpP5aG3IXDxRtUhVNZqyZdglQmY2%2BtKbSxuDk7VBf9uD%2B8DzTe2%2Bt0b8Qw8LPa%2BwU67AE9hWzLAEFN8p9Y6swX6faAtisriIR3lJ1wRQ9Xfsf8hVXJ4VRhdNSKIzMakJ186DbyHl8Jfl1MDTZ1rBke%2Fw9Gr2ggzZn9392HxfE9Gl%2FDqqpQ3oby82x2MRV2%2F7Em0SmySvTDUtWsfbYFsOGf9vwjNWoXemOjhQ9vHgTXx1YmrBi7czo4ikk6yFkdfg5GdiT%2BSZV26%2BWvWCXXa3tAHdOHSPIeu74mDLDAYInndrb1U8GRUEY7xQSVtWoZ68Tz%2Ffi61KjoJ%2BEw4WMeQU5k%2BjtVQifvtDrU3o2ek3weGa8GO%2BZJdM1fbNyT8qvdIg%3D%3D)

在二分类问题中，支持向量机(SVM)将训练集表示为空间中的点,它们被映射到一幅图中,由一条明确的、尽可能宽的间隔分开以区分两个类别。

- Pros:{Works on non linearly separable problems thanks to kernel trick}
- Cons:{Hard to train or interpret}

---

### 聚类 (Clustering)

![](https://coggle-attachments-production-eu-west-1.s3.eu-west-1.amazonaws.com/diagram/587781a8322b25193e9030d8/6d20ba3e27426264678ecc57d94ca126?AWSAccessKeyId=ASIA4YTCGXFHGSJUIQ5W&Expires=1601628643&Signature=VbLXF6gEcqkDFADYT5uBQYO5Evk%3D&response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27image.png&x-amz-security-token=IQoJb3JpZ2luX2VjEFsaCWV1LXdlc3QtMSJGMEQCICVHVqwyllKdNqlg76ballDX2LRFC5tHjfw2pXlLkySRAiAvjtBpRlWxCnEhvpWa%2BwZRbiSDykc2tEQ2jQoDTgsY2iq3AwiE%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDg3NzQ1MzAzMTc1OCIMq34uO7kolJcu24CtKosDiiEhxGS2VSjLKNEqiDIX3YGITBQ8b8m%2FcI%2Bzqj5K5%2FqMuU3TgRZMtTyz3YOrK3ubmIF7xNhLSAuOYbRl%2B5Ym4juTQOgodUkNnPdPr3weYISvSH5QwlGx%2FQhPfYBevFE9H6XHdPApfDLYKtFfmwrPK6DGKZp9JrvU76TGYg1A02Z1feiVIbsQfH5Y%2Fz47V%2Bo6Jjx8nfJ1RbZ%2Bsz0Utu4oq4yexH5hl5Nps3Tn43SbragBo5ISRwSsG6HmsKTeK68oJWRN7vfkvr3QHjSjoWzFWvBwVZmj6DOrjh7HQ%2FxYfDJwFpUi9cKtZG8H0hETYqBV0GJAb0MmE%2F61wk7a6zpvHfLA5A9A2HCs%2FY4xUI5TbjOIRxdUGSHjXb7pFfDBfl8YrNye8iwT5L8Y2dQcyzqzZhjq6V7uq1fUETOrGp3nLXyNs2FTn6DS7lImjUm%2FhroJNult6sjYJKejTEh%2BvKLnTIrpP5aG3IXDxRtUhVNZqyZdglQmY2%2BtKbSxuDk7VBf9uD%2B8DzTe2%2Bt0b8Qw8LPa%2BwU67AE9hWzLAEFN8p9Y6swX6faAtisriIR3lJ1wRQ9Xfsf8hVXJ4VRhdNSKIzMakJ186DbyHl8Jfl1MDTZ1rBke%2Fw9Gr2ggzZn9392HxfE9Gl%2FDqqpQ3oby82x2MRV2%2F7Em0SmySvTDUtWsfbYFsOGf9vwjNWoXemOjhQ9vHgTXx1YmrBi7czo4ikk6yFkdfg5GdiT%2BSZV26%2BWvWCXXa3tAHdOHSPIeu74mDLDAYInndrb1U8GRUEY7xQSVtWoZ68Tz%2Ffi61KjoJ%2BEw4WMeQU5k%2BjtVQifvtDrU3o2ek3weGa8GO%2BZJdM1fbNyT8qvdIg%3D%3D)

聚类是指对一组目标进行分类,属于同一簇(亦即一个类,cluster)的目标被划分在一组中,与其他簇目标相比,同一簇目标更加彼此相似(在某种意义上)。

- Classical Examples
  - K-Means
  - K-Medians
  - Hierarchical Clustering
  - Expectation Maximisation(EM)

- Pros:{Useful for making sense of data}
- Cons:{Results can be hard to read, May be useless on unusual datasets}

---

### 降维 (Dimensionality Reduction)

![](https://coggle-attachments-production-eu-west-1.s3.eu-west-1.amazonaws.com/diagram/587781a8322b25193e9030d8/727d1071e61c7639c0a7571c0b19e341?AWSAccessKeyId=ASIA4YTCGXFHGSJUIQ5W&Expires=1601628643&Signature=SaQfRz3yfADFswwVRPXBZ28Q%2Bv4%3D&response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27image.png&x-amz-security-token=IQoJb3JpZ2luX2VjEFsaCWV1LXdlc3QtMSJGMEQCICVHVqwyllKdNqlg76ballDX2LRFC5tHjfw2pXlLkySRAiAvjtBpRlWxCnEhvpWa%2BwZRbiSDykc2tEQ2jQoDTgsY2iq3AwiE%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDg3NzQ1MzAzMTc1OCIMq34uO7kolJcu24CtKosDiiEhxGS2VSjLKNEqiDIX3YGITBQ8b8m%2FcI%2Bzqj5K5%2FqMuU3TgRZMtTyz3YOrK3ubmIF7xNhLSAuOYbRl%2B5Ym4juTQOgodUkNnPdPr3weYISvSH5QwlGx%2FQhPfYBevFE9H6XHdPApfDLYKtFfmwrPK6DGKZp9JrvU76TGYg1A02Z1feiVIbsQfH5Y%2Fz47V%2Bo6Jjx8nfJ1RbZ%2Bsz0Utu4oq4yexH5hl5Nps3Tn43SbragBo5ISRwSsG6HmsKTeK68oJWRN7vfkvr3QHjSjoWzFWvBwVZmj6DOrjh7HQ%2FxYfDJwFpUi9cKtZG8H0hETYqBV0GJAb0MmE%2F61wk7a6zpvHfLA5A9A2HCs%2FY4xUI5TbjOIRxdUGSHjXb7pFfDBfl8YrNye8iwT5L8Y2dQcyzqzZhjq6V7uq1fUETOrGp3nLXyNs2FTn6DS7lImjUm%2FhroJNult6sjYJKejTEh%2BvKLnTIrpP5aG3IXDxRtUhVNZqyZdglQmY2%2BtKbSxuDk7VBf9uD%2B8DzTe2%2Bt0b8Qw8LPa%2BwU67AE9hWzLAEFN8p9Y6swX6faAtisriIR3lJ1wRQ9Xfsf8hVXJ4VRhdNSKIzMakJ186DbyHl8Jfl1MDTZ1rBke%2Fw9Gr2ggzZn9392HxfE9Gl%2FDqqpQ3oby82x2MRV2%2F7Em0SmySvTDUtWsfbYFsOGf9vwjNWoXemOjhQ9vHgTXx1YmrBi7czo4ikk6yFkdfg5GdiT%2BSZV26%2BWvWCXXa3tAHdOHSPIeu74mDLDAYInndrb1U8GRUEY7xQSVtWoZ68Tz%2Ffi61KjoJ%2BEw4WMeQU5k%2BjtVQifvtDrU3o2ek3weGa8GO%2BZJdM1fbNyT8qvdIg%3D%3D)

和聚类类似,降维追求并利用数据的内在结构,目的在于使用较少的信息总结或描述数据。这一算法可用于可视化高维数据或简化接下来可用于监督学习中的数据。

- Classical Examples
  - 主成分分析(Principal Component Analysis, PCA)
  - 主成分回归(Principal Component Regression, PCR)
  - 偏最小二乘回归(Partial Least Squares Regression, PLSR)
  - Sammon 映射(Sammon Mapping)
  - 多维尺度变换(Multidimensional Scaling, MDS)
  - 投影寻踪(Projection Pursuit)
  - 线性判别分析(Linear Discriminant Analysis, LDA)
  - 混合判别分析(Mixture Discriminant Analysis, MDA)
  - 二次判别分析(Quadratic Discriminant Analysis, QDA)
  - 灵活判别分析(Flexible Discriminant Analysis, FDA)

- Pros:{Handles large dataset, No assumptions on data}
- Cons:{Hard to handle non-linear data and interpret results}

---

### 贝叶斯方法 (Bayesian)

![](https://coggle-attachments-production-eu-west-1.s3.eu-west-1.amazonaws.com/diagram/587781a8322b25193e9030d8/4074242f88aa252a78f13993e344955f?AWSAccessKeyId=ASIA4YTCGXFHGSJUIQ5W&Expires=1601628643&Signature=z76Y%2BISV0JWyoQQ%2BIixRFka4LEo%3D&response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27image.png&x-amz-security-token=IQoJb3JpZ2luX2VjEFsaCWV1LXdlc3QtMSJGMEQCICVHVqwyllKdNqlg76ballDX2LRFC5tHjfw2pXlLkySRAiAvjtBpRlWxCnEhvpWa%2BwZRbiSDykc2tEQ2jQoDTgsY2iq3AwiE%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDg3NzQ1MzAzMTc1OCIMq34uO7kolJcu24CtKosDiiEhxGS2VSjLKNEqiDIX3YGITBQ8b8m%2FcI%2Bzqj5K5%2FqMuU3TgRZMtTyz3YOrK3ubmIF7xNhLSAuOYbRl%2B5Ym4juTQOgodUkNnPdPr3weYISvSH5QwlGx%2FQhPfYBevFE9H6XHdPApfDLYKtFfmwrPK6DGKZp9JrvU76TGYg1A02Z1feiVIbsQfH5Y%2Fz47V%2Bo6Jjx8nfJ1RbZ%2Bsz0Utu4oq4yexH5hl5Nps3Tn43SbragBo5ISRwSsG6HmsKTeK68oJWRN7vfkvr3QHjSjoWzFWvBwVZmj6DOrjh7HQ%2FxYfDJwFpUi9cKtZG8H0hETYqBV0GJAb0MmE%2F61wk7a6zpvHfLA5A9A2HCs%2FY4xUI5TbjOIRxdUGSHjXb7pFfDBfl8YrNye8iwT5L8Y2dQcyzqzZhjq6V7uq1fUETOrGp3nLXyNs2FTn6DS7lImjUm%2FhroJNult6sjYJKejTEh%2BvKLnTIrpP5aG3IXDxRtUhVNZqyZdglQmY2%2BtKbSxuDk7VBf9uD%2B8DzTe2%2Bt0b8Qw8LPa%2BwU67AE9hWzLAEFN8p9Y6swX6faAtisriIR3lJ1wRQ9Xfsf8hVXJ4VRhdNSKIzMakJ186DbyHl8Jfl1MDTZ1rBke%2Fw9Gr2ggzZn9392HxfE9Gl%2FDqqpQ3oby82x2MRV2%2F7Em0SmySvTDUtWsfbYFsOGf9vwjNWoXemOjhQ9vHgTXx1YmrBi7czo4ikk6yFkdfg5GdiT%2BSZV26%2BWvWCXXa3tAHdOHSPIeu74mDLDAYInndrb1U8GRUEY7xQSVtWoZ68Tz%2Ffi61KjoJ%2BEw4WMeQU5k%2BjtVQifvtDrU3o2ek3weGa8GO%2BZJdM1fbNyT8qvdIg%3D%3D)

贝叶斯方法是指明确应用了贝叶斯定理来解决如分类和回归等问题的方法。

- Classical Examples
  - 朴素贝叶斯(Naive Bayes)
  - 高斯朴素贝叶斯(Gaussian Naive Bayes)
  - 多项式朴素贝叶斯(Multinomial Naive Bayes)
  - 平均一致依赖估计器(Averaged One-Dependence Estimators, AODE)
  - 贝叶斯信念网络(Bayesian Belief Network, BBN)
  - 贝叶斯网络(Bayesian Network, BN)

- Pros:{Easy to train, Fast, Good perfomance}
- Cons:{Problems if the input variables are correlated}

---

### 基于实例 (Instance-based)

![](https://coggle-attachments-production-eu-west-1.s3.eu-west-1.amazonaws.com/diagram/587781a8322b25193e9030d8/4074242f88aa252a78f13993e344955f?AWSAccessKeyId=ASIA4YTCGXFHGSJUIQ5W&Expires=1601628643&Signature=z76Y%2BISV0JWyoQQ%2BIixRFka4LEo%3D&response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27image.png&x-amz-security-token=IQoJb3JpZ2luX2VjEFsaCWV1LXdlc3QtMSJGMEQCICVHVqwyllKdNqlg76ballDX2LRFC5tHjfw2pXlLkySRAiAvjtBpRlWxCnEhvpWa%2BwZRbiSDykc2tEQ2jQoDTgsY2iq3AwiE%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDg3NzQ1MzAzMTc1OCIMq34uO7kolJcu24CtKosDiiEhxGS2VSjLKNEqiDIX3YGITBQ8b8m%2FcI%2Bzqj5K5%2FqMuU3TgRZMtTyz3YOrK3ubmIF7xNhLSAuOYbRl%2B5Ym4juTQOgodUkNnPdPr3weYISvSH5QwlGx%2FQhPfYBevFE9H6XHdPApfDLYKtFfmwrPK6DGKZp9JrvU76TGYg1A02Z1feiVIbsQfH5Y%2Fz47V%2Bo6Jjx8nfJ1RbZ%2Bsz0Utu4oq4yexH5hl5Nps3Tn43SbragBo5ISRwSsG6HmsKTeK68oJWRN7vfkvr3QHjSjoWzFWvBwVZmj6DOrjh7HQ%2FxYfDJwFpUi9cKtZG8H0hETYqBV0GJAb0MmE%2F61wk7a6zpvHfLA5A9A2HCs%2FY4xUI5TbjOIRxdUGSHjXb7pFfDBfl8YrNye8iwT5L8Y2dQcyzqzZhjq6V7uq1fUETOrGp3nLXyNs2FTn6DS7lImjUm%2FhroJNult6sjYJKejTEh%2BvKLnTIrpP5aG3IXDxRtUhVNZqyZdglQmY2%2BtKbSxuDk7VBf9uD%2B8DzTe2%2Bt0b8Qw8LPa%2BwU67AE9hWzLAEFN8p9Y6swX6faAtisriIR3lJ1wRQ9Xfsf8hVXJ4VRhdNSKIzMakJ186DbyHl8Jfl1MDTZ1rBke%2Fw9Gr2ggzZn9392HxfE9Gl%2FDqqpQ3oby82x2MRV2%2F7Em0SmySvTDUtWsfbYFsOGf9vwjNWoXemOjhQ9vHgTXx1YmrBi7czo4ikk6yFkdfg5GdiT%2BSZV26%2BWvWCXXa3tAHdOHSPIeu74mDLDAYInndrb1U8GRUEY7xQSVtWoZ68Tz%2Ffi61KjoJ%2BEw4WMeQU5k%2BjtVQifvtDrU3o2ek3weGa8GO%2BZJdM1fbNyT8qvdIg%3D%3D)

基于实例(亦称基于记忆)不是明确归纳,直接从训练实例中建构出假设,将新的实例与训练过程中见过的实例进行对比。

- Classical Examples
  - K近邻(k-Nearest Neighbor, kNN)
  - 学习向量量化(Learning Vector Quantization, LVQ)
  - 自组织映射(Self-Organizing Map, SOM)
  - 局部加权学习(Locally Weighted Learning, LWL)

- Pros:{Simple, Easy to interpret results}
- Cons:{High memory usage, Computationally heavy, Hard to handle high-dimen feature spaces}

---

### 关联规则 (Association Rule)

![](https://coggle-attachments-production-eu-west-1.s3.eu-west-1.amazonaws.com/diagram/587781a8322b25193e9030d8/dd4d5a6a32ea04ed88a6d244bff37f13?AWSAccessKeyId=ASIA4YTCGXFHGSJUIQ5W&Expires=1601628643&Signature=Ym7uC4irppKhH1lcgNimecaylUA%3D&response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27image.png&x-amz-security-token=IQoJb3JpZ2luX2VjEFsaCWV1LXdlc3QtMSJGMEQCICVHVqwyllKdNqlg76ballDX2LRFC5tHjfw2pXlLkySRAiAvjtBpRlWxCnEhvpWa%2BwZRbiSDykc2tEQ2jQoDTgsY2iq3AwiE%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDg3NzQ1MzAzMTc1OCIMq34uO7kolJcu24CtKosDiiEhxGS2VSjLKNEqiDIX3YGITBQ8b8m%2FcI%2Bzqj5K5%2FqMuU3TgRZMtTyz3YOrK3ubmIF7xNhLSAuOYbRl%2B5Ym4juTQOgodUkNnPdPr3weYISvSH5QwlGx%2FQhPfYBevFE9H6XHdPApfDLYKtFfmwrPK6DGKZp9JrvU76TGYg1A02Z1feiVIbsQfH5Y%2Fz47V%2Bo6Jjx8nfJ1RbZ%2Bsz0Utu4oq4yexH5hl5Nps3Tn43SbragBo5ISRwSsG6HmsKTeK68oJWRN7vfkvr3QHjSjoWzFWvBwVZmj6DOrjh7HQ%2FxYfDJwFpUi9cKtZG8H0hETYqBV0GJAb0MmE%2F61wk7a6zpvHfLA5A9A2HCs%2FY4xUI5TbjOIRxdUGSHjXb7pFfDBfl8YrNye8iwT5L8Y2dQcyzqzZhjq6V7uq1fUETOrGp3nLXyNs2FTn6DS7lImjUm%2FhroJNult6sjYJKejTEh%2BvKLnTIrpP5aG3IXDxRtUhVNZqyZdglQmY2%2BtKbSxuDk7VBf9uD%2B8DzTe2%2Bt0b8Qw8LPa%2BwU67AE9hWzLAEFN8p9Y6swX6faAtisriIR3lJ1wRQ9Xfsf8hVXJ4VRhdNSKIzMakJ186DbyHl8Jfl1MDTZ1rBke%2Fw9Gr2ggzZn9392HxfE9Gl%2FDqqpQ3oby82x2MRV2%2F7Em0SmySvTDUtWsfbYFsOGf9vwjNWoXemOjhQ9vHgTXx1YmrBi7czo4ikk6yFkdfg5GdiT%2BSZV26%2BWvWCXXa3tAHdOHSPIeu74mDLDAYInndrb1U8GRUEY7xQSVtWoZ68Tz%2Ffi61KjoJ%2BEw4WMeQU5k%2BjtVQifvtDrU3o2ek3weGa8GO%2BZJdM1fbNyT8qvdIg%3D%3D)

关联规则能够提取出对数据中的变量之间的关系的最佳解释。

- Classical Examples
  - Apriori 算法(Apriori algorithm)
  - Eclat 算法(Eclat algorithm)
  - FP-growth

---

### 图模型 (Graphical Models)

![](https://coggle-attachments-production-eu-west-1.s3.eu-west-1.amazonaws.com/diagram/587781a8322b25193e9030d8/c3d9685ba0243ba9cf10c2fae4fe6050?AWSAccessKeyId=ASIA4YTCGXFHGSJUIQ5W&Expires=1601628643&Signature=JuowxWLe08nxyOsX8ulUJtHVwUc%3D&response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27image.png&x-amz-security-token=IQoJb3JpZ2luX2VjEFsaCWV1LXdlc3QtMSJGMEQCICVHVqwyllKdNqlg76ballDX2LRFC5tHjfw2pXlLkySRAiAvjtBpRlWxCnEhvpWa%2BwZRbiSDykc2tEQ2jQoDTgsY2iq3AwiE%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDg3NzQ1MzAzMTc1OCIMq34uO7kolJcu24CtKosDiiEhxGS2VSjLKNEqiDIX3YGITBQ8b8m%2FcI%2Bzqj5K5%2FqMuU3TgRZMtTyz3YOrK3ubmIF7xNhLSAuOYbRl%2B5Ym4juTQOgodUkNnPdPr3weYISvSH5QwlGx%2FQhPfYBevFE9H6XHdPApfDLYKtFfmwrPK6DGKZp9JrvU76TGYg1A02Z1feiVIbsQfH5Y%2Fz47V%2Bo6Jjx8nfJ1RbZ%2Bsz0Utu4oq4yexH5hl5Nps3Tn43SbragBo5ISRwSsG6HmsKTeK68oJWRN7vfkvr3QHjSjoWzFWvBwVZmj6DOrjh7HQ%2FxYfDJwFpUi9cKtZG8H0hETYqBV0GJAb0MmE%2F61wk7a6zpvHfLA5A9A2HCs%2FY4xUI5TbjOIRxdUGSHjXb7pFfDBfl8YrNye8iwT5L8Y2dQcyzqzZhjq6V7uq1fUETOrGp3nLXyNs2FTn6DS7lImjUm%2FhroJNult6sjYJKejTEh%2BvKLnTIrpP5aG3IXDxRtUhVNZqyZdglQmY2%2BtKbSxuDk7VBf9uD%2B8DzTe2%2Bt0b8Qw8LPa%2BwU67AE9hWzLAEFN8p9Y6swX6faAtisriIR3lJ1wRQ9Xfsf8hVXJ4VRhdNSKIzMakJ186DbyHl8Jfl1MDTZ1rBke%2Fw9Gr2ggzZn9392HxfE9Gl%2FDqqpQ3oby82x2MRV2%2F7Em0SmySvTDUtWsfbYFsOGf9vwjNWoXemOjhQ9vHgTXx1YmrBi7czo4ikk6yFkdfg5GdiT%2BSZV26%2BWvWCXXa3tAHdOHSPIeu74mDLDAYInndrb1U8GRUEY7xQSVtWoZ68Tz%2Ffi61KjoJ%2BEw4WMeQU5k%2BjtVQifvtDrU3o2ek3weGa8GO%2BZJdM1fbNyT8qvdIg%3D%3D)

图模型或概率图模型(probabilistic graphical model)是一种概率模型, 通过图结构表示随机变量之间的条件依赖结构(conditional dependence structure)。

- Classical Examples
  - 贝叶斯网络(Bayesian network)
  - 马尔可夫随机域(Markov random field)
  - 链图(Chain Graphs)
  - 祖先图(Ancestral graph)

- Pros:{Intuitively understood}
- Cons:{Determining the topology of dependence is difficult, sometimes ambiguous}

---

### 神经网络 (Neural Network)

![](https://coggle-attachments-production-eu-west-1.s3.eu-west-1.amazonaws.com/diagram/587781a8322b25193e9030d8/59953deec480cc6879b4d828fa65d403?AWSAccessKeyId=ASIA4YTCGXFHGSJUIQ5W&Expires=1601628643&Signature=CaTXOnrTq0JrfW6zzgF0MKkEQMA%3D&response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27image.png&x-amz-security-token=IQoJb3JpZ2luX2VjEFsaCWV1LXdlc3QtMSJGMEQCICVHVqwyllKdNqlg76ballDX2LRFC5tHjfw2pXlLkySRAiAvjtBpRlWxCnEhvpWa%2BwZRbiSDykc2tEQ2jQoDTgsY2iq3AwiE%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDg3NzQ1MzAzMTc1OCIMq34uO7kolJcu24CtKosDiiEhxGS2VSjLKNEqiDIX3YGITBQ8b8m%2FcI%2Bzqj5K5%2FqMuU3TgRZMtTyz3YOrK3ubmIF7xNhLSAuOYbRl%2B5Ym4juTQOgodUkNnPdPr3weYISvSH5QwlGx%2FQhPfYBevFE9H6XHdPApfDLYKtFfmwrPK6DGKZp9JrvU76TGYg1A02Z1feiVIbsQfH5Y%2Fz47V%2Bo6Jjx8nfJ1RbZ%2Bsz0Utu4oq4yexH5hl5Nps3Tn43SbragBo5ISRwSsG6HmsKTeK68oJWRN7vfkvr3QHjSjoWzFWvBwVZmj6DOrjh7HQ%2FxYfDJwFpUi9cKtZG8H0hETYqBV0GJAb0MmE%2F61wk7a6zpvHfLA5A9A2HCs%2FY4xUI5TbjOIRxdUGSHjXb7pFfDBfl8YrNye8iwT5L8Y2dQcyzqzZhjq6V7uq1fUETOrGp3nLXyNs2FTn6DS7lImjUm%2FhroJNult6sjYJKejTEh%2BvKLnTIrpP5aG3IXDxRtUhVNZqyZdglQmY2%2BtKbSxuDk7VBf9uD%2B8DzTe2%2Bt0b8Qw8LPa%2BwU67AE9hWzLAEFN8p9Y6swX6faAtisriIR3lJ1wRQ9Xfsf8hVXJ4VRhdNSKIzMakJ186DbyHl8Jfl1MDTZ1rBke%2Fw9Gr2ggzZn9392HxfE9Gl%2FDqqpQ3oby82x2MRV2%2F7Em0SmySvTDUtWsfbYFsOGf9vwjNWoXemOjhQ9vHgTXx1YmrBi7czo4ikk6yFkdfg5GdiT%2BSZV26%2BWvWCXXa3tAHdOHSPIeu74mDLDAYInndrb1U8GRUEY7xQSVtWoZ68Tz%2Ffi61KjoJ%2BEw4WMeQU5k%2BjtVQifvtDrU3o2ek3weGa8GO%2BZJdM1fbNyT8qvdIg%3D%3D)

神经网络是受生物神经网络启发而构建的算法模型,模式匹配的一种,常被用于回归和分类问题,拥有庞大的子域,由数百种算法和各类问题的变体组成。

- Classical Examples
  - 感知机(Perceptron)
  - 反向传播(Back-Propagation)
  - Hopfield 网络
  - 径向基函数网络(Radial Basis Function Network,RBFN)

- Pros:{Best-in-class performance, Adaptive to new problems}
- Cons:{Large amount of data required, Computationally expensive to train, hard to interpret "black box", hard to select metaparameter and network topology}

---

### 深度学习 (Deep Learning)

![](https://coggle-attachments-production-eu-west-1.s3.eu-west-1.amazonaws.com/diagram/587781a8322b25193e9030d8/3bfeb699a2882d7d4dd93e35d05a2589?AWSAccessKeyId=ASIA4YTCGXFHGSJUIQ5W&Expires=1601628643&Signature=yTGNoixZm2EGP5IisK%2FWOnA5r%2F4%3D&response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27image.png&x-amz-security-token=IQoJb3JpZ2luX2VjEFsaCWV1LXdlc3QtMSJGMEQCICVHVqwyllKdNqlg76ballDX2LRFC5tHjfw2pXlLkySRAiAvjtBpRlWxCnEhvpWa%2BwZRbiSDykc2tEQ2jQoDTgsY2iq3AwiE%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAAaDDg3NzQ1MzAzMTc1OCIMq34uO7kolJcu24CtKosDiiEhxGS2VSjLKNEqiDIX3YGITBQ8b8m%2FcI%2Bzqj5K5%2FqMuU3TgRZMtTyz3YOrK3ubmIF7xNhLSAuOYbRl%2B5Ym4juTQOgodUkNnPdPr3weYISvSH5QwlGx%2FQhPfYBevFE9H6XHdPApfDLYKtFfmwrPK6DGKZp9JrvU76TGYg1A02Z1feiVIbsQfH5Y%2Fz47V%2Bo6Jjx8nfJ1RbZ%2Bsz0Utu4oq4yexH5hl5Nps3Tn43SbragBo5ISRwSsG6HmsKTeK68oJWRN7vfkvr3QHjSjoWzFWvBwVZmj6DOrjh7HQ%2FxYfDJwFpUi9cKtZG8H0hETYqBV0GJAb0MmE%2F61wk7a6zpvHfLA5A9A2HCs%2FY4xUI5TbjOIRxdUGSHjXb7pFfDBfl8YrNye8iwT5L8Y2dQcyzqzZhjq6V7uq1fUETOrGp3nLXyNs2FTn6DS7lImjUm%2FhroJNult6sjYJKejTEh%2BvKLnTIrpP5aG3IXDxRtUhVNZqyZdglQmY2%2BtKbSxuDk7VBf9uD%2B8DzTe2%2Bt0b8Qw8LPa%2BwU67AE9hWzLAEFN8p9Y6swX6faAtisriIR3lJ1wRQ9Xfsf8hVXJ4VRhdNSKIzMakJ186DbyHl8Jfl1MDTZ1rBke%2Fw9Gr2ggzZn9392HxfE9Gl%2FDqqpQ3oby82x2MRV2%2F7Em0SmySvTDUtWsfbYFsOGf9vwjNWoXemOjhQ9vHgTXx1YmrBi7czo4ikk6yFkdfg5GdiT%2BSZV26%2BWvWCXXa3tAHdOHSPIeu74mDLDAYInndrb1U8GRUEY7xQSVtWoZ68Tz%2Ffi61KjoJ%2BEw4WMeQU5k%2BjtVQifvtDrU3o2ek3weGa8GO%2BZJdM1fbNyT8qvdIg%3D%3D)

深度学习是人工神经网络的最新分支,它受益于当代硬件的快速发展。众多研究者目前的方向主要集中于构建更大、更复杂的神经网络,目前有许多方法正在聚焦半监督学习问题,其中用于训练的大数据集只包含很少的标记。

- Classical Examples
  - 深度玻耳兹曼机(Deep Boltzmann Machine, DBM)
  - 深度信念网络(Deep Belief Networks, DBN)
  - 卷积神经网络(Convolutional Neural Network, ConvNN)

