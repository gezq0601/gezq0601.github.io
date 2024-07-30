# n次Bernstein基函数的绘制

## Description

> 输入$n$，在同一个图中绘制$n+1$个$Bernstein$基函数的图像

## Knowledge

$Def$（$Bernstein$基函数）：$n$次$Bernstein$基函数为$B_0^n(t), B_1^n(t),\cdots,B_i^n(t),\cdots,B_n^n(t)$，它们依次为二次项$[t+(1-t)]^n$展开后的各项，其中$B_i^n(t)$是$Bernstein$多项式。


$$
B_i^n(t) =
\begin{pmatrix} n \\ i \end{pmatrix}
(1-t)^{n-i} t^i, \quad
t\in [0,1], \quad
i = 0, 1, \cdots, n.
$$


$Thinking$：

1. 由输入的$Bernstein$基函数的次数$n$，利用$for$循环依次计算并绘制每条曲线$B_i^n(t),(i=0,1,\cdots,n)$。
2. 定义函数：实现$Bernstein$基函数的定义公式，即给定自变量$t$计算对应的函数值。

## Code

```python
# -*- coding: UTF-8 -*-
"""
    @date-time: 7/28/2024 - 11:24 PM
    @author: gezq
"""
import numpy as np
from matplotlib import pyplot as plt
from math import factorial as fac

# To solve the problem of plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def bernstein(n, i, t):
    """
    Calculate the values of Bernstein's function.
    :param n: the degree of the Bernstein's function.
    :param i: the order of the Bernstein's function.
    :param t: the array of self-variable of the Bernstein's function.
    :return: the values of Bernstein's function.
    """
    return (fac(n) / (fac(i) * fac(n - i))) * pow(t, i) * pow(1 - t, n - i)


def draw_bernstein(n):
    """
    Draw the Bernstein's function.
    :param n: the degree of the Bernstein's function.
    :return: No return.
    """
    t = np.linspace(0, 1)
    for i in range(n + 1):
        plt.plot(t, bernstein(n, i, t))
        # Use the property of max value of Bernstein basis function
        plt.text(i / n, bernstein(n, i, i / n) * 1.02, f'$ B_{i}^{n} $')
    plt.title(f'{n}-order Bernstein basis functions')
    plt.show()


if __name__ == '__main__':
    n = int(input("Please input the degree of Bernstein basis functions n="))
    draw_bernstein(n)

```

## Result

![task01-1_degree_bernstein_basis_function](https://cdn.jsdelivr.net/gh/gezq0601/Pictures/typora/task01-1_degree_bernstein_basis_function.png)

![task01-3_degree_bernstein_basis_function](https://cdn.jsdelivr.net/gh/gezq0601/Pictures/typora/task01-3_degree_bernstein_basis_function.png)

![task01-6_degree_bernstein_basis_function](https://cdn.jsdelivr.net/gh/gezq0601/Pictures/typora/task01-6_degree_bernstein_basis_function.png)

# n次Bézier曲线的绘制

## Description

> 输入$n$，鼠标输入$n+1$个点，绘制一条$Bézier$曲线，同时绘制控制多边形。
>
> 点用空心圆点表示，控制多边形与曲线用不同颜色区分。
>
> 鼠标输入控制顶点，每个点附近标记点序列号（即下标）。

## Knowledge

$Def$（$Bézier$曲线）：$n$次$Bézier$曲线的定义如下，其中$P_i$为控制点，$B_i^n(t)$为$Bernstein$基函数。


$$
\mathbf{C}(t) = \sum_{i=0}^n \mathbf{P}_i B_i^n(t), \quad
t \in [0,1].
$$


$Thinking$：

1. 导入`task01`中计算$B_i^n(t)$的函数。
2. 定义函数：计算所有$Bernstein$基函数在$t\in[0,1]$之间的所有值。
3. 思考点：由$Bézier$曲线的定义（$Bernstein$基函数与控制点）计算对应的函数值——类似矩阵相乘的思想。
4. 定义函数：绘制图形($Bézier$曲线、控制多边形及控制顶点)。
5. 思考点：多个$Bernstein$基函数的值的存放形式，及后续如何获取$Bézier$曲线的函数值——`DataFrame`。
6. 思考点：鼠标选点——未使用`figure.canvas.mpl_connect`绑定事件，而采用更简便的`ginput()`函数直接获取点的坐标。
7. 为便于后续代码中`import task02`，因此调整函数中变量以参数形式传递，避免使用全局变量，增强了代码的可重用性。

## Code

```python
# -*- coding: UTF-8 -*-
"""
    @date-time: 7/29/2024 - 8:36 AM
    @author: gezq
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import task01
import utils


def calculate_bernstein(n):
    """
    Calculate Bernstein values
    :param n: the degree of the Bernstein's function.
    :return: the DataFrame of the values of Bernstein's function.
    """
    t = np.linspace(0, 1)
    df_bernstein = pd.DataFrame()
    # The values of each Bernstein function as a column store in df_bernstein.
    for i in range(n + 1):
        df_bernstein[i] = task01.bernstein(n, i, t)
    return df_bernstein


def calculate_bezier(n, df_points):
    """
    Calculate Bezier values
    :param n: the degree of the Bézier curve.
    :param df_points: the control points.
    :return: the DataFrame of the values of Bézier curve.
    """
    df_bernstein = calculate_bernstein(n)
    # Calculate the values of Bézier curve
    df_result = df_bernstein.dot(df_points)
    return df_result


def draw_bezier(n, df_points):
    """
    Draw the Bézier curve
    :param n: the degree of the Bézier curve.
    :param df_points: the control points.
    :return: No return.
    """
    df_result = calculate_bezier(n, df_points)
    # Draw Bézier curve and control polygon.
    df_result.plot(x='x', y='y', label='Bézier curve', color='b', ax=plt.gca())
    df_points.plot(x='x', y='y', label='Control polygon', color='r', marker='o', markerfacecolor='w', ax=plt.gca())
    utils.add_text_subscript(df_points)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    n = int(input("Please input the degree of Bézier curve n="))
    draw_bezier(n, utils.selection_point(n + 1, f'{n}-order Bézier curve'))

```

## Result

![task02-1_degree_bezier_curve](https://cdn.jsdelivr.net/gh/gezq0601/Pictures/typora/task02-1_degree_bezier_curve.png)

![task02-3_degree_bezier_curve](https://cdn.jsdelivr.net/gh/gezq0601/Pictures/typora/task02-3_degree_bezier_curve.png)

![task02-6_degree_bezier_curve](https://cdn.jsdelivr.net/gh/gezq0601/Pictures/typora/task02-6_degree_bezier_curve.png)

# n次Bézier曲线的几何作图法

## Description

> 输入$n$，输入变量$t$，鼠标输入$n+1$个点，绘制一条$Bézier$曲线的控制多边形，同时绘制几何作图法过程中每个点、每条线段，最后绘制曲线上参数为$t￥的点，用不同颜色与其他点区分。
>
> 鼠标输入控制顶点，每个点附近标记点序列号（即下标）。

## Knowledge

$Def$（$de\ Casteljau$算法递推公式）：其中$k$为递归轮次。


$$
\mathbf P_i^k(t) = 
\begin{cases}
	\mathbf P_i, &k=0; \quad i=0,1,\cdots,n \\
	(1 - t) \mathbf P_{i}^{k-1}(t) 
	+ t\mathbf P_{i+1}^{k-1}(t),
    \quad &k=1,2,\cdots,n;\quad i=0,1,\cdots,n - k.
\end{cases}
$$


$Thinking$

1. 绘制$Bézier$曲线、控制多边形及控制顶点与`task02`类似。
2. 思考点：递推算法的实现——由本次控制点计算出下一轮的点，将本次控制点覆盖后再绘制，重复$n$次此操作。

## Code

```python
# -*- coding: UTF-8 -*-
"""
    @date-time: 7/29/2024 - 11:32 AM
    @author: gezq
"""
from fractions import Fraction

from matplotlib import pyplot as plt
import task02
import utils


def recursive_formula(df_points, i, t):
    """
    deCasteljau recursive formula
    :param df_points: the control points of current time.
    :param i: the time index.
    :param t: the parameter value.
    :return: the result of recursive formula.
    """
    return (1 - t) * df_points.iloc[i] + t * df_points.iloc[i + 1]


def recursion(n, df_points, i, t):
    """
    deCasteljau recursive algorithm
    :param n: the degree of the Bézier curve.
    :param df_points: the control points of current time.
    :param i: the time index.
    :param t: the parameter value.
    :return: the control points of next time.
    """
    if i == 0:
        return df_points
    for j in range(n - i + 1):
        df_points.iloc[j] = recursive_formula(df_points, j, t)
    return df_points.iloc[:n - i + 1, :]


def draw_bezier_geometry(n, df_points, t):
    """
    Draw the Bézier curve of geometric processing.
    :param n: the degree of the Bézier curve.
    :param df_points: the control points.
    :param t: the parameter value.
    :return: No return.
    """
    df_result = task02.calculate_bezier(n, df_points)
    # Unify the color of curve, 'k' is black.
    df_result.plot(x='x', y='y', label='Bézier curve', color='b', ax=plt.gca())
    # Cycle n times
    for i in range(n + 1):
        df_points = recursion(n, df_points, i, t)
        df_points.plot(x='x', y='y', label=f'The {i} round', marker='o', markerfacecolor='w', ax=plt.gca())
        utils.add_text_subscript_with_specify_superscript(df_points, i)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    n = int(input("Please input the degree of Bézier curve n="))
    # Import Fraction for handling the parameter t.
    t = Fraction(input("Please input the parameter t="))
    draw_bezier_geometry(n, utils.selection_point(n + 1, f'Geometric progressing of {n}-order Bézier curve (t={t})'),
                         float(t))

```

## Result

![task03-2_degree_bezier_curve_geometry](https://cdn.jsdelivr.net/gh/gezq0601/Pictures/typora/task03-2_degree_bezier_curve_geometry.png)

![task03-4_degree_bezier_curve_geometry](https://cdn.jsdelivr.net/gh/gezq0601/Pictures/typora/task03-4_degree_bezier_curve_geometry.png)

![task03-8_degree_bezier_curve_geometry](https://cdn.jsdelivr.net/gh/gezq0601/Pictures/typora/task03-8_degree_bezier_curve_geometry.png)

# Bézier曲线的插值

## Description

> 给定$n+1$个点，要求算出$n$次$Bézier$曲线的控制顶点，使得$n+1$个点在$n$次$Bézier$曲线上。
>
> 鼠标输入控制顶点，每个点附近标记点序列号（即下标）。

## Knowledge

$Def$（$Bézier$曲线的矩阵形式）：$n$次$Bézier$曲线的矩阵形式如下，其中$P_i$为控制点，$B_i^n(t)$为$Bernstein$基函数。


$$
\begin{align}
	\mathbf{C}(t) &= \sum_{i=0}^n \mathbf{P}_i B_i^n(t), \quad t \in [0,1]. \\
	&= \begin{pmatrix} B_0^d(t) & B_1^d(t) & \cdots & B_d^d(t) \end{pmatrix}
	\cdot
	\begin{pmatrix} \mathbf{P}_0 \\ \mathbf{P}_1 \\ \vdots \\ \mathbf{P}_d \end{pmatrix}
\end{align}
$$


若给定任意$n+1$个点$\boldsymbol{v}_i(i=0,1,\cdots,d)$，即$\boldsymbol{v}_i = \mathbf{C}(t_i)$，则有如下形式：


$$
\begin{align}
	\begin{pmatrix} \mathbf{C}(t_0) \\ \mathbf{C}(t_1) \\ \vdots \\ \mathbf{C}(t_n) \end{pmatrix}
	= 
	\begin{pmatrix} \boldsymbol{v}_0 \\ \boldsymbol{v}_1 \\ \vdots \\ \boldsymbol{v}_n \end{pmatrix}
	&=
    \begin{pmatrix} 
		B_0^n(t_0) & B_1^n(t_0) & \cdots & B_n^n(t_0) \\
		B_0^n(t_1) & B_1^n(t_1) & \cdots & B_n^n(t_1) \\
		\vdots & \vdots & \ddots & \vdots \\
		B_0^n(t_n) & B_1^n(t_n) & \cdots & B_n^n(t_n) \\
    \end{pmatrix}
    \cdot 
    \begin{pmatrix} \mathbf{P}_0 \\ \mathbf{P}_1 \\ \vdots \\ \mathbf{P}_n \end{pmatrix} 
    \\\\
    \begin{pmatrix} \mathbf{P}_0 \\ \mathbf{P}_1 \\ \vdots \\ \mathbf{P}_n \end{pmatrix}
    &= 
    {\begin{pmatrix} 
		B_0^n(t_0) & B_1^n(t_0) & \cdots & B_n^n(t_0) \\
		B_0^n(t_1) & B_1^n(t_1) & \cdots & B_n^n(t_1) \\
		\vdots & \vdots & \ddots & \vdots \\
		B_0^n(t_n) & B_1^n(t_n) & \cdots & B_n^n(t_n) \\
    \end{pmatrix}}^{-1} 
    \cdot
    \begin{pmatrix} \boldsymbol{v}_0 \\ \boldsymbol{v}_1 \\ \vdots \\ \boldsymbol{v}_n \end{pmatrix}
\end{align}
$$


为简便起见，一般取$t_i = \frac{i}{n}(i=0,1,\cdots,n)$，则可得如下形式：


$$
\begin{pmatrix} \mathbf{P}_0 \\ \mathbf{P}_1 \\ \vdots \\ \mathbf{P}_n \end{pmatrix}
= 
{\begin{pmatrix} 
    1 & 0 & \cdots & 0 \\
    B_0^n(\frac{1}{d}) & B_1^n(\frac{1}{d}) & \cdots & B_n^n(\frac{1}{d}) \\
    \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & \cdots & 1 \\
\end{pmatrix}}^{-1} 
\cdot
\begin{pmatrix} \boldsymbol{v}_0 \\ \boldsymbol{v}_1 \\ \vdots \\ \boldsymbol{v}_n \end{pmatrix}
$$


将上述矩阵形式写为线性方程组形式：


$$
\begin{cases}
	\mathbf{P}_0 &= \boldsymbol{v}_0 \\
	\mathbf{P}_n &= \boldsymbol{v}_n \\
	\sum_{i=0}^{n} 
	\begin{pmatrix} n \\ i \end{pmatrix} 
	\left( 1-\frac{i}{n} \right)^{n-i} 
	\left( \frac{i}{n} \right)^{i}
	\mathbf{P}_j &= \boldsymbol{v}_j,\quad
	j=0,1,\cdots,d-1.
\end{cases}
$$


通过求解上述线性方程组便可以求出对应$n$次$Bézier$曲线的$n+1$个控制顶点。



$Thinking$

1. 绘制$Bézier$曲线、控制多边形及控制顶点与`task02`, `task03`类似。
2. 思考点：求解控制顶点——由$Bézier$曲线的矩阵形式可由给定的$n+1$个点求解出$n+1$个控制顶点。

## Code

```python
# -*- coding: UTF-8 -*-
"""
    @date-time: 7/29/2024 - 5:53 PM
    @author: gezq
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import task02
import utils


def calculate_control_points(n, df_points):
    """
    Calculate control points from given points.
    :param n: the degree of the Bézier curve.
    :param df_points: the given points.
    :return: the control points.
    """
    df_bernstein = task02.calculate_bernstein(n, np.linspace(0, 1, n + 1))
    return pd.DataFrame(np.asmatrix(df_bernstein).I).dot(df_points)


def draw_bezier_interpolation(n, df_points):
    """
    Draw the Bézier curve of interpolation.
    :param n: the degree of the Bézier curve.
    :param df_points: the control points.
    :return: No return.
    """
    df_control_points = calculate_control_points(n, df_points)
    df_result = task02.calculate_bezier(n, df_control_points)
    df_points.plot(x='x', y='y', label='Picked point', color='w', marker='o', markerfacecolor='k', ax=plt.gca())
    df_result.plot(x='x', y='y', label='Bézier curve', color='b', ax=plt.gca())
    df_control_points.plot(x='x', y='y', label='Control polygon', color='r', marker='o', markerfacecolor='w',
                           ax=plt.gca())
    utils.add_text_subscript(df_control_points)
    plt.xlabel("t")
    plt.ylabel("u")
    plt.xlim(xmin=0, xmax=1)
    plt.ylim(ymin=0, ymax=5)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    n = int(input("Please input the degree of Bernstein basis functions n="))
    draw_bezier_interpolation(n, utils.selection_point(n + 1, f'Interpolation of {n}-order Bézier curve'))

```

## Result

![task04-2_degree_bezier_curve_interpolation](https://cdn.jsdelivr.net/gh/gezq0601/Pictures/typora/task04-2_degree_bezier_curve_interpolation.png)

![task04-3_degree_bezier_curve_interpolation](https://cdn.jsdelivr.net/gh/gezq0601/Pictures/typora/task04-3_degree_bezier_curve_interpolation.png)

![task04-4_degree_bezier_curve_interpolation](https://cdn.jsdelivr.net/gh/gezq0601/Pictures/typora/task04-4_degree_bezier_curve_interpolation.png)

# n次B样条基函数的绘制

## Description

> 输入n，在同一个图中绘制n+1个B样条基函数的图像。均匀、准均匀、非均匀的情况都需要绘制。
>
> 绘制B样条基函数时，横坐标附上绘制的节点序列下标与绘制的基函数下标。

## Knowledge

## Code

## Result

# n次B样条曲线的绘制

## Description

## Knowledge

## Code

## Result

# 三次均匀B样条曲线插值

## Description

## Knowledge

## Code

## Result

# 四点插值细分法

## Description

## Knowledge

## Code

## Result

# 平面多边形Wachspress坐标的计算及其等高线

## Description

## Knowledge

## Code

## Result

# 平面多边形中值坐标的计算及其等高线

## Description

## Knowledge

## Code

## Result

# Catmull-Clark细分曲面

## Description

## Knowledge

## Code

## Result

# Loop细分曲面

## Description

## Knowledge

## Code

## Result

# Butterfly细分曲面

## Description

## Knowledge

## Code

## Result

# 3D中值坐标的等值面

## Description

## Knowledge

## Code

## Result