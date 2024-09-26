将GA算法和PSO算法结合，用GA算法优化PSO中的参数选择（位置 和 速度）；
并且利用遗传算法的变异操作，提高了结果的全局性，解决了PSO算法局部收敛速度过快的问题；
主要代码内容为：
1）定义INDIVIDUAL类，包含calculate_fitness函数，即要求的结果的表达式。您要使用这个模型去求解别的值，需修改这个函数。
2）intialize_generation 函数，随机生成初始种群。

以下是PSO算法的核心：
3）update_best 函数，在每一次迭代中调用，更新保存全局最优以及历史最优的结果。此处求的是最小值，若要求全局最大值，需修改这个函数。
4）update_vel 函数，在一次迭代后，更新每一个个体的速度，根据公式
![image](https://github.com/user-attachments/assets/92216be8-7f85-423a-87f1-28e887763842)
5）update_pos 函数，在一次迭代后，更新每一个个体的位置

接下来将粒子群视作种群，开始交叉、变异等的遗传操作
6）crossover 函数，通过将父代的位置、速度进行一部分互相交换（类似于染色体交换、只交换位置数组/速度数组的一部分），生成两个新的子代。
7）mutation 函数，设定变异率，此处设定的mutation rate是0.01（可修改），随机对种群内个体进行变异操作。


GA_APSO函数，是最后的调用函数。需要传入的参数有：
size：种群大小
num_generation：迭代次数
dim：数据维度
max_x：坐标的最大边界值
max_vel：速度的最大值
W：在update_vel函数中，是当前速度对速度更新所占的权重
C1,C2：C1是全局最优结果对速度更新的权重；C2是该个体（点）的历史最优结果对速度更新的权重
