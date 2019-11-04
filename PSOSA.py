import matplotlib.pyplot as plt
import numpy as np
import benchmark as bmk  # 自制的benchmark函数


class PSO(object):
    def __init__(self, func, dim, bound):
        # self.w = 1
        self.Nm = 1  # 主种群数量
        self.Ns = 3  # 子种群数量
        self.fai = 0  # 迁移因子
        self.c1 = 2
        self.c2 = 2
        self.c3 = 2.05 # 学习因子
        self.population_size = 50  # 种群数量
        self.max_steps = 50  # 迭代次数
        self.w_min = 0.1
        self.w_max = 1  # 权重的最大和最小值
        self.v_max = 0.1 * bound[1]  # 粒子速度的最大值,实验证明为每一个变量范围的10%
        self.dim = dim  # 搜索空间维度,即benchmark函数的变量个数
        self.bound = bound  # 解空间的范围，即benchmark函数的定义域
        if hasattr(bmk, func.strip()):  # 传入测试函数名称
            self.func = getattr(bmk, func.strip())   # 此时，self.func即为测试函数
        else:
            exit('>>>测试函数不在benchmark函数库中！！！<<<')
        # 初始化主种群
        self.x = np.random.uniform(self.bound[0], self.bound[1], size=(self.population_size, self.dim))  # 初始化粒子群的位置
        self.v = np.zeros((self.population_size, self.dim))  # 初始速度为0，默认为0
        fitness = np.zeros((self.population_size, 1))  # 适应值初始化
        k = 0
        for x_position in self.x:
            fitness[k] = self.func(x_position)  # 计算每个粒子位置的适应值
            k = k + 1
        self.p = self.x  # 个体的最佳位置
        self.pg = [self.x[np.argmin(fitness)]]  # 全局的最佳位置
        self.individual_best_fitness = fitness  # 个体最优适应度
        self.global_best_fitness = [np.min(fitness)]  # 全局最优适应度

        # 初始化三个子种群
        self.x1 = np.random.uniform(self.bound[0], self.bound[1], size=(self.population_size, self.dim))  # 初始化粒子群的位置
        self.v1 = np.zeros((self.population_size, self.dim))  # 初始速度为0，默认为0
        fitness1 = np.zeros((self.population_size, 1))  # 适应值初始化
        k1 = 0
        for x_position1 in self.x1:
            fitness1[k1] = self.func(x_position1)  # 计算每个粒子位置的适应值
            k1 = k1 + 1
        self.p1 = self.x1  # 个体的最佳位置
        self.pg1 = [self.x1[np.argmin(fitness1)]]  # 全局的最佳位置
        self.individual_best_fitness1 = fitness1  # 个体最优适应度
        self.global_best_fitness1 = [np.min(fitness1)]  # 全局最优适应度

        self.x2 = np.random.uniform(self.bound[0], self.bound[1], size=(self.population_size, self.dim))  # 初始化粒子群的位置
        self.v2 = np.zeros((self.population_size, self.dim))  # 初始速度为0，默认为0
        fitness2 = np.zeros((self.population_size, 1))  # 适应值初始化
        k2 = 0
        for x_position2 in self.x2:
            fitness2[k2] = self.func(x_position2)  # 计算每个粒子位置的适应值
            k2 = k2 + 1
        self.p2 = self.x2  # 个体的最佳位置
        self.pg2 = [self.x2[np.argmin(fitness2)]]  # 全局的最佳位置
        self.individual_best_fitness2 = fitness2  # 个体最优适应度
        self.global_best_fitness2 = [np.min(fitness2)]  # 全局最优适应度

        self.x3 = np.random.uniform(self.bound[0], self.bound[1], size=(self.population_size, self.dim))  # 初始化粒子群的位置
        self.v3 = np.zeros((self.population_size, self.dim))  # 初始速度为0，默认为0
        fitness3 = np.zeros((self.population_size, 1))  # 适应值初始化
        k3 = 0
        for x_position3 in self.x3:
            fitness3[k3] = self.func(x_position3)  # 计算每个粒子位置的适应值
            k3 = k3 + 1
        self.p3 = self.x3  # 个体的最佳位置
        self.pg3 = [self.x3[np.argmin(fitness3)]]  # 全局的最佳位置
        self.individual_best_fitness3 = fitness3  # 个体最优适应度
        self.global_best_fitness3 = [np.min(fitness3)]  # 全局最优适应度

        self.Qpg = np.zeros((3, self.dim))
        self.Qgbf = np.zeros(3)

        # 对比种群初始化
        self.x4 = np.random.uniform(self.bound[0], self.bound[1], size=(self.population_size, self.dim))  # 初始化粒子群的位置
        self.v4 = np.zeros((self.population_size, self.dim))  # 初始速度为0，默认为0
        fitness4 = np.zeros((self.population_size, 1))  # 适应值初始化
        k4 = 0
        for x_position4 in self.x4:
            fitness4[k4] = self.func(x_position4)  # 计算每个粒子位置的适应值
            k4 = k4 + 1
        self.p4 = self.x4  # 个体的最佳位置
        self.pg4 = [self.x4[np.argmin(fitness4)]]  # 全局的最佳位置
        self.individual_best_fitness4 = fitness4  # 个体最优适应度
        self.global_best_fitness4 = [np.min(fitness4)]  # 全局最优适应度


    def evolve(self):
        for step in range(self.max_steps):
            #  对比单种群PSO算法
            R14 = np.random.random()
            R24 = np.random.random()
            self.v4 = self.v4 + self.c1 * R14 * (self.p4 - self.x4) \
                     + self.c2 * R24 * (self.pg4 - self.x4)  # 速度更新
            great_id4 = np.greater_equal(self.v4, self.v_max)  # 将速度限制在范围内
            self.v4[great_id4] = self.v_max
            less_id4 = np.less_equal(self.v4, -self.v_max)
            self.v4[less_id4] = -self.v_max
            self.x4 = self.v4 + self.x4  # 位置更新
            fitness4 = np.zeros((self.population_size, 1))
            k4 = 0
            for x_position4 in self.x4:
                fitness4[k4] = self.func(x_position4)
                k4 = k4 + 1
                # 个体最优值更新
            update_id4 = np.greater(self.individual_best_fitness4, fitness4)
            k4 = 0
            for change4 in update_id4:
                if change4 == True:
                    self.p4[k4, :] = self.x4[k4, :]
                k4 = k4 + 1
            self.individual_best_fitness4[update_id4] = fitness4[update_id4]  # 个体最优适应度更新
            # 全局最优值更新
            if np.min(fitness4) < self.global_best_fitness4[-1]:
                self.pg4 = self.x4[np.argmin(fitness4)]  # 全局最优位置更新
                self.global_best_fitness4.append(np.min(fitness4))  # 全局最优适应度更新

            # 多种群协同PSO算法    
            # 子种群更新
            R11 = np.random.random()
            R21 = np.random.random()
            self.v1 = self.v1 + self.c1 * R11 * (self.p1 - self.x1) \
                     + self.c2 * R21 * (self.pg1 - self.x1)  # 速度更新
            great_id1 = np.greater_equal(self.v1, self.v_max)  # 将速度限制在范围内
            self.v1[great_id1] = self.v_max
            less_id1 = np.less_equal(self.v1, -self.v_max)
            self.v1[less_id1] = -self.v_max
            self.x1 = self.v1 + self.x1  # 位置更新
            fitness1 = np.zeros((self.population_size, 1))
            k1 = 0
            for x_position1 in self.x1:
                fitness1[k1] = self.func(x_position1)
                k1 = k1 + 1
                # 个体最优值更新
            update_id1 = np.greater(self.individual_best_fitness1, fitness1)
            k1 = 0
            for change1 in update_id1:
                if change1 == True:
                    self.p1[k1, :] = self.x1[k1, :]
                k1 = k1 + 1
            self.individual_best_fitness1[update_id1] = fitness1[update_id1]  # 个体最优适应度更新
            # 全局最优值更新
            if np.min(fitness1) < self.global_best_fitness1[-1]:
                self.pg1 = self.x1[np.argmin(fitness1)]  # 全局最优位置更新
                self.global_best_fitness1.append(np.min(fitness1))  # 全局最优适应度更新

            R12 = np.random.random()
            R22 = np.random.random()
            self.v2 = self.v2 + self.c1 * R12 * (self.p2 - self.x2) \
                      + self.c2 * R22 * (self.pg2 - self.x2)  # 速度更新
            great_id2 = np.greater_equal(self.v2, self.v_max)  # 将速度限制在范围内
            self.v2[great_id2] = self.v_max
            less_id2 = np.less_equal(self.v2, -self.v_max)
            self.v2[less_id2] = -self.v_max
            self.x2 = self.v2 + self.x2  # 位置更新
            fitness2 = np.zeros((self.population_size, 1))
            k2 = 0
            for x_position2 in self.x2:
                fitness2[k2] = self.func(x_position2)
                k2 = k2 + 1
                # 个体最优值更新
            update_id2 = np.greater(self.individual_best_fitness2, fitness2)
            k2 = 0
            for change2 in update_id2:
                if change2 == True:
                    self.p2[k2, :] = self.x2[k2, :]
                k2 = k2 + 1
            self.individual_best_fitness2[update_id2] = fitness2[update_id2]  # 个体最优适应度更新
            # 全局最优值更新
            if np.min(fitness2) < self.global_best_fitness2[-1]:
                self.pg2 = self.x2[np.argmin(fitness2)]  # 全局最优位置更新
                self.global_best_fitness2.append(np.min(fitness2))  # 全局最优适应度更新

            R13 = np.random.random()
            R23 = np.random.random()
            self.v3 = self.v3 + self.c1 * R13 * (self.p3 - self.x3) \
                      + self.c2 * R23 * (self.pg3 - self.x3)  # 速度更新
            great_id3 = np.greater_equal(self.v3, self.v_max)  # 将速度限制在范围内
            self.v3[great_id3] = self.v_max
            less_id3 = np.less_equal(self.v3, -self.v_max)
            self.v3[less_id3] = -self.v_max
            self.x3 = self.v3 + self.x3  # 位置更新
            fitness3 = np.zeros((self.population_size, 1))
            k3 = 0
            for x_position3 in self.x3:
                fitness3[k3] = self.func(x_position3)
                k3 = k3 + 1
                # 个体最优值更新
            update_id3 = np.greater(self.individual_best_fitness3, fitness3)
            k3 = 0
            for change3 in update_id3:
                if change3 == True:
                    self.p3[k3, :] = self.x3[k3, :]
                k3 = k3 + 1
            self.individual_best_fitness3[update_id3] = fitness3[update_id3]  # 个体最优适应度更新
            # 全局最优值更新
            if np.min(fitness3) < self.global_best_fitness3[-1]:
                self.pg3 = self.x3[np.argmin(fitness3)]  # 全局最优位置更新
                self.global_best_fitness3.append(np.min(fitness3))  # 全局最优适应度更新

            self.Qpg[0, :] = self.pg1[0]
            self.Qpg[1, :] = self.pg2[0]
            self.Qpg[2, :] = self.pg3[0]
            self.Qgbf[0] = self.global_best_fitness1[-1]
            self.Qgbf[1] = self.global_best_fitness2[-1]
            self.Qgbf[2] = self.global_best_fitness3[-1]
            g_label = np.argmin(self.Qgbf)  # 子种群中适应度最佳的种群

            if self.global_best_fitness > [self.Qgbf[g_label]]:
                self.fai = 0
            elif self.global_best_fitness == [self.Qgbf[g_label]]:
                self.fai = 0.5
            else:
                self.fai = 1

            R1 = np.random.random()
            R2 = np.random.random()
            R3 = np.random.random()
            # self.w = self.w_max - step * (self.w_max - self.w_min) / self.max_steps  # 时变权重

            self.v = self.v + self.c1 * R1 * (self.p - self.x) \
                     + self.fai*self.c2 * R2 * (self.pg - self.x) \
                    + (1-self.fai)*self.c3*R3*(self.Qpg[g_label, :] - self.x)  # 速度更新

            great_id = np.greater_equal(self.v, self.v_max)  # 将速度限制在范围内
            self.v[great_id] = self.v_max
            less_id = np.less_equal(self.v, -self.v_max)
            self.v[less_id] = -self.v_max

            self.x = self.v + self.x  # 位置更新

            fitness = np.zeros((self.population_size, 1))
            k = 0
            for x_position in self.x:
                fitness[k] = self.func(x_position)
                k = k + 1
                # 个体最优值更新
            update_id = np.greater(self.individual_best_fitness, fitness)
            k = 0
            for change in update_id:
                if change == True:
                    self.p[k, :] = self.x[k, :]
                k = k + 1
            self.individual_best_fitness[update_id] = fitness[update_id]  # 个体最优适应度更新
            # 全局最优值更新
            if np.min(fitness) < self.global_best_fitness[-1]:
                self.pg = self.x[np.argmin(fitness)]  # 全局最优位置更新
                self.global_best_fitness.append(np.min(fitness))  # 全局最优适应度更新
        '''
        plt.figure('MCPSO Algorithms')
        plt.plot(list(range(len(self.global_best_fitness))), self.global_best_fitness, 'r-')
        plt.title('MCPSO Algorithms')
        plt.xlabel('iteration times')
        plt.ylabel('minimum value')
        plt.grid()
        plt.show()

        plt.figure('PSO Algorithms')
        plt.plot(list(range(len(self.global_best_fitness4))), self.global_best_fitness4, 'k-')
        plt.title('PSO Algorithms')
        plt.xlabel('iteration times')
        plt.ylabel('minimum value')
        plt.grid()
        plt.show()
        '''
        # print('PSO fitness: {0} '.format(self.global_best_fitness4[-1]))
        # print('MCPSO fitness: {0} '.format(self.global_best_fitness[-1]))
        backup = [self.global_best_fitness4[-1], self.global_best_fitness[-1]]
        return backup


Circle_Times = 30  # 测试循环次数为30次
Result = np.zeros((30, 2))  # 结果存档
TestFunName = ['Brown', 'Rosenbrock', 'Chichinadze', 'DropWave', 'EggHolder', 'Schubert']  # 所有测试函数集
Dim = [32, 8, 2, 2, 2, 2]
Bound = np.array([[-1, 4], [-5, 10], [-30, 30], [-5.12, 5.12], [-512, 512], [-10, 10]])
for m in range(6):
    for n in range(Circle_Times):
        pso = PSO(TestFunName[m], Dim[m], Bound[m, :])
        Result[n, :] = pso.evolve()
        del pso
    plt.figure('Test Result')
    plt.plot(range(30), Result[:, 0], 'k-', label='PSO')
    plt.plot(range(30), Result[:, 1], 'r-', label='MCPSO')
    plt.title(TestFunName[m] + ' Test Result')
    plt.legend()
    plt.xlabel('test times')
    plt.ylabel('best fitness')
    plt.grid()
    plt.show()
    print('the PSO for {0} mean fitness is {1}'.format(TestFunName[m], np.mean(Result[:, 0])))
    print('the MCPSO for {0} mean fitness is {1}'.format(TestFunName[m], np.mean(Result[:, 1])))
    Result = np.zeros((30, 2))



