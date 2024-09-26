import random
import numpy as np

class INDIVIDUAL():
    def __init__(self,gene_pos,vel,max_x,max_vel,dim,best_fit):
        self.gene_pos = gene_pos
        self.max_x = max_x
        self.dim = dim
        self.max_vel =max_vel
        self.vel = vel
        self.bestpos = np.zeros((1, dim))
        self.best_fit = best_fit
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        sum = 0
        for i in range(self.dim):
            # print(i)
            # print(self.gene_pos)
            t = self.gene_pos[i]
            sum = t**2+sum
        return sum

def intialize_generation(size,max_x,max_vel,dim):
    generation = []
    for _ in range(size):
        gene = []
        vel = []
        for i in range(dim):
            # print(i)
            gene.append(random.randint(-max_x,max_x))
            vel.append(random.randint(-max_vel,max_vel))
        generation.append(INDIVIDUAL(gene,vel,max_x,max_vel,dim,10000))

    return generation


def update_best(part,best_one):
    ans = best_one.fitness
    print(part.gene_pos)
    t = part.fitness
    if t < part.best_fit :
        part.bestpos = part.gene_pos
        part.best_fit = t
    elif t < ans :
        best_one.fitness = t
        best_one.gene_pos = part.gene_pos
        ans = t
    print("更新全局最优以及历史最优成功")
    return best_one.gene_pos


def update_vel(part,W,C1,C2,best):
    np_vel = np.array(part.vel)
    np_globalbestpos = np.array(best.gene_pos)
    np_gene_pos = np.array(part.gene_pos)
    np_pointbestpos = np.array(part.bestpos)

    print(np_vel)
    vel_value = (W * np_vel + C1 * np.random.rand() * (np_globalbestpos - np_gene_pos)
                + C2 * np.random.rand() * (np_pointbestpos - np_gene_pos))
    # 速度更新公式
    vel_value[vel_value > part.max_vel] = part.max_vel
    vel_value[vel_value < -part.max_vel] = -part.max_vel
    part.vel = vel_value
    print("更新速度成功")

def update_pos(part):
    temp = part.gene_pos+part.vel
    temp[temp > part.max_x] = part.max_x
    temp[temp< - part.max_x] = - part.max_x
    part.gene_pos = temp
    part.fitness = part.calculate_fitness()
    print("更新坐标及fitness成功")

def crossover(parent1, parent2,max_x,max_vel,dim):
    # 单点交叉
    # parent1, parent2: 选择的两个父本个体
    # 随机选择交叉点，交换父本基因，生成两个子代
    point = random.randint(1, len(parent1.gene_pos) - 1)
    # child1_genes = (parent1.gene_pos[:point] )
    child1_genes = np.hstack((parent1.gene_pos[:point],parent2.gene_pos[point:]))
    child2_genes = np.hstack((parent2.gene_pos[:point] , parent1.gene_pos[point:]))
    vel1 = np.hstack((parent1.vel[:point],parent2.vel[point:]))
    vel2 = np.hstack((parent2.vel[:point],parent1.vel[point:]))
    return INDIVIDUAL(child1_genes,vel1,max_x,max_vel,dim,10000), INDIVIDUAL(child2_genes,vel2,max_x,max_vel,dim,10000)

def mutation(individual, mutation_rate=0.01):
    # 对个体的基因序列进行随机变异
    # individual: 要变异的个体
    # mutation_rate: 变异概率
    for i in range(len(individual.gene_pos)):
        if random.random() < mutation_rate:
            # 对每个基因位以一定的概率进行增减操作
            individual.gene_pos[i] += random.randint(-1, 1)
    # 更新个体的适应度
    individual.fitness = individual.calculate_fitness()
    return individual



def GA_APSO(size,num_generation,dim,max_x,max_vel,W,C1,C2):

    generation = intialize_generation(size,max_x,max_vel,dim)
    vel = generation[0].vel
    best_generation = []
    best_generation.append(generation[0])
    bestbest_individual = generation[0]
    crossover_rate = 0.8
    mutation_rate = 0.02
    cnt = 0
    for _ in range(num_generation):
        new_generation = []
        cnt = 0
        bestbest_individual = best_generation[0]
        print("目前全局最优",bestbest_individual.gene_pos)
        for i in range(size):
            cnt += 1
            print(cnt)
            print(bestbest_individual.fitness)
            update_best(generation[i],bestbest_individual)
        for i in range(size):
            update_vel(generation[i],W,C1,C2,bestbest_individual)
            update_pos(generation[i])

        # 选择
        generation.sort(key=lambda x: x.fitness)
        elite_count = int(size*0.2)
        # print(elite_count)
        new_generation.extend(generation[:elite_count])

        # 交叉
        for _ in range(int(size*crossover_rate*0.5)):
            parent1 = random.choice(generation[:elite_count])
            parent2 = random.choice(generation[:elite_count])
            child1,child2 = crossover(parent1,parent2,max_x,max_vel,dim)
            new_generation.append(child1)
            new_generation.append(child2)

        # 变异
        for i in range(len(new_generation)):
            if random.random() < mutation_rate:
                new_generation[i] = mutation(new_generation[i])

        generation = new_generation
        generation = sorted(generation, key=lambda x: x.fitness,reverse=False)
        print(f"最优适应度: {generation[0].fitness}")
        best_generation.append(generation[0])
        print("加入完毕")
        best_generation = sorted(best_generation,key=lambda x:x.fitness,reverse=False)
    for i in range(num_generation):
        print(best_generation[i].fitness)
    bestbest_individual = best_generation[0]
    return bestbest_individual


best = GA_APSO(100,500,10,30.0,1.0,1.0,5.0,2.0)
print(f"最小位置："+str(best.gene_pos))
print("最优解：",best.fitness)





