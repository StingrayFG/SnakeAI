import numpy as np
import pygame
from math import sqrt as square
from random import random
from random import randint

pygame.init()
pygame.font.init()

win = pygame.display.set_mode((960, 656))

pygame.display.set_caption('name')

end = False

score = 0
high_score = 0

width = 14
height = 14
speed = 16

moves = 200
life_time = 0

time_delay = 50

pos = [[648, 312], [648, 328], [648, 344]]
fruits_pos = []

ate_fruit = False

direction = 'u'
moving = 'u'

# fruit


def random_fruit():
    global xf
    xf = 328 + 16 * (randint(1, 38))
    global yf
    yf = 8 + 16 * (randint(1, 38))

    while not all(xf != pos[i][0] for i in range(len(pos))) and not all(
            yf != pos[i][1] for i in range(len(pos))):
        xf = 328 + 16 * (randint(1, 38))
        yf = 8 + 16 * (randint(1, 38))


random_fruit()

# text

font = pygame.font.SysFont('Calibri', 24)
learning_font = pygame.font.SysFont('Calibri', 80)
score_font = pygame.font.SysFont('Calibri', 32)

# losing

displaying = False
learning = True
first_gen = True


def losing():
    global pos
    pos = [[648, 312], [648, 328], [648, 344]]
    global end
    end = False

    global learning
    global displaying
    learning = True
    displaying = False

    global life_time
    global moves
    moves = 200
    life_time = 0
    global score
    global high_score
    if score > high_score:
        high_score = score
    score = 0

    random_fruit()
    global direction
    direction = 'u'
    global moving
    moving = 'u'


# NEURAL NET

syn0 = np.empty([24, 18])
syn1 = np.empty([18, 18])
syn2 = np.empty([18, 4])

res_syn0 = []
res_syn1 = []
res_syn2 = []

best_score = 0

mutation_rate = 5.0

l0 = np.empty([1, 24])


def exponential(x, pow):
    for i in range(pow):
        x = x * x
    return x


def find_max(a, b):
    if a > b:
        return a
    else:
        return b


def relu(a):
    return np.maximum(a, 0)


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def crossover(mtrx1, mtrx2, a, b):
    mtrx3 = np.empty([a, b])

    random_a = randint(0, a - 1)
    random_b = randint(0, b - 1)

    for x in range(b):
        for y in range(a):
            if(x < random_b) or (x == random_b and y <= random_a):
                mtrx3[y][x] = mtrx1[y][x]
            else:
                mtrx3[y][x] = mtrx2[y][x]

    return mtrx3


def mutating(mtrx, a, b, mutation):

    mutation = mutation / 100

    for x in range(b):
        for y in range(a):
            rand = random()
            if rand < mutation:
                mtrx[y][x] += np.random.normal(loc=0.0, scale=1.0, size=None) / 5

                if mtrx[y][x] > 1:
                    mtrx[y][x] = 1
                if mtrx[y][x] < -1:
                    mtrx[y][x] = -1

    return mtrx


def parent(iters):
    sum_fitness = 0

    for i in range(len(iters)):
        sum_fitness += iters[i][0]

    summation = 0
    rand = randint(0, sum_fitness - 1)

    for i in range(len(iters)):
        summation += iters[i][0]
        if summation > rand:
            return iters[i]

    return iters[0]


def poses(pos, fr, syn3):
    w = open('poses.txt', 'a')
    for i in range(len(pos)):
        w.write(str(i) + ' ' + str(pos[i]) + '\n')

    for j in range(18):
        for m in range(4):
            w.write(str(syn3[j][m]))

    w.write('\n' + str(fr))
    w.write('\n' + '\n')
    w.close()


def u_loading(syn0, syn1, syn2, fr_pos, length):
    f = open('save.txt', 'w')
    for a in range(24):
        for b in range(18):
            f.write(str(syn0[a][b]) + '\n')

    for a in range(18):
        for b in range(18):
            f.write(str(syn1[a][b]) + '\n')

    for a in range(18):
        for b in range(4):
            f.write(str(syn2[a][b]) + '\n')

    f.write(str(length) + '\n')

    for a in range(len(fr_pos)):
        for b in range(2):
            f.write((str(fr_pos[a][b]) + '\n'))

    f.close()


f = open('poses.txt', 'w')
f.close()


def d_loading():
    f = open('save.txt', 'r')

    fr_pos = []

    for a in range(24):
        for b in range(18):
            syn0[a][b] = float(f.readline())

    for a in range(18):
        for b in range(18):
            syn1[a][b] = float(f.readline())

    for a in range(18):
        for b in range(4):
            syn2[a][b] = float(f.readline())

    local_length = int(f.readline())

    for a in range(local_length):
        fr_pos.append([int(f.readline()), int(f.readline())])

    f.close

    return syn0, syn1, syn2, fr_pos


is_loaded = False

l1 = relu(np.dot(l0, syn0))
l2 = relu(np.dot(l1, syn1))
l3 = relu(np.dot(l2, syn2))

num_of_iterations = 4000
saved_iteration = [0]
iteration = 0
iterations = []

generation = 0
manual_control = True


def view():
    global l0
    global l1
    global l2
    global l3

    global pos
    global direction

    # DISTANCES

    # [0] - fruit [1] - is_tail [2] - is_wall

    # is_fruit 90d:

    if pos[0][0] == xf and pos[0][1] - yf > 0:  # up
        l0[0][0] = 1
    else:
        l0[0][0] = 0

    if pos[0][0] == xf and pos[0][1] - yf < 0:  # down
        l0[0][3] = 1
    else:
        l0[0][3] = 0

    if pos[0][1] == yf and pos[0][0] - xf < 0:  # right
        l0[0][6] = 1
    else:
        l0[0][6] = 0

    if pos[0][1] == yf and pos[0][0] - xf > 0:  # left
        l0[0][9] = 1
    else:
        l0[0][9] = 0

    # is_tail 90d:

    for i in range(len(pos)):
        if pos[0][0] == pos[i][0] and pos[0][1] - pos[i][1] > 0:  # up
            l0[0][1] = 1
        else:
            l0[0][1] = 0

        if pos[0][0] == pos[i][0] and pos[0][1] - pos[i][1] < 0:  # down
            l0[0][4] = 1
        else:
            l0[0][4] = 0

        if pos[0][1] == pos[i][1] and pos[0][0] - pos[i][0] < 0:  # right
            l0[0][7] = 1
        else:
            l0[0][7] = 0

        if pos[0][1] == pos[i][1] and pos[0][0] - pos[i][0] > 0:  # left
            l0[0][10] = 1
        else:
            l0[0][10] = 0

    # is_wall 90d:

    l0[0][2] = 1 / ((pos[0][1]) / 16)  # up
    l0[0][5] = 1 / ((640 - pos[0][1]) / 16)  # down
    l0[0][8] = 1 / ((960 - pos[0][0]) / 16)  # right
    l0[0][11] = 1 / ((pos[0][0] - 320) / 16)  # left

    # is_fruit 45d:

    if xf - pos[0][0] == pos[0][1] and pos[0][1] - yf > 0:  # up-right
        l0[0][12] = 1
    else:
        l0[0][12] = 0

    if pos[0][0] - xf == pos[0][1] and pos[0][1] - yf > 0:  # up-left
        l0[0][15] = 1
    else:
        l0[0][15] = 0

    if xf - pos[0][0] == yf - pos[0][1] and yf - pos[0][1] > 0:  # down-right
        l0[0][18] = 1
    else:
        l0[0][18] = 0

    if pos[0][0] - xf == yf - pos[0][1] and yf - pos[0][1] > 0:  # down-left
        l0[0][21] = 1
    else:
        l0[0][21] = 0

    # is_tail 45d:

    for i in range(len(pos)):
        if pos[i][0] - pos[0][0] == pos[0][1] - pos[i][1] and pos[0][1] - pos[i][1] > 0:  # up-right
            l0[0][13] = 1
        else:
            l0[0][13] = 0

        if pos[0][0] - pos[i][0] == pos[0][1] - pos[i][1] and pos[0][1] - pos[i][1] > 0:  # up-left
            l0[0][13] = 1
        else:
            l0[0][13] = 0

        if pos[i][0] - pos[0][0] == pos[i][1] - pos[0][1] and pos[i][1] - pos[0][1] > 0:  # down-right
            l0[0][13] = 1
        else:
            l0[0][13] = 0

        if pos[0][0] - pos[i][0] == pos[i][1] - pos[0][1] and pos[i][1] - pos[0][1] > 0:  # down-left
            l0[0][13] = 1
        else:
            l0[0][13] = 0

    # is_wall 45d:

    l0[0][14] = 1 / (find_max(pos[0][1], 960 - pos[0][0]) / 16)  # up-right
    l0[0][17] = 1 / (find_max(pos[0][1], pos[0][0] - 320) / 16)  # up-left
    l0[0][20] = 1 / (find_max(640 - pos[0][1], 960 - pos[0][0]) / 16)  # down-right
    l0[0][23] = 1 / (find_max(640 - pos[0][1], pos[0][0] - 320) / 16)  # down-left

    # DISTANCES END

    l1 = relu(np.dot(l0, syn0))
    l2 = relu(np.dot(l1, syn1))
    l3 = relu(np.dot(l2, syn2))

    ways = [['u', l3[0][0]], ['d', l3[0][1]], ['r', l3[0][2]], ['l', l3[0][3]]]
    ways.sort(key=lambda f: f[1], reverse=True)
    direction = ways[0][0]

    if displaying:
        print(l3)

# END OF NEURAL NET


# MOVING

run = True

fruits_pos_display = []
poses_display = []

while run:
    pygame.time.delay(time_delay)

    keys = pygame.key.get_pressed()

    if keys[pygame.K_t]:
        mutation_rate += 0.2
        mutation_rate = round(mutation_rate, 1)
        if mutation_rate < 0:
            mutation_rate = 0

    if keys[pygame.K_g]:
        mutation_rate -= 0.2
        mutation_rate = round(mutation_rate, 1)
        if mutation_rate < 0:
            mutation_rate = 0

    if keys[pygame.K_y]:
        time_delay = int(time_delay / 1.5)

    if keys[pygame.K_h]:
        time_delay = int(time_delay * 1.5)

    if keys[pygame.K_n]:
        time_delay = 50

    if keys[pygame.K_b]:
        mutation_rate = 5.0

    if keys[pygame.K_j]:
        is_loaded = True

    if keys[pygame.K_u]:
        u_loading(syn0, syn1, syn2, fruits_pos)

    if keys[pygame.K_m]:
        if manual_control:
            manual_control = False
        elif not manual_control:
            manual_control = True

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    if learning:

        win.fill((0, 0, 0))

        generation_text = font.render('Generation:  ' + str(generation), 1, (200, 200, 200))
        win.blit(generation_text, (100, 100))

        learning_text = learning_font.render('LEARNING...', 1, (200, 200, 200))
        win.blit(learning_text, (470, 280))

        mutation_rate_text = font.render('Mutation rate: ' + str(mutation_rate) + '%', 1, (200, 200, 200))
        win.blit(mutation_rate_text, (100, 70))

        high_score_text = score_font.render('Highscore: ' + str(high_score), 1, (200, 200, 200))
        win.blit(high_score_text, (100, 610))

        pygame.draw.rect(win, (210, 210, 210), (954, 2, 2, 650))
        pygame.draw.rect(win, (210, 210, 210), (338, 2, 2, 650))
        pygame.draw.rect(win, (210, 210, 210), (340, 2, 614, 2))
        pygame.draw.rect(win, (210, 210, 210), (340, 650, 614, 2))

        living = True
        iteration = 0

        best_pos = []
        best_syn2 = np.empty([18, 4])

        fruits_pos_best = []

        for i in range(num_of_iterations):

            if i == 0 and not first_gen:
                continue

            if first_gen:
                syn0 = 2 * np.random.random((24, 18)) - 1
                syn1 = 2 * np.random.random((18, 18)) - 1
                syn2 = 2 * np.random.random((18, 4)) - 1
            else:
                syn0 = mutating(res_syn0[i], 24, 18, mutation_rate)
                syn1 = mutating(res_syn1[i], 18, 18, mutation_rate)
                syn2 = mutating(res_syn2[i], 18, 4, mutation_rate)

            local_life_time = 0
            local_score = 0
            fitness = 0
            fruits_pos = []

            living = True

            pos = [[648, 312], [648, 328], [648, 344]]

            moves = 200

            direction = 'u'
            moving = 'u'

            random_fruit()

            ate_fruit = False

            fruits_pos.append([xf, yf])

            pos_snake = []

            while living:

                pos_snake.append(pos)

                view()

                last_pos = pos[-1]

                if direction == 'l' and moving == 'r':
                    direction = 'r'

                if direction == 'r' and moving == 'l':
                    direction = 'l'

                if direction == 'u' and moving == 'd':
                    direction = 'd'

                if direction == 'd' and moving == 'u':
                    direction = 'u'

                if direction == 'l':
                    moving = 'l'
                    local_life_time += 1
                    moves -= 1
                    del pos[-1]
                    head = pos[0]
                    pos = [[head[0] - 16, head[1]]] + pos

                if direction == 'r':
                    moving = 'r'
                    local_life_time += 1
                    moves -= 1
                    del pos[-1]
                    head = pos[0]
                    pos = [[head[0] + 16, head[1]]] + pos

                if direction == 'u':
                    moving = 'u'
                    local_life_time += 1
                    moves -= 1
                    del pos[-1]
                    head = pos[0]
                    pos = [[head[0], head[1] - 16]] + pos

                if direction == 'd':
                    moving = 'd'
                    local_life_time += 1
                    moves -= 1
                    del pos[-1]
                    head = pos[0]
                    pos = [[head[0], head[1] + 16]] + pos

                if ate_fruit:  # fruit spawning
                    xf = 328 + 16 * (randint(1, 38))
                    yf = 8 + 16 * (randint(1, 38))

                    while not all(xf != pos[i][0] for i in range(len(pos))) and not all(
                            yf != pos[i][1] for i in range(len(pos))):
                        xf = 328 + 16 * (randint(1, 38))
                        yf = 8 + 16 * (randint(1, 38))

                    fruits_pos.append([xf, yf])

                    ate_fruit = False

                if pos[0] == [xf, yf]:  # fruit eating
                    if moves < 500:
                        if moves > 400:
                            moves = 500
                        else:
                            moves += 100

                    local_score += 1

                    pos.append([last_pos[0]])
                    pos[-1].append(last_pos[1])

                    ate_fruit = True

                if moves <= 0:
                    living = False

                if not all(pos[0] != pos[i] for i in range(1, len(pos))):
                    living = False

                if pos[0][0] <= 328 or pos[0][0] >= 952 or pos[0][1] <= -8 or pos[0][1] >= 648:
                    living = False

            if score < 10:
                fitness = int(local_life_time * local_life_time) * exponential(2, local_score)
            else:
                fitness = 10000 * local_score * local_score * local_life_time

            if high_score > 18:
                fitness = int(square(fitness))

            '''
            if fitness >= best_score:
                best_score = fitness
                u_loading(syn0, syn1, syn2, fruits_pos, len(fruits_pos))
            '''

            iteration += 1
            iterations.append([fitness, syn0, syn1, syn2, fruits_pos])

            print(iteration)

            pygame.display.update()

        generation += 1
        learning = False
        first_gen = False

    if not learning and not displaying:

        iterations.sort(key=lambda f: f[0], reverse=True)

        # syn0 = for_display[1]
        # syn1 = for_display[2]
        # syn2 = for_display[3]

        # fruits_pos = for_display[4]

        ''' syn0, syn1, syn2, fruits_pos = d_loading() '''

        if iterations[0][0] <= 80000 and generation == 1:
            losing()
            generation -= 1
            first_gen = True
            continue

        new_iterations = iterations[::-1]

        res_syn0.clear()
        res_syn1.clear()
        res_syn2.clear()

        for i in range(num_of_iterations):

            if (i + 1) % 20 == 0:
                print(i + 1)

            '''
            n = randint(0, 827)

            if n <= 431:
                res_syn0.append(crossover(parent1[1], parent2[1], 24, 18))
                res_syn1.append(parent2[2])
                res_syn2.append(parent2[3])
            elif n >= 432 <= 755:
                res_syn0.append(parent1[1])
                res_syn1.append(crossover(parent1[2], parent2[2], 18, 18))
                res_syn2.append(parent2[3])
            elif n >= 756:
                res_syn0.append(parent1[1])
                res_syn1.append(parent1[2])
                res_syn2.append(crossover(parent1[3], parent2[3], 18, 4))
            '''

            parent1 = parent(new_iterations)
            parent2 = parent(new_iterations)

            res_syn0.append(crossover(parent1[1], parent2[1], 24, 18))
            res_syn1.append(crossover(parent1[2], parent2[2], 18, 18))
            res_syn2.append(crossover(parent1[3], parent2[3], 18, 4))

        # syn0 = crossover(iterations[0][1], iterations[1][1], 24, 18)

        reserved = False

        displaying = True
        pos = [[648, 312], [648, 328], [648, 344]]

        moves = 200
        life_time = 0

        score = 0

        '''
        if is_loaded:
            syn0, syn1, syn2, fruits_pos = d_loading()
            is_loaded = False
        '''

        iterations = iterations[:1:]

        useless = 0

        useless, syn0, syn1, syn2, fruits_pos = iterations[0]

        xf = fruits_pos[0][0]
        yf = fruits_pos[0][1]

        direction = 'u'
        moving = 'u'

        print(iterations[0][0])

        ate_fruit = False

    if displaying:

        if generation == 2:
            poses_display.append(pos)

        print('Max score: ' + str(len(fruits_pos) - 1) + '     Score: ' + str(score))

        print([xf, yf])

        view()

        last_pos = pos[-1]

        if manual_control:
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                direction = 'l'

            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                direction = 'r'

            if keys[pygame.K_UP] or keys[pygame.K_w]:
                direction = 'u'

            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                direction = 'd'

        if end:
            direction = ''
            pygame.time.delay(800)
            losing()

        if direction == 'l' and moving == 'r':
            direction = 'r'

        if direction == 'r' and moving == 'l':
            direction = 'l'

        if direction == 'u' and moving == 'd':
            direction = 'd'

        if direction == 'd' and moving == 'u':
            direction = 'u'

        if direction == 'l':
            moving = 'l'
            life_time += 1
            moves -= 1
            del pos[-1]
            head = pos[0]
            pos = [[head[0] - 16, head[1]]] + pos

        if direction == 'r':
            moving = 'r'
            life_time += 1
            moves -= 1
            del pos[-1]
            head = pos[0]
            pos = [[head[0] + 16, head[1]]] + pos

        if direction == 'u':
            moving = 'u'
            life_time += 1
            moves -= 1
            del pos[-1]
            head = pos[0]
            pos = [[head[0], head[1] - 16]] + pos

        if direction == 'd':
            moving = 'd'
            life_time += 1
            moves -= 1
            del pos[-1]
            head = pos[0]
            pos = [[head[0], head[1] + 16]] + pos

        if ate_fruit:  # fruit spawning
            try:
                xf = fruits_pos[score][0]
                yf = fruits_pos[score][1]
            except IndexError:
                xf = 328 + 16 * (randint(1, 38))
                yf = 8 + 16 * (randint(1, 38))

            while not all(xf != pos[i][0] for i in range(len(pos))) and not all(
                    yf != pos[i][1] for i in range(len(pos))):
                xf = 328 + 16 * (randint(1, 38))
                yf = 8 + 16 * (randint(1, 38))

            ate_fruit = False

        if pos[0] == [xf, yf]:  # fruit eating
            if moves < 500:
                if moves > 400:
                    moves = 500
                else:
                    moves += 100

            score += 1

            pos.append([last_pos[0]])
            pos[-1].append(last_pos[1])

            ate_fruit = True

        if moves <= 0:
            end = True

        if not all(pos[0] != pos[i] for i in range(1, len(pos))):
            end = True

        if pos[0][0] <= 328 or pos[0][0] >= 952 or pos[0][1] <= -8 or pos[0][1] >= 648:
            end = True

        win.fill((0, 0, 0))
        for x, y in pos:
            pygame.draw.rect(win, (255, 255, 255), (x, y, width, height))
            if end or moves <= 0:
                pygame.draw.rect(win, (120, 120, 120), (pos[0][0], pos[0][1], width, height))

        if not ate_fruit:
            pygame.draw.rect(win, (255, 0, 0), (xf, yf, width, height))

        pygame.draw.rect(win, (210, 210, 210), (954, 2, 2, 650))
        pygame.draw.rect(win, (210, 210, 210), (338, 2, 2, 650))
        pygame.draw.rect(win, (210, 210, 210), (340, 2, 614, 2))
        pygame.draw.rect(win, (210, 210, 210), (340, 650, 614, 2))

        score_text = score_font.render('Score: ' + str(score), 1, (200, 200, 200))
        win.blit(score_text, (100, 570))

        high_score_text = score_font.render('Highscore: ' + str(high_score), 1, (200, 200, 200))
        win.blit(high_score_text, (100, 610))

        moves_text = font.render('Moves left: ' + str(moves), 1, (200, 200, 200))
        win.blit(moves_text, (100, 10))

        life_time_text = font.render('Life time: ' + str(life_time), 1, (200, 200, 200))
        win.blit(life_time_text, (100, 40))

        mutation_rate_text = font.render('Mutation rate: ' + str(mutation_rate) + '%', 1, (200, 200, 200))
        win.blit(mutation_rate_text, (100, 70))

        generation_text = font.render('Generation:  ' + str(generation), 1, (200, 200, 200))
        win.blit(generation_text, (100, 100))

    pygame.display.update()




