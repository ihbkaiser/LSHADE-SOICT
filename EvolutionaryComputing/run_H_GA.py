
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as ptc
import typing
import tqdm
import pickle
import math
import warnings

# %%
class Config(typing.NamedTuple):
    pop_size: int
    temperature: float
    threshold: float
    useless_penalty: float
    active_penalty: float


# %%
with open("fix_sensors.pkl", 'rb') as f:
  fs_data = pickle.load(f)

with open("fix_targets.pkl", 'rb') as f:
  ft_data = pickle.load(f)


# %%
def make_pan_boundaries(centroid, q, radius):
    # first line always lies on x_axis
    ans = []
    theta = 2*np.pi/q
    for i in range(q):
        x = radius*np.cos(theta*i)
        y = radius*np.sin(theta*i)
        ans.append((centroid[0] + x, centroid[1] + y))
    return ans

# %%
def show_network(network, sensors_mask=None, figsize=None):
    n = network['n']
    m = network['m']
    q = network['q']

    if figsize is not None:
      plt.figure(figsize=(figsize, figsize))

    margin = network['margin']

    targets = network['targets']
    sensors = network['sensors']
    radius = network['radius']

    tar_x = [i[0] for i in targets]
    tar_y = [i[1] for i in targets]
    plt.plot(tar_x, tar_y, 'r^', label='targets')
    sen_x = [i[0] for i in sensors]
    sen_y = [i[1] for i in sensors]
    plt.plot(sen_x, sen_y, 'go', label='sensors')
    ax = plt.gca()

    for i in range(len(sensors)):
        sensor = sensors[i]
        active = True

        if sensors_mask is not None:
            active = False
            if sensors_mask[i] != q:
                theta = 360.0/q
                dir = sensors_mask[i]
                active = True
                theta1, theta2 = theta*dir, theta*(dir+1)
                wedge = ptc.Wedge(sensors[i], radius, theta1, theta2, color='#34e1eb', alpha=0.45)
                ax.add_artist(wedge)
        if active:
            circle = plt.Circle(sensor, radius, color='m', fill=False, linewidth=1)
            ax.add_artist(circle)
            pan_boundaries = make_pan_boundaries(sensor, q, radius)
            for point in pan_boundaries:
                plt.plot([sensor[0], point[0]], [sensor[1], point[1]], 'b--', alpha=0.2)

    plt.xlim(margin[0])
    plt.ylim(margin[1])
    ax.set_aspect(1.0)  # make aspect ratio square

    plt.legend()

    plt.show()

# %%
def is_within_FoV(bisector, target, sensor, radius):
    target = np.asarray(target)
    sensor = np.asarray(sensor)
    bisector = np.asarray(bisector)
    v = target - sensor
    dist = np.linalg.norm(v)
    if dist <= 1e-12:  # target exactly at sensor
        return True
    scalar = bisector.dot(v)
    # FoV half-angle pi/8 as in your original (q assumed to discretize 2π)
    return scalar + 1e-7 >= radius*dist*np.cos(np.pi/8) and dist - 1e-7 <= radius

def init_T(network):
    sensors = network['sensors']
    targets = network['targets']
    radius = network['radius']
    n = network['n']
    m = network['m']
    q = network['q']
    T = np.zeros((n, m, q), dtype=bool)

    bisectors = []
    for i in range(q):
        bisectors.append((radius*np.cos(np.pi*(1 + i*2)/q), radius*np.sin(np.pi*(1 + i*2)/q)))

    for i in range(n):
        for j in range(m):
            for p in range(q):
                T[i, j, p] = is_within_FoV(bisectors[p], targets[j], sensors[i], radius)
    return T


# %% [markdown]
# # Model

# %%
def cooldown(temperature):
  return temperature - temperature*max(0.02, np.exp(-np.square(temperature)))

def get_prob(fitness, temperature):
  return np.exp(-1.5*fitness/(temperature+10))


# %%
class Result(typing.NamedTuple):
  genome: np.ndarray
  fitness: float
  achieved_coverage: np.ndarray
  useless: int
  active: int


# %%
class Individual:
    def __init__(self, genome, fitness) -> None:
        self.genome = genome
        self.fitness = float(fitness)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def copy(self):
      # fitness is a float; make an explicit float copy to avoid attribute errors
      return Individual(self.genome.copy(), float(self.fitness))

# --- UCB1 helper (Selection Mechanism as MAB) ---
class UCB1:
    def __init__(self, n_arms, c=math.sqrt(2), decay=1.0):
        self.n = np.zeros(n_arms, dtype=float)   # pulls per arm
        self.mu = np.zeros(n_arms, dtype=float)  # mean reward per arm
        self.t = 0
        self.c = c
        self.decay = decay  # 1.0 = stationary; <1.0 = exponential forgetting

    def select(self):
        self.t += 1
        untried = np.where(self.n == 0)[0]
        if len(untried) > 0:
            return int(np.random.choice(untried))
        ucb = self.mu + self.c * np.sqrt(np.log(self.t) / self.n)
        return int(np.argmax(ucb))

    def update(self, arm, r):
        # optional decay for non-stationarity
        if self.decay < 1.0:
            self.n *= self.decay
            self.mu *= self.decay
        self.n[arm] += 1.0
        self.mu[arm] += (r - self.mu[arm]) / self.n[arm]

# %%
class GAModel:
  def __init__(self):
    self.__compiled = False

  def adapt(self, network):
    self.n = network['n']
    self.m = network['m']
    self.q = network['q']
    self.K = network['K']
    self.T = init_T(network)    # shape (n, m, q) -> bool
    # Cache a per-sensor orientation coverage count (for heuristics/CBC priors)
    self.coverage_count = np.zeros((self.n, self.q+1), dtype=int)  # +1 for 'inactive' slot q
    for i in range(self.n):
      for p in range(self.q):
        self.coverage_count[i, p] = int(np.sum(self.T[i, :, p]))
      # 'inactive' covers 0 by definition
      self.coverage_count[i, self.q] = 0

    # None value attributes
    self.POPULATION_SIZE = None
    self.temperature = None
    self.threshold = None

    # CBC/TGEM weights are set per-generation
    self._cbc_w = None

    self.__compiled = False

  def compile(self, config):
    self.POPULATION_SIZE = config.pop_size
    self.temperature = config.temperature
    self.threshold = config.threshold
    self.delta = 20
    self.useless_penalty = config.useless_penalty
    self.active_penalty = config.active_penalty

    # --- UCB selection (rank buckets as arms) ---
    self.parent_buckets = max(3, int(round(0.5 * np.log2(self.POPULATION_SIZE) + 2)))  # 3–6 typically
    self.ucb_c = math.sqrt(2.0)
    self.ucb_decay = 0.98      # set 1.0 to disable forgetting
    self.ucb_par = UCB1(self.parent_buckets, c=self.ucb_c, decay=self.ucb_decay)

    # CBC knobs
    self.cbc_allow_deactivate = True
    self.cbc_diversity_prior = 1e-3  # tiny prior to avoid zero-prob ties

    # TGEM knobs
    self.tgem_explore_prob = 0.2     # chance to do a pure random flip
    self.tgem_deactivate_if_useless = True

    self.__compiled = True

  # ---------- Hybrid GA main loop (H-GA) ----------
  def solve(self, cross_ratio=0.4, mutation_ratio=0.05, max_gens=100, verbose=2):
    if not self.__compiled:
      return RuntimeError("Model has not been compiled. Please execute GAModel.compile method")
    if verbose > 2 or verbose < 0 or max_gens < 1:
      raise ValueError("Invalid verbose or negative max_gens at 'GAModel.solve'")

    initial_genomes = self.init_genome(size=(self.POPULATION_SIZE, self.n), bound=self.q + 1)
    population = []
    for i in range(self.POPULATION_SIZE):
      temp_fitness = self.cal_fitness(initial_genomes[i])
      population.append(Individual(initial_genomes[i], temp_fitness))

    if verbose == 1:
      print(f"Initial population with {self.POPULATION_SIZE} individuals:")
      print('---------------------------------------------------')
    elif verbose == 2:
      print(f"Initial population with {self.POPULATION_SIZE} individuals:")
      for i in range(self.POPULATION_SIZE):
        print(f"{i + 1}. {population[i].genome}, fitness: {population[i].fitness}")
      print('---------------------------------------------------')

    history = []

    temperature = self.temperature
    gen = 0
    not_grow_gens = 0
    while temperature > 4 and gen <= max_gens:
      population.sort()
      best_ind = population[0]
      # Prepare CBC/TGEM weights from current best (deficit vector)
      self._cbc_w = self._compute_deficit_weights(best_ind.genome)

      if verbose == 2:
        print("\nCurrent temp: ", temperature)

      new_population = []
      old_fitness = best_ind.fitness
      cross_size = int(self.POPULATION_SIZE*cross_ratio)
      mutation_size = int(self.POPULATION_SIZE*mutation_ratio)

      # ---- UCB-based parent selection over rank buckets ----
      # Build rank buckets (arms)
      B = self.parent_buckets
      bounds = np.linspace(0, len(population), B + 1, dtype=int)
      bucket_slices = [(bounds[b], bounds[b+1]) for b in range(B)]
      def sample_from_bucket(b):
        lo, hi = bucket_slices[b]
        if hi <= lo:
          return population[np.random.randint(len(population))]
        return population[np.random.randint(lo, hi)]

      for _ in range(cross_size):
          b1 = self.ucb_par.select()
          b2 = self.ucb_par.select()

          pA = sample_from_bucket(b1)
          pB = sample_from_bucket(b2)

          # Coverage-Biased Crossover (CBC)
          g1, g2 = self.cross(pA.genome, pB.genome)

          c1 = Individual(g1, self.cal_fitness(g1))
          c2 = Individual(g2, self.cal_fitness(g2))
          new_population.append(c1)
          new_population.append(c2)

          # reward = improvement of best child vs best parent (min fitness = better)
          parent_best = min(pA.fitness, pB.fitness)
          child_best  = min(c1.fitness, c2.fitness)
          r = self._reward_from_improvement(parent_best, child_best)
          self.ucb_par.update(b1, r)
          self.ucb_par.update(b2, r)

      # ---- Target-Guided Exploratory Mutation (TGEM) ----
      mutation_list = np.random.choice(np.arange(self.POPULATION_SIZE), mutation_size, replace=False)
      for i in mutation_list:
          p = population[i].copy()
          new_genome = self.mutate(p.genome, self.q+1)
          new_ind = Individual(new_genome, self.cal_fitness(new_genome))
          new_population.append(new_ind)

      # ---- Elitist + probabilistic survivor selection (unchanged) ----
      direct_count = int(self.POPULATION_SIZE*self.threshold)
      prob_count = self.POPULATION_SIZE - direct_count

      temp_pop = population + new_population
      temp_pop.sort()

      population = temp_pop[:direct_count]
      remain = temp_pop[direct_count:]

      prob = np.array([get_prob(tp.fitness, temperature) for tp in remain])
      prob = prob/np.sum(prob)
      prob_list = list(np.random.choice(remain, prob_count, p=prob))

      population = population + prob_list
      temperature = cooldown(temperature)

      new_best = temp_pop[0].fitness

      if verbose > 0:
        print("Generation:", gen)
        if verbose == 2:
          print("Old best fitness value:", old_fitness)
        print("New best fitness value:", new_best)
      history.append(temp_pop[0].fitness)
      if new_best < old_fitness:
        not_grow_gens = 0
      else:
        not_grow_gens += 1
      if not_grow_gens > self.delta:
        break
      gen += 1

    f, useless, active = self.achieved_coverage(population[0].genome)
    result = Result(
          population[0].genome,
          population[0].fitness,
          f,
          useless,
          active
      )

    return {'result': result,
            'history': history}

  # ---------- Initialization (hybrid: heuristic + random) ----------
  def init_genome(self, size, bound, heu_init=0.5):
    l = size[1]
    heuristic_size = int(size[0]*heu_init)
    rand_size = size[0] - heuristic_size

    # reuse precomputed coverage_count (n, q+1)
    probs = self.coverage_count.astype(float) + 0.5  # Laplace smoothing
    probs = probs/np.sum(probs, axis=1, keepdims=True)

    heu = np.zeros((heuristic_size, self.n), dtype=int)
    for i in range(self.n):
      heu[:, i] = np.random.choice(np.arange(self.q+1), p=probs[i], size=heuristic_size)

    rand = np.random.randint(bound, size=(rand_size, l))
    return np.concatenate((heu, rand), axis=0)

  # ---------- Target-Guided Exploratory Mutation (TGEM) ----------
  def mutate(self, genome, bound):
    # bound == q+1 (including inactive slot)
    idx = np.random.randint(genome.shape[0])
    old = genome[idx]

    # exploration step
    if np.random.rand() < self.tgem_explore_prob:
      # pure random flip to a different value
      while True:
        new_val = np.random.randint(bound)
        if new_val != old:
          genome[idx] = new_val
          return genome

    # Exploit: push toward under-covered targets
    if self._cbc_w is None:
      warnings.warn("CBC weights not set; falling back to random mutation")
      while True:
        new_val = np.random.randint(bound)
        if new_val != old:
          genome[idx] = new_val
          return genome

    # score each orientation by weighted deficit it can reduce
    scores = np.zeros(self.q+1, dtype=float)  # include inactive
    for p in range(self.q):
      # sum of deficits covered by (idx, p)
      scores[p] = float(np.dot(self.T[idx, :, p], self._cbc_w))
    # optionally deactivate if it doesn't help
    if self.tgem_deactivate_if_useless:
      scores[self.q] = 0.0  # deactivation has 0 immediate gain, but saves active penalty

    # choose best different from old; if tie, random among best
    best = np.argmax(scores)
    if scores[best] <= 1e-12:
      # nothing helpful -> either deactivate or random different
      if self.tgem_deactivate_if_useless:
        best = self.q
      else:
        # random different
        candidates = list(range(self.q+1))
        candidates.remove(int(old))
        best = np.random.choice(candidates)

    genome[idx] = best
    return genome

  # ---------- Coverage-Biased Crossover (CBC) ----------
  def cross(self, genome_A, genome_B):
    # gene-wise biased picks using current deficit weights self._cbc_w
    g1 = np.empty_like(genome_A)
    g2 = np.empty_like(genome_A)

    # small prior to preserve diversity & avoid zero prob
    prior = self.cbc_diversity_prior

    for i in range(self.n):
      a = int(genome_A[i])
      b = int(genome_B[i])

      # candidate set includes a, b; optionally also inactive q if neither is inactive
      candidates = {a, b}
      if self.cbc_allow_deactivate:
        candidates.add(self.q)

      cand = sorted(list(candidates))
      # compute score for each candidate: weighted deficit reduction
      sc = []
      for p in cand:
        if p == self.q:
          sc.append(0.0)  # deactivate -> 0 immediate coverage improvement
        else:
          sc.append(float(np.dot(self.T[i, :, p], self._cbc_w)))
      sc = np.array(sc, dtype=float)

      # add a tiny prior that prefers orientations with historically higher raw coverage
      prior_vec = np.array([self.coverage_count[i, p] if p != self.q else 0.0 for p in cand], dtype=float)
      prior_vec = (prior_vec / (prior_vec.sum() + 1e-9)) if prior_vec.sum() > 0 else np.ones_like(prior_vec)/len(prior_vec)
      scores = sc + prior * prior_vec

      # probabilities (softmax over positive scores)
      if np.all(scores <= 0):
        # all zero -> uniform over candidates
        probs = np.ones_like(scores) / len(scores)
      else:
        # normalize to probabilities
        scores = np.maximum(scores, 0.0)
        probs = scores / (scores.sum() + 1e-12)

      pick1 = int(np.random.choice(cand, p=probs))
      # second child: independent draw (keeps diversity)
      pick2 = int(np.random.choice(cand, p=probs))

      g1[i] = pick1
      g2[i] = pick2

    return g1, g2

  # ---------- Optional k-point (unchanged, spare) ----------
  def k_point_cross(self, genome_A, genome_B, k=1):
    new_genome1 = np.zeros_like(genome_A)
    new_genome2 = np.zeros_like(genome_A)
    k_points = list(np.random.randint(0, self.n, size=k))
    k_points.append(self.n)
    k_points.insert(0, 0)

    bit = True
    for i in range(1, k+2):
      if bit:
        for j in range(k_points[i-1], k_points[i]):
          new_genome1[j] = genome_A[j]
          new_genome2[j] = genome_B[j]
      else:
        for j in range(k_points[i-1], k_points[i]):
          new_genome1[j] = genome_B[j]
          new_genome2[j] = genome_A[j]
      bit = not bit

    return new_genome1, new_genome2

  # ---------- Utilities ----------
  def _compute_deficit_weights(self, genome):
    # deficit vector w_j = max(0, K_j - f_j)
    f, _, _ = self.achieved_coverage(genome)
    deficit = np.maximum(0, np.array(self.K, dtype=float) - np.array(f, dtype=float))
    # emphasize larger gaps; normalize to sum=1 to make scores comparable
    if deficit.sum() <= 1e-12:
      return np.ones(self.m, dtype=float) / float(self.m)  # fully satisfied -> uniform
    return deficit / deficit.sum()

  def _reward_from_improvement(self, parent_best, child_best, eps=1e-9):
    # r = max{0, (f_par - f_child) / (|f_par| + ε)}  (fitness minimized)
    gain = parent_best - child_best
    denom = abs(parent_best) + eps
    return max(0.0, gain / denom)

  def achieved_coverage(self, genome):
    f = np.zeros((self.m,), dtype=int)
    useless = 0
    active_sensors = 0

    for i in range(self.n):
      if genome[i] != self.q:
        track = False
        for j in range(self.m):
          if self.T[i, j, genome[i]]:
            track = True
            f[j] += 1

        if track:
          active_sensors += 1
        else:
          useless += 1

    return f, useless, active_sensors

  def cal_fitness(self, genome):
    f, useless, active_sensors = self.achieved_coverage(genome)
    f = np.minimum(f, self.K)
    priority_factors = np.sqrt(self.K)
    # squared deficit + penalties (minimize)
    return np.sum(priority_factors*np.square(f - self.K)) + self.useless_penalty*useless + self.active_penalty*active_sensors


# %% [markdown]
# # Solve problem

# %% [markdown]
# # Metrics

# %%
def distance_index(k, x):
  a = np.sum(k*k)
  b = k - x
  b = np.sum(b*b)
  return 1 -b/a

# %%
def variance(k, x):
  m = len(x)
  mk = np.zeros_like(x)
  for t in range(m):
    mk[t] = np.sum(k == k[t])
  nu_k = np.zeros_like(x)
  for t in range(m):
    ans = 0
    for i in range(m):
      ans += x[i]*(k[i] == k[t])
    nu_k[t] = ans/mk[t]

  a = (x - nu_k)
  return np.sum(a*a/mk)


# %%
def activated_sensors(genome, bound):
  cnt = 0
  for i in genome:
    if i != bound:
      cnt += 1
  return cnt

# %% [markdown]
# (plots elided for brevity)

# %%
def coverage_quality(mask, network):
  sensors = network['sensors']
  targets = network['targets']
  radius = network['radius']
  n = network['n']
  m = network['m']
  q = network['q']
  T = np.zeros((n, q, m), dtype=bool)

  bisectors = []
  for i in range(q):
      bisectors.append((radius*np.cos(np.pi*(1 + i*2)/q), radius*np.sin(np.pi*(1 + i*2)/q)))

  for i in range(n):
      for j in range(m):
          for p in range(q):
              T[i, p, j] = is_within_FoV(bisectors[p], targets[j], sensors[i], radius)

  U = np.zeros((n, q, m), dtype=float)
  for i in range(n):
    for j in range(m):
        for p in range(q):
          if T[i, p, j]:
            target = np.asarray(targets[j])
            sensor = np.asarray(sensors[i])
            v = target - sensor
            U[i, p, j] = 1 - np.square(np.linalg.norm(v)/radius)

  S = np.zeros((n, q), dtype=bool)
  for i in range(n):
    if mask[i] != q:
      S[i, mask[i]] = True

  return np.sum(np.sum(U, axis=2)*S)


# %% [markdown]
# # Evaluate (same evaluation scaffold you provided)
# Existing code up to the evaluation for fs_small is assumed to be run first.

# For fs_small (fix sensors small)
DI_fs_small = []
VAR_fs_small = []
CQ_fs_small = []
ACT_fs_small = []
for i in range(10):
    di = []
    var = []
    cq = []
    act = []
    for dt in tqdm.tqdm(fs_data[i]['small']):
        model_fs = GAModel()  # Create a new instance each time
        model_fs.adapt(dt)
        config_fs = Config(pop_size=max(50, int(dt['n']*dt['q'])), temperature=1000, threshold=.7, useless_penalty=1., active_penalty=1./(dt['n'] + 1))
        model_fs.compile(config_fs)
        result = model_fs.solve(verbose=0)
        DI_score = distance_index(np.asarray(dt['K']), result['result'].achieved_coverage)
        var_score = variance(np.asarray(dt['K']), result['result'].achieved_coverage)
        cq_score = coverage_quality(result['result'].genome, dt)
        act_score = activated_sensors(result['result'].genome, 8)
        di.append(DI_score)
        var.append(var_score)
        cq.append(cq_score)
        act.append(act_score)
    DI_fs_small.append(di)
    VAR_fs_small.append(var)
    CQ_fs_small.append(cq)
    ACT_fs_small.append(act)
GA_fs_small = [DI_fs_small, VAR_fs_small, CQ_fs_small, ACT_fs_small]
import pickle
with open("HGA_fs_small.pkl", 'wb') as f:
    pickle.dump(GA_fs_small, f)

# For fs_large (fix sensors large)
DI_fs_large = []
VAR_fs_large = []
CQ_fs_large = []
ACT_fs_large = []
for i in range(10):
    di = []
    var = []
    cq = []
    act = []
    for dt in tqdm.tqdm(fs_data[i]['large']):
        model_fs = GAModel()  # Create a new instance each time
        model_fs.adapt(dt)
        config_fs = Config(pop_size=max(50, int(dt['n']*dt['q'])), temperature=1000, threshold=.7, useless_penalty=1., active_penalty=1./(dt['n'] + 1))
        model_fs.compile(config_fs)
        result = model_fs.solve(verbose=0)
        DI_score = distance_index(np.asarray(dt['K']), result['result'].achieved_coverage)
        var_score = variance(np.asarray(dt['K']), result['result'].achieved_coverage)
        cq_score = coverage_quality(result['result'].genome, dt)
        act_score = activated_sensors(result['result'].genome, 8)
        di.append(DI_score)
        var.append(var_score)
        cq.append(cq_score)
        act.append(act_score)
    DI_fs_large.append(di)
    VAR_fs_large.append(var)
    CQ_fs_large.append(cq)
    ACT_fs_large.append(act)
GA_fs_large = [DI_fs_large, VAR_fs_large, CQ_fs_large, ACT_fs_large]
with open("HGA_fs_large.pkl", 'wb') as f:
    pickle.dump(GA_fs_large, f)

# For ft_small (fix targets small)
DI_ft_small = []
VAR_ft_small = []
CQ_ft_small = []
ACT_ft_small = []
for i in range(10):
    di = []
    var = []
    cq = []
    act = []
    for dt in tqdm.tqdm(ft_data[i]['small']):
        model_ft = GAModel()  # Create a new instance each time
        model_ft.adapt(dt)
        config_ft = Config(pop_size=max(50, int(dt['n']*dt['q'])), temperature=1000, threshold=.7, useless_penalty=1., active_penalty=1./(dt['n'] + 1))
        model_ft.compile(config_ft)
        result = model_ft.solve(verbose=0)
        DI_score = distance_index(np.asarray(dt['K']), result['result'].achieved_coverage)
        var_score = variance(np.asarray(dt['K']), result['result'].achieved_coverage)
        cq_score = coverage_quality(result['result'].genome, dt)
        act_score = activated_sensors(result['result'].genome, 8)
        di.append(DI_score)
        var.append(var_score)
        cq.append(cq_score)
        act.append(act_score)
    DI_ft_small.append(di)
    VAR_ft_small.append(var)
    CQ_ft_small.append(cq)
    ACT_ft_small.append(act)
GA_ft_small = [DI_ft_small, VAR_ft_small, CQ_ft_small, ACT_ft_small]
with open("HGA_ft_small.pkl", 'wb') as f:
    pickle.dump(GA_ft_small, f)

# For ft_large (fix targets large)
DI_ft_large = []
VAR_ft_large = []
CQ_ft_large = []
ACT_ft_large = []
for i in range(10):
    di = []
    var = []
    cq = []
    act = []
    for dt in tqdm.tqdm(ft_data[i]['large']):
        model_ft = GAModel()  # Create a new instance each time
        model_ft.adapt(dt)
        config_ft = Config(pop_size=max(50, int(dt['n']*dt['q'])), temperature=1000, threshold=.7, useless_penalty=1., active_penalty=1./(dt['n'] + 1))
        model_ft.compile(config_ft)
        result = model_ft.solve(verbose=0)
        DI_score = distance_index(np.asarray(dt['K']), result['result'].achieved_coverage)
        var_score = variance(np.asarray(dt['K']), result['result'].achieved_coverage)
        cq_score = coverage_quality(result['result'].genome, dt)
        act_score = activated_sensors(result['result'].genome, 8)
        di.append(DI_score)
        var.append(var_score)
        cq.append(cq_score)
        act.append(act_score)
    DI_ft_large.append(di)
    VAR_ft_large.append(var)
    CQ_ft_large.append(cq)
    ACT_ft_large.append(act)
GA_ft_large = [DI_ft_large, VAR_ft_large, CQ_ft_large, ACT_ft_large]
with open("HGA_ft_large.pkl", 'wb') as f:
    pickle.dump(GA_ft_large, f)

