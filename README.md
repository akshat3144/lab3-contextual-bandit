# Lab 3: Contextual Bandit-Based News Article Recommendation

**Student:** Akshat  
**Roll Number:** U20230093  
**Branch:** akshat_U20230093  
**Notebook:** `lab3_results_U20230093.ipynb`

---

## Overview

This lab implements a contextual multi-armed bandit system for personalized news article recommendation. The pipeline consists of:

1. **User Classification** — classify users into one of three contexts (user_1, user_2, user_3)
2. **Contextual Bandit** — learn the best news category for each user context using three exploration strategies
3. **Recommendation Engine** — recommend articles based on the learned bandit policy

---

## Approach & Design Decisions

### Data Preprocessing (Section 5.1)
- **Missing values:** Age column (698 missing in train) filled with median imputation
- **Categorical encoding:** `region_code` encoded via frequency encoding (handles unseen values in test set gracefully); `subscriber` encoded as int
- **Dropped columns:** `user_id` (non-predictive), `browser_version` (high-cardinality string), `label` (target)
- **Scaling:** StandardScaler applied to all 30 numeric features
- **Split:** 80/20 train-validation split with stratification

### User Classification (Section 5.2)
- **Model:** GradientBoostingClassifier (n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8)
- **Validation Accuracy:** 91.5% on the held-out 20% validation set
- **Final model** retrained on 100% of training data before predicting test user contexts

### Bandit Algorithms (Section 5.3)

**Arm Mapping:**
- 3 contexts × 4 news categories = 12 arms
- Arm index `j = context_idx × 4 + category_idx`
- Categories: [ENTERTAINMENT=0, EDUCATION=1, TECH=2, CRIME=3]
- Contexts: [user_1=0, user_2=1, user_3=2]

**Sampler:** Initialized with roll number `i=93`, rewards obtained via `sampler.sample(j)`

**Algorithms implemented:**

| Algorithm | Hyperparameters Tested | Best Config |
|-----------|----------------------|-------------|
| Epsilon-Greedy | ε ∈ {0.01, 0.1, 0.3} | ε = 0.01 |
| UCB | c ∈ {0.5, 1.0, 2.0} | c = 0.5 |
| SoftMax | τ ∈ {0.1, 1.0, 5.0} | τ = 0.1 |

### Recommendation Engine (Section 5.4)
- Classifies each test user → context
- Uses learned bandit policy (UCB c=1.0) to select the best news category for that context
- Samples a random article from the recommended category in the news dataset

---

## Key Results

### Best Category per Context
All three bandit algorithms consistently identify **CRIME** as the highest-reward category across all user contexts.

### Mean Rewards (T=10,000 steps, best hyperparameters)

| Context | ε-Greedy (ε=0.01) | UCB (c=0.5) | SoftMax (τ=0.1) |
|---------|-------------------|-------------|-----------------|
| user_1  | 3.010 | 3.185 | 3.179 |
| user_2  | 9.188 | 9.319 | 9.320 |
| user_3  | 3.728 | 3.771 | 3.762 |

### Observations
1. **UCB and SoftMax (low τ)** converge fastest to optimal arms, achieving the highest cumulative rewards
2. **UCB is the most robust** to hyperparameter choice — all c values yield similar performance
3. **Epsilon-Greedy** suffers slightly from continued random exploration even after convergence
4. **Higher exploration** (ε=0.3, τ=5.0) significantly degrades performance by wasting pulls on suboptimal arms
5. **user_2 receives much higher rewards** (~9.3) than user_1 (~3.1) and user_3 (~3.8), demonstrating context-dependent reward distributions

---

## Reproducing the Experiments

### Prerequisites
```bash
pip install numpy pandas scikit-learn matplotlib rlcmab-sampler
```

### Run
1. Open `lab3_results_U20230093.ipynb` in Jupyter/VS Code
2. Run all cells top to bottom
3. The notebook is self-contained — all data loading, preprocessing, training, bandit simulation, and plotting are included

### Data Files
- `data/train_users.csv` — 2000 labeled users (33 columns, labels: user_1/user_2/user_3)
- `data/test_users.csv` — 2000 unlabeled users (32 columns)
- `data/news_articles.csv` — 209,527 news articles (6 columns, 42 categories)

---

## References
- Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 2 (Multi-Armed Bandits)
- rlcmab-sampler package (provided by course instructor)
