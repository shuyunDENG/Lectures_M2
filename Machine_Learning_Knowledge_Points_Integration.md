# Machine Learning (SPLEX) - 知识点整合与考核内容

## 基于ER1考试的核心考点分析

### ER1 2022-2023 考核内容
1. **Spectral clustering and graph Laplacian** (3pts)
2. **Support Vector Machines (SVM)** (3pts)
3. **Linear classification/regression, Training vs Testing** (2pts)
4. **k-plus proches voisins (k-NN)** (3pts)
5. **Naive Bayes** (7pts)
6. **Perceptron** (4pts)

### ER1 2023-2024 考核内容
1. **Short questions** - classifier selection, accuracy interpretation, regularization (3pts)
2. **Canonical Correlation Analysis (CCA)** (4pts)
3. **Gradient descent for k-means** (3pts)
4. **Contingency tables, probabilities, logistic regression, independence** (8pts)
5. **Decision trees with entropy** (5pts)

---

## Cours 1: Introduction, Evaluation et Agrégation de Classifieurs

### 1.1 核心概念

#### Classification Task (分类任务形式化)
- **Population** Π, **Descriptive space** D ⊆ ℝ^d, **Classes** C
- **Goal**: Find Ĉ: D → C such that ∀π, Ĉ(D(π)) ≈ C(π)

#### Supervised Learning Methodology (监督学习方法论)
1. **Training** (Apprentissage): Learn Ĉ on Πₐ
2. **Validation**: Evaluate Ĉ on Πᵥ
3. **Prediction**: Apply Ĉ on new data

### 1.2 关键推导

#### Error Metrics (错误度量)

**Erreur en généralisation** (泛化误差):
```
e(Ĉ) = 𝔼ₓ[C(x) ≠ Ĉ(x)]
```

**Erreur en apprentissage**:
```
eₐ(Ĉ) = (1/|Πₐ|) Σ_{x∈Πₐ} δ(C(x) - Ĉ(x))
```

**Erreur en validation**:
```
eᵥ(Ĉ) = (1/|Πᵥ|) Σ_{x∈Πᵥ} δ(C(x) - Ĉ(x))
```

#### Confusion Matrix (混淆矩阵) 及指标

|  | C(x)=⊖ | C(x)=⊕ |
|---|---|---|
| **Ĉ(x)=⊖** | VN (True Negative) | FN (False Negative) |
| **Ĉ(x)=⊕** | FP (False Positive) | VP (True Positive) |

**关键指标计算**:
- **Accuracy** (准确率): (VP + VN) / N
- **Precision** (精确率): VP / (VP + FP)
- **Recall/Sensitivity** (召回率/灵敏度): VP / (VP + FN)
- **Specificity** (特异性): VN / (VN + FP)
- **F1-score**: 2 × (Precision × Recall) / (Precision + Recall)

#### Leave-One-Out Cross Validation (LOOCV)

**算法**:
```
For each π ∈ Πₐ:
    1. Train Ĉ₋π on Πₐ \ {π}
    2. Calculate eπ = |Ĉ₋π(π) - C(π)|

eLOOCV(Ĉ) = (1/|Πₐ|) Σ_{π∈Πₐ} eπ
```

### 1.3 ROC Curve (ROC曲线)

#### 构造方法
1. Sort data by g(x) values (classifier score)
2. For k = 0 to |Πₐ|, define Ĉₖ that classifies ⊖ for rank ≤ k
3. Plot (1-Specificity, Sensitivity) for each Ĉₖ

#### AUC (Area Under Curve)
- **Random classifier**: AUC = 0.5
- **Perfect classifier**: AUC = 1.0
- **Interpretation**: Probability that classifier ranks a positive higher than a negative

### 1.4 Bootstrap 与聚合方法

#### Bootstrap原理
```
For b = 1 to B:
    1. Draw Lₒ sample with replacement from L
    2. Estimate θ̂ₒ on sample Lₒ

L̃ = {θ̂₁, ..., θ̂ᵦ}
```

**方差估计**:
```
σ²θ̂ ≈ σ̂²ᵦ = (1/(B-1)) Σᵦ(θ̂ₒ - (1/B)Σᵦθ̂ₒ)²
```

### 1.5 Bagging (Bootstrap Aggregating)

**算法**:
```
For i = 1 to I:
    1. Create bootstrap sample Π⁽ⁱ⁾ₐ
    2. Train Ĉᵢ on Π⁽ⁱ⁾ₐ

Ĉbagging(x) = arg max_{c} |{i: Ĉᵢ(x) = c}|
```

**特点**: 降低方差，对不稳定分类器有效（如决策树）

### 1.6 AdaBoost 推导

**核心更新公式**:

**Step t**:
1. Calculate error: εₜ = eᴅₜ(Ĉₜ) = Σᵢ Dₜ(xᵢ) · 𝟙[Ĉₜ(xᵢ) ≠ C(xᵢ)]

2. Calculate βₜ:
```
βₜ = εₜ / (1 - εₜ)
```

3. Update weights:
```
Dₜ₊₁(xᵢ) ∝ {
    Dₜ(xᵢ)      if Ĉₜ(xᵢ) = C(xᵢ)
    βₜ·Dₜ(xᵢ)    if Ĉₜ(xᵢ) ≠ C(xᵢ)
}
```

4. Calculate αₜ:
```
αₜ = (1/2) log(1/βₜ) = (1/2) log((1-εₜ)/εₜ)
```

**最终分类器**:
```
Ĉboosting(x) = sign(Σₜ αₜ·Ĉₜ(x))
```

**关键性质**: AdaBoost增加margin，减少bias，即使训练误差为0也能继续提升泛化性能

---

## Cours 2: Tests d'Hypothèses Statistiques

### 2.1 假设检验框架

#### Basic Setup
- **H₀**: Null hypothesis (待检验假设)
- **H₁**: Alternative hypothesis (备择假设)
- **Test statistic**: 统计量用于决策

#### Error Types (错误类型)

|  | H₀ is true | H₁ is true |
|---|---|---|
| **Reject H₀** | Type I error (α) | Correct |
| **Don't reject H₀** | Correct | Type II error (β) |

- **α**: Significance level (显著性水平)
- **Power**: 1 - β (检验效力)

### 2.2 Maximum Likelihood Estimation (MLE)

**Likelihood function**:
```
L(θ : D) = P(D | θ) = Πₘ P(xₘ | θ)
```

**MLE**:
```
θ̂_MLE = arg max_θ L(θ : D)
```

**对于二项分布**:
```
L(θ : D) = θᵖ(1-θ)ᵍ  where p+q=n

dL/dθ = 0 ⟹ θ̂ = p/(p+q)
```

### 2.3 Neyman-Pearson Lemma (简单假设最优检验)

For simple hypotheses H₀: θ = θ₀ vs H₁: θ = θ₁:

**Likelihood ratio test**:
```
λ(x) = L(x, θ₀)/L(x, θ₁)

Decision:
  λ(x) > k  ⟹  Accept H₀
  λ(x) < k  ⟹  Reject H₀
  λ(x) = k  ⟹  Accept H₀ with probability ρ
```

k and ρ are determined by desired α level.

### 2.4 χ² Distribution (卡方分布)

**定义**: 若 Xᵢ ~ N(0, 1) i.i.d., 则:
```
Σᵢ₌₁ʳ Xᵢ² ~ χ²(r)
```

**Properties**:
- Mean = r (degrees of freedom)
- Variance = 2r
- For r > 100: χ²(r) ≈ N(r, 2r)

#### Corrected Variance Distribution

若 Xᵢ ~ N(μ, σ²), 则:
```
S² = Σᵢ(Xᵢ - X̄)² / (n-1)

(n-1)S²/σ² ~ χ²(n-1)
```

### 2.5 χ² Goodness-of-Fit Test (拟合优度检验)

**Test statistic**:
```
D²(n) = Σₗ₌₁ᵏ (Nₗ - n·pₗ)² / (n·pₗ)
```

where:
- Nₗ = observed count in class l
- n·pₗ = expected count under H₀
- k = number of classes

**Distribution**: D²(n) ~ χ²(k-1) when n → ∞

**Decision rule**: Reject H₀ if D² > χ²_{k-1,α}

### 2.6 χ² Independence Test (独立性检验)

**Contingency table** (列联表):

|  | B₁ | B₂ | ... | Bⱼ | Total |
|---|---|---|---|---|---|
| A₁ | n₁₁ | n₁₂ | ... | n₁ⱼ | n₁· |
| A₂ | n₂₁ | n₂₂ | ... | n₂ⱼ | n₂· |
| ... | ... | ... | ... | ... | ... |
| Aᵢ | nᵢ₁ | nᵢ₂ | ... | nᵢⱼ | nᵢ· |
| Total | n·₁ | n·₂ | ... | n·ⱼ | n |

**Under independence**:
```
E[nᵢⱼ] = (nᵢ· × n·ⱼ) / n
```

**Test statistic**:
```
χ² = ΣᵢΣⱼ (nᵢⱼ - nᵢ·n·ⱼ/n)² / (nᵢ·n·ⱼ/n)
```

**Distribution**: χ² ~ χ²((I-1)×(J-1))

### 2.7 Student's t-Distribution

**Definition**: For X ~ N(μ, σ²), sample size n:
```
T = (X̄ - μ)/(S/√n) ~ t(n-1)
```

**Properties**:
- E[T] = 0
- Var(T) = n/(n-2) for n > 2
- As n → ∞, t(n) → N(0,1)

#### Confidence Interval for μ

**σ² known, large n**:
```
CI = [x̄ ± z_{α/2} × σ/√n]
```

**σ² unknown, X ~ N**:
```
CI = [x̄ ± t_{n-1,α/2} × s/√n]
```

### 2.8 Two-Sample t-Test (两样本比较)

**Setup**:
- Sample 1: n₁, X̄₁, s₁²
- Sample 2: n₂, X̄₂, s₂²

**H₀**: μ₁ = μ₂

**Pooled variance**:
```
s² = [(n₁-1)s₁² + (n₂-1)s₂²] / (n₁+n₂-2)
```

**Test statistic**:
```
t = (X̄₁ - X̄₂) / √[s²(1/n₁ + 1/n₂)] ~ t(n₁+n₂-2)
```

---

## Cours 3: Classification Non-paramétrique

### 3.1 K-means Clustering

#### Algorithm
```
1. Initialize k cluster centers (randomly or k-means++)
2. Repeat until convergence:
   a. Assign each point to nearest center
   b. Update centers as mean of assigned points
```

#### 关键推导

**Objective function** (Inertia):
```
I_G = Σₖ₌₁ᴷ Σ_{xᵢ∈Gₖ} d²(xᵢ, gₖ)
```

where gₖ = (1/|Gₖ|) Σ_{xᵢ∈Gₖ} xᵢ

**Minimization**:
```
∂Iₖ/∂gₖ = Σ_{xᵢ∈Gₖ} 2(xᵢ - gₖ) = 0

⟹ gₖ = (1/|Gₖ|) Σ_{xᵢ∈Gₖ} xᵢ
```

**Inter-cluster inertia** (要最大化):
```
I_X = Σₖ₌₁ᴷ |Gₖ| · d²(gₖ, g)
```

where g = (1/n) Σᵢ xᵢ (global centroid)

#### Gradient Descent for K-means (ER2023题目)

**Loss function**:
```
ℓ = Σⱼ₌₁ᴷ Σ_{xᵢ∈Sⱼ} ‖xᵢ - cⱼ‖²
```

**Gradient**:
```
∂ℓ/∂cⱼ = Σ_{xᵢ∈Sⱼ} 2(cⱼ - xᵢ) = 2(|Sⱼ|·cⱼ - Σ_{xᵢ∈Sⱼ} xᵢ)
```

**Update rule**:
```
cⱼ^(t+1) = cⱼ^(t) - η·∂ℓ/∂cⱼ
         = cⱼ^(t) - 2η(|Sⱼ|·cⱼ^(t) - Σ_{xᵢ∈Sⱼ} xᵢ)
```

### 3.2 Hierarchical Clustering (分层聚类)

#### Linkage Criteria (距离度量)

**Single linkage** (最小距离):
```
D(G₁, G₂) = min_{x∈G₁, y∈G₂} d(x, y)
```

**Complete linkage** (最大距离):
```
D(G₁, G₂) = max_{x∈G₁, y∈G₂} d(x, y)
```

**Average linkage** (平均距离):
```
D(G₁, G₂) = (1/(|G₁|·|G₂|)) Σ_{x∈G₁, y∈G₂} d(x, y)
```

**Ward's method** (Ward方法):
```
D(G₁, G₂) = (|G₁|·|G₂|)/(|G₁|+|G₂|) · d(g₁, g₂)²
```

Ward方法最小化intra-cluster inertia increase.

### 3.3 k-Nearest Neighbors (k-NN)

#### Classification Rule
```
Ĉ(x) = (1/|V(x)|) Σ_{xᵢ∈V(x)} yᵢ
```

where V(x) = k nearest neighbors of x

#### 距离度量

**Euclidean (L₂)**:
```
d(a,b) = √(Σᵢ(aᵢ-bᵢ)²)
```

**Manhattan (L₁)**:
```
d(a,b) = Σᵢ|aᵢ-bᵢ|
```

**Minkowski (Lₚ)**:
```
d(a,b) = (Σᵢ|aᵢ-bᵢ|ᵖ)^(1/p)
```

#### Parzen Window (Kernel Density Estimation)

**General form**:
```
g(x) = Σᵢ₌₁ⁿ Φ(d(x,xᵢ)/h) · yᵢ
```

**Gaussian kernel**:
```
Φ(u) = (1/√(2π)) exp(-u²/2)
```

### 3.4 Decision Trees (决策树)

#### Impurity Measures (不纯度度量)

**Error rate**:
```
Error(N) = 1 - max_c P(c|N)
```

**Gini index**:
```
Gini(N) = 1 - Σ_c P(c|N)²
```

**Entropy** (Shannon entropy):
```
Entropy(N) = -Σ_c P(c|N)·log₂(P(c|N))
```

#### Information Gain (信息增益)

**定义**:
```
Δ(N, V) = I(N) - Σᵥ (|Rᵥ|/|R|)·I(Nᵥ)
```

where:
- N = parent node
- V = splitting variable
- Nᵥ = child node for value v
- I(·) = impurity measure

#### Gain Ratio (增益比率)

To avoid bias towards high-cardinality features:

```
Δ_Ratio(N, V) = Δ(N, V) / H(V)

where H(V) = -Σᵥ P(V=v)·log₂(P(V=v))
```

#### Entropy 理论基础

**Hartley information**:
```
H(n) = log₂(n) = -log₂(1/n)
```

**Shannon entropy** for p = (p₁, ..., pₙ):
```
h(p₁, ..., pₙ) = -Σᵢ pᵢ·log₂(pᵢ)
```

**Properties**:
- H(1) = 0 (no uncertainty)
- H(2) = 1 (one bit)
- H(n·m) = H(n) + H(m) (additivity)

#### MDL (Minimum Description Length)

```
MDL(T) = α·Size(T) + Σ_{f∈leaves(T)} I(f)
```

平衡树的复杂度和叶子节点的不纯度。

---

## Cours 5: Classification Probabiliste et Linéaire Binaire

### 5.1 Probabilistic Classification Framework

#### Bayes' Theorem
```
P(Y|X) = P(X|Y)·P(Y) / P(X)
```

#### Maximum A Posteriori (MAP)
```
y*_MAP = arg max_y P(y|x) = arg max_y P(x|y)·P(y)
```

#### Maximum Likelihood (ML)
```
y*_ML = arg max_y P(x|y)
```

### 5.2 Naive Bayes Classifier

#### 独立性假设
```
∀k≠l, Xₖ ⊥⊥ Xₗ | Y
```

即: P(X|Y) = Πₖ P(Xₖ|Y)

#### 分类规则
```
y* = arg max_y P(y) · Πₖ₌₁ᵈ P(xₖ|y)
```

#### Gaussian Naive Bayes

若 P(Xₖ|Y=y) ~ N(μₖ,y, σ²ₖ,y):

```
P(xₖ|y) = (1/√(2πσ²ₖ,y)) exp(-(xₖ-μₖ,y)²/(2σ²ₖ,y))
```

**参数估计**:
```
μₖ,y = (1/ny) Σ_{i:yᵢ=y} xᵢₖ

σ²ₖ,y = (1/ny) Σ_{i:yᵢ=y} (xᵢₖ - μₖ,y)²
```

where ny = |{i: yᵢ = y}|

### 5.3 Linear Binary Classification (CLB)

#### General Form
```
Ĉ(x) = σ(w'·x + w₀) = σ(Σᵢ wᵢxᵢ + w₀)
```

where σ is sign function:
```
σ(u) = {-1  if u < 0
        +1  if u ≥ 0
```

#### 几何解释

**Hyperplane equation**: w'·x + w₀ = 0

**Distance from x to hyperplane**:
```
r = |w'·x + w₀| / ‖w‖
```

**Normal vector**: w (perpendicular to hyperplane)

### 5.4 Logistic Regression

#### Logit Function
```
logit(p) = log(p/(1-p)) = log(P(⊕|x)/P(⊖|x))
```

#### Model
```
log(P(⊕|x)/P(⊖|x)) = w'·x + w₀

⟹ P(⊕|x) = exp(w'·x+w₀)/(1+exp(w'·x+w₀))

⟹ P(⊖|x) = 1/(1+exp(w'·x+w₀))
```

#### Log-Likelihood

For dataset (X, Y) with yᵢ ∈ {0,1}:

```
LL(β⁺) = Σᵢ [yᵢ·(β'·xᵢ⁺) - log(1 + exp(β'·xᵢ⁺))]
```

where β⁺ = (β, β₀), xᵢ⁺ = (xᵢ, 1)

#### Gradient
```
∂LL/∂β⁺ = Σᵢ xᵢ·(yᵢ - p(xᵢ; β⁺))
```

where p(xᵢ; β⁺) = exp(β'·xᵢ⁺)/(1+exp(β'·xᵢ⁺))

#### Newton-Raphson Update
```
β⁺_{t+1} = β⁺_t - [∂²LL/∂β⁺∂β⁺']⁻¹ · ∂LL/∂β⁺
```

### 5.5 Fisher's Linear Discriminant

#### Between-class separation
```
Δw = w'·(M⊕ - M⊖)
```

where Mₖ = (1/nₖ) Σ_{i∈k} xᵢ

#### Within-class variance
```
sₖ = Σ_{i∈k} (yᵢ - w'·Mₖ)²
```

where yᵢ = w'·xᵢ

**Objective**: Maximize Δw² / (s⊕ + s⊖)

### 5.6 Gaussian Discriminant Analysis

若 P(x|c) ~ N(μc, Σc):

**Density**:
```
p(x|c) = (1/((2π)^(d/2)|Σc|^(1/2))) exp(-½(x-μc)'Σc⁻¹(x-μc))
```

#### Linear Discriminant (Homoscedastic, Σ⊕ = Σ⊖ = Σ)

```
g(x) = (μ⊕ - μ⊖)'Σ⁻¹(x - x₀)
```

where:
```
x₀ = ½(μ⊕ + μ⊖) + [log(P(⊕)/P(⊖)) / ((μ⊕-μ⊖)'Σ⁻¹(μ⊕-μ⊖))] · (μ⊕ - μ⊖)
```

---

## Spectral Clustering (谱聚类)

### 基本概念

#### Graph Representation
- **G = (V, E)**: undirected graph
- **W**: weighted adjacency matrix (wᵢⱼ ≥ 0)
- **D**: degree matrix (diagonal, dᵢ = Σⱼ wᵢⱼ)

#### Similarity Graphs

**ε-neighborhood**:
```
wᵢⱼ = {1  if ‖xᵢ-xⱼ‖ < ε
       0  otherwise
```

**k-nearest neighbors**: Connect if xⱼ among k-NN of xᵢ

**Fully connected with Gaussian kernel**:
```
wᵢⱼ = exp(-‖xᵢ-xⱼ‖²/(2σ²))
```

### Graph Laplacian

#### Unnormalized Laplacian
```
L = D - W
```

**Properties**:
1. Symmetric and positive semi-definite
2. Smallest eigenvalue λ₁ = 0, eigenvector = 𝟙 (all ones)
3. For any f ∈ ℝⁿ:
   ```
   f'Lf = ½ Σᵢⱼ wᵢⱼ(fᵢ - fⱼ)²
   ```

#### Normalized Laplacian
```
L_sym = D⁻¹/²(D - W)D⁻¹/² = I - D⁻¹/²WD⁻¹/²
```

or:
```
L_rw = D⁻¹(D - W) = I - D⁻¹W
```

### Spectral Clustering Algorithm

**Unnormalized version**:
```
1. Compute adjacency matrix W
2. Compute Laplacian L = D - W
3. Compute first k eigenvectors v₁,...,vₖ of L
4. Form matrix V = [v₁ | ... | vₖ] ∈ ℝⁿˣᵏ
5. Treat each row of V as point yᵢ ∈ ℝᵏ
6. Run k-means on {y₁,...,yₙ}
```

**Normalized version (Ng-Jordan-Weiss)**:
```
1. Compute adjacency matrix W
2. Compute normalized Laplacian L_sym
3. Compute first k eigenvectors v₁,...,vₖ of L_sym
4. Form matrix V ∈ ℝⁿˣᵏ
5. Normalize rows: uᵢⱼ = vᵢⱼ / (Σₖ vᵢₖ²)^(1/2)
6. Run k-means on rows of U
```

### Silhouette Score (轮廓系数)

**For point i**:
```
a(i) = (1/(nₐ-1)) Σ_{j∈Cₐ,j≠i} d(i,j)  (average within-cluster distance)

b(i) = min_{k≠a} (1/nₖ) Σ_{j∈Cₖ} d(i,j)  (distance to nearest cluster)

s(i) = (b(i) - a(i)) / max{a(i), b(i)}
```

**Interpretation**:
- s(i) → +1: well clustered
- s(i) ≈ 0: on cluster boundary
- s(i) → -1: possibly misclassified

**Average for cluster k**:
```
s̄ₖ = (1/nₖ) Σ_{i∈Cₖ} s(i)
```

**Global average**:
```
Cₖ = (1/n) Σₖ nₖ·s̄ₖ
```

---

## 特殊主题: Canonical Correlation Analysis (CCA)

### 定义与目标

给定两组变量:
- **X = (X₁,...,Xₚ)**: 第一组变量
- **Y = (Y₁,...,Yᵧ)**: 第二组变量

**Goal**: Find linear combinations:
```
U = a'X
V = b'Y
```

that maximize correlation ρ = corr(U, V)

### Canonical Correlation

**First canonical correlation**:
```
ρ₁ = max_{a,b} corr(a'X, b'Y)
```

subject to Var(a'X) = Var(b'Y) = 1

**Subsequent canonical correlations**: orthogonal to previous ones

**Properties**:
- ρ ∈ [-1, 1]
- Number of canonical correlations = min(p, q)

### Interpretation

**Weights (a, b)**: indicate importance of each variable in the relationship

**Unsupervised method**: explores relationships between variable groups without class labels

---

## Support Vector Machines (SVM) - ER2022考点

### Linear SVM (Hard Margin)

#### Primal Problem
```
min_{w,w₀} ½‖w‖²

subject to: yᵢ(w'xᵢ + w₀) ≥ 1, ∀i
```

**Geometric interpretation**:
- Maximize margin 2/‖w‖
- Support vectors: points where yᵢ(w'xᵢ + w₀) = 1

#### Decision Boundary
```
Ĉ(x) = sign(w'x + w₀)
```

**Margin**:
```
M = 2/‖w‖
```

### Soft Margin SVM

**Primal with slack variables**:
```
min_{w,w₀,ξ} ½‖w‖² + C·Σᵢξᵢ

subject to:
  yᵢ(w'xᵢ + w₀) ≥ 1 - ξᵢ
  ξᵢ ≥ 0
```

**Parameter C**: trade-off between margin and misclassification

---

## Perceptron - ER2022考点

### Algorithm
```
Initialize: w⁽⁰⁾, w₀⁽⁰⁾
For each epoch:
  For each (xᵢ, yᵢ):
    if yᵢ(w'xᵢ + w₀) ≤ 0:  // misclassified
      w ← w + η·yᵢ·xᵢ
      w₀ ← w₀ + η·yᵢ
```

**Learning rate**: η (often set to 1)

**Convergence**: Guaranteed if data is linearly separable

### Example Calculation (ER2022)

Given:
- Decision boundary: y = 0.5
- x-axis: X₁, y-axis: X₂
- Error: ½

**Perceptron weights** for y = 0.5:
```
w'x + w₀ = 0
If boundary is y = 0.5: -w₂/w₁ = slope, -w₀/w₁ = intercept

For horizontal line y = 0.5:
w₁ = 0, w₂ = 1, w₀ = -0.5
```

**Update with (w₀,w₁,w₂) = (1,0,-2)**:
```
Point misclassified with ε = 1:
w ← w + ε·y·x
```

---

## 考试题型总结

### 1. 概念性问题
- 选择合适的分类器（概率估计 → probabilistic classifier）
- 判断分类器性能（imbalanced data → check baseline）
- 正则化效果（all coefficients = 0 → λ太大）

### 2. 推导题
- **k-means gradient descent**: ∂ℓ/∂cⱼ, update rule
- **Logistic regression**: likelihood, gradient, Newton-Raphson
- **Perceptron**: weight updates, convergence

### 3. 概率计算题
- **Naive Bayes**: P(X,Y), P(Y), P(X|Y), decision rules
- **Contingency tables**: probabilities, independence tests
- **Logistic regression**: coefficients from contingency table

### 4. 图论/聚类题
- **Spectral clustering**: adjacency matrix, degree matrix, Laplacian, eigenvectors
- **Graph properties**: connectivity, similarity measures

### 5. 决策树题
- **Entropy calculation**: H(N) = -Σ p·log₂(p)
- **Information gain**: Δ(N,V)
- **Optimal tree**: compare gains

---

## 重要公式速查

### Probability & Statistics
```
Bayes: P(Y|X) = P(X|Y)P(Y)/P(X)
Entropy: H(X) = -Σ p(x)log₂p(x)
χ² test: χ² = Σ(O-E)²/E
t-test: t = (x̄-μ)/(s/√n)
```

### Clustering
```
K-means objective: Σₖ Σ_{x∈Cₖ} ‖x-μₖ‖²
Silhouette: s(i) = (b(i)-a(i))/max{a(i),b(i)}
Laplacian: L = D - W
```

### Classification
```
Logistic: P(y=1|x) = 1/(1+exp(-w'x))
Perceptron update: w ← w + η·y·x (if error)
SVM margin: M = 2/‖w‖
```

### Evaluation
```
Accuracy: (TP+TN)/(TP+TN+FP+FN)
Precision: TP/(TP+FP)
Recall: TP/(TP+FN)
F1: 2·Precision·Recall/(Precision+Recall)
```

---

## 学习建议

1. **掌握推导**: 特别是logistic regression, k-means gradient, information gain
2. **理解概念**: Naive Bayes假设, spectral clustering原理, SVM margin
3. **熟练计算**: 概率表, 混淆矩阵, entropy, χ²统计量
4. **实践应用**: 知道何时用哪个算法，理解trade-offs
5. **复习ER题**: 两份ER覆盖了大部分考点

祝考试顺利！
