# Missing Data Preprocessing Ablation

## 1. 실험 목적

Missing data 처리에서 다음 네 가지 질문을 실험적으로 확인했다.

1. **Missing 비율이 높은 feature를 제거할 것인가?**
2. **NaN을 어떤 값으로 imputation할 것인가?**
3. **Imputation 대신 LightGBM의 native missing 처리를 사용하는 것이 좋은가?**
4. **Missing 비율이 매우 높은 feature라도 predictive signal이 남아 있을 수 있는가?**

평가 지표는 **AUC ↑ / LogLoss ↓**를 사용했다.

---

# 2. Experiment 1: Missing Preprocessing Ablation

첫 번째 실험에서는 missing feature가 많은 **Horse Colic** 데이터를 이용해 column filtering, imputation, native missing handling을 비교했다.

## 2.1 Column Filtering 영향

각 feature의 missing rate를 다음과 같이 정의한다.

$$
\text{MissingRate}(x_j)
=
\frac{\#\{x_{ij}\text{ is NaN}\}}{N}
$$

그리고 missing rate가 특정 threshold를 초과하면 해당 feature를 제거했다.

---

### Logistic Regression + Median

| Drop threshold | Features |        AUC |    LogLoss |
| -------------- | -------: | ---------: | ---------: |
| None           |       27 | **0.8939** | **0.3852** |
| > 90%          |       27 |     0.8939 |     0.3852 |
| > 70%          |       26 |     0.8925 |     0.3874 |
| > 50%          |       24 |     0.8748 |     0.4236 |
| > 30%          |       20 |     0.8762 |     0.4425 |

**결과**

* Logistic Regression에서는 feature를 제거하지 않는 것이 가장 좋았다.
* 70% threshold까지는 거의 차이가 없었다.
* 30~50% 수준에서 feature를 공격적으로 제거하면 성능이 하락했다.

즉,

> **Missing 비율이 높다는 이유만으로 feature를 제거하면 유용한 signal까지 잃을 수 있다.**

---

### MLP + Zero Imputation

| Drop threshold |        AUC |
| -------------- | ---------: |
| None           |     0.8503 |
| > 70%          |     0.8789 |
| **> 50%**      | **0.8816** |
| > 30%          |     0.8639 |

MLP에서는 적당한 filtering이 효과가 있었다.

$$
0.8503 \rightarrow \mathbf{0.8816}
$$

하지만 30%까지 threshold를 낮춰 너무 많은 feature를 제거하면 다시 성능이 감소했다.

> **적당한 sparse-feature 제거는 도움이 될 수 있지만, 너무 공격적인 제거는 정보 손실을 발생시킬 수 있다.**

---

### LightGBM + Native NaN

| Drop threshold | Features |        AUC |    LogLoss |
| -------------- | -------: | ---------: | ---------: |
| None           |       27 |     0.8884 |     0.4006 |
| > 90%          |       27 |     0.8884 |     0.4006 |
| > 70%          |       26 |     0.8884 |     0.3941 |
| > 50%          |       24 |     0.8857 |     0.4062 |
| **> 30%**      |   **20** | **0.9252** | **0.3198** |

LightGBM은 NaN을 직접 처리할 수 있지만, 이번 데이터에서는 missing이 많은 feature까지 모두 유지하는 것이 최선은 아니었다.

> 30% missing인 7개 feature를 제거했을 때 가장 높은 성능을 기록했다.

$$
AUC:\;0.8884\rightarrow\mathbf{0.9252}
$$

$$
LogLoss:\;0.4006\rightarrow\mathbf{0.3198}
$$

따라서:

> **Native missing 처리를 지원한다고 해서 매우 sparse한 feature까지 반드시 유지해야 하는 것은 아니다.**

---

# 3. Imputation 방법의 영향

Logistic Regression에서 column을 제거하지 않고 imputation 방법만 비교했다.

| Imputation            |        AUC |    LogLoss |
| --------------------- | ---------: | ---------: |
| **Median**            | **0.8939** | **0.3852** |
| Mean                  |     0.8871 |     0.4241 |
| Zero                  |     0.8639 |     0.4694 |
| Median + Missing Flag |     0.8435 |     0.5237 |

이번 데이터에서는:

$$
\boxed{
Median > Mean > Zero > Median+Flag
}
$$

순으로 나타났다.

## Median

$$
x_{\text{missing}}
\leftarrow
\operatorname{median}(X_{\text{train}})
$$

가장 좋은 결과를 기록했다.

Median은 mean보다 extreme value의 영향을 덜 받기 때문에 skewed feature에서 안정적일 수 있다.

## Mean

$$
x_{\text{missing}}
\leftarrow
\frac{1}{N}\sum_i x_i
$$

Median보다 AUC와 LogLoss 모두 나빴다.

## Zero

$$
x_{\text{missing}}\leftarrow0
$$

Logistic Regression에서는 성능이 상당히 떨어졌다.

반면 MLP에서는 `drop >50% + zero`가 MLP 중 가장 좋았다.

따라서:

> **동일한 imputation 방법이라도 모델에 따라 효과가 달라질 수 있다.**

---

# 4. Missing Indicator 영향

Median imputation과 함께 다음 binary feature를 추가했다.

$$
m_j =
\begin{cases}
1 & x_j \text{ was missing}\\
0 & \text{otherwise}
\end{cases}
$$

예를 들어:

```text
age = NaN

↓

age = median(age)
age_missing = 1
```

Logistic Regression 결과:

```text
Median
27 features
AUC = 0.8939

Median + Missing Indicator
54 features
AUC = 0.8435
```

Missing indicator를 추가하면서 feature 수는 두 배가 되었지만 성능은 오히려 감소했다.

따라서:

> **Missing 여부 자체를 feature로 추가한다고 항상 성능이 향상되는 것은 아니다.**

---

# 5. Native Missing Handling 영향

Logistic Regression이나 MLP는 일반적으로 NaN을 직접 입력할 수 없기 때문에 먼저 값을 채워야 한다.

```text
Logistic / MLP

NaN
 ↓
Mean / Median / Zero
 ↓
Model
```

반면 LightGBM은:

```text
LightGBM

NaN
 ↓
Native missing handling
 ↓
Tree
```

처럼 NaN을 직접 처리할 수 있다.

전체 최고 결과는 다음과 같다.

| Model        | Column filtering | Missing 처리     |        AUC |    LogLoss |
| ------------ | ---------------- | -------------- | ---------: | ---------: |
| **LightGBM** | **>30% drop**    | **Native NaN** | **0.9252** | **0.3198** |
| Logistic     | None             | Median         |     0.8939 |     0.3852 |
| MLP          | >50% drop        | Zero           |     0.8816 |     0.5184 |

이번 Horse Colic 실험에서는:

$$
\boxed{
LightGBM + Sparse\ Feature\ Filtering + Native\ NaN
}
$$

조합이 가장 높은 성능을 기록했다.

---

# 6. Experiment 2: Sparse하지만 유용할 수 있는 Feature

Experiment 1에서는 sparse feature를 제거하는 것이 LightGBM 성능을 크게 향상시켰다.

그러나 여기서 중요한 질문이 생긴다.

> **Missing rate가 높다고 해서 그 feature가 정말 필요 없는 feature인가?**

이를 확인하기 위해 Titanic 데이터의 `deck` feature를 이용해 추가 실험했다.

`deck`은 Cabin 정보에서 파생된 feature이며 전체 데이터의 약 **77.2%가 missing**이었다.

```text
deck missing rate = 77.22%
```

즉 일반적인 missing-rate rule을 사용하면 제거 대상으로 분류될 수 있는 매우 sparse한 feature다.

---

## 6.1 실험 방법

다음 네 가지 방법을 비교했다.

| Variant         | 설명                                        |
| --------------- | ----------------------------------------- |
| `drop_sparse`   | deck 정보를 완전히 제거                           |
| `presence_only` | deck 값이 존재하는지 여부만 사용                      |
| `deck_only`     | 실제 deck category 사용, missing은 별도 category |
| `deck_presence` | deck category + presence indicator 모두 사용  |

단일 split의 우연을 줄이기 위해 **5개 random seed**에서 반복하고 mean ± std를 계산했다.

---

## 6.2 LightGBM 결과

| Variant       |   Mean AUC | Std AUC | Mean LogLoss |
| ------------- | ---------: | ------: | -----------: |
| **deck_only** | **0.8484** |  0.0211 |   **0.4513** |
| drop_sparse   |     0.8470 |  0.0195 |       0.4529 |
| presence_only |     0.8437 |  0.0272 |       0.4570 |
| deck_presence |     0.8427 |  0.0208 |       0.4594 |

5-seed 평균에서 `deck` category를 유지한 것이 가장 높은 AUC를 기록했다.

Sparse feature를 완전히 제거한 경우와 비교하면:

$$
AUC:
0.8470
\rightarrow
0.8484
$$

따라서:

$$
\Delta AUC \approx +0.0014
$$

이다.

즉 **77%가 missing인 feature라도 predictive signal이 완전히 없는 것은 아니었다.**

다만 개선 폭이 매우 작기 때문에 `deck`을 **매우 중요한 sparse feature**라고 해석하기에는 근거가 부족하다.

---

## 6.3 Logistic Regression 결과

| Variant         |   Mean AUC | Std AUC | Mean LogLoss |
| --------------- | ---------: | ------: | -----------: |
| **drop_sparse** | **0.8424** |  0.0194 |   **0.4565** |
| presence_only   |     0.8394 |  0.0198 |       0.4599 |
| deck_only       |     0.8374 |  0.0185 |       0.4641 |
| deck_presence   |     0.8368 |  0.0181 |       0.4651 |

Logistic Regression에서는 오히려 sparse feature를 완전히 제거하는 것이 가장 좋았다.

즉 같은 sparse feature라도:

```text
LightGBM
deck 유지 → 아주 작은 개선

Logistic
deck 유지 → 성능 하락
```

으로 모델에 따라 결과가 달라졌다.

---

# 7. Missing 여부 자체가 중요한가?

`presence_only`는 실제 deck 값을 사용하지 않고 다음 정보만 제공한다.

$$
deck\_known =
\begin{cases}
1 & deck\ exists\\
0 & deck\ missing
\end{cases}
$$

그러나 결과는:

### LightGBM

$$
0.8470_{\text{drop}}
\rightarrow
0.8437_{\text{presence}}
$$

### Logistic

$$
0.8424_{\text{drop}}
\rightarrow
0.8394_{\text{presence}}
$$

로 두 모델 모두 성능이 하락했다.

따라서 이번 Titanic 데이터에서는:

> **Missing이라는 사실 자체가 강한 추가 signal은 아니었다.**

또한 `deck + presence`를 동시에 사용하는 것도 두 모델 모두 개선되지 않았다.

---

# 8. Sparse Feature의 Seed별 영향

LightGBM에서는 일부 split에서 `deck`의 효과가 상대적으로 크게 나타났다.

예를 들어 seed 46:

```text
drop_sparse
AUC = 0.8403

deck_only
AUC = 0.8529

ΔAUC = +0.0126
```

반면 seed 42에서는:

```text
drop_sparse
AUC = 0.8333

deck_only
AUC = 0.8287
```

오히려 성능이 감소했다.

따라서 평균적으로는 약간의 signal이 존재했지만 split에 따른 variance도 상당했다.

---

# 9. 두 실험을 합친 핵심 결과

두 데이터셋은 서로 다른 결과를 보여준다.

## Horse Colic

LightGBM:

```text
모든 feature 유지
AUC = 0.8884

>30% missing feature 제거
AUC = 0.9252
```

즉:

$$
\Delta AUC = +0.0368
$$

**Sparse feature filtering이 크게 도움이 되었다.**

---

## Titanic

LightGBM:

```text
77% missing deck 제거
Mean AUC = 0.8470

77% missing deck 유지
Mean AUC = 0.8484
```

즉:

$$
\Delta AUC \approx +0.0014
$$

**Sparse feature를 유지했을 때 아주 작은 개선이 있었다.**

---

## 비교

```text
                  High Missing Feature
                          │
            ┌─────────────┴─────────────┐
            │                           │
       Horse Colic                  Titanic deck
            │                           │
    sparse feature 제거          sparse feature 유지
            │                           │
       성능 크게 ↑                  성능 약간 ↑
```

따라서 가장 중요한 결론은:

$$
\boxed{
\text{Missing Rate}
\neq
\text{Feature Importance}
}
$$

이다.

**Missing 비율은 feature를 제거할지 판단하기 위한 하나의 신호일 뿐, feature importance 자체를 의미하지 않는다.**

---

# 10. 영향도 종합

## ① Model

Horse Colic에서 최적 preprocessing 적용 시:

```text
LightGBM    0.9252
Logistic    0.8939
MLP         0.8816
```

모델 선택 자체가 큰 영향을 보였다.

---

## ② Column Filtering

모델과 데이터에 따라 효과가 달랐다.

```text
Horse Colic / LightGBM
0.8884 → 0.9252
큰 개선

Horse Colic / MLP
0.8503 → 0.8816
개선

Horse Colic / Logistic
0.8939 → 0.8939
제거할 필요 없음

Titanic / LightGBM
0.8470 → 0.8484
sparse deck 유지 시 아주 작은 개선
```

즉 **고정된 missing-rate threshold를 모든 데이터와 모델에 적용해서는 안 된다.**

---

## ③ Imputation

Horse Colic Logistic Regression:

```text
Median       0.8939
Mean         0.8871
Zero         0.8639
Median+Flag  0.8435
```

따라서 imputation 방법도 모델 성능에 상당한 영향을 줄 수 있다.

---

## ④ Missing Indicator

Horse Colic의 `median + missing flag`와 Titanic의 `deck presence` 모두 성능 향상을 보이지 않았다.

따라서:

> **Missing 여부 자체가 유용한 signal인지도 validation을 통해 확인해야 한다.**

---

## ⑤ Native Missing Handling

LightGBM은 NaN을 직접 처리할 수 있어 별도의 imputation이 필요하지 않았다.

하지만 Horse Colic에서는 native missing handling만 사용하는 것보다 sparse feature filtering까지 결합했을 때 더 높은 성능을 기록했다.

즉:

$$
\boxed{
Native\ Missing
\neq
Keep\ Every\ Sparse\ Feature
}
$$

이다.

---

# 11. 최종 결론

이번 실험에서 얻은 가장 중요한 결론은 **missing rate만으로 feature를 제거하면 안 된다는 것**이다.

```text
Missing Feature
      │
      ├── 1. Missing rate 확인
      │
      ├── 2. Sparse feature 제거 ablation
      │
      ├── 3. Imputation 방법 비교
      │       ├── Median
      │       ├── Mean
      │       ├── Zero
      │       └── Missing Indicator
      │
      ├── 4. Native missing 모델 비교
      │       └── LightGBM
      │
      └── 5. Validation에서 실제 predictive value 확인
```

따라서 실무에서는 다음과 같은 단순 규칙을 피하는 것이 좋다.

```text
missing > 90%
→ 무조건 drop              X

NaN
→ 무조건 median            X

LightGBM native missing
→ 모든 sparse feature 유지  X
```

대신 다음 조합을 validation으로 결정해야 한다.

$$
\boxed{
\text{Missing Rate}
+
\text{Feature Predictive Value}
+
\text{Imputation Strategy}
+
\text{Model Missing Handling}
}
$$

특히 두 실험을 함께 보면:

> **Sparse feature는 noise일 수도 있고, 값이 적더라도 predictive signal을 포함할 수도 있다. 따라서 missing rate는 feature filtering의 후보를 만드는 기준으로 사용하고, 최종 제거 여부는 validation ablation 또는 feature importance를 통해 결정하는 것이 적절하다.**

### 주의사항

* Horse Colic은 작은 데이터셋이므로 `30%`, `50%` threshold 자체를 일반적인 최적값으로 해석해서는 안 된다.
* Titanic `deck` 역시 77%가 missing이지만 평균 AUC 개선은 약 `+0.0014`에 불과하므로 **“매우 중요한 sparse feature”를 발견한 것은 아니다.**
* Titanic 실험에서 5개 seed의 결과가 흔들렸으므로 mean ± std를 함께 해석해야 한다.
* 따라서 실무에서 중요한 것은 특정 threshold를 외우는 것이 아니라 **데이터와 모델별로 missing preprocessing을 validation하는 과정 자체**이다.
