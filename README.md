# Missing Data Preprocessing Ablation

## 1. 실험 목적

Missing data 처리에서 다음 세 가지의 영향을 분리해서 비교했다.

1. **Missing 비율이 높은 feature를 제거할 것인가?**
2. **NaN을 어떤 값으로 imputation할 것인가?**
3. **Imputation 대신 LightGBM의 native missing 처리를 사용하는 것이 좋은가?**

평가 지표는 **AUC ↑ / LogLoss ↓**를 사용했다.

---

## 2. Column Filtering 영향

Missing rate가 특정 threshold보다 높은 feature를 제거했다.

$$
\text{MissingRate}(x_j)
=
\frac{\#\{x_{ij}\text{ is NaN}\}}{N}
$$

### Logistic Regression + Median

| Drop threshold | Features |        AUC |    LogLoss |
| -------------- | -------: | ---------: | ---------: |
| None           |       27 | **0.8939** | **0.3852** |
| > 90%          |       27 |     0.8939 |     0.3852 |
| > 70%          |       26 |     0.8925 |     0.3874 |
| > 50%          |       24 |     0.8748 |     0.4236 |
| > 30%          |       20 |     0.8762 |     0.4425 |

**해석**

* Logistic Regression에서는 feature를 제거하지 않는 것이 가장 좋았다.
* 70% threshold까지는 거의 차이가 없었다.
* 30~50% 수준에서 feature를 공격적으로 제거하면 오히려 성능이 하락했다.
* 따라서 **missing 비율이 높다는 이유만으로 feature를 제거하면 유용한 signal까지 잃을 수 있다.**

---

### MLP + Zero Imputation

| Drop threshold |        AUC |
| -------------- | ---------: |
| None           |     0.8503 |
| > 70%          |     0.8789 |
| **> 50%**      | **0.8816** |
| > 30%          |     0.8639 |

**해석**

MLP에서는 적당한 filtering이 효과가 있었다.

$$
0.8503 \rightarrow \mathbf{0.8816}
$$

하지만 30%까지 threshold를 낮춰 너무 많은 feature를 제거하면 다시 성능이 감소했다.

즉,

> **적당한 sparse-feature 제거는 도움이 될 수 있지만, 너무 공격적인 제거는 정보 손실을 발생시킨다.**

---

### LightGBM + Native NaN

| Drop threshold | Features |        AUC |    LogLoss |
| -------------- | -------: | ---------: | ---------: |
| None           |       27 |     0.8884 |     0.4006 |
| > 90%          |       27 |     0.8884 |     0.4006 |
| > 70%          |       26 |     0.8884 |     0.3941 |
| > 50%          |       24 |     0.8857 |     0.4062 |
| **> 30%**      |   **20** | **0.9252** | **0.3198** |

LightGBM은 NaN을 직접 처리할 수 있지만, 이번 실험에서는 missing이 많은 feature까지 모두 유지하는 것이 최선은 아니었다. >30% missing인 7개 feature를 제거했을 때 가장 높은 성능을 기록했다.

$$
AUC:\;0.8884\rightarrow\mathbf{0.9252}
$$

$$
LogLoss:\;0.4006\rightarrow\mathbf{0.3198}
$$

따라서:

> **Native missing 처리를 지원한다고 해서 매우 sparse한 feature까지 반드시 유지해야 하는 것은 아니다.**

---

## 3. Imputation 방법의 영향

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

### Median

$$
x_{\text{missing}}
\leftarrow
\operatorname{median}(X_{\text{train}})
$$

가장 좋은 결과를 기록했다.

Median은 mean보다 extreme value의 영향을 덜 받기 때문에 skewed feature에서 안정적일 수 있다.

### Mean

$$
x_{\text{missing}}
\leftarrow
\frac{1}{N}\sum_i x_i
$$

Median보다 AUC와 LogLoss 모두 조금 나빴다.

### Zero

$$
x_{\text{missing}}\leftarrow0
$$

Logistic Regression에서는 성능이 상당히 떨어졌다.

반면 MLP에서는 `drop >50% + zero`가 MLP 중 가장 좋았다.

따라서 **동일한 imputation이라도 모델에 따라 결과가 달라질 수 있다.**

---

## 4. Missing Indicator 영향

Median imputation과 함께 다음 binary feature를 추가했다.

$$
m_j =
\begin{cases}
1 & x_j \text{ was missing}\\
0 & \text{otherwise}
\end{cases}
$$

즉,

```text
age = NaN

↓

age = median(age)
age_missing = 1
```

### Logistic Regression

```text
Median
27 features
AUC = 0.8939

Median + Missing Indicator
54 features
AUC = 0.8435
```

Missing indicator를 추가했지만 오히려 성능이 크게 감소했다.

따라서:

> **Missing 여부 자체를 feature로 추가한다고 항상 성능이 향상되는 것은 아니다.**

---

## 5. Native Missing Handling 영향

LightGBM은 NaN을 미리 숫자로 바꾸지 않고 직접 처리할 수 있다.

```text
Logistic / MLP

NaN
 ↓
Mean / Median / Zero
 ↓
Model
```

반면:

```text
LightGBM

NaN
 ↓
Native missing handling
 ↓
Tree
```

전체 최고 결과는:

| Model        | Column filtering | Missing 처리     |        AUC |    LogLoss |
| ------------ | ---------------- | -------------- | ---------: | ---------: |
| **LightGBM** | **>30% drop**    | **Native NaN** | **0.9252** | **0.3198** |
| Logistic     | None             | Median         |     0.8939 |     0.3852 |
| MLP          | >50% drop        | Zero           |     0.8816 |     0.5184 |

LightGBM 최고 결과와 Logistic 최고 결과는 실제 실험에서도 각각 AUC 0.9252와 0.8939를 기록했다.

---

# 6. 영향도 요약

이번 실험 결과만 기준으로 보면:

### ① 모델 선택

가장 큰 차이를 만들었다.

```text
LightGBM + 최적 preprocessing    0.9252
Logistic + 최적 preprocessing   0.8939
MLP + 최적 preprocessing        0.8816
```

### ② Column filtering

모델마다 영향이 달랐다.

```text
LightGBM
0.8884 → 0.9252
큰 개선

MLP
0.8503 → 0.8816
개선

Logistic + median
0.8939 → 0.8939
제거할 필요 없음
```

즉 **고정된 missing-rate threshold를 모든 모델에 적용하면 안 된다.**

### ③ Imputation

Logistic에서:

```text
Median       0.8939
Mean         0.8871
Zero         0.8639
Median+Flag  0.8435
```

imputation 방법에 따라서도 상당한 차이가 발생했다.

### ④ Missing Indicator

이번 데이터에서는 효과가 없었고 오히려 악화됐다.

---

# 7. 최종 결론

이번 실험에서는 다음 전략이 가장 좋았다.

```text
Missing Data
     │
     ├── Missing-rate filtering
     │       │
     │       ├── Logistic → aggressive drop 불필요
     │       ├── MLP      → >50% drop best
     │       └── LightGBM → >30% drop best
     │
     ├── Imputation
     │       │
     │       ├── Logistic → Median best
     │       └── MLP      → Zero best
     │
     └── Native Missing
             │
             └── LightGBM
                   ↓
             Imputation 없이 NaN 처리
                   ↓
             전체 최고 AUC 0.9252
```

따라서 실무에서는 **`missing > 90%면 무조건 drop`, `NaN이면 무조건 median` 같은 고정 규칙을 사용하기보다는**, 다음을 validation ablation으로 결정하는 것이 적절하다.

$$
\boxed{
\text{Column threshold}
+
\text{Imputation strategy}
+
\text{Model's native missing support}
}
$$

특히 이번 실험은 **LightGBM의 native missing handling이 강력하지만, 매우 sparse한 column을 제거하는 feature filtering과 함께 사용할 때 더 좋은 결과가 나올 수 있음**을 보여준다.

> **주의:** Horse Colic은 작은 데이터셋이므로 30%/50%라는 threshold 자체를 일반적인 최적값으로 해석해서는 안 된다. 여러 random seed에서 mean ± std를 확인한 뒤 효과의 재현성을 판단해야 한다.
