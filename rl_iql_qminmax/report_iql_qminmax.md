# 1. Περιγραφή Προβλήματος

Στην παρούσα εργασία μελετάμε τη μάθηση ισορροπίας σε επαναλαμβανόμενο παιχνίδι δύο παικτών. Συγκεκριμένα, εξετάζουμε:

- Επαναλαμβανόμενο **Prisoner’s Dilemma**
- Μετατροπή του σε **zero-sum μορφή μέσω reward shaping**
- Σύγκριση δύο multi-agent RL προσεγγίσεων:
  - Independent Q-Learning (IQL)
  - Minimax Q-Learning

## Περιβάλλον

- 2 παίκτες
- 2 δράσεις:
  - **C (Cooperate)**
  - **D (Defect)**
- Horizon: 200 βήματα ανά episode
- Episodes: 50,000
- Κατάσταση: τελευταία κοινή δράση

Η κατάσταση ορίζεται ως:

$$
s_t = (a_A^{t-1}, a_B^{t-1})
$$

Άρα υπάρχουν συνολικά 4 states.

## Αρχικό Payoff Matrix

|        | C        | D        |
|--------|----------|----------|
| **C**  | (3,3)    | (0,5)    |
| **D**  | (5,0)    | (1,1)    |

---

## Zero-Sum Reward Shaping

Για να εφαρμοστεί Minimax Q-learning, μετατρέπουμε το παιχνίδι σε zero-sum:

$$
r_A^{zs} = r_A - r_B
$$

$$
r_B^{zs} = - r_A^{zs}
$$

Έτσι ισχύει:

$$
r_A^{zs} + r_B^{zs} = 0
$$

Το underlying game παραμένει Prisoner’s Dilemma, αλλά το optimization objective γίνεται ανταγωνιστικό.

---

# 2. RL Αλγόριθμοι

## 2.1 Independent Q-Learning (IQL)

Κάθε agent:

- Αγνοεί ότι ο αντίπαλος μαθαίνει
- Μαθαίνει $Q(s,a)$
- Χρησιμοποιεί $\varepsilon$-greedy exploration

Update rule:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Big( r + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big)
$$

### Παράμετροι

- $\alpha = 0.1$
- $\gamma = 0.95$
- $\varepsilon$: γραμμική μείωση από 1.0 σε 0.03 σε 20,000 episodes

---

## 2.2 Minimax Q-Learning

Σε zero-sum setting, ο agent μαθαίνει:

$$
Q(s, a_{self}, a_{opp})
$$

Η state value υπολογίζεται ως:

$$
V(s) = \max_a \min_{a'} Q(s,a,a')
$$

Policy:

- Επιλέγει δράση που μεγιστοποιεί το ελάχιστο payoff.

Update rule:

$$
Q(s,a_i,a_j) \leftarrow Q(s,a_i,a_j) + \alpha \Big( r + \gamma V(s') - Q(s,a_i,a_j) \Big)
$$

Ο αλγόριθμος προσεγγίζει το minimax equilibrium.

---

# 3. Metrics Αξιολόγησης

## 3.1 Rolling Joint-Action Occupancy

Μετράται το ποσοστό εμφάνισης των:

- (C,C)
- (D,D)
- (C,D)
- (D,C)

Με rolling window 200 episodes.

---

## 3.2 Rolling Reward Variance

$$
\mathrm{Var}(R_A), \quad \mathrm{Var}(R_B)
$$

Χαμηλή διακύμανση → σταθεροποίηση στρατηγικής.

---

## 3.3 L1 Occupancy Change

$$
L1_t = \sum_i \left| p_t(i) - p_{t-1}(i) \right|
$$

---

## 3.4 Rolling Episode Return

Μετράται το shaped zero-sum reward:

$$
E[R_A] = -E[R_B]
$$

Σε equilibrium:

$$
E[R_A] = E[R_B] = 0
$$

---

# 4. Ανάλυση Γραφημάτων

## 4.1 Joint-Action Occupancy

Παρατηρείται:

- Το (D,D) αυξάνεται μονοτονικά
- Συγκλίνει ≈ 1 μετά τα ~20,000 episodes
- Οι άλλες καταστάσεις → 0

Ερμηνεία:

Το σύστημα συγκλίνει στο ασφαλές minimax αποτέλεσμα (D,D).

Ο Minimax learner επιβάλλει στρατηγική προστασίας.  
Ο IQL προσαρμόζεται.

---

## 4.2 Reward Variance

- Υψηλή διακύμανση στην αρχή (exploration)
- Απότομη πτώση μετά ~20k
- Σταθεροποίηση

Δείχνει policy stabilization.

---

## 4.3 L1 Occupancy Change

- Υψηλό spike αρχικά
- Σχεδόν μηδενικό μετά ~20k

Άρα:

$$
\text{Convergence time} \approx 20,000 \text{ episodes}
$$

---

## 4.4 Rolling Episode Return

- Μεγάλες διακυμάνσεις στην αρχή
- Μετά τη σύγκλιση → returns ≈ 0

Αναμενόμενο σε zero-sum equilibrium.

---

# 5. Συμπεράσματα

- Το σύστημα συγκλίνει καθαρά.
- Η σύγκλιση συμβαίνει περίπου στα 20,000 episodes.
- Η τελική στρατηγική είναι (D,D).
- Το Minimax Q-learning οδηγεί σε ασφαλές equilibrium.
- Ο Independent Q-Learner προσαρμόζεται.
- Το zero-sum shaping μετατρέπει ένα social dilemma σε καθαρά ανταγωνιστικό παιχνίδι.

---
