Absolutely. This is worth doing before we keep tuning the potential, because otherwise we're tweaking formulas without having the full model in our head.

I went back to the `config.py` and the latest `theory.py`. The `config.py` explicitly says it is the single source of truth for the game's physics, while your `theory.py` is only allowed to add belief features and shaping. 
$\frac{d}{f}$

# 1. Private type: (\theta_i=(\bar\rho_i,\bar m_i))

Each player has two hidden structural parameters:

[
\boxed{\theta_i=(\bar\rho_i,\bar m_i)}
]

### Resolve

[
\boxed{
\bar\rho_i\sim \operatorname{Beta}(\rho_a,\rho_b)
}
]

For Iran:

[
\bar\rho_I\sim Beta(5,2),
\qquad
E[\bar\rho_I]=\frac57\approx0.714.
]

For the U.S.:

[
\bar\rho_U\sim Beta(3,3),
\qquad
E[\bar\rho_U]=0.5.
]

This is why your current theory starts its opponent-resolve estimate from the prior mean. 

---

### Cost-management type

First draw

[
z_i\sim Beta(m_a,m_b),
]

then transform it:

[
\boxed{
\bar m_i
========

m_{\rm lo}
+
(m_{\rm hi}-m_{\rm lo})z_i.
}
]

So:

**Iran**

[
z_I\sim Beta(2,2),\qquad
\bar m_I=0.8+0.9z_I.
]

**U.S.**

[
z_U\sim Beta(3,2),\qquad
\bar m_U=0.9+0.9z_U.
]

The normalized version used throughout the code is

[
\boxed{
\hat m_i=
\frac{\bar m_i-m_{\rm lo}}
{m_{\rm hi}-m_{\rm lo}}
}
]

so approximately (\hat m_i\in[0,1]).

---

# 2. Initial endurance (e_i^0)

Once (\bar\rho_i) and (\hat m_i) are known:

[
\boxed{
e_i^0
=====

E_0
\left(
e_{0,\text{base}}
+
e_{0,\rho}\bar\rho_i
+
e_{0,m}\hat m_i
\right).
}
]

The common parameters are:

[
E_0=100,
\qquad
e_{0,\text{base}}=0.50,
\qquad
e_{0,\rho}=0.35,
\qquad
e_{0,m}=0.15.
]

Therefore:

[
e_i^0
=====

100(0.50+0.35\bar\rho_i+0.15\hat m_i).
]

So higher resolve and better cost management give greater initial endurance.

Your current opponent-endurance reconstruction uses exactly this relationship. 

---

# 3. Normalized remaining endurance

At any later time:

[
\boxed{
\hat e_i^t=\frac{e_i^t}{e_i^0}
}
]

with the implementation generally clipping it to a sensible range such as ([0,1]).

This is the quantity appearing in (m_i^t), (B_i), and the signaling cost.

---

# 4. Effective cost-management capacity (m_i^t)

The player's current ability to absorb costs is:

[
\boxed{
m_i^t
=====

\bar m_i
\left[
m_{\rm floor}
+
(1-m_{\rm floor})\hat e_i^t
\right].
}
]

Common:

[
m_{\rm floor}=0.35.
]

So:

* when (\hat e_i=1),

[
m_i^t=\bar m_i;
]

* when (\hat e_i=0),

[
m_i^t=0.35\bar m_i.
]

This is why depletion of endurance makes future costs worse. Your latest code uses this same formula when reconstructing the opponent's hidden state. 

---

# 5. Material / economic flow cost

The baseline material cost is affected by the opponent's current standing signal:

[
\boxed{
c_i^t
=====

c_{0,i}
+
\kappa_i\sigma_j.
}
]

Parameters:

### Iran

[
c_0=1.4,\qquad\kappa=3.0.
]

### U.S.

[
c_0=1.8,\qquad\kappa=2.5.
]

So if the opponent raises (\sigma_j), your material pressure rises.

Your latest code uses exactly this structure in `_own_flow_cost()`. 

---

# 6. Audience / commitment cost

Each player's accumulated commitment is:

[
\boxed{
K_i^{t+1}
=========

\delta_KK_i^t+\sigma_i^t
}
]

with

[
\boxed{\delta_K=0.75}.
]

So previous commitment decays by 25% each stage.

Then audience cost is:

[
\boxed{
a_i^{aud}=\alpha_iK_i^t.
}
]

Parameters:

### Iran

[
\alpha_I=0.8.
]

### U.S.

[
\alpha_U=1.3.
]

So the U.S. suffers more audience pressure from the same commitment level.

---

# 7. Escalation risk

The risk term is:

[
\boxed{
r_i^t
=====

\eta_i\sigma_i\sigma_j.
}
]

For both players:

[
\eta=2.5.
]

So if both players have high standing signals, risk becomes large.

Your current endurance reconstruction uses exactly

[
\eta\sigma_i\sigma_j.
]



---

# 8. Total effective flow cost (g_i^t)

Put the previous pieces together:

[
\boxed{
g_i^t
=====

\text{ramp}_t
\frac{
c_i^t+a_i^{aud,t}+r_i^t
}{
m_i^t
}.
}
]

The early-stage ramp is

[
\boxed{
\text{ramp}_t
=============

\min\left(1,\frac{t+1}{4}\right).
}
]

Therefore:

|     (t) | ramp |
| ------: | ---: |
|       0 | 0.25 |
|       1 | 0.50 |
|       2 | 0.75 |
| (t\ge3) |    1 |

Your current `_own_flow_cost()` implements this calculation. 

---

# 9. Endurance dynamics

After the current-stage cost is charged:

[
\boxed{
e_i^{t+1}
=========

e_i^t-g_i^t+\xi_i^t
}
]

where

[
\boxed{
\xi_i^t\sim N(0,0.5^2).
}
]

So even if two players have identical types and history, their endurance isn't perfectly deterministic.

This is important for the hidden-state inference problem.

---

# 10. Signaling cost (\phi_i)

When a player produces a new signal:

[
\boxed{
\phi_i^t
========

\frac{
\phi_0(\sigma_i^t)^2
}{
[w+(1-w)\bar\rho_i]
[w+(1-w)\hat e_i^t]
[v+(1-v)\hat m_i]
}.
}
]

Parameters:

[
\phi_0=1.2,
\qquad
w=0.40,
\qquad
v=0.50.
]

So:

* higher (\sigma) → much more expensive;
* higher resolve → cheaper signaling;
* higher endurance → cheaper signaling;
* better cost management → cheaper signaling.

And because of (\sigma^2), strong escalation becomes disproportionately expensive.

---

# 11. Value of outlasting (B_i)

The payoff from winning/outlasting is:

[
\boxed{
B_i
===

b_{0,i}
\left(
b_{\rm base}
+b_e\hat e_i
+b_\rho\bar\rho_i
\right).
}
]

Common coefficients:

[
b_{\rm base}=0.50,
\qquad
b_e=0.30,
\qquad
b_\rho=0.20.
]

Player-specific scale:

### Iran

[
b_{0,I}=175.
]

### U.S.

[
b_{0,U}=175.
]

So:

[
B_i
===

175(0.50+0.30\hat e_i+0.20\bar\rho_i).
]

Your potential calculates this from (e) and (\rho). 

---

# 12. Exit value (X_i)

This one is especially important for our potential.

[
\boxed{
X_i
===

## x_{0,i}

## \lambda_iK_i

h_i\sigma_j
+
w_{\rm med}M_t.
}
]

Parameters:

### Iran

[
x_{0,I}=26.
]

### U.S.

[
x_{0,U}=28.
]

Both:

[
\lambda=5,
\qquad
h=8,
\qquad
w_{\rm med}=20.
]

Thus your own commitment directly reduces your exit value:

[
\frac{\partial X_i}{\partial K_i}=-5.
]

This is exactly why your earlier objection about including (K_{\rm opp}) in the **own commitment penalty** was correct: (K_i), not (K_j), appears here. Your current code implements this exact exit-value formula. 

---

# 13. Mediation (M_t)

(M_t) is public and binary:

[
M_t\in{0,1}.
]

If mediation is **closed**:

[
P(M_{t+1}=1)
============

\begin{cases}
0.35,&\sigma_I+\sigma_U\le0.50,\
0.05,&\sigma_I+\sigma_U>0.50.
\end{cases}
]

If mediation is **open**:

[
P(M_{t+1}=0)=0.25.
]

Therefore calm behavior makes mediation much more likely.

Your current feature code explicitly computes the expected mediation outlook using these probabilities. 

---

# 14. Signals and action structure

Signals are not continuous in the implementation. They are:

[
\boxed{
\sigma\in{0,.25,.5,.75,1}.
}
]

A signal is considered **high** when:

[
\boxed{\sigma\ge0.75}.
]

That high signal can create a response turn.

The seven actions are essentially:

[
\begin{array}{c|c}
0& C,\sigma=0\
1& C,\sigma=.25\
2& C,\sigma=.5\
3& C,\sigma=.75\
4& C,\sigma=1\
5& EXIT\
6& HOLD
\end{array}
]

This is why your TFT detector looks specifically at response turns following high signals. 

---

# 15. Horizon

The game lasts at most:

[
\boxed{\bar T=40}.
]

At timeout, both players receive their respective exit value (X_i).

---

# 16. The complete dependency chain

This is probably the most useful way to remember the model:

[
\boxed{
(\bar\rho_i,\bar m_i)
\longrightarrow
e_i^0
}
]

then, at every stage,

[
\boxed{
e_i^t
\longrightarrow
\hat e_i^t
\longrightarrow
m_i^t
}
]

and

[
\boxed{
(\sigma_i,\sigma_j,K_i,e_i^t,\rho_i,m_i)
\longrightarrow
g_i^t
}
]

then:

[
\boxed{
e_i^{t+1}=e_i^t-g_i^t+\xi_i
}
]

while commitment evolves independently as

[
\boxed{
K_i^{t+1}=0.75K_i^t+\sigma_i.
}
]

Finally, the strategic terminal quantities are:

[
\boxed{
B_i
===

b_{0,i}(0.50+0.30\hat e_i+0.20\rho_i)
}
]

and

[
\boxed{
X_i=x_{0,i}-5K_i-8\sigma_j+20M.
}
]

So the central decision problem is essentially:

[
\boxed{
V_i^C(t,h^t)
\quad\text{vs.}\quad
X_i(t,h^t).
}
]

Your current theory explicitly uses (B), (X), and (g) in the potential, and its continuation feature has already been upgraded to the more sensible approximate form (p_{\rm win}(B-X)-g). 

---

## The parameters I would keep in our mental "core"

For improving `potential()`, you really only need to remember this:

[
\boxed{
\begin{aligned}
e^0 &= E_0(e_{base}+e_\rho\rho+e_m\hat m)\
m^t &= \bar m[m_{floor}+(1-m_{floor})\hat e]\
g^t &= \text{ramp}\frac{c_0+\kappa\sigma_j+\alpha K_i+\eta\sigma_i\sigma_j}{m^t}\
K^{t+1}&=.75K^t+\sigma^t\
B&=b_0(.5+.3\hat e+.2\rho)\
X&=x_0-5K-8\sigma_j+20M
\end{aligned}
}
]

Those six equations are basically **the economic engine of the game**.

And this gives us a useful insight for the next potential terms: **(K_i) should matter because it lowers (X_i) and raises (g_i); (K_j) should not simply be treated as our own "bad commitment."** That's exactly why your criticism of the current Part 4 was correct. 
