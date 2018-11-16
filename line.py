

import numpy as np
import matplotlib.pyplot as plt

import stan_utils as stan


x, y, x_err, y_err = np.loadtxt("data.txt")

x_err, y_err = (np.sqrt(np.abs(x_err)), np.sqrt(np.abs(y_err)))

model = stan.load_stan_model("line.stan")

data_dict = dict(x=x, y=y, x_err=x_err, y_err=y_err, N=x.size)


# Don't really need an initialisation but fuck it.
DM = np.vstack((np.ones_like(x), x)).T
C = np.diag(y_err**2)
cov = np.linalg.inv(DM.T @ np.linalg.solve(C, DM))
c, m = cov @ (DM.T @ np.linalg.solve(C, y))

init_dict = dict(c=c, m=m, x_t=x)

# Optimize
p_opt = model.optimizing(data=data_dict, init=init_dict, iter=2000)

# Sample
samples = model.sampling(**stan.sampling_kwds(
    data=data_dict, chains=2, init=p_opt, iter=2000))

fig = samples.traceplot()
fig.savefig("trace.png")


fig, ax = plt.subplots()
ax.scatter(x, y, facecolor="k")
ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt="None", ecolor="k")

s = samples.extract(["c", "m", "x_t", "x_err_int"])
c_s, m_s, xt_s, x_err_int = s["c"], s["m"], s["x_t"], s["x_err_int"]


idx = np.random.choice(c_s.size, 100, replace=False)
for i in idx:

    xi = np.random.normal(xt_s[i], x_err_int[i])

    ax.plot(xi, c_s[i] + m_s[i] * xi, c="tab:blue", zorder=-1, alpha=0.1)

ax.plot(xi, xi * 5 + 2, c="r", lw=2)

fig.savefig("draws.png")


m_l, m, m_u = np.percentile(m_s, [16, 50, 84])
m_err_neg, m_err_pos = (m_l - m, m_u - m)

c_l, c, c_u = np.percentile(c_s, [16, 50, 84])
c_err_neg, c_err_pos = (c_l - c, c_u - c)

print(f"m = {m:.2f} ({m_err_pos:+.2f}, {m_err_neg:+.2f})")
print(f"c = {c:.2f} ({c_err_pos:+.2f}, {c_err_neg:+.2f})")


from corner import corner

Z = np.vstack([c_s, m_s]).T
fig = corner(Z, truths=[2, 5])

fig.savefig("corner.png")