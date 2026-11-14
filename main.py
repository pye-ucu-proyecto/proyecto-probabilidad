#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caos determinista y pseudoaleatoriedad con el mapa logístico
Probabilidad y Estadística Aplicada - Universidad Católica del Uruguay

Programa principal. Al ejecutarlo (python main.py) se reproducen TODOS los
resultados, tablas y figuras del informe:
  - figuras  -> carpeta  salida/figuras/*.png
  - números  -> archivo  salida/resultados.json  (y un resumen por consola)

No se usa ninguna fuente de aleatoriedad: todas las semillas son
deterministas, por lo que el resultado es idéntico en cada ejecución.
Dependencias: numpy, scipy, matplotlib, mpmath (ver requirements.txt).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")  # backend sin ventana: sólo guarda archivos
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# Configuración general
# --------------------------------------------------------------------------
AQUI = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(AQUI, "salida")
FIGS = os.path.join(SALIDA, "figuras")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "font.family": "serif",
})
C1, C2, C3 = "#1f4e79", "#c0504d", "#4f9d5d"   # paleta sobria

RES = {}          # diccionario con todos los resultados numéricos


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, nombre))
    plt.close(fig)


# --------------------------------------------------------------------------
# Funciones básicas del mapa logístico
# --------------------------------------------------------------------------
def orbita(x0, r, n):
    """Devuelve x_0, x_1, ..., x_n del mapa logístico x_{n+1}=r x_n (1-x_n)."""
    x = np.empty(n + 1)
    x[0] = x0
    for i in range(n):
        x[i + 1] = r * x[i] * (1.0 - x[i])
    return x


def iterar_vec(x, r, n):
    """Aplica n veces el mapa a un arreglo (vectorizado, misma aritmética)."""
    x = np.array(x, dtype=float)
    for _ in range(n):
        x = r * x * (1.0 - x)
    return x


def lyapunov(r, x0, trans=1000, n=5000):
    """Exponente de Lyapunov: promedio de ln|f'(x_i)| = ln|r(1-2x_i)|.
    Acepta r escalar o arreglo."""
    r = np.asarray(r, dtype=float)
    x = iterar_vec(np.full_like(r, x0), r, trans)
    s = np.zeros_like(r)
    for _ in range(n):
        x = r * x * (1.0 - x)
        s += np.log(np.abs(r * (1.0 - 2.0 * x)) + 1e-300)
    return s / n


def detectar_periodo(r, x0, trans=5000, maxp=64, tol=1e-7):
    """Período del atractor (None si no se detecta período <= maxp)."""
    x = iterar_vec(np.array([x0]), r, trans)[0]
    tr = orbita(x, r, 4 * maxp)
    for p in range(1, maxp + 1):
        if np.max(np.abs(tr[p:] - tr[:-p])) < tol:
            return p
    return None


# ==========================================================================
# PARTE 1 - Estudio propedéutico
# ==========================================================================
X0 = 0.23432
print("PARTE 1: dinámica del mapa logístico")

# --- 1. Trayectorias para distintos regímenes (60 iteraciones) -----------
regimenes = [(0.8, "extinción"), (2.5, "punto fijo"), (3.2, "período 2"),
             (3.5, "período 4"), (3.7, "caos"), (3.83, "ventana período 3"),
             (3.9, "caos"), (4.0, "caos pleno")]
fig, axs = plt.subplots(4, 2, figsize=(7, 8), sharex=True)
for ax, (r, nom) in zip(axs.ravel(), regimenes):
    t = orbita(X0, r, 60)
    ax.plot(range(61), t, "o-", ms=2.5, lw=0.8, color=C1)
    ax.set_title(f"$r={r}$ ({nom})")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("$x_n$")
for ax in axs[-1]:
    ax.set_xlabel("iteración $n$")
guardar(fig, "fig1_trayectorias.png")

# --- 2. Comportamiento asintótico con varias condiciones iniciales -------
r_tabla = [0.8, 2.5, 3.2, 3.5, 3.7, 3.83, 3.9, 4.0]
x0_tabla = [0.05, 0.23432, 0.37, 0.61, 0.93]
filas = []
for r in r_tabla:
    periodos, medias, lyap = [], [], []
    for x0 in x0_tabla:
        periodos.append(detectar_periodo(r, x0))
        cola = orbita(iterar_vec(np.array([x0]), r, 2000)[0], r, 2000)
        medias.append(cola.mean())
        lyap.append(float(lyapunov(r, x0, trans=1000, n=20000)))
    filas.append({
        "r": r,
        "periodos": [p if p is not None else 0 for p in periodos],  # 0 = sin período
        "media_media": float(np.mean(medias)), "media_sd": float(np.std(medias)),
        "lyap_media": float(np.mean(lyap)), "lyap_sd": float(np.std(lyap)),
        "x0": x0_tabla,
    })
RES["asintotico"] = filas
for f_ in filas:
    print(f"  r={f_['r']:<5} períodos={f_['periodos']}  "
          f"<x>={f_['media_media']:.4f}  lambda={f_['lyap_media']:+.4f}")

# --- 3. Diagrama de bifurcación y exponente de Lyapunov ------------------
N_R = 1500                                   # >= 100 valores de r (consigna)
rs = np.linspace(2.5, 4.0, N_R)
x = iterar_vec(np.full(N_R, X0), rs, 1000)   # se descarta el transitorio
pts_r, pts_x = [], []
for _ in range(150):
    x = rs * x * (1.0 - x)
    pts_r.append(rs.copy()); pts_x.append(x.copy())
pts_r, pts_x = np.concatenate(pts_r), np.concatenate(pts_x)
lam = lyapunov(rs, X0, trans=1000, n=3000)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6.2), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.3]})
a1.plot(pts_r, pts_x, ",", color="k", alpha=0.5)
a1.axvline(3.56995, color=C2, lw=0.8, ls="--", label="$r^*\\approx 3{,}56995$")
a1.axvline(3.0, color=C3, lw=0.8, ls=":", label="$r=3$")
a1.set_ylabel("$x_n$ (atractor)"); a1.legend(loc="lower left")
a2.plot(rs, lam, lw=0.6, color=C1)
a2.axhline(0, color=C2, lw=0.8)
a2.set_ylabel("$\\lambda$"); a2.set_xlabel("$r$"); a2.set_ylim(-2, 1)
guardar(fig, "fig2_bifurcacion.png")

# --- 4. Ventanas periódicas posteriores a r* ------------------------------
rf = np.linspace(3.5699, 4.0, 4001)
per = np.array([detectar_periodo(r, X0, trans=3000, maxp=32, tol=1e-6)
                or 0 for r in rf])
lam_f = lyapunov(rf, X0, trans=2000, n=6000)
ventanas, i = [], 0
while i < len(rf):
    if per[i] > 0:
        j = i
        while j + 1 < len(rf) and per[j + 1] > 0:
            j += 1
        if rf[j] - rf[i] >= 0.0005:
            ventanas.append({"r_ini": float(rf[i]), "r_fin": float(rf[j]),
                             "periodo_min": int(per[i:j + 1].min())})
        i = j + 1
    else:
        i += 1
RES["ventanas"] = ventanas
print("  ventanas periódicas detectadas (r > r*):")
for v in ventanas:
    print(f"    [{v['r_ini']:.4f}, {v['r_fin']:.4f}]  período base {v['periodo_min']}")

# acercamiento a la ventana principal (período 3)
rz = np.linspace(3.80, 3.90, 1200)
x = iterar_vec(np.full(len(rz), X0), rz, 1500)
zr, zx = [], []
for _ in range(120):
    x = rz * x * (1.0 - x)
    zr.append(rz.copy()); zx.append(x.copy())
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(np.concatenate(zr), np.concatenate(zx), ",", color="k", alpha=0.5)
ax.axvline(1 + np.sqrt(8), color=C2, ls="--", lw=0.8,
           label="$1+\\sqrt{8}\\approx 3{,}8284$")
ax.set_xlabel("$r$"); ax.set_ylabel("$x_n$ (atractor)"); ax.legend(loc="upper left")
guardar(fig, "fig3_ventana.png")

# --- 5. Caso r = 4: solución explícita ------------------------------------
# Tanto la recurrencia como la fórmula x_n = sin^2(2^n theta_0) se calculan en
# doble precisión, y ambas acumulan error (el de redondeo se multiplica por 2^n).
# Por eso NO se comparan entre sí como si una fuera exacta: se usa como
# referencia la recurrencia calculada con 120 dígitos desde x0 = 0,23432 exacto.
import mpmath as mp
mp.mp.dps = 120
n_max = 60
num = orbita(X0, 4.0, n_max)                              # recurrencia, doble precisión
theta = np.arcsin(np.sqrt(X0))
ana = np.sin(2.0 ** np.arange(n_max + 1) * theta) ** 2    # fórmula, doble precisión
x0_hp = mp.mpf("0.23432")
ref = [x0_hp]
for _ in range(n_max):
    ref.append(4 * ref[-1] * (1 - ref[-1]))               # referencia (120 dígitos)
th_hp = mp.asin(mp.sqrt(x0_hp))
ana_hp = [mp.sin(2 ** n * th_hp) ** 2 for n in range(n_max + 1)]
ref_f = np.array([float(v) for v in ref])
err_num = np.array([float(abs(mp.mpf(float(num[n])) - ref[n])) for n in range(n_max + 1)])
err_ana = np.array([float(abs(mp.mpf(float(ana[n])) - ref[n])) for n in range(n_max + 1)])
dif_na = np.abs(num - ana)
primero = lambda e: int(np.argmax(e > 1e-3))
RES["r4_explicito"] = {
    "theta": float(theta),
    "tabla": [{"n": n, "ref": float(ref[n]), "num": float(num[n]), "ana": float(ana[n]),
               "err_num": float(err_num[n]), "err_ana": float(err_ana[n])}
              for n in (1, 10, 20, 30, 40, 50)],
    "n_err_num_1e3": primero(err_num), "n_err_ana_1e3": primero(err_ana),
    "n_dif_num_ana_1e3": primero(dif_na),
    "err_formula_hp_max": float(max(abs(ref[n] - ana_hp[n]) for n in range(n_max + 1))),
    "err_x0_double": float(abs(mp.mpf(X0) - x0_hp)),
}
print(f"  r=4: error > 1e-3 en n={RES['r4_explicito']['n_err_num_1e3']} (recurrencia double), "
      f"n={RES['r4_explicito']['n_err_ana_1e3']} (fórmula double); "
      f"fórmula en 120 dígitos vs referencia: {RES['r4_explicito']['err_formula_hp_max']:.1e}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 3.1))
a1.plot(range(51), ref_f[:51], "-", lw=1.0, color="k", label="referencia (120 dígitos)")
a1.plot(range(51), num[:51], "o", ms=3, color=C1, label="recurrencia (doble)")
a1.plot(range(51), ana[:51], "x", ms=3.5, color=C2, label="fórmula (doble)")
a1.set_xlabel("$n$"); a1.set_ylabel("$x_n$"); a1.legend(loc="lower left", fontsize=7)
a2.semilogy(range(n_max + 1), np.maximum(err_num, 1e-18), "o-", ms=2.5, lw=0.7, color=C1,
            label="recurrencia (doble)")
a2.semilogy(range(n_max + 1), np.maximum(err_ana, 1e-18), "x-", ms=3, lw=0.7, color=C2,
            label="fórmula (doble)")
a2.set_xlabel("$n$"); a2.set_ylabel("error absoluto vs referencia"); a2.legend(fontsize=7)
guardar(fig, "fig4_r4_explicito.png")

# ==========================================================================
# PARTE 2 - Experimento de sensibilidad (r = 4)
# ==========================================================================
print("PARTE 2: sensibilidad a las condiciones iniciales")
BASE = "314159265"                 # x0 = 0,314159265
X0S = float("0." + BASE)
UMBRAL, NIT = 0.1, 100
semillas = []
for k in range(1, 10):             # se cambia sólo la k-ésima cifra decimal
    d = int(BASE[k - 1])
    nuevo = d + 1 if d < 9 else d - 1
    s = BASE[:k - 1] + str(nuevo) + BASE[k:]
    semillas.append({"k": k, "txt": "0," + s, "val": float("0." + s)})

tray_x = orbita(X0S, 4.0, NIT)
tabla_sep, difs = [], []
for s in semillas:
    ty = orbita(s["val"], 4.0, NIT)
    dif = np.abs(tray_x - ty)
    difs.append(dif)
    sobre = np.where(dif > UMBRAL)[0]
    s["T"] = int(sobre[0]) if len(sobre) else None
    s["delta"] = abs(s["val"] - X0S)
    tabla_sep.append({"k": s["k"], "semilla": s["txt"], "delta": s["delta"], "T": s["T"]})
RES["separacion"] = tabla_sep
for t in tabla_sep:
    print(f"  cifra {t['k']}: delta={t['delta']:.1e}  primera iteración con |x-y|>0,1: {t['T']}")

fig, axs = plt.subplots(3, 3, figsize=(7.2, 6.4), sharex=True, sharey=True)
for ax, s in zip(axs.ravel(), semillas):
    ax.plot(range(NIT + 1), tray_x, lw=0.7, color=C1, label="$x_n$")
    ax.plot(range(NIT + 1), orbita(s["val"], 4.0, NIT), lw=0.7, color=C2,
            label="$y_n$")
    ax.set_title(f"cifra {s['k']}: $y_0$ = {s['txt']}", fontsize=7.5)
for ax in axs[-1]:
    ax.set_xlabel("$n$")
for ax in axs[:, 0]:
    ax.set_ylabel("valor")
axs[0, 0].legend(loc="lower left")
guardar(fig, "fig5_pares.png")

fig, ax = plt.subplots(figsize=(7, 3.8))
cmap = plt.cm.viridis(np.linspace(0, 0.9, 9))
for s, dif, c in zip(semillas, difs, cmap):
    m = dif > 0                               # sólo diferencias representables
    ax.semilogy(np.arange(NIT + 1)[m], dif[m], lw=0.8, color=c,
                label=f"$\\delta=10^{{-{s['k']}}}$")
ax.axhline(UMBRAL, color="k", ls="--", lw=0.8)
ax.text(NIT * 0.01, UMBRAL * 1.3, "umbral 0,1", fontsize=8, bbox=dict(fc="white", ec="none", pad=1))
ax.set_xlabel("$n$"); ax.set_ylabel("$|x_n-y_n|$")
ax.legend(ncol=3, loc="lower right", fontsize=7)
guardar(fig, "fig6_distancias.png")

# --- Complemento: tiempo de separación promedio sobre 500 semillas base ----
bases = 0.05 + 0.85 * (np.arange(500) + 0.5) / 500   # evita que base+delta > 1
ks = np.arange(1, 15)
Tmed, Tmedia = [], []
for k in ks:
    a = bases.copy(); b = bases + 10.0 ** (-k)
    T = np.full(len(a), np.nan)
    vivo = np.ones(len(a), bool)
    for n in range(1, 300):
        a = 4 * a * (1 - a); b = 4 * b * (1 - b)
        cruza = vivo & (np.abs(a - b) > UMBRAL)
        T[cruza] = n; vivo &= ~cruza
        if not vivo.any():
            break
    Tmedia.append(float(np.nanmean(T))); Tmed.append(float(np.nanmedian(T)))
pend, ordn, rr, _, _ = stats.linregress(ks, Tmedia)
RES["separacion_promedio"] = {"k": ks.tolist(), "T_media": Tmedia, "T_mediana": Tmed,
                              "pendiente": float(pend), "ordenada": float(ordn),
                              "pendiente_teorica": float(np.log2(10))}
print(f"  complemento: pendiente T vs k = {pend:.3f} (teórica log2(10)={np.log2(10):.3f})")

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(ks, Tmedia, "o-", ms=4, color=C1, label="promedio (500 semillas base)")
ax.plot([t["k"] for t in tabla_sep], [t["T"] for t in tabla_sep], "s", ms=5,
        color=C2, label="semilla 0,314159265")
ax.plot(ks, pend * ks + ordn, "--", color="k", lw=0.8,
        label=f"ajuste lineal (pendiente {pend:.2f})")
ax.set_xlabel("$k$ ($\\delta=10^{-k}$)"); ax.set_ylabel("$T$: primera iteración con $|x_n-y_n|>0{,}1$")
ax.legend()
guardar(fig, "fig7_tiempo_separacion.png")

# ==========================================================================
# PARTE 3 - Pseudoaleatoriedad
# ==========================================================================
print("PARTE 3: distribución y dependencia de x_1000")
N = 10000
# Semillas determinísticas y equiespaciadas en (0; 0,5). Se evita (0,5; 1)
# porque f(s) = f(1-s): semillas simétricas darían trayectorias idénticas.
semillas_A = 0.5 * np.arange(1, N + 1) / (N + 1)
xA = iterar_vec(semillas_A, 4.0, 1000)       # observación: x_1000 de cada semilla
RES["muestraA"] = {"N": N, "unicos": int(len(np.unique(xA))),
                   "ceros_o_unos": int(np.sum((xA <= 0) | (xA >= 1)))}
print(f"  valores distintos: {RES['muestraA']['unicos']} de {N}; en {{0,1}}: "
      f"{RES['muestraA']['ceros_o_unos']}")

PROBS = [0.10, 0.25, 0.50, 0.75, 0.90]


def resumen(x, nombre):
    """Media, varianza, cuantiles y teóricos (Uniforme y Beta(1/2,1/2))."""
    d = {"media": float(x.mean()), "var": float(x.var(ddof=1)),
         "cuantiles": [float(q) for q in np.quantile(x, PROBS)]}
    ks_u = stats.kstest(x, "uniform")
    ks_b = stats.kstest(x, "beta", args=(0.5, 0.5))
    d["ks_unif"] = [float(ks_u.statistic), float(ks_u.pvalue)]
    d["ks_beta"] = [float(ks_b.statistic), float(ks_b.pvalue)]
    return d


RES["x"] = resumen(xA, "x")
RES["teorico"] = {
    "unif": {"media": 0.5, "var": 1 / 12, "cuantiles": PROBS},
    "beta": {"media": 0.5, "var": 1 / 8,
             "cuantiles": [float(np.sin(np.pi * p / 2) ** 2) for p in PROBS]},
}
print(f"  x: media={RES['x']['media']:.4f} var={RES['x']['var']:.4f}  "
      f"KS-unif D={RES['x']['ks_unif'][0]:.4f} p={RES['x']['ks_unif'][1]:.2e}  "
      f"KS-beta D={RES['x']['ks_beta'][0]:.4f} p={RES['x']['ks_beta'][1]:.3f}")

# histograma + densidades
g = np.linspace(0.002, 0.998, 800)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.hist(xA, bins=50, range=(0, 1), density=True, color="#9db7d5", edgecolor="white",
        label="histograma de $x_{1000}$")
ax.plot(g, np.ones_like(g), "--", color=C2, lw=1.4, label="Uniforme(0,1)")
ax.plot(g, stats.beta.pdf(g, 0.5, 0.5), color="k", lw=1.2, label="Beta(1/2, 1/2)")
hmax = np.histogram(xA, bins=50, range=(0, 1), density=True)[0].max()
ax.set_ylim(0, 1.15 * hmax)          # que ninguna barra quede recortada
ax.set_xlabel("$x_{1000}$"); ax.set_ylabel("densidad")
ax.legend(loc="upper center")
guardar(fig, "fig8_hist_x.png")   # la densidad Beta diverge en 0 y 1: sale del marco

# función de distribución empírica
def fig_ecdf(x, nombre, fig_nombre, etiqueta, con_beta=True):
    xs = np.sort(x); Fn = np.arange(1, len(xs) + 1) / len(xs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 3.1))
    a1.step(xs, Fn, where="post", color=C1, lw=1.2, label="empírica")
    a1.plot([0, 1], [0, 1], "--", color=C2, label="Uniforme(0,1)")
    if con_beta:
        a1.plot(g, stats.beta.cdf(g, 0.5, 0.5), color="k", lw=0.9, label="Beta(1/2, 1/2)")
    a1.set_xlabel(etiqueta); a1.set_ylabel("$F(x)$"); a1.legend(loc="lower right")
    a2.plot(xs, Fn - xs, color=C2, lw=0.8, label="$F_n-F_{unif}$")
    if con_beta:
        a2.plot(xs, Fn - stats.beta.cdf(xs, 0.5, 0.5), color="k", lw=0.8,
                label="$F_n-F_{Beta}$")
    a2.axhline(0, color="gray", lw=0.6)
    a2.set_xlabel(etiqueta); a2.set_ylabel("diferencia"); a2.legend()
    guardar(fig, fig_nombre)


fig_ecdf(xA, "x", "fig9_ecdf_x.png", "$x_{1000}$")

# --- Dependencia -----------------------------------------------------------
def acf(x, m):
    xc = x - x.mean(); den = np.sum(xc * xc)
    return np.array([np.sum(xc[:-k] * xc[k:]) / den for k in range(1, m + 1)])


def ljung_box(x, m=20):
    n = len(x); rho = acf(x, m)
    Q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, m + 1)))
    return float(Q), float(stats.chi2.sf(Q, m))


def runs_test(x):
    """Test de rachas de Wald-Wolfowitz respecto de la mediana."""
    s = x[x != np.median(x)] > np.median(x)
    n1 = int(s.sum()); n2 = len(s) - n1
    R = 1 + int(np.sum(s[1:] != s[:-1]))
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    z = (R - mu) / np.sqrt(var)
    return R, float(mu), float(z), float(2 * stats.norm.sf(abs(z)))


def serial_chi2(x, k=10):
    """Chi-cuadrado de independencia entre (x_i, x_{i+1}), k x k celdas
    equiprobables en cada marginal (por rangos)."""
    rk = stats.rankdata(x, method="ordinal")
    b = np.minimum(((rk - 1) * k // len(x)).astype(int), k - 1)
    tab = np.zeros((k, k)); np.add.at(tab, (b[:-1], b[1:]), 1)
    chi2, p, dof, esp = stats.chi2_contingency(tab, correction=False)
    return float(chi2), float(p), int(dof), float(esp.min())


def dependencia(x):
    Q, pQ = ljung_box(x, 20)
    R, mu, z, pR = runs_test(x)
    c2, pC, dof, emin = serial_chi2(x, 10)
    return {"r1": float(acf(x, 1)[0]), "LB": [Q, pQ], "runs": [R, mu, z, pR],
            "chi2": [c2, pC, dof, emin]}


# Muestra de control B: iteraciones consecutivas de UNA sola trayectoria
xB = orbita(X0, 4.0, 1000 + N - 1)[1000:]
RES["dep_A"] = dependencia(xA)
RES["dep_B"] = dependencia(xB)
RES["dep_B"]["ks_beta"] = list(map(float, stats.kstest(xB, "beta", args=(0.5, 0.5))))
RES["dep_B"]["ks_unif"] = list(map(float, stats.kstest(xB, "uniform")))
for nom in ("A", "B"):
    d = RES["dep_" + nom]
    print(f"  muestra {nom}: r1={d['r1']:+.4f}  LjungBox p={d['LB'][1]:.3g}  "
          f"rachas p={d['runs'][3]:.3g}  chi2 serial p={d['chi2'][1]:.3g}")

fig, axs = plt.subplots(2, 2, figsize=(7, 6))
for col, (x, nom) in enumerate([(xA, "A"), (xB, "B")]):
    rho = acf(x, 30); cota = 1.96 / np.sqrt(len(x))
    ax = axs[0, col]
    ax.bar(range(1, 31), rho, color=C1, width=0.7)
    ax.axhline(cota, color=C2, ls="--", lw=0.8); ax.axhline(-cota, color=C2, ls="--", lw=0.8)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel("retardo $k$"); ax.set_ylabel("autocorrelación")
    ax.set_title("Muestra A (semillas distintas)" if nom == "A"
                 else "Muestra B (trayectoria única)")
    ax = axs[1, col]
    ax.plot(x[:-1], x[1:], ".", ms=1.2, color=C1, alpha=0.5)
    ax.set_xlabel("$x_i$"); ax.set_ylabel("$x_{i+1}$")
guardar(fig, "fig10_dependencia.png")

# --- Robustez: 40 conjuntos de semillas deterministas distintos ---------------
# El p-valor de un único conjunto de semillas es una realización puntual;
# se repite el experimento desplazando la grilla de semillas para ver si el
# resultado es sistemático.
pv_beta, pv_unif, medias = [], [], []
for j in range(40):
    sem = 0.5 * (np.arange(N) + (j + 0.5) / 40) / N
    xj = iterar_vec(sem, 4.0, 1000)
    pv_beta.append(float(stats.kstest(xj, "beta", args=(0.5, 0.5)).pvalue))
    pv_unif.append(float(stats.kstest(xj, "uniform").pvalue))
    medias.append(float(xj.mean()))
RES["robustez"] = {
    "n_conjuntos": 40, "p_beta_min": min(pv_beta), "p_beta_max": max(pv_beta),
    "rechazos_beta_5pct": int(np.sum(np.array(pv_beta) < 0.05)),
    "rechazos_unif_5pct": int(np.sum(np.array(pv_unif) < 0.05)),
    "media_de_medias": float(np.mean(medias)),
}
print(f"  robustez (40 conjuntos): rechazos Beta al 5% = "
      f"{RES['robustez']['rechazos_beta_5pct']}, rechazos Uniforme = "
      f"{RES['robustez']['rechazos_unif_5pct']}")

# ==========================================================================
# PARTE 4 - Uniformización
# ==========================================================================
print("PARTE 4: uniformización u = (2/pi) arcsin(sqrt(x))")
uA = (2.0 / np.pi) * np.arcsin(np.sqrt(xA))
uB = (2.0 / np.pi) * np.arcsin(np.sqrt(xB))
RES["u"] = resumen(uA, "u")
RES["u"]["dep"] = dependencia(uA)
RES["uB"] = resumen(uB, "uB")
RES["uB"]["dep"] = dependencia(uB)
print(f"  u: media={RES['u']['media']:.4f} var={RES['u']['var']:.4f} "
      f"KS-unif D={RES['u']['ks_unif'][0]:.4f} p={RES['u']['ks_unif'][1]:.3f}")

fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.hist(uA, bins=50, range=(0, 1), density=True, color="#9db7d5", edgecolor="white",
        label="histograma de $u$")
ax.plot([0, 1], [1, 1], "--", color=C2, lw=1.4, label="Uniforme(0,1)")
ax.set_ylim(0, 2); ax.set_xlabel("$u$"); ax.set_ylabel("densidad"); ax.legend()
guardar(fig, "fig11_hist_u.png")
fig_ecdf(uA, "u", "fig12_ecdf_u.png", "$u$", con_beta=False)

with open(os.path.join(SALIDA, "resultados.json"), "w", encoding="utf-8") as fh:
    json.dump(RES, fh, ensure_ascii=False, indent=1)
print(f"\nListo. Figuras en {FIGS}\nResultados numéricos en {SALIDA}/resultados.json")
