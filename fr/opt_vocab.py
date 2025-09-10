import argparse, pandas as pd, numpy as np, math
from typing import Tuple
import matplotlib.pyplot as plt

def _parse_sheet1(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    df = df.copy()
    cols = list(df.columns)
    if len(cols) >= 2:
        def _isnum(x):
            try:
                float(str(x).replace(',', '')); return True
            except: return False
        if _isnum(cols[0]) and _isnum(cols[1]):
            first = pd.DataFrame([{df.columns[0]: float(cols[0]), df.columns[1]: float(cols[1])}])
            df = pd.concat([first, df], ignore_index=True)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) >= 2:
        Kcol, Fcol = num_cols[0], num_cols[1]
    else:
        Kcol, Fcol = df.columns[:2]
    K = pd.to_numeric(df[Kcol], errors='coerce').to_numpy()
    f = pd.to_numeric(df[Fcol], errors='coerce').to_numpy()
    m = np.isfinite(K) & np.isfinite(f)
    K = K[m].astype(np.int64); f = np.clip(f[m].astype(np.float64), 0.0, 1.0)
    idx = np.argsort(K)
    K = K[idx]; f = f[idx]
    uniqK, uniq_idx = np.unique(K, return_index=True)
    return uniqK, f[uniq_idx]

def _parse_sheet2(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    df = df.copy()
    vals = None
    for _, row in df.iterrows():
        arr = pd.to_numeric(row, errors='coerce').to_numpy()
        if np.isfinite(arr).any():
            vals = arr; break
    if vals is None:
        raise RuntimeError("Sheet2 has no numeric row")
    Ks = []
    for c in df.columns:
        cs = str(c).strip().replace(',', '')
        try:
            Ks.append(int(float(cs)))
        except:
            Ks.append(np.nan)
    K = np.array(Ks, dtype=np.float64)
    mask = np.isfinite(K) & np.isfinite(vals)
    K = K[mask].astype(np.int64)
    CH = np.array(vals[mask], dtype=np.float64)
    idx = np.argsort(K)
    return K[idx], CH[idx]

def _parse_sheet3(df: pd.DataFrame):
    """
    Sheet3: 列名=K，首个数值行=吞吐(tokens/s)；返回 (K3, TPS)（按K升序）
    """
    df = df.copy()
    vals = None
    for _, row in df.iterrows():
        arr = pd.to_numeric(row, errors='coerce').to_numpy()
        if np.isfinite(arr).any():
            vals = arr; break
    if vals is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    Ks = []
    for c in df.columns:
        cs = str(c).strip().replace(',', '')
        try:
            Ks.append(int(float(cs)))
        except:
            Ks.append(np.nan)
    K = np.array(Ks, dtype=np.float64)
    mask = np.isfinite(K) & np.isfinite(vals)
    K = K[mask].astype(np.int64)
    TPS = np.array(vals[mask], dtype=np.float64)
    idx = np.argsort(K)
    return K[idx], TPS[idx]

def fit_linear_CH(K: np.ndarray, CH: np.ndarray):
    """最小二乘拟合 C_H(K) = a + b K（均值中心化以改善数值条件）"""
    K = K.astype(np.float64); CH = CH.astype(np.float64)
    K_mean = K.mean()
    Kc = K - K_mean
    A = np.vstack([np.ones_like(Kc), Kc]).T
    coef, *_ = np.linalg.lstsq(A, CH, rcond=None)
    a_c, b = coef
    a = a_c - b * K_mean
    return float(a), float(b)

def interp_f(Kq: np.ndarray, K1: np.ndarray, f1: np.ndarray) -> np.ndarray:
    """对 f(K) 做分段线性插值；两端钳位"""
    return np.interp(Kq, K1, f1, left=f1[0], right=f1[-1])

def expected_cost(Kq: np.ndarray, a: float, b: float, fK: np.ndarray, CF: float) -> np.ndarray:
    """E[T](K) = CF + f(K) * (CH(K) - CF)"""
    CH = a + b * Kq
    return CF + fK * (CH - CF)

def save_plots_single_pages(prefix, K1, f1, K2, CH2, a, b, CF, Kq, fK, G, K_star):
    # 1) f(K)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(Kq, fK, linewidth=2)
    ax1.scatter(K1, f1, s=12)
    ax1.set_xlabel("K (hot vocabulary size)")
    ax1.set_ylabel("Hit rate f(K)")
    ax1.set_title("Hit-rate curve (CDF)")
    ax1.grid(True, linestyle="--", linewidth=0.5)
    out1 = f"{prefix}_hit_rate.pdf"
    fig1.savefig(out1, bbox_inches="tight")
    plt.close(fig1)

    # 2) C_H(K) fit
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    Kfit = np.linspace(min(K2.min(), Kq.min()), max(K2.max(), Kq.max()), 256)
    CHfit = a + b * Kfit
    ax2.plot(Kfit, CHfit, linewidth=2)
    ax2.scatter(K2, CH2, s=12)
    ax2.set_xlabel("K (hot vocabulary size)")
    ax2.set_ylabel("Hot sampling cost C_H(K)")
    ax2.set_title("Hot sampling cost: data & linear fit")
    ax2.grid(True, linestyle="--", linewidth=0.5)
    out2 = f"{prefix}_cost_fit.pdf"
    fig2.savefig(out2, bbox_inches="tight")
    plt.close(fig2)

    # 3) E[T](K)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(Kq, G, linewidth=2)
    ax3.axvline(K_star, linestyle="--", linewidth=1.5, color="red")
    ax3.set_xlabel("K (hot vocabulary size)")
    ax3.set_ylabel("Expected cost E[T](K)")
    ax3.set_title("Objective and optimal K*")
    ax3.grid(True, linestyle="--", linewidth=0.5)
    out3 = f"{prefix}_objective.pdf"
    fig3.savefig(out3, bbox_inches="tight")
    plt.close(fig3)
    return out1, out2, out3

def save_objective_vs_throughput(prefix, Kq, G, K_star, K3, TPS):
    """第四张：左轴 1/E[T](K)；右轴实测吞吐为散点（不连线，显著区分配色/样式）"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    eps = 1e-12
    invG = 1.0 / np.maximum(G, eps)
    # 理论代理：1/E[T](K)
    ax.plot(Kq, invG, linewidth=2, label="1 / E[T](K)")
    ax.axvline(K_star, linestyle="--", linewidth=1.5, color="red", label="K*")
    ax.set_xlabel("K (hot vocabulary size)")
    ax.set_ylabel("1 / Expected cost (proxy throughput)")
    ax.grid(True, linestyle="--", linewidth=0.5)

    # 实测吞吐：高区分度散点（橙色方块 + 黑描边）
    if K3.size > 0:
        order = np.argsort(K3)
        K3s = np.array(K3)[order]
        TPSs = np.array(TPS)[order]
        ax2 = ax.twinx()
        ax2.scatter(K3s, TPSs, s=40, marker='s',
                    color='tab:orange', edgecolors='black', linewidth=0.7,
                    alpha=0.95, label="Throughput (tokens/s)")
        ax2.set_ylabel("Throughput (tokens/s)")

        # 合并图例（含 1/E[T]、K*、Throughput）
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

    out = f"{prefix}_objective_vs_throughput.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to Excel file")
    ap.add_argument("--sheet1", default="Sheet1", help="Sheet name with hit-rate f(K)")
    ap.add_argument("--sheet2", default="Sheet2", help="Sheet name with hot cost C_H(K)")
    ap.add_argument("--sheet3", default="Sheet3", help="Sheet name with measured throughput (tokens/s)")
    ap.add_argument("--grid_step", type=int, default=1_000, help="Search step for K grid")
    ap.add_argument("--cf_mode", choices=["maxK", "medianK"], default="maxK",
                    help="How to estimate C_F: use CH at max K (default) or at median K")
    ap.add_argument("--plot_pdf_prefix", default=None, help="Prefix path for single-page PDFs")
    args = ap.parse_args()

    xls = pd.ExcelFile(args.excel)
    s1 = pd.read_excel(xls, sheet_name=args.sheet1)
    s2 = pd.read_excel(xls, sheet_name=args.sheet2)
    try:
        s3 = pd.read_excel(xls, sheet_name=args.sheet3)
    except Exception:
        s3 = None

    K1, f1 = _parse_sheet1(s1)
    K2, CH2 = _parse_sheet2(s2)
    a, b = fit_linear_CH(K2, CH2)

    # 估计 C_F（默认取最大 K 的热词成本近似）
    if args.cf_mode == "medianK":
        K_med = int(np.median(K2))
        CF = a + b * K_med
    else:
        K_max = int(K2.max())
        CF = a + b * K_max

    # 搜索区间（两张表交集；如需并集/自定义可加开关）
    K_min = max(int(K1.min()), int(K2.min()))
    K_maxv = min(int(K1.max()), int(K2.max()))
    if K_min >= K_maxv:
        K_min = int(min(K1.min(), K2.min()))
        K_maxv = int(max(K1.max(), K2.max()))
    step = max(1, int(args.grid_step))
    Kq = np.arange(K_min, K_maxv + 1, step, dtype=np.int64)

    fK = interp_f(Kq, K1, f1)
    G = expected_cost(Kq, a, b, fK, CF)
    i_star = int(np.argmin(G))
    K_star = int(Kq[i_star])
    G_star = float(G[i_star])

    print("=== Parsed ===")
    print(f"Sheet1 f(K): {len(K1)} points, K in [{K1.min()}..{K1.max()}]")
    print(f"Sheet2 C_H(K): {len(K2)} points, K in [{K2.min()}..{K2.max()}]")
    print("\n=== Fit C_H(K) = a + b*K ===")
    print(f"a = {a:.6g}, b = {b:.6g}")
    print(f"Estimated C_F = {CF:.6g}  (mode={args.cf_mode})")
    print("\n=== Search ===")
    print(f"Grid: K in [{K_min}..{K_maxv}] step={step}")
    print(f"Optimal K* = {K_star}, Expected cost = {G_star:.6g}")
    left = max(0, i_star-3); right = min(len(Kq), i_star+4)
    for i in range(left, right):
        mark = "*" if i==i_star else " "
        print(f"{mark} K={int(Kq[i])}  f={fK[i]:.4f}  CH={a + b*Kq[i]:.6g}  E[T]={G[i]:.6g}")

    K3 = np.array([], dtype=np.int64); TPS = np.array([], dtype=np.float64)
    if s3 is not None:
        try:
            K3, TPS = _parse_sheet3(s3)
            print(f"\nSheet3 throughput: {len(K3)} points, K in [{K3.min()}..{K3.max()}]")
        except Exception as e:
            print(f"\nSheet3 parse error: {e}")

    if args.plot_pdf_prefix:
        outs = save_plots_single_pages(args.plot_pdf_prefix, K1, f1, K2, CH2, a, b, CF, Kq, fK, G, K_star)
        joint = save_objective_vs_throughput(args.plot_pdf_prefix, Kq, G, K_star, K3, TPS)
        outs = list(outs) + [joint]
        print("\nSaved single-page PDFs:")
        for p in outs: print(" ", p)

if __name__ == "__main__":
    main()
