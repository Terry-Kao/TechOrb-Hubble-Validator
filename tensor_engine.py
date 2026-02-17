import sympy
from sympy import symbols, diag, diff, simplify, Function

# 定義坐標與變數
t, chi, theta, phi = symbols('t chi theta phi')
a = Function('a')(t)
P = Function('P')(chi)  # 放射散射投影函數

# 定義度規 g_munu
# ds^2 = -dt^2 + a(t)^2 * P(chi)^2 * (dchi^2 + chi^2*dOmega^2)
g = diag(-1, a**2 * P**2, a**2 * P**2 * chi**2, a**2 * P**2 * chi**2 * sympy.sin(theta)**2)

# 計算逆度規 g^munu
ginv = g.inv()

# --- 計算克氏符號 (Christoffel Symbols) ---
def get_christoffel(i, j, k):
    res = 0
    coords = [t, chi, theta, phi]
    for m in range(4):
        term = 0.5 * ginv[i, m] * (diff(g[m, j], coords[k]) + diff(g[m, k], coords[j]) - diff(g[j, k], coords[m]))
        res += term
    return simplify(res)

print("正在計算核心張量組件...")
# 這裡僅演示 G_00 (能量密度項) 的推導邏輯
# 完整的 Ricci 與 Einstein 張量將以此類推產出
