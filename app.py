"""
Gauss-Seidel Solver Web Application
Integrates the original Python solver with the redesigned premium HTML interface.
"""

from flask import Flask, render_template_string, request, jsonify
import re
import json
import sqlite3
from datetime import datetime
from math import isclose

app = Flask(__name__)

DB_PATH = 'solver_history.db'


# ────────────────────────────────────────────────
# Database helpers
# ────────────────────────────────────────────────

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS solve_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                mode TEXT NOT NULL,
                tol REAL NOT NULL,
                max_iter INTEGER NOT NULL,
                variables TEXT NOT NULL,
                input_payload TEXT NOT NULL,
                converged INTEGER NOT NULL,
                iter_count INTEGER NOT NULL,
                solution TEXT NOT NULL,
                diagnostics TEXT NOT NULL
            )
        ''')
        conn.commit()


def save_history_entry(entry: dict, result: dict) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            INSERT INTO solve_history (
                created_at, mode, tol, max_iter, variables, input_payload,
                converged, iter_count, solution, diagnostics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            entry['mode'], entry['tol'], entry['maxIter'],
            json.dumps(entry['variables']), json.dumps(entry['input']),
            int(result['converged']), int(result['iterCount']),
            json.dumps(result['solution']), json.dumps(result.get('diagnostics', {})),
        ))
        conn.commit()


def fetch_history(limit: int = 20) -> list:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute('''
            SELECT id, created_at, mode, tol, max_iter, variables, input_payload,
                   converged, iter_count, solution, diagnostics
            FROM solve_history ORDER BY id DESC LIMIT ?
        ''', (limit,)).fetchall()
    return [{
        'id': r['id'], 'createdAt': r['created_at'], 'mode': r['mode'],
        'tol': r['tol'], 'maxIter': r['max_iter'],
        'variables': json.loads(r['variables']), 'input': json.loads(r['input_payload']),
        'converged': bool(r['converged']), 'iterCount': r['iter_count'],
        'solution': json.loads(r['solution']), 'diagnostics': json.loads(r['diagnostics']),
    } for r in rows]


def clear_history() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('DELETE FROM solve_history')
        conn.commit()


init_db()


# ────────────────────────────────────────────────
# Solver core
# ────────────────────────────────────────────────

def parse_equation(eq: str, var_to_idx: dict) -> tuple:
    eq = eq.replace(" ", "")
    if "=" not in eq:
        raise ValueError("Equation must contain '='")
    left, right_str = eq.split("=", 1)
    try:
        rhs = float(right_str)
    except ValueError:
        raise ValueError(f"Right-hand side is not a number: {right_str}")

    coeffs = [0.0] * len(var_to_idx)
    pattern = r"([+-]?)(\d*\.?\d*)([a-zA-Z])|([+-])([a-zA-Z])"
    i = 0
    implicit_sign = "+"

    while i < len(left):
        m = re.match(pattern, left[i:])
        if not m:
            i += 1
            continue
        sign_str, coeff_str, var, sign2, var2 = m.groups()
        if var2:
            sign, coeff, var = sign2, 1.0, var2
        else:
            sign = sign_str if sign_str else implicit_sign
            coeff = float(coeff_str) if coeff_str and coeff_str != "." else 1.0
        if sign == "-":
            coeff = -coeff
        if var in var_to_idx:
            coeffs[var_to_idx[var]] = coeff
        else:
            raise ValueError(f"Unknown variable '{var}' in equation")
        i += m.end()
        implicit_sign = "+"

    return coeffs, rhs


def prepare_matrix(A, b):
    n = len(A)
    if len(b) != n:
        raise ValueError("Length of b must match number of rows in A")
    for row in A:
        if len(row) != n:
            raise ValueError("Matrix A must be square")
    return [[float(x) for x in row] for row in A], [float(x) for x in b]


def gauss_seidel(A, b, variables, tol=1e-4, max_iter=1000):
    n = len(A)
    x = [0.0] * n
    iterations_log = []

    for iteration in range(1, max_iter + 1):
        max_delta = 0.0
        details = []
        for i in range(n):
            old_xi = x[i]
            sum_other = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sum_other) / A[i][i]
            delta = abs(x[i] - old_xi)
            max_delta = max(max_delta, delta)
            details.append({'variable': variables[i], 'value': x[i], 'delta': delta})

        if iteration <= 10 or iteration % 20 == 0:
            iterations_log.append({'number': iteration, 'details': details, 'maxDelta': max_delta})

        if max_delta < tol:
            return {'converged': True, 'iterations': iterations_log, 'solution': x, 'iterCount': iteration}

    return {'converged': False, 'iterations': iterations_log, 'solution': x, 'iterCount': max_iter}


def analyze_system(A: list, b: list, x: list) -> dict:
    n = len(A)
    row_metrics, min_margin, strict_rows, weak_rows = [], float('inf'), 0, 0

    for i in range(n):
        diag = abs(A[i][i])
        off_diag = sum(abs(A[i][j]) for j in range(n) if j != i)
        margin = diag - off_diag
        ratio = (diag / off_diag) if off_diag > 0 else float('inf')
        min_margin = min(min_margin, margin)
        if diag > off_diag:
            strict_rows += 1; status = 'strict'
        elif isclose(diag, off_diag, rel_tol=1e-12, abs_tol=1e-12):
            weak_rows += 1; status = 'weak'
        else:
            status = 'not_dominant'
        row_metrics.append({'row': i+1, 'diagAbs': diag, 'offDiagAbs': off_diag,
                             'margin': margin, 'ratio': ratio, 'status': status})

    residual_vector = [sum(A[i][j] * x[j] for j in range(n)) - b[i] for i in range(n)]
    inf_norm = max(abs(r) for r in residual_vector) if residual_vector else 0.0

    if strict_rows == n:        stability = 'excellent'
    elif strict_rows + weak_rows == n: stability = 'fair'
    else:                       stability = 'risky'

    return {
        'rowMetrics': row_metrics, 'strictDominantRows': strict_rows,
        'weakDominantRows': weak_rows, 'minimumMargin': min_margin,
        'residualVector': residual_vector, 'residualInfinityNorm': inf_norm,
        'stability': stability,
    }


def build_domain_system(domain: str, params: dict) -> tuple:
    domain = (domain or '').strip().lower()

    if domain == 'electrical':
        r1,r2,r3,rm = float(params.get('r1',12)), float(params.get('r2',10)), float(params.get('r3',8)), float(params.get('rm',2))
        v1,v2,v3    = float(params.get('v1',24)), float(params.get('v2',18)), float(params.get('v3',12))
        variables = ['I1','I2','I3']
        A = [[r1+rm+1,-rm,0],[-rm,r2+(2*rm)+1,-rm],[0,-rm,r3+rm+1]]
        b = [v1,v2,v3]
        title = 'Electrical Circuit Solver (Mesh Currents)'

    elif domain == 'structural':
        w,xbar  = float(params.get('w',120)), float(params.get('xbar',4))
        l1,l2   = float(params.get('l1',3)),  float(params.get('l2',8))
        ka,kb,kc = float(params.get('ka',2.0)), float(params.get('kb',1.6)), float(params.get('kc',1.2))
        lateral = float(params.get('lateral',18))
        variables = ['RA','RB','RC']
        A = [[1,1,1],[0,l1,l2],[ka+1,-(kb+0.5),kc+1]]
        b = [w, w*xbar, lateral]
        title = 'Structural Engineering Load Reactions'

    elif domain == 'economics':
        d1,d2,d3 = float(params.get('d1',130)), float(params.get('d2',125)), float(params.get('d3',120))
        variables = ['P1','P2','P3']
        A = [[10,-2,-1],[-1,11,-2],[-2,-1,12]]
        b = [d1,d2,d3]
        title = 'Economics Equilibrium Model (Market Prices)'

    elif domain == 'chemical':
        c1,c2,c3 = float(params.get('c1',42)), float(params.get('c2',38)), float(params.get('c3',30))
        variables = ['A','B','C']
        A = [[9,-2,-1],[-1,10,-2],[-2,-1,9]]
        b = [c1,c2,c3]
        title = 'Chemical Balance System (Species Rates)'

    else:
        raise ValueError('Unknown domain model selected')

    return A, b, variables, title


# ────────────────────────────────────────────────
# Redesigned Premium HTML Template
# ────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>GaussSolve &mdash; Iterative Linear System Solver</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Nunito:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<style>
:root{
  --bg:#F8F7F4;--surface:#FFFFFF;--surface2:#F2F1EE;
  --border:#E8E6E1;--border2:#D4D0C8;
  --ink:#1A1916;--ink2:#4A4740;--ink3:#8A8680;
  --accent:#2563EB;--accent-lt:#EEF4FF;--accent2:#7C3AED;
  --success:#059669;--success-lt:#ECFDF5;
  --warn:#D97706;--warn-lt:#FFFBEB;--danger:#DC2626;
  --r-md:14px;--r-lg:20px;--r-xl:28px;
  --shadow-sm:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
  --shadow-md:0 4px 16px rgba(0,0,0,.08),0 1px 4px rgba(0,0,0,.04);
  --shadow-lg:0 12px 40px rgba(0,0,0,.10),0 2px 8px rgba(0,0,0,.05);
  --font-sans:'Sora',sans-serif;--font-mono:'JetBrains Mono',monospace;--font-fun:'Nunito',sans-serif;
  --ease-spring:cubic-bezier(.34,1.56,.64,1);--ease-smooth:cubic-bezier(.25,.46,.45,.94);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:var(--font-sans);background:var(--bg);color:var(--ink);font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased;overflow-x:hidden}
::selection{background:#BFDBFE;color:var(--ink)}
a{color:inherit;text-decoration:none}
button{cursor:pointer;border:none;background:none;font-family:inherit}
input,select,textarea{font-family:inherit}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:99px}

/* NAVBAR */
.navbar{position:fixed;top:0;left:0;right:0;z-index:100;display:flex;align-items:center;justify-content:space-between;padding:0 40px;height:64px;background:rgba(248,247,244,.88);backdrop-filter:blur(16px);border-bottom:1px solid var(--border);transition:box-shadow .3s}
.navbar.scrolled{box-shadow:var(--shadow-md)}
.nav-logo{display:flex;align-items:center;gap:10px;font-weight:700;font-size:17px;letter-spacing:-.3px}
.nav-logo-icon{width:32px;height:32px;border-radius:8px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:flex;align-items:center;justify-content:center;font-family:var(--font-fun);font-weight:800;font-size:14px;color:#fff}
.nav-links{display:flex;align-items:center;gap:32px}
.nav-links a{font-size:14px;font-weight:500;color:var(--ink2);transition:color .2s}
.nav-links a:hover{color:var(--ink)}
.nav-cta{padding:8px 20px;border-radius:99px;background:var(--ink);color:#fff;font-size:13px;font-weight:600;transition:background .2s,transform .15s var(--ease-spring),box-shadow .2s;box-shadow:0 2px 8px rgba(0,0,0,.15)}
.nav-cta:hover{background:#2d2b27;transform:translateY(-1px);box-shadow:0 4px 16px rgba(0,0,0,.2)}

/* HERO */
.hero{min-height:100vh;display:flex;align-items:center;padding:120px 40px 80px;position:relative;overflow:hidden}
.hero-bg{position:absolute;inset:0;z-index:0;background:radial-gradient(ellipse 70% 60% at 60% 50%,rgba(37,99,235,.06) 0%,transparent 70%),radial-gradient(ellipse 50% 40% at 20% 80%,rgba(124,58,237,.05) 0%,transparent 70%)}
.hero-grid{position:absolute;inset:0;background-image:linear-gradient(var(--border) 1px,transparent 1px),linear-gradient(90deg,var(--border) 1px,transparent 1px);background-size:60px 60px;opacity:.4;mask-image:radial-gradient(ellipse 80% 80% at 50% 50%,black 30%,transparent 80%)}
.hero-inner{max-width:1100px;margin:0 auto;width:100%;display:grid;grid-template-columns:1fr 1fr;gap:80px;align-items:center;position:relative;z-index:1}
.hero-badge{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border-radius:99px;background:var(--accent-lt);border:1px solid #BFDBFE;font-size:12px;font-weight:600;color:var(--accent);letter-spacing:.02em;text-transform:uppercase;margin-bottom:24px;animation:fadeInUp .6s var(--ease-smooth) both}
.hero-badge-dot{width:6px;height:6px;border-radius:50%;background:var(--accent);animation:pulse-dot 2s infinite}
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.6;transform:scale(.7)}}
.hero-title{font-size:clamp(36px,5vw,58px);font-weight:800;letter-spacing:-.03em;line-height:1.1;margin-bottom:20px;animation:fadeInUp .6s .1s var(--ease-smooth) both}
.hero-title em{font-style:normal;color:var(--accent)}
.hero-desc{font-size:17px;color:var(--ink2);line-height:1.7;max-width:460px;margin-bottom:36px;animation:fadeInUp .6s .2s var(--ease-smooth) both}
.hero-actions{display:flex;align-items:center;gap:16px;flex-wrap:wrap;animation:fadeInUp .6s .3s var(--ease-smooth) both}
.btn-primary{display:inline-flex;align-items:center;gap:8px;padding:14px 28px;border-radius:12px;background:var(--ink);color:#fff;font-size:15px;font-weight:600;box-shadow:0 4px 16px rgba(0,0,0,.2);transition:transform .2s var(--ease-spring),box-shadow .2s}
.btn-primary:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.25)}
.btn-secondary{display:inline-flex;align-items:center;gap:8px;padding:14px 28px;border-radius:12px;background:var(--surface);color:var(--ink);font-size:15px;font-weight:600;border:1.5px solid var(--border);box-shadow:var(--shadow-sm);transition:transform .2s var(--ease-spring),box-shadow .2s,border-color .2s}
.btn-secondary:hover{transform:translateY(-2px);box-shadow:var(--shadow-md);border-color:var(--border2)}
.hero-stats{display:flex;gap:32px;margin-top:48px;animation:fadeInUp .6s .4s var(--ease-smooth) both}
.hero-stat-num{font-size:22px;font-weight:800;color:var(--ink);font-family:var(--font-fun)}
.hero-stat-label{font-size:12px;color:var(--ink3);font-weight:500}
.hero-visual{position:relative;display:flex;justify-content:center;align-items:center;animation:fadeInRight .8s .2s var(--ease-smooth) both}
.math-stage{width:100%;max-width:440px;aspect-ratio:1}
.math-stage svg{width:100%;height:100%;filter:drop-shadow(0 20px 60px rgba(37,99,235,.12))}

/* FEATURES STRIP */
.features-strip{padding:20px 40px;border-top:1px solid var(--border);border-bottom:1px solid var(--border);background:var(--surface);overflow:hidden}
.features-scroll{display:flex;gap:48px;align-items:center;animation:marquee 30s linear infinite;white-space:nowrap}
.features-scroll span{display:flex;align-items:center;gap:10px;font-size:13px;font-weight:600;color:var(--ink2);letter-spacing:.02em;flex-shrink:0}
.features-scroll .dot{color:var(--accent);font-size:18px}
@keyframes marquee{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}

/* SECTIONS */
section{padding:96px 40px}
.section-label{display:inline-flex;align-items:center;gap:6px;font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);margin-bottom:14px}
.section-title{font-size:clamp(26px,3.5vw,40px);font-weight:800;letter-spacing:-.025em;line-height:1.15;margin-bottom:16px}
.section-desc{font-size:17px;color:var(--ink2);line-height:1.7;max-width:520px}
.section-inner{max-width:1100px;margin:0 auto}

/* SOLVER */
.solver-section{background:var(--surface)}
.solver-layout{display:grid;grid-template-columns:340px 1fr;gap:32px;margin-top:56px}
.mode-tabs{display:flex;gap:4px;background:var(--surface2);border-radius:var(--r-md);padding:4px;margin-bottom:24px}
.mode-tab{flex:1;padding:9px 14px;border-radius:10px;font-size:13px;font-weight:600;color:var(--ink2);transition:all .2s;text-align:center}
.mode-tab.active{background:var(--surface);color:var(--ink);box-shadow:var(--shadow-sm)}
.mode-tab:hover:not(.active){color:var(--ink)}
.panel-card{background:var(--surface);border-radius:var(--r-xl);border:1.5px solid var(--border);box-shadow:var(--shadow-md);padding:28px;transition:box-shadow .3s}
.panel-card:hover{box-shadow:var(--shadow-lg)}
.form-group{margin-bottom:20px}
.form-label{display:block;font-size:12px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:var(--ink2);margin-bottom:8px}
.form-input{width:100%;padding:11px 14px;border-radius:var(--r-md);border:1.5px solid var(--border);background:var(--bg);font-size:14px;color:var(--ink);transition:border-color .2s,box-shadow .2s;outline:none}
.form-input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(37,99,235,.12)}
.form-select{width:100%;padding:11px 14px;border-radius:var(--r-md);border:1.5px solid var(--border);background:var(--bg);font-size:14px;color:var(--ink);appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%238A8680' d='M6 8L1 3h10z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 14px center;outline:none;cursor:pointer;transition:border-color .2s,box-shadow .2s}
.form-select:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(37,99,235,.12)}
.matrix-grid{display:grid;gap:6px;margin-bottom:8px}
.matrix-cell{padding:9px 10px;border-radius:8px;border:1.5px solid var(--border);background:var(--bg);font-family:var(--font-mono);font-size:13px;color:var(--ink);text-align:center;outline:none;width:100%;transition:border-color .2s,box-shadow .2s}
.matrix-cell:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(37,99,235,.12)}
.matrix-label-row,.matrix-label-col{font-family:var(--font-mono);font-size:11px;font-weight:600;color:var(--accent);text-align:center;padding:4px;letter-spacing:.04em}
.result-panel{display:flex;flex-direction:column;gap:20px}
.result-card{background:var(--surface);border-radius:var(--r-xl);border:1.5px solid var(--border);box-shadow:var(--shadow-md);padding:28px;animation:fadeInUp .4s var(--ease-smooth) both}
.result-status{display:flex;align-items:center;gap:10px;padding:12px 16px;border-radius:var(--r-md);font-size:14px;font-weight:600;margin-bottom:20px}
.result-status.converged{background:var(--success-lt);color:var(--success)}
.result-status.diverged{background:var(--warn-lt);color:var(--warn)}
.result-status .status-icon{font-size:18px}
.solution-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px;margin-bottom:20px}
.solution-var{background:var(--surface2);border-radius:var(--r-md);padding:14px 16px;border:1.5px solid var(--border);transition:transform .2s var(--ease-spring),box-shadow .2s}
.solution-var:hover{transform:translateY(-2px);box-shadow:var(--shadow-sm)}
.solution-var-name{font-family:var(--font-mono);font-size:11px;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
.solution-var-val{font-family:var(--font-mono);font-size:18px;font-weight:600;color:var(--ink)}
.diag-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.diag-item{background:var(--surface2);border-radius:var(--r-md);padding:12px 14px;border:1.5px solid var(--border)}
.diag-key{font-size:11px;font-weight:700;color:var(--ink3);text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}
.diag-val{font-family:var(--font-mono);font-size:14px;font-weight:600;color:var(--ink)}
.stability-excellent{color:var(--success)!important}
.stability-fair{color:var(--warn)!important}
.stability-risky{color:var(--danger)!important}
.iter-table-wrap{overflow-x:auto;border-radius:var(--r-md);border:1.5px solid var(--border)}
.iter-table{width:100%;border-collapse:collapse;font-size:13px}
.iter-table th{background:var(--surface2);font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink2);padding:10px 14px;text-align:left;border-bottom:1px solid var(--border)}
.iter-table td{padding:9px 14px;border-bottom:1px solid var(--border)}
.iter-table tr:last-child td{border-bottom:none}
.iter-table tr:nth-child(even) td{background:var(--surface2)}
.mono{font-family:var(--font-mono)}
.history-section{background:var(--bg)}
.history-list{display:flex;flex-direction:column;gap:10px;margin-top:24px}
.history-item{background:var(--surface);border:1.5px solid var(--border);border-radius:var(--r-lg);padding:16px 20px;display:flex;align-items:center;justify-content:space-between;gap:16px;transition:border-color .2s,box-shadow .2s;cursor:pointer}
.history-item:hover{border-color:var(--accent);box-shadow:0 0 0 3px rgba(37,99,235,.08)}
.history-item-meta{font-size:12px;color:var(--ink3)}
.history-badge{padding:3px 10px;border-radius:99px;font-size:11px;font-weight:700;letter-spacing:.04em}
.badge-converged{background:var(--success-lt);color:var(--success)}
.badge-diverged{background:var(--warn-lt);color:var(--warn)}

/* HOW + DOMAINS */
.how-section{background:var(--bg)}
.steps-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin-top:56px}
.step-card{background:var(--surface);border-radius:var(--r-xl);padding:32px 28px;border:1.5px solid var(--border);box-shadow:var(--shadow-sm);position:relative;overflow:hidden;transition:transform .3s var(--ease-spring),box-shadow .3s}
.step-card:hover{transform:translateY(-4px);box-shadow:var(--shadow-lg)}
.step-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--accent),var(--accent2))}
.step-num{font-family:var(--font-fun);font-size:48px;font-weight:800;color:var(--border);position:absolute;top:16px;right:20px;line-height:1;user-select:none}
.step-icon{width:48px;height:48px;border-radius:12px;background:var(--accent-lt);display:flex;align-items:center;justify-content:center;font-size:22px;margin-bottom:20px}
.step-title{font-size:17px;font-weight:700;margin-bottom:10px;letter-spacing:-.01em}
.step-desc{font-size:14px;color:var(--ink2);line-height:1.65}
.domains-section{background:var(--surface)}
.domains-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:20px;margin-top:56px}
.domain-card{border-radius:var(--r-xl);border:1.5px solid var(--border);padding:28px;background:var(--bg);display:flex;align-items:flex-start;gap:18px;transition:all .3s var(--ease-spring);cursor:pointer}
.domain-card:hover{border-color:var(--accent);background:var(--surface);transform:translateY(-3px);box-shadow:var(--shadow-lg)}
.domain-icon{width:52px;height:52px;border-radius:14px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:24px}
.domain-title{font-size:16px;font-weight:700;margin-bottom:6px;letter-spacing:-.01em}
.domain-desc{font-size:13px;color:var(--ink2);line-height:1.6}

/* FOOTER */
footer{background:var(--ink);color:rgba(255,255,255,.7);padding:64px 40px 40px}
.footer-inner{max-width:1100px;margin:0 auto;display:grid;grid-template-columns:1fr 1fr;gap:48px;border-bottom:1px solid rgba(255,255,255,.08);padding-bottom:48px;margin-bottom:32px}
.footer-brand-name{font-size:18px;font-weight:800;color:#fff;margin-bottom:10px;letter-spacing:-.02em}
.footer-brand-desc{font-size:13px;line-height:1.7;max-width:280px}
.footer-links{display:flex;gap:48px}
.footer-col-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:rgba(255,255,255,.4);margin-bottom:16px}
.footer-col a{display:block;font-size:13px;color:rgba(255,255,255,.6);margin-bottom:10px;transition:color .2s}
.footer-col a:hover{color:#fff}
.footer-bottom{max-width:1100px;margin:0 auto;display:flex;align-items:center;justify-content:space-between}
.footer-copy,.footer-made{font-size:12px}

/* ANIMATIONS */
@keyframes fadeInUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeInRight{from{opacity:0;transform:translateX(32px)}to{opacity:1;transform:translateX(0)}}
.appear{opacity:0;transform:translateY(20px);transition:opacity .6s var(--ease-smooth),transform .6s var(--ease-smooth)}
.appear.visible{opacity:1;transform:translateY(0)}
.btn-solve{width:100%;padding:14px;background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;border-radius:var(--r-md);font-size:15px;font-weight:700;letter-spacing:.01em;box-shadow:0 4px 16px rgba(37,99,235,.3);transition:transform .2s var(--ease-spring),box-shadow .2s,opacity .2s;display:flex;align-items:center;justify-content:center;gap:8px}
.btn-solve:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 8px 24px rgba(37,99,235,.4)}
.btn-solve:disabled{opacity:.6;cursor:not-allowed}
.btn-solve.loading::after{content:'';width:16px;height:16px;border:2px solid rgba(255,255,255,.4);border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.result-placeholder{display:flex;flex-direction:column;align-items:center;justify-content:center;height:320px;border-radius:var(--r-xl);border:2px dashed var(--border);gap:16px;color:var(--ink3)}
.result-placeholder svg{opacity:.3}
.size-row{display:flex;align-items:center;gap:8px;margin-bottom:16px}
.size-btn{width:30px;height:30px;border-radius:8px;border:1.5px solid var(--border);background:var(--surface);font-size:16px;font-weight:700;color:var(--ink2);display:flex;align-items:center;justify-content:center;transition:all .15s;line-height:1}
.size-btn:hover{border-color:var(--accent);color:var(--accent);background:var(--accent-lt)}
.size-label{font-size:13px;font-weight:600;color:var(--ink);min-width:80px}
.input-mode{display:none}
.input-mode.active{display:block}

/* RESPONSIVE */
@media(max-width:900px){
  .navbar{padding:0 20px}.nav-links{display:none}
  .hero{padding:100px 20px 60px}
  .hero-inner{grid-template-columns:1fr;gap:48px}
  .hero-visual{order:-1}
  .solver-layout{grid-template-columns:1fr}
  .steps-grid{grid-template-columns:1fr}
  .domains-grid{grid-template-columns:1fr}
  section{padding:64px 20px}
  .footer-inner{grid-template-columns:1fr}
  .footer-links{gap:32px}
  .footer-bottom{flex-direction:column;gap:12px;text-align:center}
  .diag-grid{grid-template-columns:1fr}
  .solution-grid{grid-template-columns:1fr 1fr}
}
@media(max-width:500px){
  .hero-stats{gap:20px}
  .hero-actions{flex-direction:column;align-items:flex-start}
  .btn-primary,.btn-secondary{width:100%;justify-content:center}
}
</style>
</head>
<body>

<!-- NAVBAR -->
<nav class="navbar" id="navbar">
  <div class="nav-logo">
    <div class="nav-logo-icon">G&#x2211;</div>
    GaussSolve
  </div>
  <div class="nav-links">
    <a href="#solver">Solver</a>
    <a href="#how">How it works</a>
    <a href="#domains">Domains</a>
    <a href="#history">History</a>
  </div>
  <button class="nav-cta" onclick="document.getElementById('solver').scrollIntoView({behavior:'smooth'})">
    Start Solving &rarr;
  </button>
</nav>

<!-- HERO -->
<section class="hero" id="hero">
  <div class="hero-bg"></div>
  <div class="hero-grid"></div>
  <div class="hero-inner">
    <div class="hero-content">
      <div class="hero-badge"><span class="hero-badge-dot"></span>Iterative Numerical Methods</div>
      <h1 class="hero-title">Solve Linear<br/>Systems with<br/><em>Elegance</em></h1>
      <p class="hero-desc">Harness the power of Gauss-Seidel iteration to solve complex linear equations across engineering, economics, and science &mdash; instantly, visually, precisely.</p>
      <div class="hero-actions">
        <button class="btn-primary" onclick="document.getElementById('solver').scrollIntoView({behavior:'smooth'})">&#x26A1; Launch Solver</button>
        <button class="btn-secondary" onclick="document.getElementById('how').scrollIntoView({behavior:'smooth'})">See how it works</button>
      </div>
      <div class="hero-stats">
        <div><div class="hero-stat-num">&#x221E;</div><div class="hero-stat-label">Iterations logged</div></div>
        <div style="width:1px;background:var(--border);height:36px;margin-top:4px;"></div>
        <div><div class="hero-stat-num">4</div><div class="hero-stat-label">Domain presets</div></div>
        <div style="width:1px;background:var(--border);height:36px;margin-top:4px;"></div>
        <div><div class="hero-stat-num">10&#x207B;&#x2074;</div><div class="hero-stat-label">Default tolerance</div></div>
      </div>
    </div>
    <!-- Cartoon Math SVG Animation -->
    <div class="hero-visual">
      <div class="math-stage">
        <svg viewBox="0 0 440 440" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <radialGradient id="bg-grd" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="#EEF4FF"/><stop offset="100%" stop-color="#F8F7F4"/></radialGradient>
            <filter id="shadow"><feDropShadow dx="0" dy="4" stdDeviation="8" flood-color="#2563EB" flood-opacity=".15"/></filter>
            <filter id="shadow-sm"><feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#000" flood-opacity=".08"/></filter>
          </defs>
          <rect x="20" y="20" width="400" height="400" rx="28" fill="url(#bg-grd)" stroke="#E8E6E1" stroke-width="1.5"/>
          <g stroke="#E8E6E1" stroke-width="1" opacity=".5">
            <line x1="20" y1="100" x2="420" y2="100"/><line x1="20" y1="180" x2="420" y2="180"/>
            <line x1="20" y1="260" x2="420" y2="260"/><line x1="20" y1="340" x2="420" y2="340"/>
            <line x1="100" y1="20" x2="100" y2="420"/><line x1="180" y1="20" x2="180" y2="420"/>
            <line x1="260" y1="20" x2="260" y2="420"/><line x1="340" y1="20" x2="340" y2="420"/>
          </g>
          <polyline points="50,380 100,280 150,210 180,165 200,145 215,136 225,132 232,130 237,129.5"
            fill="none" stroke="#2563EB" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"
            stroke-dasharray="400" stroke-dashoffset="400" opacity=".8">
            <animate attributeName="stroke-dashoffset" values="400;0" dur="2.5s" begin="0.5s" fill="freeze" calcMode="spline" keySplines="0.25 0.46 0.45 0.94"/>
          </polyline>
          <line x1="50" y1="129" x2="390" y2="129" stroke="#7C3AED" stroke-width="1.5" stroke-dasharray="6 4" opacity=".4"/>
          <text x="394" y="133" font-family="JetBrains Mono,monospace" font-size="10" fill="#7C3AED" opacity=".7" font-weight="600">x*</text>
          <circle cx="50"  cy="380" r="5" fill="#2563EB" opacity="0"><animate attributeName="opacity" values="0;1" dur=".1s" begin="0.5s" fill="freeze"/><animate attributeName="r" values="0;5" dur=".3s" begin="0.5s" fill="freeze" calcMode="spline" keySplines=".34 1.56 .64 1"/></circle>
          <circle cx="100" cy="280" r="5" fill="#2563EB" opacity="0"><animate attributeName="opacity" values="0;1" dur=".1s" begin="1.0s" fill="freeze"/><animate attributeName="r" values="0;5" dur=".3s" begin="1.0s" fill="freeze" calcMode="spline" keySplines=".34 1.56 .64 1"/></circle>
          <circle cx="150" cy="210" r="5" fill="#2563EB" opacity="0"><animate attributeName="opacity" values="0;1" dur=".1s" begin="1.4s" fill="freeze"/><animate attributeName="r" values="0;5" dur=".3s" begin="1.4s" fill="freeze" calcMode="spline" keySplines=".34 1.56 .64 1"/></circle>
          <circle cx="200" cy="145" r="5" fill="#2563EB" opacity="0"><animate attributeName="opacity" values="0;1" dur=".1s" begin="1.8s" fill="freeze"/><animate attributeName="r" values="0;5" dur=".3s" begin="1.8s" fill="freeze" calcMode="spline" keySplines=".34 1.56 .64 1"/></circle>
          <circle cx="237" cy="129" r="7" fill="#059669" opacity="0"><animate attributeName="opacity" values="0;1" dur=".1s" begin="2.8s" fill="freeze"/><animate attributeName="r" values="0;7" dur=".4s" begin="2.8s" fill="freeze" calcMode="spline" keySplines=".34 1.56 .64 1"/></circle>
          <g font-family="Nunito,sans-serif" font-weight="800" font-size="11">
            <g opacity="0"><animate attributeName="opacity" values="0;1" dur=".2s" begin="0.7s" fill="freeze"/>
              <rect x="56" y="354" width="28" height="18" rx="5" fill="#DBEAFE"/><text x="70" y="366" text-anchor="middle" fill="#2563EB">i=1</text></g>
            <g opacity="0"><animate attributeName="opacity" values="0;1" dur=".2s" begin="1.2s" fill="freeze"/>
              <rect x="106" y="256" width="28" height="18" rx="5" fill="#DBEAFE"/><text x="120" y="268" text-anchor="middle" fill="#2563EB">i=2</text></g>
            <g opacity="0"><animate attributeName="opacity" values="0;1" dur=".2s" begin="1.6s" fill="freeze"/>
              <rect x="156" y="186" width="28" height="18" rx="5" fill="#DBEAFE"/><text x="170" y="198" text-anchor="middle" fill="#2563EB">i=3</text></g>
            <g opacity="0"><animate attributeName="opacity" values="0;1" dur=".2s" begin="2.0s" fill="freeze"/>
              <rect x="206" y="120" width="28" height="18" rx="5" fill="#DBEAFE"/><text x="220" y="132" text-anchor="middle" fill="#2563EB">i=4</text></g>
          </g>
          <g opacity="0"><animate attributeName="opacity" values="0;1" dur=".5s" begin="3.0s" fill="freeze" calcMode="spline" keySplines=".34 1.56 .64 1"/>
            <rect x="250" y="100" width="130" height="42" rx="12" fill="#059669" filter="url(#shadow)"/>
            <text x="315" y="118" font-family="Nunito,sans-serif" font-weight="800" font-size="11" fill="#fff" text-anchor="middle">&#x2713; CONVERGED!</text>
            <text x="315" y="133" font-family="JetBrains Mono,monospace" font-size="10" fill="rgba(255,255,255,.8)" text-anchor="middle">&#x03B4; &lt; 1&#xD7;10&#x207B;&#x2074;</text>
          </g>
          <g font-family="Nunito,sans-serif" font-weight="800">
            <g filter="url(#shadow-sm)"><rect x="32" y="36" width="54" height="36" rx="10" fill="white" stroke="#E8E6E1" stroke-width="1"/>
              <text x="59" y="59" text-anchor="middle" font-size="20" fill="#7C3AED">&#x2211;</text>
              <animateTransform attributeName="transform" type="translate" values="0,0;0,-4;0,0" dur="3s" repeatCount="indefinite" calcMode="spline" keySplines=".45 0 .55 1;.45 0 .55 1"/></g>
            <g filter="url(#shadow-sm)"><rect x="340" y="44" width="52" height="32" rx="10" fill="white" stroke="#E8E6E1" stroke-width="1"/>
              <text x="366" y="65" text-anchor="middle" font-size="16" font-weight="800" fill="#2563EB">=</text>
              <animateTransform attributeName="transform" type="translate" values="0,0;0,4;0,0" dur="2.5s" repeatCount="indefinite" calcMode="spline" keySplines=".45 0 .55 1;.45 0 .55 1"/></g>
            <g filter="url(#shadow-sm)"><rect x="290" y="360" width="100" height="36" rx="10" fill="white" stroke="#E8E6E1" stroke-width="1"/>
              <text x="340" y="383" text-anchor="middle" font-size="14" font-family="JetBrains Mono,monospace" fill="#1A1916" font-weight="600">Ax = b</text>
              <animateTransform attributeName="transform" type="translate" values="0,0;0,-3;0,0" dur="3.5s" repeatCount="indefinite" calcMode="spline" keySplines=".45 0 .55 1;.45 0 .55 1"/></g>
            <g filter="url(#shadow-sm)"><rect x="32" y="370" width="90" height="32" rx="10" fill="white" stroke="#E8E6E1" stroke-width="1"/>
              <text x="77" y="391" text-anchor="middle" font-size="11" font-family="JetBrains Mono,monospace" fill="#D97706" font-weight="600">tol=1e-4</text>
              <animateTransform attributeName="transform" type="translate" values="0,0;0,3;0,0" dur="2.8s" repeatCount="indefinite" calcMode="spline" keySplines=".45 0 .55 1;.45 0 .55 1"/></g>
          </g>
        </svg>
      </div>
    </div>
  </div>
</section>

<!-- FEATURES STRIP -->
<div class="features-strip">
  <div class="features-scroll">
    <span><span class="dot">&#x25CF;</span>Gauss-Seidel Iteration</span>
    <span><span class="dot">&#x25CF;</span>Matrix Mode</span>
    <span><span class="dot">&#x25CF;</span>Equation Parser</span>
    <span><span class="dot">&#x25CF;</span>Domain Presets</span>
    <span><span class="dot">&#x25CF;</span>Convergence Diagnostics</span>
    <span><span class="dot">&#x25CF;</span>History Logging</span>
    <span><span class="dot">&#x25CF;</span>Residual Analysis</span>
    <span><span class="dot">&#x25CF;</span>Diagonal Dominance Check</span>
    <span><span class="dot">&#x25CF;</span>Gauss-Seidel Iteration</span>
    <span><span class="dot">&#x25CF;</span>Matrix Mode</span>
    <span><span class="dot">&#x25CF;</span>Equation Parser</span>
    <span><span class="dot">&#x25CF;</span>Domain Presets</span>
    <span><span class="dot">&#x25CF;</span>Convergence Diagnostics</span>
    <span><span class="dot">&#x25CF;</span>History Logging</span>
    <span><span class="dot">&#x25CF;</span>Residual Analysis</span>
    <span><span class="dot">&#x25CF;</span>Diagonal Dominance Check</span>
  </div>
</div>

<!-- SOLVER -->
<section class="solver-section" id="solver">
  <div class="section-inner">
    <div class="section-label">&#x26A1; Interactive Tool</div>
    <h2 class="section-title">The Solver</h2>
    <p class="section-desc">Choose your input mode, configure your system, and watch it converge &mdash; step by step.</p>
    <div class="solver-layout">
      <div>
        <div class="mode-tabs">
          <button class="mode-tab active" onclick="setMode('matrix',this)">Matrix</button>
          <button class="mode-tab" onclick="setMode('equation',this)">Equations</button>
          <button class="mode-tab" onclick="setMode('domain',this)">Domain</button>
        </div>
        <div class="panel-card">
          <div class="input-mode active" id="mode-matrix">
            <div class="size-row">
              <button class="size-btn" onclick="resizeMatrix(-1)">&#x2212;</button>
              <span class="size-label" id="size-label">3 &times; 3 system</span>
              <button class="size-btn" onclick="resizeMatrix(1)">+</button>
            </div>
            <div class="form-label">Matrix A</div>
            <div id="matrix-A-wrap"></div>
            <div class="form-label" style="margin-top:16px;">Vector b</div>
            <div id="matrix-b-wrap"></div>
          </div>
          <div class="input-mode" id="mode-equation">
            <div class="form-label">Equations (one per line)</div>
            <div style="font-size:12px;color:var(--ink3);margin-bottom:8px;">e.g. <span style="font-family:var(--font-mono);color:var(--accent)">10x-2y=16</span></div>
            <textarea class="form-input" id="eq-input" rows="5" placeholder="10x-2y=16&#10;-2x+10y=24&#10;" style="resize:vertical;font-family:var(--font-mono);font-size:13px;"></textarea>
          </div>
          <div class="input-mode" id="mode-domain">
            <div class="form-group">
              <label class="form-label">Domain Preset</label>
              <select class="form-select" id="domain-select" onchange="renderDomainFields()">
                <option value="electrical">&#x26A1; Electrical Circuit</option>
                <option value="structural">&#x1F3D7; Structural Engineering</option>
                <option value="economics">&#x1F4C8; Economics Model</option>
                <option value="chemical">&#x2697; Chemical Balance</option>
              </select>
            </div>
            <div id="domain-params"></div>
          </div>
          <div style="margin-top:20px;padding-top:20px;border-top:1px solid var(--border);">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px;">
              <div class="form-group" style="margin:0"><label class="form-label">Tolerance</label>
                <input class="form-input" type="number" id="tol" value="0.0001" step="0.0001" min="0.000001"/></div>
              <div class="form-group" style="margin:0"><label class="form-label">Max Iterations</label>
                <input class="form-input" type="number" id="maxIter" value="1000" min="1" max="10000"/></div>
            </div>
            <button class="btn-solve" id="solve-btn" onclick="solve()">
              <span id="solve-label">&#x26A1; Solve System</span>
            </button>
          </div>
        </div>
      </div>
      <div class="result-panel" id="result-panel">
        <div class="result-placeholder" id="result-placeholder">
          <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
            <rect x="8" y="8" width="48" height="48" rx="12" stroke="#8A8680" stroke-width="2"/>
            <path d="M20 32h24M32 20v24" stroke="#8A8680" stroke-width="2" stroke-linecap="round"/>
          </svg>
          <div style="font-size:14px;font-weight:600;color:var(--ink3);">Configure and solve a system</div>
          <div style="font-size:12px;">Results will appear here</div>
        </div>
        <div id="result-content" style="display:none;"></div>
      </div>
    </div>
  </div>
</section>

<!-- HOW IT WORKS -->
<section class="how-section" id="how">
  <div class="section-inner">
    <div class="section-label appear">&#x1F522; Method</div>
    <h2 class="section-title appear">How Gauss-Seidel Works</h2>
    <p class="section-desc appear">An elegant iterative technique that updates each variable using the most recent values &mdash; converging faster than Jacobi.</p>
    <div class="steps-grid">
      <div class="step-card appear"><span class="step-num">01</span>
        <div class="step-icon">&#x1F3AF;</div>
        <div class="step-title">Initial Guess</div>
        <div class="step-desc">Start with all variables set to zero. The method finds its way to the solution regardless of the starting point.</div>
      </div>
      <div class="step-card appear"><span class="step-num">02</span>
        <div class="step-icon">&#x1F504;</div>
        <div class="step-title">Iterative Update</div>
        <div class="step-desc">For each variable, compute a new value using the latest known values of all other variables &mdash; faster than Jacobi.</div>
      </div>
      <div class="step-card appear"><span class="step-num">03</span>
        <div class="step-icon">&#x2705;</div>
        <div class="step-title">Convergence Check</div>
        <div class="step-desc">After every sweep, check if max change &delta; is below the tolerance. If yes &mdash; done!</div>
      </div>
    </div>
  </div>
</section>

<!-- DOMAINS -->
<section class="domains-section" id="domains">
  <div class="section-inner">
    <div class="section-label appear">&#x1F310; Applications</div>
    <h2 class="section-title appear">Built for Real Engineering</h2>
    <p class="section-desc appear">Pre-configured domain models let you solve real-world problems without manual setup.</p>
    <div class="domains-grid">
      <div class="domain-card appear" onclick="activateDomain('electrical')">
        <div class="domain-icon" style="background:#EEF4FF;">&#x26A1;</div>
        <div><div class="domain-title">Electrical Circuits</div>
        <div class="domain-desc">Mesh current analysis using KVL. Solve multi-loop circuits with resistors and voltage sources instantly.</div></div>
      </div>
      <div class="domain-card appear" onclick="activateDomain('structural')">
        <div class="domain-icon" style="background:#FEF9C3;">&#x1F3D7;</div>
        <div><div class="domain-title">Structural Engineering</div>
        <div class="domain-desc">Compute reaction forces and moments for static structures under distributed and point loads.</div></div>
      </div>
      <div class="domain-card appear" onclick="activateDomain('economics')">
        <div class="domain-icon" style="background:#ECFDF5;">&#x1F4C8;</div>
        <div><div class="domain-title">Economic Equilibrium</div>
        <div class="domain-desc">Solve simultaneous supply/demand price models with cross-elasticity for multi-market equilibrium.</div></div>
      </div>
      <div class="domain-card appear" onclick="activateDomain('chemical')">
        <div class="domain-icon" style="background:#FDF4FF;">&#x2697;</div>
        <div><div class="domain-title">Chemical Kinetics</div>
        <div class="domain-desc">Balance multi-species reaction networks and steady-state concentration systems.</div></div>
      </div>
    </div>
  </div>
</section>

<!-- HISTORY -->
<section class="history-section" id="history">
  <div class="section-inner">
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px;">
      <div>
        <div class="section-label appear">&#x1F4CB; Logs</div>
        <h2 class="section-title appear" style="margin-bottom:0;">Solve History</h2>
      </div>
      <div style="display:flex;gap:10px;">
        <button class="btn-secondary" style="padding:10px 18px;font-size:13px;" onclick="loadHistory()">&#x21BB; Refresh</button>
        <button class="btn-secondary" style="padding:10px 18px;font-size:13px;color:var(--danger);border-color:var(--danger);" onclick="clearHistory()">&#x1F5D1; Clear All</button>
      </div>
    </div>
    <div class="history-list" id="history-list">
      <div style="color:var(--ink3);font-size:14px;padding:32px 0;text-align:center;">Loading history&hellip;</div>
    </div>
  </div>
</section>

<!-- FOOTER -->
<footer>
  <div class="footer-inner">
    <div>
      <div class="footer-brand-name">GaussSolve</div>
      <div class="footer-brand-desc">A precision-engineered numerical solver built for scientists, engineers, and analysts who demand accuracy and speed.</div>
    </div>
    <div class="footer-links">
      <div class="footer-col">
        <div class="footer-col-title">Tool</div>
        <a href="#solver">Matrix Solver</a>
        <a href="#solver">Equation Mode</a>
        <a href="#domains">Domain Presets</a>
        <a href="#history">History</a>
      </div>
      <div class="footer-col">
        <div class="footer-col-title">Learn</div>
        <a href="#how">How it works</a>
        <a href="#domains">Applications</a>
      </div>
    </div>
  </div>
  <div class="footer-bottom">
    <div class="footer-copy">&copy; 2025 GaussSolve. All rights reserved.</div>
    <div class="footer-made">Built with precision &starf; Powered by Python &amp; Flask</div>
  </div>
</footer>

<script>
var currentMode='matrix', matrixSize=3;

window.addEventListener('scroll',function(){
  document.getElementById('navbar').classList.toggle('scrolled',window.scrollY>20);
});

var obs=new IntersectionObserver(function(entries){
  entries.forEach(function(e,i){
    if(e.isIntersecting){setTimeout(function(){e.target.classList.add('visible')},i*80);obs.unobserve(e.target);}
  });
},{threshold:.12});
document.querySelectorAll('.appear').forEach(function(el){obs.observe(el);});

function setMode(mode,btn){
  currentMode=mode;
  document.querySelectorAll('.mode-tab').forEach(function(t){t.classList.remove('active');});
  btn.classList.add('active');
  document.querySelectorAll('.input-mode').forEach(function(el){el.classList.remove('active');});
  document.getElementById('mode-'+mode).classList.add('active');
}

function buildMatrix(){
  var n=matrixSize;
  document.getElementById('size-label').textContent=n+' \u00D7 '+n+' system';
  var html='<div class="matrix-grid" style="grid-template-columns:repeat('+(n+1)+',1fr)"><div></div>';
  for(var j=0;j<n;j++) html+='<div class="matrix-label-col">x'+(j+1)+'</div>';
  for(var i=0;i<n;i++){
    html+='<div class="matrix-label-row">r'+(i+1)+'</div>';
    for(var j=0;j<n;j++){
      var def=i===j?10:(Math.abs(i-j)===1?-1:0);
      html+='<input class="matrix-cell" type="number" step="any" id="a'+i+j+'" value="'+def+'"/>';
    }
  }
  html+='</div>';
  document.getElementById('matrix-A-wrap').innerHTML=html;
  var bDef=[16,24,15,8];
  var bHtml='<div class="matrix-grid" style="grid-template-columns:repeat('+n+',1fr)">';
  for(var i=0;i<n;i++) bHtml+='<input class="matrix-cell" type="number" step="any" id="b'+i+'" value="'+(bDef[i]||10)+'"/>';
  bHtml+='</div>';
  document.getElementById('matrix-b-wrap').innerHTML=bHtml;
}

function resizeMatrix(d){matrixSize=Math.max(2,Math.min(6,matrixSize+d));buildMatrix();}

var DOMAIN_FIELDS={
  electrical:[{id:'r1',label:'R1 (\u03A9)',def:12},{id:'r2',label:'R2 (\u03A9)',def:10},{id:'r3',label:'R3 (\u03A9)',def:8},{id:'rm',label:'Rm (\u03A9)',def:2},{id:'v1',label:'V1 (V)',def:24},{id:'v2',label:'V2 (V)',def:18},{id:'v3',label:'V3 (V)',def:12}],
  structural:[{id:'w',label:'Load W (kN)',def:120},{id:'xbar',label:'x\u0305 (m)',def:4},{id:'l1',label:'L1 (m)',def:3},{id:'l2',label:'L2 (m)',def:8},{id:'ka',label:'Ka',def:2.0},{id:'kb',label:'Kb',def:1.6},{id:'kc',label:'Kc',def:1.2},{id:'lateral',label:'Lateral (kN)',def:18}],
  economics:[{id:'d1',label:'Demand D1',def:130},{id:'d2',label:'Demand D2',def:125},{id:'d3',label:'Demand D3',def:120}],
  chemical:[{id:'c1',label:'Conc. C1',def:42},{id:'c2',label:'Conc. C2',def:38},{id:'c3',label:'Conc. C3',def:30}]
};

function renderDomainFields(){
  var domain=document.getElementById('domain-select').value;
  var fields=DOMAIN_FIELDS[domain]||[];
  var cols=fields.length>4?2:1;
  var html='<div style="display:grid;grid-template-columns:repeat('+cols+',1fr);gap:10px;">';
  fields.forEach(function(f){
    html+='<div class="form-group" style="margin:0"><label class="form-label">'+f.label+'</label>'
        +'<input class="form-input" type="number" step="any" id="dp-'+f.id+'" value="'+f.def+'"/></div>';
  });
  html+='</div>';
  document.getElementById('domain-params').innerHTML=html;
}

async function solve(){
  var btn=document.getElementById('solve-btn');
  var label=document.getElementById('solve-label');
  btn.disabled=true; btn.classList.add('loading'); label.textContent='Solving\u2026';
  var tol=parseFloat(document.getElementById('tol').value)||1e-4;
  var maxIter=parseInt(document.getElementById('maxIter').value)||1000;
  var payload={mode:currentMode,tol:tol,maxIter:maxIter};
  try{
    if(currentMode==='matrix'){
      var n=matrixSize, A=[], b=[];
      for(var i=0;i<n;i++){
        var row=[];
        for(var j=0;j<n;j++) row.push(parseFloat(document.getElementById('a'+i+j).value)||0);
        A.push(row);
        b.push(parseFloat(document.getElementById('b'+i).value)||0);
      }
      payload.A=A; payload.b=b;
    } else if(currentMode==='equation'){
      var text=document.getElementById('eq-input').value;
      var equations=text.split('\n').map(function(s){return s.trim();}).filter(Boolean);
      if(!equations.length) throw new Error('Enter at least one equation');
      payload.equations=equations;
    } else {
      var domainType=document.getElementById('domain-select').value;
      var domainParams={};
      (DOMAIN_FIELDS[domainType]||[]).forEach(function(f){
        var el=document.getElementById('dp-'+f.id);
        domainParams[f.id]=el?(parseFloat(el.value)||f.def):f.def;
      });
      payload.domainType=domainType; payload.domainParams=domainParams;
    }
    var res=await fetch('/solve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    var data=await res.json();
    if(data.error) throw new Error(data.error);
    renderResult(data);
    loadHistory();
  } catch(e){ renderError(e.message); }
  finally{ btn.disabled=false; btn.classList.remove('loading'); label.textContent='\u26A1 Solve System'; }
}

function renderResult(data){
  document.getElementById('result-placeholder').style.display='none';
  var rc=document.getElementById('result-content'); rc.style.display='block';
  var converged=data.converged, vars=data.variables||[], sol=data.solution||[], diag=data.diagnostics||{}, iters=data.iterations||[];
  var html='<div class="result-card">';
  if(data.domainTitle) html+='<div style="font-size:13px;font-weight:700;color:var(--accent);margin-bottom:12px;letter-spacing:.04em;text-transform:uppercase;">'+data.domainTitle+'</div>';
  html+='<div class="result-status '+(converged?'converged':'diverged')+'">'
      +'<span class="status-icon">'+(converged?'\u2705':'\u26A0\uFE0F')+'</span>'
      +'<span>'+(converged?'Converged in '+data.iterCount+' iterations':'Did not converge after '+data.iterCount+' iterations')+'</span></div>';
  html+='<div class="form-label">Solution</div><div class="solution-grid">';
  vars.forEach(function(v,i){
    html+='<div class="solution-var"><div class="solution-var-name">'+v+'</div>'
        +'<div class="solution-var-val">'+(sol[i]!==undefined?Number(sol[i]).toFixed(6):'\u2014')+'</div></div>';
  });
  html+='</div>';
  if(diag.stability){
    html+='<div class="form-label" style="margin-top:16px;">Diagnostics</div><div class="diag-grid">'
        +'<div class="diag-item"><div class="diag-key">Stability</div><div class="diag-val stability-'+diag.stability+'">'+diag.stability.toUpperCase()+'</div></div>'
        +'<div class="diag-item"><div class="diag-key">Residual \u2016r\u2016\u221E</div><div class="diag-val mono">'+Number(diag.residualInfinityNorm).toExponential(3)+'</div></div>'
        +'<div class="diag-item"><div class="diag-key">Strict Dominant Rows</div><div class="diag-val">'+diag.strictDominantRows+' / '+vars.length+'</div></div>'
        +'<div class="diag-item"><div class="diag-key">Min Margin</div><div class="diag-val mono">'+Number(diag.minimumMargin).toFixed(4)+'</div></div>'
        +'</div>';
  }
  html+='</div>';
  if(iters.length){
    html+='<div class="result-card"><div class="form-label">Iteration Log</div><div class="iter-table-wrap"><table class="iter-table"><thead><tr><th>Iter</th><th>Max \u03B4</th>';
    vars.forEach(function(v){html+='<th>'+v+'</th>';});
    html+='</tr></thead><tbody>';
    iters.slice(0,15).forEach(function(it){
      html+='<tr><td class="mono">'+it.number+'</td><td class="mono">'+Number(it.maxDelta).toExponential(3)+'</td>';
      (it.details||[]).forEach(function(d){html+='<td class="mono">'+Number(d.value).toFixed(6)+'</td>';});
      html+='</tr>';
    });
    if(iters.length>15) html+='<tr><td colspan="'+(vars.length+2)+'" style="text-align:center;color:var(--ink3);font-size:12px;padding:12px;">\u2026 '+(iters.length-15)+' more rows</td></tr>';
    html+='</tbody></table></div></div>';
  }
  rc.innerHTML=html;
  rc.scrollIntoView({behavior:'smooth',block:'nearest'});
}

function renderError(msg){
  document.getElementById('result-placeholder').style.display='none';
  var rc=document.getElementById('result-content'); rc.style.display='block';
  rc.innerHTML='<div class="result-card" style="border-color:#FCA5A5;">'
             +'<div class="result-status" style="background:#FEF2F2;color:#DC2626;">'
             +'<span class="status-icon">\u274C</span> Error: '+msg+'</div></div>';
}

async function loadHistory(){
  try{
    var res=await fetch('/history?limit=20');
    var data=await res.json();
    var list=document.getElementById('history-list');
    if(!data.history||!data.history.length){
      list.innerHTML='<div style="color:var(--ink3);font-size:14px;padding:32px 0;text-align:center;">No solves yet \u2014 run your first system above!</div>';
      return;
    }
    list.innerHTML=data.history.map(function(h){
      var vars=h.variables||[];
      var preview=vars.slice(0,3).map(function(v,i){return v+'='+Number(h.solution[i]).toFixed(4);}).join(', ');
      return '<div class="history-item">'
           +'<div><div style="font-size:14px;font-weight:600;color:var(--ink);margin-bottom:4px;">'+h.mode.toUpperCase()+' \u00B7 '+h.iterCount+' iters</div>'
           +'<div class="history-item-meta mono">'+preview+'</div>'
           +'<div class="history-item-meta" style="margin-top:3px;">'+h.createdAt+'</div></div>'
           +'<span class="history-badge '+(h.converged?'badge-converged':'badge-diverged')+'">'+(h.converged?'Converged':'Diverged')+'</span>'
           +'</div>';
    }).join('');
  } catch(e){
    document.getElementById('history-list').innerHTML='<div style="color:var(--ink3);font-size:14px;padding:24px 0;text-align:center;">Could not load history.</div>';
  }
}

async function clearHistory(){
  if(!confirm('Clear all solve history?')) return;
  await fetch('/history/clear',{method:'POST'});
  loadHistory();
}

function activateDomain(d){
  setMode('domain',document.querySelectorAll('.mode-tab')[2]);
  document.getElementById('domain-select').value=d;
  renderDomainFields();
  document.getElementById('solver').scrollIntoView({behavior:'smooth'});
}

buildMatrix();
renderDomainFields();
loadHistory();
</script>
</body>
</html>"""


# ────────────────────────────────────────────────
# Flask Routes
# ────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.json
        mode = data.get('mode')
        tol = float(data.get('tol', 1e-4))
        max_iter = int(data.get('maxIter', 1000))
        if tol <= 0:
            raise ValueError('Tolerance must be positive')
        if max_iter < 1:
            raise ValueError('Maximum iterations must be at least 1')

        if mode == 'matrix':
            A = data.get('A')
            b = data.get('b')
            n = len(A)
            variables = [chr(120 + i) for i in range(n)]
            input_payload = {'A': A, 'b': b}

        elif mode == 'equation':
            equations = data.get('equations')
            all_vars = set()
            for eq in equations:
                all_vars.update(re.findall(r'[a-zA-Z]', eq))
            variables = sorted(all_vars)
            var_to_idx = {v: i for i, v in enumerate(variables)}
            A, b = [], []
            for eq in equations:
                coeffs, const = parse_equation(eq, var_to_idx)
                A.append(coeffs)
                b.append(const)
            input_payload = {'equations': equations}

        elif mode == 'domain':
            domain_type = data.get('domainType')
            domain_params = data.get('domainParams', {})
            A, b, variables, domain_title = build_domain_system(domain_type, domain_params)
            input_payload = {'domainType': domain_type, 'domainParams': domain_params}

        else:
            return jsonify({'error': 'Invalid mode'}), 400

        A, b = prepare_matrix(A, b)
        result = gauss_seidel(A, b, variables, tol=tol, max_iter=max_iter)
        result['variables'] = variables
        result['diagnostics'] = analyze_system(A, b, result['solution'])
        if mode == 'domain':
            result['domainTitle'] = domain_title

        save_history_entry({'mode': mode, 'tol': tol, 'maxIter': max_iter,
                            'variables': variables, 'input': input_payload}, result)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/history', methods=['GET'])
def history():
    try:
        limit = max(1, min(100, int(request.args.get('limit', 20))))
        return jsonify({'history': fetch_history(limit)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/history/clear', methods=['POST'])
def clear_history_route():
    try:
        clear_history()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  GAUSS-SEIDEL SOLVER - WEB APPLICATION")
    print("=" * 50)
    print("\nStarting server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server\n")
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)