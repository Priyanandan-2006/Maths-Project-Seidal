"""
Gauss-Seidel Solver Web Application
Integrates the original Python solver with the HTML interface
"""

from flask import Flask, render_template_string, request, jsonify
import re
import json
import sqlite3
from datetime import datetime
from math import isclose

app = Flask(__name__)


DB_PATH = 'solver_history.db'


def init_db() -> None:
    """Create local SQLite table for solve history."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            '''
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
            '''
        )
        conn.commit()


def save_history_entry(entry: dict, result: dict) -> None:
    """Persist one successful solve call into the history table."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            '''
            INSERT INTO solve_history (
                created_at, mode, tol, max_iter, variables, input_payload,
                converged, iter_count, solution, diagnostics
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                entry['mode'],
                entry['tol'],
                entry['maxIter'],
                json.dumps(entry['variables']),
                json.dumps(entry['input']),
                int(result['converged']),
                int(result['iterCount']),
                json.dumps(result['solution']),
                json.dumps(result.get('diagnostics', {})),
            ),
        )
        conn.commit()


def fetch_history(limit: int = 20) -> list:
    """Return latest solve history entries."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            '''
            SELECT id, created_at, mode, tol, max_iter, variables, input_payload,
                   converged, iter_count, solution, diagnostics
            FROM solve_history
            ORDER BY id DESC
            LIMIT ?
            ''',
            (limit,),
        ).fetchall()

    items = []
    for row in rows:
        items.append({
            'id': row['id'],
            'createdAt': row['created_at'],
            'mode': row['mode'],
            'tol': row['tol'],
            'maxIter': row['max_iter'],
            'variables': json.loads(row['variables']),
            'input': json.loads(row['input_payload']),
            'converged': bool(row['converged']),
            'iterCount': row['iter_count'],
            'solution': json.loads(row['solution']),
            'diagnostics': json.loads(row['diagnostics']),
        })
    return items


def clear_history() -> None:
    """Delete all history entries."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('DELETE FROM solve_history')
        conn.commit()

init_db()

# ────────────────────────────────────────────────
# Parse single equation → coefficients + constant
# ────────────────────────────────────────────────
def parse_equation(eq: str, var_to_idx: dict) -> tuple:
    """Parse an equation string into coefficients and constant"""
    eq = eq.replace(" ", "")  # remove spaces
    if "=" not in eq:
        raise ValueError("Equation must contain '='")

    left, right_str = eq.split("=", 1)
    try:
        rhs = float(right_str)
    except ValueError:
        raise ValueError(f"Right-hand side is not a number: {right_str}")

    coeffs = [0.0] * len(var_to_idx)

    # Pattern matches:  -, +, 2.5x, -3y, .4z, x, -x, +x, etc.
    pattern = r"([+-]?)(\d*\.?\d*)([a-zA-Z])|([+-])([a-zA-Z])"
    i = 0
    implicit_sign = "+"  # first term is positive if no sign

    while i < len(left):
        m = re.match(pattern, left[i:])
        if not m:
            if left[i].isspace():
                i += 1
                continue
            raise ValueError(f"Cannot parse at position {i}: '{left[i:]}'")

        sign_str, num_str, var1, sign2, var2 = m.groups()
        sign = sign_str or sign2 or implicit_sign
        var = var1 or var2
        num = num_str.strip()

        if var not in var_to_idx:
            raise ValueError(f"Unknown variable '{var}'")

        idx = var_to_idx[var]

        if num == "" or num == ".":
            coeff = 1.0
        else:
            try:
                coeff = float(num)
            except ValueError:
                raise ValueError(f"Invalid coefficient '{num}' before {var}")

        if sign == "-":
            coeff = -coeff

        coeffs[idx] += coeff
        i += m.end()
        implicit_sign = "+"  # after first term

    return coeffs, rhs


# ────────────────────────────────────────────────
# Reorder rows → largest possible diagonal elements
# ────────────────────────────────────────────────
def prepare_matrix(A: list, b: list) -> tuple:
    """Prepare matrix by reordering rows for numerical stability"""
    n = len(A)
    A = [row[:] for row in A]  # deep copy
    b = b[:]

    for i in range(n):
        # Find row with largest abs value in column i (from current row down)
        max_row = i
        max_val = abs(A[i][i])

        for k in range(i + 1, n):
            if abs(A[k][i]) > max_val:
                max_val = abs(A[k][i])
                max_row = k

        # Swap rows
        if max_row != i:
            A[i], A[max_row] = A[max_row], A[i]
            b[i], b[max_row] = b[max_row], b[i]

        # Final safety check
        if isclose(A[i][i], 0, abs_tol=1e-12):
            raise ValueError(
                f"Near-zero pivot on diagonal at position {i} after reordering. "
                "System may be singular or requires different method."
            )

    return A, b


# ────────────────────────────────────────────────
# Gauss-Seidel iteration
# ────────────────────────────────────────────────
def gauss_seidel(
    A: list,
    b: list,
    variables: list,
    tol: float = 1e-4,
    max_iter: int = 1000,
) -> dict:
    """
    Solve system using Gauss-Seidel iteration
    Returns dict with solution and iteration details
    """
    n = len(A)
    x = [0.0] * n
    iteration = 0
    iterations_log = []

    while iteration < max_iter:
        iteration += 1
        x_new = x.copy()
        max_delta = 0.0
        details = []

        for i in range(n):
            s = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            if abs(A[i][i]) < 1e-12:
                raise ValueError(f"Zero division prevented at variable {variables[i]}")

            x_new[i] = (b[i] - s) / A[i][i]
            delta = abs(x_new[i] - x[i])
            max_delta = max(max_delta, delta)

            details.append({
                'variable': variables[i],
                'value': x_new[i],
                'delta': delta
            })

        x = x_new

        # Store iteration details (first 10 or every 20th)
        if iteration <= 10 or iteration % 20 == 0:
            iterations_log.append({
                'number': iteration,
                'details': details,
                'maxDelta': max_delta
            })

        if max_delta < tol:
            return {
                'converged': True,
                'iterations': iterations_log,
                'solution': x,
                'iterCount': iteration
            }

    return {
        'converged': False,
        'iterations': iterations_log,
        'solution': x,
        'iterCount': max_iter
    }


def analyze_system(A: list, b: list, x: list) -> dict:
    """Return diagnostics that explain numerical stability and solution quality."""
    n = len(A)
    row_metrics = []
    min_margin = float('inf')
    strict_rows = 0
    weak_rows = 0

    for i in range(n):
        diag = abs(A[i][i])
        off_diag = sum(abs(A[i][j]) for j in range(n) if j != i)
        margin = diag - off_diag
        ratio = (diag / off_diag) if off_diag > 0 else float('inf')
        min_margin = min(min_margin, margin)

        if diag > off_diag:
            strict_rows += 1
            status = 'strict'
        elif isclose(diag, off_diag, rel_tol=1e-12, abs_tol=1e-12):
            weak_rows += 1
            status = 'weak'
        else:
            status = 'not_dominant'

        row_metrics.append({
            'row': i + 1,
            'diagAbs': diag,
            'offDiagAbs': off_diag,
            'margin': margin,
            'ratio': ratio,
            'status': status,
        })

    residual_vector = []
    for i in range(n):
        lhs = sum(A[i][j] * x[j] for j in range(n))
        residual_vector.append(lhs - b[i])

    inf_norm = max(abs(r) for r in residual_vector) if residual_vector else 0.0

    if strict_rows == n:
        stability = 'excellent'
    elif strict_rows + weak_rows == n:
        stability = 'fair'
    else:
        stability = 'risky'

    return {
        'rowMetrics': row_metrics,
        'strictDominantRows': strict_rows,
        'weakDominantRows': weak_rows,
        'minimumMargin': min_margin,
        'residualVector': residual_vector,
        'residualInfinityNorm': inf_norm,
        'stability': stability,
    }




def build_domain_system(domain: str, params: dict) -> tuple:
    """Build domain-specific linear systems to solve with Gauss-Seidel."""
    domain = (domain or '').strip().lower()

    if domain == 'electrical':
        r1 = float(params.get('r1', 12))
        r2 = float(params.get('r2', 10))
        r3 = float(params.get('r3', 8))
        rm = float(params.get('rm', 2))
        v1 = float(params.get('v1', 24))
        v2 = float(params.get('v2', 18))
        v3 = float(params.get('v3', 12))

        variables = ['I1', 'I2', 'I3']
        A = [
            [r1 + rm + 1, -rm, 0],
            [-rm, r2 + (2 * rm) + 1, -rm],
            [0, -rm, r3 + rm + 1],
        ]
        b = [v1, v2, v3]
        title = 'Electrical Circuit Solver (Mesh Currents)'

    elif domain == 'structural':
        w = float(params.get('w', 120))
        xbar = float(params.get('xbar', 4))
        l1 = float(params.get('l1', 3))
        l2 = float(params.get('l2', 8))
        ka = float(params.get('ka', 2.0))
        kb = float(params.get('kb', 1.6))
        kc = float(params.get('kc', 1.2))
        lateral = float(params.get('lateral', 18))

        variables = ['RA', 'RB', 'RC']
        A = [
            [1, 1, 1],
            [0, l1, l2],
            [ka + 1, -(kb + 0.5), kc + 1],
        ]
        b = [w, w * xbar, lateral]
        title = 'Structural Engineering Load Reactions'

    elif domain == 'economics':
        d1 = float(params.get('d1', 130))
        d2 = float(params.get('d2', 125))
        d3 = float(params.get('d3', 120))
        variables = ['P1', 'P2', 'P3']
        A = [
            [10, -2, -1],
            [-1, 11, -2],
            [-2, -1, 12],
        ]
        b = [d1, d2, d3]
        title = 'Economics Equilibrium Model (Market Prices)'

    elif domain == 'chemical':
        c1 = float(params.get('c1', 42))
        c2 = float(params.get('c2', 38))
        c3 = float(params.get('c3', 30))
        variables = ['A', 'B', 'C']
        A = [
            [9, -2, -1],
            [-1, 10, -2],
            [-2, -1, 9],
        ]
        b = [c1, c2, c3]
        title = 'Chemical Balance System (Species Rates)'

    else:
        raise ValueError('Unknown domain model selected')

    return A, b, variables, title


# ────────────────────────────────────────────────
# Web Routes
# ────────────────────────────────────────────────

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gauss-Seidel Solver</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: radial-gradient(circle at top left, #7c3aed 0%, transparent 45%),
                        radial-gradient(circle at 85% 15%, #0ea5e9 0%, transparent 35%),
                        linear-gradient(135deg, #1e1b4b 0%, #312e81 45%, #4c1d95 100%);
            min-height: 100vh;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            position: relative;
        }

        body::before {
            content: '';
            position: fixed;
            inset: 0;
            background-image: linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px);
            background-size: 30px 30px;
            mask-image: radial-gradient(circle at center, black, transparent 85%);
            pointer-events: none;
        }

        .container {
            background: linear-gradient(145deg, rgba(255,255,255,0.9), rgba(224,231,255,0.85));
            backdrop-filter: blur(12px);
            border-radius: 22px;
            box-shadow: 0 30px 70px rgba(15, 23, 42, 0.45), 0 0 0 1px rgba(129,140,248,0.35), 0 0 40px rgba(56,189,248,0.2);
            max-width: 1100px;
            width: 100%;
            padding: 48px;
            animation: slideIn 0.4s ease-out;
            position: relative;
            z-index: 2;
            border: 1px solid rgba(255, 255, 255, 0.8);
            overflow: hidden;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .header {
            margin-bottom: 28px;
            padding: 28px;
            border-radius: 18px;
            background: linear-gradient(135deg, #312e81 0%, #5b21b6 52%, #0284c7 100%);
            box-shadow: 0 12px 28px rgba(49, 46, 129, 0.35);
            color: white;
            position: relative;
            overflow: hidden;
        }

        .header::after {
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 80% 20%, rgba(255,255,255,0.25), transparent 50%);
            pointer-events: none;
        }

        h1 {
            color: #ffffff;
            font-size: 34px;
            font-weight: 800;
            margin-bottom: 8px;
            letter-spacing: -0.6px;
            text-shadow: 0 4px 20px rgba(15, 23, 42, 0.25);
        }

        .subtitle {
            color: rgba(255,255,255,0.92);
            font-size: 14px;
            font-weight: 500;
            max-width: 640px;
        }

        .hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 16px;
            position: relative;
            z-index: 1;
        }

        .hero-tag {
            border: 1px solid rgba(255,255,255,0.45);
            color: white;
            background: rgba(255,255,255,0.14);
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: 700;
            backdrop-filter: blur(3px);
            letter-spacing: 0.2px;
        }

        .mode-selector {
            display: inline-flex;
            background: #eef2ff;
            border-radius: 12px;
            padding: 5px;
            margin-bottom: 26px;
            box-shadow: inset 0 2px 6px rgba(79, 70, 229, 0.15);
            border: 1px solid #c7d2fe;
        }

        .mode-btn {
            padding: 12px 28px;
            border: none;
            background: transparent;
            color: #6b7280;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s ease;
            position: relative;
        }

        .mode-btn:hover:not(.active) {
            color: #111827;
            background: rgba(255, 255, 255, 0.5);
        }

        .mode-btn.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4), 0 2px 4px rgba(102, 126, 234, 0.2);
            transform: translateY(-1px);
        }

        .input-section {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #dbeafe;
            position: relative;
            padding: 28px;
            border-radius: 14px;
            margin-bottom: 24px;
            animation: expandIn 0.3s ease-out;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.08);
        }

        .input-section::before {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: 14px;
            border: 1px solid rgba(99, 102, 241, 0.15);
            pointer-events: none;
        }

        @keyframes expandIn {
            from {
                opacity: 0;
                transform: scaleY(0.95);
            }
            to {
                opacity: 1;
                transform: scaleY(1);
            }
        }

        .form-group {
            margin-bottom: 24px;
        }

        .form-group:last-child {
            margin-bottom: 0;
        }

        label {
            display: block;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }

        input[type="number"],
        input[type="text"] {
            width: 100%;
            padding: 11px 14px;
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            font-size: 14px;
            font-family: 'SF Mono', Monaco, Consolas, monospace;
            transition: all 0.3s ease;
            background: white;
            color: #111827;
        }

        input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15), 0 4px 12px rgba(102, 126, 234, 0.2);
            transform: translateY(-1px);
        }

        input:hover:not(:focus) {
            border-color: #d1d5db;
        }

        input::placeholder {
            color: #9ca3af;
        }

        .matrix-input {
            display: grid;
            gap: 8px;
            margin-top: 12px;
        }

        .matrix-row {
            display: flex;
            gap: 8px;
            animation: slideInRow 0.25s ease-out backwards;
        }

        @keyframes slideInRow {
            from {
                opacity: 0;
                transform: translateX(-15px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        .matrix-row input {
            flex: 1;
            text-align: center;
        }

        .equation-input {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
            animation: slideInRow 0.25s ease-out backwards;
        }

        .equation-input input {
            flex: 1;
        }

        .equation-number {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 36px;
            height: 36px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 14px;
            flex-shrink: 0;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }

        .solve-btn {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
            position: relative;
            overflow: hidden;
        }

        .solve-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            transition: left 0.5s ease;
        }

        .solve-btn:hover:not(:disabled)::before {
            left: 100%;
        }

        .solve-btn:hover:not(:disabled) {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
            box-shadow: 0 6px 24px rgba(102, 126, 234, 0.5);
            transform: translateY(-2px);
        }

        .solve-btn:active:not(:disabled) {
            transform: translateY(0);
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
        }

        .solve-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .results {
            margin-top: 32px;
            animation: slideIn 0.4s ease-out;
        }

        .results-header {
            background: linear-gradient(135deg, #3730a3 0%, #7c3aed 60%, #0284c7 100%);
            color: white;
            padding: 18px 24px;
            border-radius: 12px 12px 0 0;
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 0.5px;
            box-shadow: 0 8px 20px rgba(67, 56, 202, 0.35);
        }

        .results-content {
            background: white;
            border: 2px solid #e5e7eb;
            border-top: none;
            padding: 24px;
            border-radius: 0 0 10px 10px;
            max-height: 500px;
            overflow-y: auto;
        }

        .results-content::-webkit-scrollbar {
            width: 10px;
        }

        .results-content::-webkit-scrollbar-track {
            background: linear-gradient(180deg, #f3f4f6 0%, #e5e7eb 100%);
            border-radius: 5px;
        }

        .results-content::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
            border-radius: 5px;
        }

        .results-content::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(180deg, #764ba2 0%, #667eea 100%);
        }

        .iteration-block {
            background: linear-gradient(135deg, #fafbfc 0%, #f8f9fa 100%);
            border: 2px solid transparent;
            border-left: 4px solid;
            border-image: linear-gradient(180deg, #667eea, #764ba2) 1;
            padding: 20px;
            margin-bottom: 16px;
            border-radius: 10px;
            animation: slideInRow 0.25s ease-out;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.08);
        }

        .iteration-header {
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 16px;
            font-size: 15px;
            padding-bottom: 12px;
            border-bottom: 2px solid #f3f4f6;
        }

        .variable-value {
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #f3f4f6;
            font-family: 'SF Mono', Monaco, Consolas, monospace;
            font-size: 13px;
        }

        .variable-value:last-child {
            border-bottom: none;
        }

        .variable-name {
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .variable-number {
            color: #764ba2;
            font-weight: 600;
        }

        .delta {
            color: #6b7280;
            font-size: 12px;
            background: #f3f4f6;
            padding: 2px 8px;
            border-radius: 4px;
        }

        .max-change {
            margin-top: 16px;
            padding-top: 16px;
            border-top: 2px solid #f3f4f6;
            color: #374151;
            font-size: 13px;
            font-family: 'SF Mono', Monaco, Consolas, monospace;
            font-weight: 600;
        }

        .final-solution {
            background: linear-gradient(135deg, #10b981 0%, #059669 50%, #047857 100%);
            color: white;
            padding: 28px;
            border-radius: 12px;
            margin-top: 16px;
            box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4);
            position: relative;
            overflow: hidden;
        }

        .final-solution::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
            pointer-events: none;
        }

        .final-solution h3 {
            margin-bottom: 20px;
            font-size: 18px;
            font-weight: 700;
            letter-spacing: 0.5px;
            position: relative;
        }

        .solution-item {
            padding: 14px 0;
            font-family: 'SF Mono', Monaco, Consolas, monospace;
            font-size: 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            display: flex;
            justify-content: space-between;
            position: relative;
        }

        .solution-item:last-child {
            border-bottom: none;
        }

        .error-message {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border: 2px solid #fca5a5;
            color: #991b1b;
            padding: 16px;
            border-radius: 10px;
            margin-top: 16px;
            font-size: 14px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
        }

        .error-message strong {
            font-weight: 700;
            color: #7f1d1d;
        }

        .loading {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
            margin-right: 8px;
            vertical-align: middle;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .info-box {
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border: 2px solid #93c5fd;
            padding: 14px;
            border-radius: 8px;
            margin-top: 16px;
            color: #1e40af;
            font-size: 13px;
        }

        .info-box strong {
            font-weight: 700;
            color: #1e3a8a;
        }

        .badge {
            display: inline-block;
            padding: 6px 14px;
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(10px);
            color: white;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            margin-left: 12px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }



        .action-row {
            display: grid;
            grid-template-columns: 1fr 180px 180px;
            gap: 10px;
            margin-top: 6px;
        }

        .secondary-btn {
            padding: 14px;
            border: none;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            color: #1f2937;
            background: linear-gradient(135deg, #e5e7eb 0%, #d1d5db 100%);
        }

        .secondary-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(17, 24, 39, 0.15);
        }

        .clear-btn {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            color: #7f1d1d;
        }

        .history-panel {
            margin-top: 20px;
            background: linear-gradient(135deg, #eef2ff 0%, #f0f9ff 100%);
            border: 1px solid #bfdbfe;
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 8px 24px rgba(14, 116, 144, 0.12);
        }

        .history-item {
            border: 1px solid #c7d2fe;
            background: white;
            border-radius: 12px;
            padding: 14px;
            margin-bottom: 10px;
            font-size: 13px;
            color: #1f2937;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.08);
        }

        .history-item:last-child {
            margin-bottom: 0;
        }

        .history-title {
            font-weight: 800;
            color: #312e81;
            margin-bottom: 10px;
            font-size: 15px;
            letter-spacing: 0.3px;
        }

        .history-meta {
            font-family: 'SF Mono', Monaco, Consolas, monospace;
            color: #374151;
            margin-bottom: 6px;
        }

        .history-status {
            display: inline-block;
            margin-top: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.2px;
            text-transform: uppercase;
        }

        .history-status.ok {
            background: #dcfce7;
            color: #166534;
        }

        .history-status.warn {
            background: #fee2e2;
            color: #991b1b;
        }


        .domain-card {
            background: linear-gradient(135deg, #ecfeff 0%, #eef2ff 100%);
            border: 1px solid #bae6fd;
            border-radius: 12px;
            padding: 16px;
            margin-top: 12px;
        }

        .domain-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }

        .domain-note {
            margin-top: 10px;
            color: #0f172a;
            font-size: 13px;
        }

        .result-domain {
            background: linear-gradient(135deg, #ede9fe 0%, #dbeafe 100%);
            border: 1px solid #c4b5fd;
            color: #3730a3;
            border-radius: 10px;
            padding: 10px 14px;
            margin-top: 14px;
            font-size: 13px;
            font-weight: 700;
        }


        .explain-panel {
            margin-top: 16px;
            background: linear-gradient(135deg, #ecfeff 0%, #eef2ff 100%);
            border: 1px solid #93c5fd;
            border-radius: 12px;
            padding: 16px;
            color: #0f172a;
        }

        .explain-panel h3 {
            color: #1e3a8a;
            margin-bottom: 8px;
            font-size: 16px;
        }

        .explain-panel ul {
            margin-left: 18px;
            margin-top: 8px;
            font-size: 13px;
            line-height: 1.6;
        }

        .explain-panel code {
            background: #e0e7ff;
            border-radius: 4px;
            padding: 1px 6px;
            font-size: 12px;
        }

        .diagnostics-panel {
            margin-top: 16px;
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border: 2px solid #fcd34d;
            border-radius: 12px;
            padding: 20px;
        }

        .diagnostics-panel h3 {
            color: #92400e;
            margin-bottom: 10px;
            font-size: 16px;
        }

        .diagnostics-panel p {
            color: #78350f;
            font-size: 13px;
            margin-bottom: 8px;
        }

        .diagnostics-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-family: 'SF Mono', Monaco, Consolas, monospace;
            font-size: 12px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }

        .diagnostics-table th,
        .diagnostics-table td {
            border: 1px solid #fde68a;
            padding: 8px;
            text-align: center;
        }

        .diagnostics-table th {
            background: #fef3c7;
            color: #78350f;
            font-weight: 700;
        }

        .status-pill {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }

        .status-strict {
            background: #dcfce7;
            color: #166534;
        }

        .status-weak {
            background: #fef3c7;
            color: #92400e;
        }

        .status-not_dominant {
            background: #fee2e2;
            color: #991b1b;
        }

        .math-sky {
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            overflow: hidden;
        }

        .math-symbol {
            position: absolute;
            color: rgba(255,255,255,0.8);
            font-weight: 800;
            text-shadow: 0 0 16px rgba(147,197,253,0.9);
            animation: floatMath var(--dur, 16s) linear infinite;
            animation-delay: var(--delay, 0s);
            transform: translateY(120vh) rotate(0deg);
            user-select: none;
        }

        @keyframes floatMath {
            0% {
                transform: translateY(115vh) translateX(0) rotate(0deg) scale(0.9);
                opacity: 0;
            }
            10% { opacity: 0.85; }
            50% {
                transform: translateY(50vh) translateX(20px) rotate(180deg) scale(1.05);
            }
            100% {
                transform: translateY(-20vh) translateX(-14px) rotate(360deg) scale(0.9);
                opacity: 0;
            }
        }

        .header::before {
            content: '🤖 solving equations in hyperspace...';
            position: absolute;
            right: 18px;
            bottom: 12px;
            font-size: 12px;
            font-weight: 700;
            color: rgba(255,255,255,0.9);
            background: rgba(15,23,42,0.28);
            border: 1px solid rgba(255,255,255,0.35);
            border-radius: 999px;
            padding: 6px 12px;
            animation: pulseTag 1.8s ease-in-out infinite;
        }

        @keyframes pulseTag {
            0%, 100% { transform: scale(1); opacity: 0.8; }
            50% { transform: scale(1.04); opacity: 1; }
        }

        @media (max-width: 768px) {
            body {
                padding: 20px 16px;
            }

            .container {
                padding: 28px 24px;
            }

            h1 {
                font-size: 24px;
            }

            .mode-selector {
                width: 100%;
            }

            .mode-btn {
                flex: 1;
            }

            .action-row {
                grid-template-columns: 1fr;
            }

            .header::before {
                position: static;
                display: inline-block;
                margin-top: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="math-sky" aria-hidden="true">
        <span class="math-symbol" style="left:6%;font-size:32px;--dur:18s;--delay:0s">∑</span>
        <span class="math-symbol" style="left:14%;font-size:24px;--dur:14s;--delay:2s">π</span>
        <span class="math-symbol" style="left:22%;font-size:28px;--dur:17s;--delay:4s">√</span>
        <span class="math-symbol" style="left:34%;font-size:26px;--dur:20s;--delay:1s">∞</span>
        <span class="math-symbol" style="left:46%;font-size:30px;--dur:15s;--delay:3s">x²</span>
        <span class="math-symbol" style="left:58%;font-size:22px;--dur:16s;--delay:5s">Δ</span>
        <span class="math-symbol" style="left:68%;font-size:30px;--dur:19s;--delay:2.5s">∫</span>
        <span class="math-symbol" style="left:78%;font-size:24px;--dur:13s;--delay:6s">≈</span>
        <span class="math-symbol" style="left:88%;font-size:28px;--dur:21s;--delay:1.5s">θ</span>
    </div>
    <div class="container">
        <div class="header">
            <h1>Gauss-Seidel Iterative Solver</h1>
            <p class="subtitle">Numerical solution for systems of linear equations with live diagnostics and saved history.</p>
            <div class="hero-tags">
                <span class="hero-tag">Fast Iteration Tracking</span>
                <span class="hero-tag">Residual Diagnostics</span>
                <span class="hero-tag">Persistent History</span>
            </div>
        </div>

        <div class="mode-selector">
            <button class="mode-btn active" onclick="setMode('matrix')" id="matrixBtn">
                Matrix Form
            </button>
            <button class="mode-btn" onclick="setMode('equation')" id="equationBtn">
                Equation Form
            </button>
            <button class="mode-btn" onclick="setMode('domain')" id="domainBtn">
                Domain Models
            </button>
        </div>

        <div id="matrixMode" class="input-section">
            <div class="form-group">
                <label>Number of Variables:</label>
                <input type="number" id="matrixSize" min="2" max="10" value="3" onchange="generateMatrixInputs()">
            </div>

            <div class="form-group">
                <label>Coefficient Matrix (A):</label>
                <div id="matrixInputs"></div>
            </div>

            <div class="form-group">
                <label>Constants Vector (b):</label>
                <div id="constantInputs"></div>
            </div>
        </div>

        <div id="equationMode" class="input-section" style="display: none;">
            <div class="form-group">
                <label>Number of Equations:</label>
                <input type="number" id="numEquations" min="2" max="10" value="3" onchange="generateEquationInputs()">
            </div>

            <div class="form-group">
                <label>Enter Equations (e.g., 2x + 3y - z = 5):</label>
                <div id="equationInputs"></div>
            </div>

            <div class="info-box">
                <strong>Input format:</strong> Enter equations in standard form (e.g., 2x + 3y - z = 5). Variables can be any letters.
            </div>
        </div>

        <div id="domainMode" class="input-section" style="display: none;">
            <div class="form-group">
                <label>Engineering/Science Model:</label>
                <select id="domainType" onchange="generateDomainInputs()" style="width:100%;padding:11px 14px;border:2px solid #e5e7eb;border-radius:8px;font-size:14px;background:white;">
                    <option value="electrical">Electrical Circuit Solver</option>
                    <option value="structural">Structural Engineering Loads</option>
                    <option value="economics">Economics Equilibrium Model</option>
                    <option value="chemical">Chemical Balance Systems</option>
                </select>
            </div>
            <div class="domain-card">
                <div id="domainInputs" class="domain-grid"></div>
                <div id="domainDescription" class="domain-note"></div>
            </div>


        <div class="input-section" style="margin-top: 0;">
            <div class="form-group">
                <label>Tolerance:</label>
                <input type="number" id="tolerance" step="any" min="0.0000001" value="0.0001">
            </div>
            <div class="form-group">
                <label>Maximum Iterations:</label>
                <input type="number" id="maxIterations" min="1" max="5000" value="1000">
            </div>
        </div>

        <div class="action-row">
            <button class="solve-btn" onclick="solve()">
                <span id="solveText">Compute Solution</span>
            </button>
            <button class="secondary-btn" onclick="toggleHistory()">View History</button>
            <button class="secondary-btn clear-btn" onclick="clearHistoryRecords()">Clear History</button>
        </div>
        <div style="margin-top:10px;">
            <button class="secondary-btn" onclick="toggleExplanation()" style="width:100%;">📘 Explanation: How this was solved</button>
        </div>

        <div id="results" style="display: none;"></div>
        <div id="explanationPanel" class="explain-panel" style="display: none;"></div>
        <div id="historyPanel" class="history-panel" style="display: none;"></div>
    </div>

    <script>
        let currentMode = 'matrix';
        let lastResult = null;

        function setMode(mode) {
            currentMode = mode;
            document.getElementById('matrixBtn').classList.toggle('active', mode === 'matrix');
            document.getElementById('equationBtn').classList.toggle('active', mode === 'equation');
            document.getElementById('domainBtn').classList.toggle('active', mode === 'domain');
            document.getElementById('matrixMode').style.display = mode === 'matrix' ? 'block' : 'none';
            document.getElementById('equationMode').style.display = mode === 'equation' ? 'block' : 'none';
            document.getElementById('domainMode').style.display = mode === 'domain' ? 'block' : 'none';
            
            if (mode === 'matrix') {
                generateMatrixInputs();
            } else if (mode === 'equation') {
                generateEquationInputs();
            } else {
                generateDomainInputs();
            }
        }

        function generateMatrixInputs() {
            const n = parseInt(document.getElementById('matrixSize').value);
            const matrixInputs = document.getElementById('matrixInputs');
            const constantInputs = document.getElementById('constantInputs');
            
            matrixInputs.innerHTML = '';
            constantInputs.innerHTML = '';

            const matrixContainer = document.createElement('div');
            matrixContainer.className = 'matrix-input';

            for (let i = 0; i < n; i++) {
                const row = document.createElement('div');
                row.className = 'matrix-row';
                row.style.animationDelay = `${i * 0.05}s`;

                for (let j = 0; j < n; j++) {
                    const input = document.createElement('input');
                    input.type = 'number';
                    input.step = 'any';
                    input.placeholder = `a${i+1}${j+1}`;
                    input.id = `a${i}_${j}`;
                    input.value = i === j ? '10' : '1';
                    row.appendChild(input);
                }
                matrixContainer.appendChild(row);
            }
            matrixInputs.appendChild(matrixContainer);

            const constantRow = document.createElement('div');
            constantRow.className = 'matrix-row';
            for (let i = 0; i < n; i++) {
                const input = document.createElement('input');
                input.type = 'number';
                input.step = 'any';
                input.placeholder = `b${i+1}`;
                input.id = `b${i}`;
                input.value = '10';
                constantRow.appendChild(input);
            }
            constantInputs.appendChild(constantRow);
        }

        function generateEquationInputs() {
            const n = parseInt(document.getElementById('numEquations').value);
            const container = document.getElementById('equationInputs');
            container.innerHTML = '';

            const examples = [
                '10x + y + z = 12',
                'x + 10y + z = 12',
                'x + y + 10z = 12'
            ];

            for (let i = 0; i < n; i++) {
                const div = document.createElement('div');
                div.className = 'equation-input';
                div.style.animationDelay = `${i * 0.05}s`;

                const number = document.createElement('div');
                number.className = 'equation-number';
                number.textContent = i + 1;

                const input = document.createElement('input');
                input.type = 'text';
                input.id = `eq${i}`;
                input.placeholder = examples[i] || `Enter equation ${i+1}`;
                input.value = examples[i] || '';

                div.appendChild(number);
                div.appendChild(input);
                container.appendChild(div);
            }
        }

        function getDomainConfig(type) {
            const configs = {
                electrical: {
                    description: 'Solve mesh currents from Kirchhoff voltage laws. Great for multi-loop circuit analysis.',
                    fields: [
                        ['r1', 'R1 (Ω)', 12], ['r2', 'R2 (Ω)', 10], ['r3', 'R3 (Ω)', 8],
                        ['rm', 'Mutual Rm (Ω)', 2], ['v1', 'V1 (V)', 24], ['v2', 'V2 (V)', 18], ['v3', 'V3 (V)', 12],
                    ],
                },
                structural: {
                    description: 'Estimate support reactions using load balance + moment equilibrium + lateral compatibility.',
                    fields: [
                        ['w', 'Total Load W (kN)', 120], ['xbar', 'Centroid x̄ (m)', 4], ['l1', 'L1 (m)', 3], ['l2', 'L2 (m)', 8],
                        ['ka', 'Stiffness kA', 2.0], ['kb', 'Stiffness kB', 1.6], ['kc', 'Stiffness kC', 1.2], ['lateral', 'Lateral eq. term', 18],
                    ],
                },
                economics: {
                    description: 'Find market equilibrium prices for three interdependent sectors from linearized demand-supply equations.',
                    fields: [
                        ['d1', 'Demand Constant D1', 130], ['d2', 'Demand Constant D2', 125], ['d3', 'Demand Constant D3', 120],
                    ],
                },
                chemical: {
                    description: 'Balance three species rates under coupled reaction constraints (steady-state approximation).',
                    fields: [
                        ['c1', 'Constraint C1', 42], ['c2', 'Constraint C2', 38], ['c3', 'Constraint C3', 30],
                    ],
                },
            };
            return configs[type] || configs.electrical;
        }

        function generateDomainInputs() {
            const type = document.getElementById('domainType').value;
            const config = getDomainConfig(type);
            const container = document.getElementById('domainInputs');
            const description = document.getElementById('domainDescription');
            container.innerHTML = '';
            description.textContent = config.description;

            for (const [name, label, val] of config.fields) {
                const wrap = document.createElement('div');
                const l = document.createElement('label');
                l.textContent = label;
                l.style.marginBottom = '6px';
                const input = document.createElement('input');
                input.type = 'number';
                input.step = 'any';
                input.id = `domain_${name}`;
                input.value = val;
                wrap.appendChild(l);
                wrap.appendChild(input);
                container.appendChild(wrap);
            }
        }

        async function solve() {
            const resultsDiv = document.getElementById('results');
            const solveBtn = document.querySelector('.solve-btn');
            const solveText = document.getElementById('solveText');

            solveBtn.disabled = true;
            solveText.innerHTML = '<span class="loading"></span>Computing...';
            resultsDiv.style.display = 'none';

            try {
                const tolerance = parseFloat(document.getElementById('tolerance').value);
                const maxIterations = parseInt(document.getElementById('maxIterations').value);

                if (isNaN(tolerance) || tolerance <= 0) throw new Error('Tolerance must be a positive number');
                if (isNaN(maxIterations) || maxIterations < 1) throw new Error('Maximum iterations must be at least 1');

                let requestData = { mode: currentMode, tol: tolerance, maxIter: maxIterations };

                if (currentMode === 'matrix') {
                    const n = parseInt(document.getElementById('matrixSize').value);
                    const A = [];
                    const b = [];

                    for (let i = 0; i < n; i++) {
                        const row = [];
                        for (let j = 0; j < n; j++) {
                            const val = parseFloat(document.getElementById(`a${i}_${j}`).value);
                            if (isNaN(val)) throw new Error('Invalid matrix entry');
                            row.push(val);
                        }
                        A.push(row);

                        const bVal = parseFloat(document.getElementById(`b${i}`).value);
                        if (isNaN(bVal)) throw new Error('Invalid constant');
                        b.push(bVal);
                    }

                    requestData.A = A;
                    requestData.b = b;
                } else if (currentMode === 'equation') {
                    const n = parseInt(document.getElementById('numEquations').value);
                    const equations = [];

                    for (let i = 0; i < n; i++) {
                        const eq = document.getElementById(`eq${i}`).value.trim();
                        if (!eq) throw new Error(`Equation ${i+1} is empty`);
                        equations.push(eq);
                    }

                    requestData.equations = equations;
                } else {
                    const domainType = document.getElementById('domainType').value;
                    const config = getDomainConfig(domainType);
                    const params = {};
                    for (const [name] of config.fields) {
                        const val = parseFloat(document.getElementById(`domain_${name}`).value);
                        if (isNaN(val)) throw new Error(`Invalid domain value: ${name}`);
                        params[name] = val;
                    }
                    requestData.domainType = domainType;
                    requestData.domainParams = params;
                }

                const response = await fetch('/solve', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestData)
                });

                const result = await response.json();

                if (result.error) {
                    throw new Error(result.error);
                }

                lastResult = result;
                displayResults(result);
                await loadHistory();

            } catch (error) {
                resultsDiv.innerHTML = `<div class="error-message"><strong>Error:</strong> ${error.message}</div>`;
                resultsDiv.style.display = 'block';
            } finally {
                solveBtn.disabled = false;
                solveText.innerHTML = 'Compute Solution';
            }
        }


        async function loadHistory() {
            const historyPanel = document.getElementById('historyPanel');
            const response = await fetch('/history');
            const payload = await response.json();

            if (payload.error) {
                historyPanel.innerHTML = `<div class="error-message"><strong>Error:</strong> ${payload.error}</div>`;
                return;
            }

            const items = payload.history || [];
            if (items.length === 0) {
                historyPanel.innerHTML = '<div class="history-item">No saved calculations yet. Solve a system to build your history.</div>';
                return;
            }

            let html = '<div class="history-title">Saved Calculation History</div>';
            for (const item of items) {
                const solutionText = item.variables
                    .map((name, idx) => `${name}=${Number(item.solution[idx]).toFixed(4)}`)
                    .join(', ');

                html += '<div class="history-item">';
                html += `<div class="history-meta">#${item.id} • ${item.createdAt}</div>`;
                html += `<div><strong>Mode:</strong> ${item.mode} | <strong>tol:</strong> ${item.tol} | <strong>maxIter:</strong> ${item.maxIter}</div>`;
                html += `<div><strong>Status:</strong> ${item.converged ? 'Converged' : 'Not converged'} in ${item.iterCount} iterations</div>`;
                html += `<span class="history-status ${item.converged ? 'ok' : 'warn'}">${item.converged ? 'stable result' : 'needs tuning'}</span>`;
                html += `<div><strong>Solution:</strong> ${solutionText}</div>`;
                if (item.diagnostics && item.diagnostics.residualInfinityNorm !== undefined) {
                    html += `<div><strong>Residual ||Ax-b||∞:</strong> ${Number(item.diagnostics.residualInfinityNorm).toExponential(3)}</div>`;
                }
                html += '</div>';
            }
            historyPanel.innerHTML = html;
        }

        async function toggleHistory() {
            const historyPanel = document.getElementById('historyPanel');
            const show = historyPanel.style.display === 'none';
            historyPanel.style.display = show ? 'block' : 'none';
            if (show) {
                await loadHistory();
            }
        }

        async function clearHistoryRecords() {
            await fetch('/history/clear', { method: 'POST' });
            const historyPanel = document.getElementById('historyPanel');
            if (historyPanel.style.display !== 'none') {
                await loadHistory();
            }
        }

        function buildExplanationHtml(result) {
            const domainText = result.domainTitle
                ? `This run used the domain template: <code>${result.domainTitle}</code>.`
                : 'This run used your direct matrix/equation inputs.';

            const firstLogged = (result.iterations && result.iterations.length > 0) ? result.iterations[0] : null;
            let firstIterHtml = '';
            if (firstLogged) {
                const samples = firstLogged.details
                    .map(d => `<li><code>${d.variable}</code> updated to <code>${Number(d.value).toFixed(6)}</code> (Δ=${Number(d.delta).toExponential(2)})</li>`)
                    .join('');
                firstIterHtml = `<p><strong>First logged iteration (Iteration ${firstLogged.number}):</strong></p><ul>${samples}</ul>`;
            }

            const diagnostics = result.diagnostics || {};
            const stability = diagnostics.stability ? diagnostics.stability.toUpperCase() : 'N/A';
            const residual = diagnostics.residualInfinityNorm !== undefined
                ? Number(diagnostics.residualInfinityNorm).toExponential(3)
                : 'N/A';

            return `
                <h3>How to do it (Gauss-Seidel method)</h3>
                <ul>
                    <li>Start with an initial guess for all unknowns (this app uses 0 for each variable).</li>
                    <li>Rearrange each equation in iterative form and update each variable sequentially.</li>
                    <li>After each full pass, compute max change <code>max |xᵢ(new)-xᵢ(old)|</code>.</li>
                    <li>Stop when max change is below tolerance or when max iterations is reached.</li>
                </ul>
                <p>${domainText}</p>
                ${firstIterHtml}
                <p><strong>Convergence summary:</strong> ${result.converged ? 'Converged' : 'Not converged'} in ${result.iterCount} iterations.</p>
                <p><strong>Diagnostics summary:</strong> Stability <code>${stability}</code>, residual norm <code>${residual}</code>.</p>
            `;
        }

        function toggleExplanation() {
            const panel = document.getElementById('explanationPanel');
            const show = panel.style.display === 'none';
            panel.style.display = show ? 'block' : 'none';

            if (!show) return;

            if (!lastResult) {
                panel.innerHTML = `
                    <h3>How to do it (Gauss-Seidel method)</h3>
                    <p>Run a solve first, then click this button again to see a step-by-step explanation using your own result.</p>
                    <ul>
                        <li>Pick mode (Matrix, Equation, or Domain Model).</li>
                        <li>Enter coefficients/constants and solver settings.</li>
                        <li>Press <code>Compute Solution</code>.</li>
                        <li>Open this explanation panel for a guided breakdown.</li>
                    </ul>
                `;
                return;
            }

            panel.innerHTML = buildExplanationHtml(lastResult);
        }

        function displayResults(result) {
            const resultsDiv = document.getElementById('results');
            let html = '<div class="results-header">Iteration Log</div>';
            html += '<div class="results-content">';

            for (const iter of result.iterations) {
                html += `<div class="iteration-block">`;
                html += `<div class="iteration-header">Iteration ${iter.number}</div>`;
                
                for (const detail of iter.details) {
                    html += `<div class="variable-value">`;
                    html += `<span><span class="variable-name">${detail.variable}</span> = <span class="variable-number">${detail.value.toFixed(6)}</span></span>`;
                    html += `<span class="delta">Δ ${detail.delta.toExponential(2)}</span>`;
                    html += `</div>`;
                }

                html += `<div class="max-change">`;
                html += `Maximum change: ${iter.maxDelta.toExponential(2)}`;
                html += `</div>`;
                html += `</div>`;
            }

            html += '</div>';

            if (result.domainTitle) {
                html += `<div class="result-domain">🚀 Domain Solver: ${result.domainTitle}</div>`;
            }

            if (result.diagnostics) {
                const d = result.diagnostics;
                const stabilityText = {
                    excellent: 'Excellent stability (strict diagonal dominance in every row).',
                    fair: 'Fair stability (some rows are weakly dominant).',
                    risky: 'Risky stability (non-dominant rows may slow or prevent convergence).'
                };

                html += '<div class="diagnostics-panel">';
                html += '<h3>Competition Edge: Convergence Diagnostics</h3>';
                html += `<p><strong>Stability grade:</strong> ${d.stability.toUpperCase()} — ${stabilityText[d.stability] || ''}</p>`;
                html += `<p><strong>Minimum dominance margin:</strong> ${d.minimumMargin.toExponential(3)}</p>`;
                html += `<p><strong>Residual ||Ax-b||∞:</strong> ${d.residualInfinityNorm.toExponential(3)}</p>`;

                html += '<table class="diagnostics-table">';
                html += '<tr><th>Row</th><th>|aᵢᵢ|</th><th>Σ|aᵢⱼ|, j≠i</th><th>Margin</th><th>Ratio</th><th>Status</th></tr>';
                for (const row of d.rowMetrics) {
                    const ratioText = Number.isFinite(row.ratio) ? row.ratio.toFixed(3) : '∞';
                    html += '<tr>';
                    html += `<td>${row.row}</td>`;
                    html += `<td>${row.diagAbs.toFixed(4)}</td>`;
                    html += `<td>${row.offDiagAbs.toFixed(4)}</td>`;
                    html += `<td>${row.margin.toExponential(2)}</td>`;
                    html += `<td>${ratioText}</td>`;
                    html += `<td><span class="status-pill status-${row.status}">${row.status.replace('_', ' ')}</span></td>`;
                    html += '</tr>';
                }
                html += '</table>';
                html += '</div>';
            }

            if (result.converged) {
                html += '<div class="final-solution">';
                html += `<h3>Solution <span class="badge">Converged in ${result.iterCount} iterations</span></h3>`;
                for (let i = 0; i < result.variables.length; i++) {
                    html += `<div class="solution-item">`;
                    html += `<span>${result.variables[i]}</span>`;
                    html += `<span>${result.solution[i].toFixed(6)}</span>`;
                    html += `</div>`;
                }
                html += '</div>';
            } else {
                html += '<div class="error-message">Solution did not converge within the maximum number of iterations.</div>';
            }

            resultsDiv.innerHTML = html;
            resultsDiv.style.display = 'block';
        }

        // Initialize
        generateMatrixInputs();
        generateDomainInputs();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/solve', methods=['POST'])
def solve():
    """Solve the linear system using Gauss-Seidel method"""
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
            variables = [chr(120 + i) for i in range(n)]  # x, y, z, ...
            input_payload = {'A': A, 'b': b}

        elif mode == 'equation':
            equations = data.get('equations')
            
            # Extract all variables
            all_vars = set()
            for eq in equations:
                matches = re.findall(r'[a-zA-Z]', eq)
                all_vars.update(matches)
            
            variables = sorted(all_vars)
            var_to_idx = {v: i for i, v in enumerate(variables)}
            
            # Parse equations
            A = []
            b = []
            for eq in equations:
                coeffs, const = parse_equation(eq, var_to_idx)
                A.append(coeffs)
                b.append(const)
            input_payload = {'equations': equations}
        

        elif mode == 'domain':
            domain_type = data.get('domainType')
            domain_params = data.get('domainParams', {})
            A, b, variables, domain_title = build_domain_system(domain_type, domain_params)
            input_payload = {
                'domainType': domain_type,
                'domainParams': domain_params,
            }

        else:
            return jsonify({'error': 'Invalid mode'}), 400

        # Prepare matrix
        A, b = prepare_matrix(A, b)

        # Solve
        result = gauss_seidel(A, b, variables, tol=tol, max_iter=max_iter)
        result['variables'] = variables
        result['diagnostics'] = analyze_system(A, b, result['solution'])
        if mode == 'domain':
            result['domainTitle'] = domain_title

        save_history_entry(
            {
                'mode': mode,
                'tol': tol,
                'maxIter': max_iter,
                'variables': variables,
                'input': input_payload,
            },
            result,
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/history', methods=['GET'])
def history():
    """Return latest solver calculations from SQLite history."""
    try:
        limit = int(request.args.get('limit', 20))
        limit = max(1, min(limit, 100))
        return jsonify({'history': fetch_history(limit)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/history/clear', methods=['POST'])
def clear_history_route():
    """Clear all persisted history entries."""
    try:
        clear_history()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  GAUSS-SEIDEL SOLVER - WEB APPLICATION")
    print("="*50)
    print("\nStarting server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server\n")
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
