"""
Gauss-Seidel Solver Web Application
Integrates the original Python solver with the HTML interface
"""

from flask import Flask, render_template_string, request, jsonify
import re
from math import isclose

app = Flask(__name__)

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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
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
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(252, 163, 17, 0.3) 0%, transparent 50%),
                radial-gradient(circle at 40% 20%, rgba(59, 130, 246, 0.3) 0%, transparent 50%);
            pointer-events: none;
        }

        .container {
            background: rgba(255, 255, 255, 0.98);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1), 0 2px 8px rgba(0, 0, 0, 0.05);
            max-width: 1000px;
            width: 100%;
            padding: 48px;
            animation: slideIn 0.4s ease-out;
            position: relative;
            border: 1px solid rgba(255, 255, 255, 0.5);
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
            margin-bottom: 40px;
            border-bottom: 2px solid transparent;
            border-image: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
            border-image-slice: 1;
            padding-bottom: 24px;
        }

        h1 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }

        .subtitle {
            color: #6b7280;
            font-size: 14px;
            font-weight: 400;
        }

        .mode-selector {
            display: inline-flex;
            background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            border-radius: 10px;
            padding: 4px;
            margin-bottom: 32px;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
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
            background: linear-gradient(135deg, #fafbfc 0%, #f8f9fa 100%);
            border: 2px solid transparent;
            background-clip: padding-box;
            position: relative;
            padding: 28px;
            border-radius: 12px;
            margin-bottom: 24px;
            animation: expandIn 0.3s ease-out;
        }

        .input-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            border-radius: 12px;
            padding: 2px;
            background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 18px 24px;
            border-radius: 10px 10px 0 0;
            font-size: 16px;
            font-weight: 700;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
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
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Gauss-Seidel Iterative Solver</h1>
            <p class="subtitle">Numerical solution for systems of linear equations</p>
        </div>

        <div class="mode-selector">
            <button class="mode-btn active" onclick="setMode('matrix')" id="matrixBtn">
                Matrix Form
            </button>
            <button class="mode-btn" onclick="setMode('equation')" id="equationBtn">
                Equation Form
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

        <button class="solve-btn" onclick="solve()">
            <span id="solveText">Compute Solution</span>
        </button>

        <div id="results" style="display: none;"></div>
    </div>

    <script>
        let currentMode = 'matrix';

        function setMode(mode) {
            currentMode = mode;
            document.getElementById('matrixBtn').classList.toggle('active', mode === 'matrix');
            document.getElementById('equationBtn').classList.toggle('active', mode === 'equation');
            document.getElementById('matrixMode').style.display = mode === 'matrix' ? 'block' : 'none';
            document.getElementById('equationMode').style.display = mode === 'equation' ? 'block' : 'none';
            
            if (mode === 'matrix') {
                generateMatrixInputs();
            } else {
                generateEquationInputs();
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

        async function solve() {
            const resultsDiv = document.getElementById('results');
            const solveBtn = document.querySelector('.solve-btn');
            const solveText = document.getElementById('solveText');

            solveBtn.disabled = true;
            solveText.innerHTML = '<span class="loading"></span>Computing...';
            resultsDiv.style.display = 'none';

            try {
                let requestData = { mode: currentMode };

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
                } else {
                    const n = parseInt(document.getElementById('numEquations').value);
                    const equations = [];

                    for (let i = 0; i < n; i++) {
                        const eq = document.getElementById(`eq${i}`).value.trim();
                        if (!eq) throw new Error(`Equation ${i+1} is empty`);
                        equations.push(eq);
                    }

                    requestData.equations = equations;
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

                displayResults(result);

            } catch (error) {
                resultsDiv.innerHTML = `<div class="error-message"><strong>Error:</strong> ${error.message}</div>`;
                resultsDiv.style.display = 'block';
            } finally {
                solveBtn.disabled = false;
                solveText.innerHTML = 'Compute Solution';
            }
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

        if mode == 'matrix':
            A = data.get('A')
            b = data.get('b')
            n = len(A)
            variables = [chr(120 + i) for i in range(n)]  # x, y, z, ...

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
        
        else:
            return jsonify({'error': 'Invalid mode'}), 400

        # Prepare matrix
        A, b = prepare_matrix(A, b)

        # Solve
        result = gauss_seidel(A, b, variables)
        result['variables'] = variables

        return jsonify(result)

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
