from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from typing import Any
import json
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import numba
from numba import njit

# Create your views here.

def index(request) -> Any:
    html = '''
        <html>
        <head>
            <title>2D Симуляция жидкости</title>
            <style>
                body { font-family: Arial; }
                #sim-canvas { border: 1px solid #333; }
            </style>
        </head>
        <body>
            <h1>2D Симуляция жидкости</h1>
            <label>Размер поля:
                <select id="grid-size">
                    <option value="50">50x50</option>
                    <option value="100">100x100</option>
                    <option value="200">200x200</option>
                    <option value="500" selected>500x500</option>
                </select>
            </label>
            <label>Скорость потока:
                <input id="flow-speed" type="number" value="1" min="0.01" max="100" step="0.01" style="width:60px;">
            </label>
            <label>Вязкость:
                <input id="viscosity" type="number" value="0.1" min="0" max="10" step="0.01" style="width:60px;">
            </label>
            <label>Итераций:
                <input id="steps" type="number" value="30" min="1" max="1000" step="1" style="width:60px;">
            </label>
            <canvas id="sim-canvas" width="600" height="600"></canvas>
            <script>
                let N = 500;
                const canvas = document.getElementById('sim-canvas');
                const ctx = canvas.getContext('2d');
                let drawing = false;
                let obstacles = Array.from({length: N}, () => Array(N).fill(0));
                const cell = () => canvas.width / N;
                let u_in = 1.0;
                let viscosity = 0.1;
                let steps = 30;
                // Изменение размера сетки
                document.getElementById('grid-size').addEventListener('change', function() {
                    N = parseInt(this.value);
                    obstacles = Array.from({length: N}, () => Array(N).fill(0));
                    drawObstacles();
                    updateFlow();
                });
                // Рисование препятствий
                canvas.addEventListener('mousedown', () => drawing = true);
                canvas.addEventListener('mouseup', () => drawing = false);
                canvas.addEventListener('mouseleave', () => drawing = false);
                canvas.addEventListener('mousemove', function(e) {
                    if (!drawing) return;
                    const rect = canvas.getBoundingClientRect();
                    const x = Math.floor((e.clientX - rect.left) / cell());
                    const y = Math.floor((e.clientY - rect.top) / cell());
                    if (x >= 0 && x < N && y >= 0 && y < N) {
                        // Увеличиваем препятствие до 2x2 ячеек
                        for (let dy = 0; dy < 2; dy++) {
                            for (let dx = 0; dx < 2; dx++) {
                                let yy = y + dy;
                                let xx = x + dx;
                                if (xx >= 0 && xx < N && yy >= 0 && yy < N) {
                                    obstacles[yy][xx] = 1;
                                }
                            }
                        }
                        drawObstacles();
                    }
                });
                function drawObstacles() {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = '#222';
                    for (let i = 0; i < N; i++) {
                        for (let j = 0; j < N; j++) {
                            if (obstacles[i][j]) {
                                ctx.fillRect(j*cell(), i*cell(), cell(), cell());
                            }
                        }
                    }
                }
                // Визуализация потока
                function drawFlow(vx, vy) {
                    // Цветовое отображение скорости для каждой ячейки
                    for (let i = 0; i < N; i++) {
                        for (let j = 0; j < N; j++) {
                            if (obstacles[i][j]) continue;
                            const v = Math.sqrt(vx[i][j]*vx[i][j] + vy[i][j]*vy[i][j]);
                            ctx.fillStyle = `hsl(${240 - Math.min(240, v*240)}, 100%, 50%)`;
                            ctx.fillRect(j*cell(), i*cell(), cell(), cell());
                        }
                    }
                    // Стрелки с шагом
                    const step = N > 200 ? 10 : 4;
                    for (let i = 0; i < N; i += step) {
                        for (let j = 0; j < N; j += step) {
                            if (obstacles[i][j]) continue;
                            ctx.save();
                            ctx.translate(j*cell()+cell()/2, i*cell()+cell()/2);
                            ctx.rotate(Math.atan2(vy[i][j], vx[i][j]));
                            ctx.beginPath();
                            ctx.moveTo(0, 0);
                            ctx.lineTo(cell()*0.8, 0);
                            ctx.lineTo(cell()*0.6, cell()*0.2);
                            ctx.moveTo(cell()*0.8, 0);
                            ctx.lineTo(cell()*0.6, -cell()*0.2);
                            ctx.strokeStyle = '#fff';
                            ctx.lineWidth = 1;
                            ctx.stroke();
                            ctx.restore();
                        }
                    }
                }
                // Отправка препятствий и обновление потока
                document.getElementById('flow-speed').addEventListener('input', function() {
                    u_in = parseFloat(this.value);
                    updateFlow();
                });
                document.getElementById('viscosity').addEventListener('input', function() {
                    viscosity = parseFloat(this.value);
                    updateFlow();
                });
                document.getElementById('steps').addEventListener('input', function() {
                    steps = parseInt(this.value);
                    updateFlow();
                });
                async function updateFlow() {
                    const response = await fetch('api/flow', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({obstacles: obstacles, N: N, u_in: u_in, viscosity: viscosity, steps: steps})
                    });
                    const data = await response.json();
                    drawObstacles();
                    drawFlow(data.vx, data.vy);
                }
                setInterval(updateFlow, 1000);
            </script>
        </body>
        </html>
    '''
    return HttpResponse(html, content_type='text/html; charset=utf-8')

@njit(fastmath=True)
def lbm_simulate(N, u_in, viscosity, steps, obstacles):
    w = np.array([4/9] + [1/9]*4 + [1/36]*4)
    c = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]])
    tau = 3*viscosity + 0.5
    f = np.zeros((N, N, 9))
    rho = np.ones((N, N))
    ux = np.zeros((N, N))
    uy = np.zeros((N, N))
    for k in range(9):
        f[:,:,k] = w[k]
    for t in range(steps):
        rho = np.sum(f, axis=2)
        ux = np.sum(f * c[:,0], axis=2) / rho
        uy = np.sum(f * c[:,1], axis=2) / rho
        ux[:,0] = u_in
        uy[:,0] = 0
        rho[:,0] = 1/(1-ux[:,0]) * (
            f[:,0,0] + f[:,0,2] + f[:,0,4] + 2*(f[:,0,3] + f[:,0,6] + f[:,0,7])
        )
        # No-slip препятствия (bounce-back) через явные циклы
        for i in range(N):
            for j in range(N):
                if obstacles[i, j] == 1:
                    for k, opp in enumerate([0,3,4,1,2,7,8,5,6]):
                        f[i, j, k] = f[i, j, opp]
        for k, (dx,dy) in enumerate(c):
            if dx == 0 and dy == 0:
                continue
            f[0,:,k] = f[0,:,8-k]
            f[-1,:,k] = f[-1,:,8-k]
        eu_dot = np.zeros((N,N,9))
        for k in range(9):
            eu_dot[:,:,k] = 3*(c[k,0]*ux + c[k,1]*uy)
        feq = np.zeros((N,N,9))
        for k in range(9):
            feq[:,:,k] = w[k]*rho*(1 + eu_dot[:,:,k] + 0.5*eu_dot[:,:,k]**2 - 1.5*(ux**2 + uy**2))
        f += -(f - feq)/tau
        f_stream = np.zeros_like(f)
        for k, (dx, dy) in enumerate(c):
            temp = f[:, :, k]
            # Сдвиг по y (оси 0)
            if dy > 0:
                temp_y = np.zeros_like(temp)
                temp_y[dy:, :] = temp[:-dy, :]
            elif dy < 0:
                temp_y = np.zeros_like(temp)
                temp_y[:dy, :] = temp[-dy:, :]
            else:
                temp_y = temp.copy()
            # Сдвиг по x (оси 1)
            if dx > 0:
                temp_x = np.zeros_like(temp_y)
                temp_x[:, dx:] = temp_y[:, :-dx]
            elif dx < 0:
                temp_x = np.zeros_like(temp_y)
                temp_x[:, :dx] = temp_y[:, -dx:]
            else:
                temp_x = temp_y.copy()
            f_stream[:, :, k] = temp_x
        f = f_stream
        ux = np.sum(f * c[:,0], axis=2) / np.sum(f, axis=2)
        uy = np.sum(f * c[:,1], axis=2) / np.sum(f, axis=2)
        ux[:,-1] = 0
        uy[:,-1] = 0
    rho = np.sum(f, axis=2)
    ux = np.sum(f * c[:,0], axis=2) / rho
    uy = np.sum(f * c[:,1], axis=2) / rho
    for i in range(N):
        for j in range(N):
            if obstacles[i, j] == 1:
                ux[i, j] = 0
                uy[i, j] = 0
    return ux, uy

@csrf_exempt
def flow_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    data = json.loads(request.body)
    N = int(data.get('N', 100))
    u_in = float(data.get('u_in', 0.1))
    viscosity = float(data.get('viscosity', 0.02))
    steps = int(data.get('steps', 100))
    obstacles = np.array(data.get('obstacles', []))
    if obstacles.shape != (N, N):
        return JsonResponse({'error': f'obstacles must be {N}x{N}'}, status=400)
    ux, uy = lbm_simulate(N, u_in, viscosity, steps, obstacles)
    return JsonResponse({'vx': ux.tolist(), 'vy': uy.tolist()})
