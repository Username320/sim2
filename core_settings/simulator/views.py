from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from typing import Any
import json
from django.views.decorators.csrf import csrf_exempt
import numpy as np

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
                            ctx.strokeStyle = '#000';
                            ctx.lineWidth = 1;
                            ctx.stroke();
                            ctx.restore();
                        }
                    }
                }
                // Отправка препятствий и обновление потока
                document.getElementById('flow-speed').addEventListener('input', function() {
                    u_in = parseFloat(this.value);
                });
                document.getElementById('viscosity').addEventListener('input', function() {
                    viscosity = parseFloat(this.value);
                });
                document.getElementById('steps').addEventListener('input', function() {
                    steps = parseInt(this.value);
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

@csrf_exempt
def flow_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    data = json.loads(request.body)
    N = int(data.get('N', 500))
    u_in = float(data.get('u_in', 1.0))
    viscosity = float(data.get('viscosity', 0.1))
    steps = int(data.get('steps', 30))
    obstacles = np.array(data.get('obstacles', []))
    if obstacles.shape != (N, N):
        return JsonResponse({'error': f'obstacles must be {N}x{N}'}, status=400)
    # Параметры потока
    vx = np.zeros((N, N))
    vy = np.zeros((N, N))
    vx[:, 0] = u_in  # входной поток слева
    mask = (obstacles == 0)
    for step in range(steps):
        vx_new = vx.copy()
        vy_new = vy.copy()
        m = mask[1:-1, 1:-1]
        # Определяем, есть ли рядом препятствие
        near_obstacle = (
            (obstacles[2:, 1:-1] == 1) |
            (obstacles[:-2, 1:-1] == 1) |
            (obstacles[1:-1, 2:] == 1) |
            (obstacles[1:-1, :-2] == 1)
        )
        # Вязкость только рядом с препятствиями
        viscosity_matrix = np.zeros((N-2, N-2))
        viscosity_matrix[near_obstacle] = viscosity
        vx_new[1:-1, 1:-1][m] = (
            vx[1:-1, 1:-1][m] + viscosity_matrix[m] * (
                vx[2:, 1:-1][m] + vx[:-2, 1:-1][m] + vx[1:-1, 2:][m] + vx[1:-1, :-2][m] - 4*vx[1:-1, 1:-1][m]
            )
        )
        vy_new[1:-1, 1:-1][m] = (
            vy[1:-1, 1:-1][m] + viscosity_matrix[m] * (
                vy[2:, 1:-1][m] + vy[:-2, 1:-1][m] + vy[1:-1, 2:][m] + vy[1:-1, :-2][m] - 4*vy[1:-1, 1:-1][m]
            )
        )
        vx, vy = vx_new, vy_new
    # Применяем препятствия
    vx[obstacles == 1] = 0
    vy[obstacles == 1] = 0
    # Возвращаем результат
    return JsonResponse({
        'vx': vx.tolist(),
        'vy': vy.tolist(),
    })
