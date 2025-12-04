import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

# 設定浮點數精度為 float64 (物理模擬建議使用雙精度)
tf.keras.backend.set_floatx("float64")

# =================================
# 1. Parameter Definitions
# =================================
DTYPE = 'float64'

# Training Hyperparameters
EPOCHS_PHASE_1 = 2000   # 只訓練電位
EPOCHS_PHASE_2 = 3000   # 只訓練熱 PDE
EPOCHS_PHASE_3 = 10000  # 全開 + Global Loss
TOTAL_EPOCHS = EPOCHS_PHASE_1 + EPOCHS_PHASE_2 + EPOCHS_PHASE_3

BATCH_SIZE_COLLOC = 10000
BATCH_SIZE_BOUNDARY = 2000

# Learning Rates
LR_FAST = 1e-3
LR_SLOW = 5e-4

# Physical Coefficients
SIGMA_ELEC = 10.0
K_THERM = 2.0
V_DD = 1.0

# =================================
# 2. Model Builder
# =================================
def DNN_builder(in_shape=2, out_shape=2, n_hidden_layers=6, neuron_per_layer=64, actfn="swish"):
    input_layer = tf.keras.layers.Input(shape=(in_shape,))
    hidden = input_layer
    for _ in range(n_hidden_layers):
        hidden = tf.keras.layers.Dense(neuron_per_layer, activation=actfn)(hidden)
    output_layer = tf.keras.layers.Dense(out_shape, activation=None)(hidden)
    model = tf.keras.Model(input_layer, output_layer, name=f"PINN-{n_hidden_layers}layers")
    return model

# =================================
# 3. Data Generator
# =================================
@tf.function
def generate_data():
    x = tf.random.uniform((BATCH_SIZE_COLLOC, 1), -1, 1, dtype=DTYPE)
    y = tf.random.uniform((BATCH_SIZE_COLLOC, 1), -1, 1, dtype=DTYPE)

    n_b = BATCH_SIZE_BOUNDARY
    ones = tf.ones((n_b, 1), dtype=DTYPE)
    vals = tf.cast(tf.linspace(-1.0, 1.0, n_b)[:, None], DTYPE)

    x_r, y_r = ones, vals
    x_l, y_l = -ones, vals
    x_t, y_t = vals, ones
    x_b, y_b = vals, -ones
    
    return x, y, x_t, y_t, x_b, y_b, x_l, y_l, x_r, y_r

print("Plotting data point distribution...")
# 生成一批資料
x, y, x_t, y_t, x_b, y_b, x_l, y_l, x_r, y_r = generate_data()
    
plt.figure(figsize=(8, 8))
    
# 1. 畫內部點 (Domain Points) - 藍色點
plt.scatter(x.numpy(), y.numpy(), c='blue', s=1, alpha=0.5, label='Interior Domain')
    
# 2. 畫邊界點 (Boundary Points) - 紅色點
# 為了看清楚，我們把四個邊的點合併畫
x_bound = tf.concat([x_t, x_b, x_l, x_r], axis=0)
y_bound = tf.concat([y_t, y_b, y_l, y_r], axis=0)
plt.scatter(x_bound.numpy(), y_bound.numpy(), c='red', s=1, label='Boundaries')
    
plt.title(f"Data Distribution\n(Interior: {len(x)}, Boundary: {len(x_bound)})")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='upper right')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig("data_distribution.png", dpi=300)
plt.close()

# =================================
# 4. Physics Helper Functions
# =================================
def hard_constraint_T(x, y):
    return (1.0 - x ** 2) * (1.0 - y ** 2)

def get_chip_layout_heat(x, y):
    q_dyn_1 = 15.0 * tf.exp(-(x**2 + y**2) / (2 * 0.2**2))
    q_dyn_2 = 5.0 * tf.exp(-((x**2 + y**2 - 0.5)**2) / (2 * 0.1**2))
    return q_dyn_1 + q_dyn_2

# =================================
# 5. Core Physics Engine (Loss)
# =================================
@tf.function
def compute_physics_loss(model, x, y, x_top, y_top, x_bot, y_bot, x_left, y_left, x_right, y_right):
    
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x, y])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x, y])
            
            outputs = model(tf.concat([x, y], axis=1))
            phi = outputs[:, 0:1]
            T_raw = outputs[:, 1:2]
            D_vals = hard_constraint_T(x, y)
            T = T_raw * D_vals
        
        grad_phi_x = tape1.gradient(phi, x)
        grad_phi_y = tape1.gradient(phi, y)
        grad_T_x = tape1.gradient(T, x)
        grad_T_y = tape1.gradient(T, y)
        
    grad2_phi_xx = tape2.gradient(grad_phi_x, x)
    grad2_phi_yy = tape2.gradient(grad_phi_y, y)
    grad2_T_xx = tape2.gradient(grad_T_x, x)
    grad2_T_yy = tape2.gradient(grad_T_y, y)
    
    del tape1, tape2

    # --- Physics 1: Elec ---
    res_elec = SIGMA_ELEC * (grad2_phi_xx + grad2_phi_yy)
    loss_elec = tf.reduce_mean(tf.square(res_elec))

    # --- Physics 2: Therm ---
    J_x = -SIGMA_ELEC * grad_phi_x
    J_y = -SIGMA_ELEC * grad_phi_y
    Q_joule = (1.0 / SIGMA_ELEC) * (J_x**2 + J_y**2)
    Q_logic = get_chip_layout_heat(x, y)
    Q_total = Q_joule + Q_logic
    
    res_therm = K_THERM * (grad2_T_xx + grad2_T_yy) + Q_total
    loss_therm = tf.reduce_mean(tf.square(res_therm))

    # --- Physics 3: Global ---
    total_gen = tf.reduce_mean(Q_total) * 4.0
    
    def get_boundary_flux(x_b, y_b, nx, ny):
        with tf.GradientTape(persistent=True) as t:
            t.watch([x_b, y_b])
            out = model(tf.concat([x_b, y_b], axis=1))
            T_r = out[:, 1:2]
            D = hard_constraint_T(x_b, y_b)
            T_b = T_r * D
        grad_x = t.gradient(T_b, x_b)
        grad_y = t.gradient(T_b, y_b)
        return -K_THERM * (grad_x * nx + grad_y * ny)

    flux_r = tf.reduce_mean(get_boundary_flux(x_right, y_right, 1.0, 0.0)) * 2.0
    flux_l = tf.reduce_mean(get_boundary_flux(x_left, y_left, -1.0, 0.0)) * 2.0
    flux_t = tf.reduce_mean(get_boundary_flux(x_top, y_top, 0.0, 1.0)) * 2.0
    flux_b = tf.reduce_mean(get_boundary_flux(x_bot, y_bot, 0.0, -1.0)) * 2.0
    
    total_flux_out = flux_r + flux_l + flux_t + flux_b
    loss_global = tf.square(total_gen - total_flux_out)

    # --- BCs ---
    phi_top = model(tf.concat([x_top, y_top], axis=1))[:, 0:1]
    phi_bot = model(tf.concat([x_bot, y_bot], axis=1))[:, 0:1]
    loss_bc_elec = tf.reduce_mean(tf.square(phi_top - V_DD)) + \
                   tf.reduce_mean(tf.square(phi_bot - 0.0))

    return loss_elec, loss_therm, loss_global, loss_bc_elec, total_gen, total_flux_out

# =================================
# 6. Training Loop (Updated for History)
# =================================
model = DNN_builder(out_shape=2)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    LR_SLOW, decay_steps=2000, decay_rate=0.95, staircase=True)
optimizer_fast = tf.keras.optimizers.Adam(learning_rate=LR_FAST)
optimizer_slow = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

print(f"=== Starting Coupled Electro-Thermal Simulation ===")
print(f"Device: {tf.config.list_physical_devices('GPU') or 'CPU'}")

start_time = time.time()

# 字典用於儲存詳細歷史
history = {
    'total': [], 'elec': [], 'therm': [], 'glob': [], 'bc': [],
    'gen_val': [], 'flux_val': []
}

pbar = tqdm(range(1, TOTAL_EPOCHS + 1), desc="Training", unit="ep")

for epoch in pbar:
    x, y, x_t, y_t, x_b, y_b, x_l, y_l, x_r, y_r = generate_data()
    
    if epoch <= EPOCHS_PHASE_1:
        phase = "Ph1:Elec"
        w_e, w_t, w_g, w_bc = 1.0, 0.0, 0.0, 20.0
        opt = optimizer_fast
    elif epoch <= EPOCHS_PHASE_1 + EPOCHS_PHASE_2:
        phase = "Ph2:Therm"
        w_e, w_t, w_g, w_bc = 1.0, 1.0, 0.0, 20.0
        opt = optimizer_fast
    else:
        phase = "Ph3:Global"
        w_e, w_t, w_g, w_bc = 1.0, 1.0, 5.0, 20.0
        opt = optimizer_slow

    with tf.GradientTape(persistent=True) as tape:
        l_e, l_t, l_g, l_bc, val_gen, val_flux = compute_physics_loss(
            model, x, y, x_t, y_t, x_b, y_b, x_l, y_l, x_r, y_r
        )
        total_loss = w_e*l_e + w_t*l_t + w_g*l_g + w_bc*l_bc
        
    grads = tape.gradient(total_loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    
    # 紀錄數據 (每 10 epochs 記一次)
    if epoch % 10 == 0:
        history['total'].append(total_loss.numpy())
        history['elec'].append(l_e.numpy())
        history['therm'].append(l_t.numpy())
        history['glob'].append(l_g.numpy())
        history['bc'].append(l_bc.numpy())
        history['gen_val'].append(val_gen.numpy())
        history['flux_val'].append(val_flux.numpy())

    if epoch % 100 == 0:
        err_p = 0.0
        if val_gen > 1e-5:
            err_p = abs(val_gen - val_flux) / val_gen * 100
            
        pbar.set_postfix({
            "Ph": phase,
            "L_T": f"{l_t:.1e}",
            "Gen": f"{val_gen:.1f}",
            "Flux": f"{val_flux:.1f}",
            "Err": f"{err_p:.1f}%"
        })

elapsed = time.time() - start_time
print(f"\nTraining Finished in {elapsed:.2f} seconds.")

# =================================
# 7. Validation
# =================================
print("\n=== Validation on Unseen Grid ===")
n_val = 200
x_v = tf.cast(tf.linspace(-1.0, 1.0, n_val), DTYPE)
y_v = tf.cast(tf.linspace(-1.0, 1.0, n_val), DTYPE)
X_val, Y_val = tf.meshgrid(x_v, y_v)
x_val_flat = tf.reshape(X_val, [-1, 1])
y_val_flat = tf.reshape(Y_val, [-1, 1])

def get_residuals(model, x, y):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x, y])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x, y])
            out = model(tf.concat([x, y], axis=1))
            phi, T_raw = out[:, 0:1], out[:, 1:2]
            T = T_raw * hard_constraint_T(x, y)
        gp_x, gp_y = tape1.gradient(phi, x), tape1.gradient(phi, y)
        gt_x, gt_y = tape1.gradient(T, x), tape1.gradient(T, y)
    g2p_xx = tape2.gradient(gp_x, x)
    g2p_yy = tape2.gradient(gp_y, y)
    g2t_xx = tape2.gradient(gt_x, x)
    g2t_yy = tape2.gradient(gt_y, y)
    
    res_e = SIGMA_ELEC * (g2p_xx + g2p_yy)
    J2 = (SIGMA_ELEC*gp_x)**2 + (SIGMA_ELEC*gp_y)**2
    Q = (1.0/SIGMA_ELEC)*J2 + get_chip_layout_heat(x, y)
    res_t = K_THERM * (g2t_xx + g2t_yy) + Q
    return res_e, res_t

res_e_val, res_t_val = get_residuals(model, x_val_flat, y_val_flat)
mae_t = tf.reduce_mean(tf.abs(res_t_val))
print(f"Validation Thermal PDE Error: Mean={mae_t:.2e}")

# =================================
# 8. Advanced Visualization (最終修復版)
# =================================
print("Generating Advanced Visualization Plots...")

# --- 1. 準備繪圖資料 ---
n_grid = 400
x_vals = np.linspace(-1, 1, n_grid)
y_vals = np.linspace(-1, 1, n_grid)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

# [關鍵修正] 準備給 TF 的輸入，必須是 (N, 1)
# 但後續計算全部轉回 (N,) 一維陣列以避免廣播錯誤
x_flat_input = tf.cast(X_grid.flatten()[:, None], DTYPE)
y_flat_input = tf.cast(Y_grid.flatten()[:, None], DTYPE)

# --- 2. 預測與物理量計算 ---
# Model Prediction
out = model(tf.concat([x_flat_input, y_flat_input], axis=1))

# [關鍵修正] 全部強制壓扁成 1D array (N,)
phi_pred_flat = out[:, 0].numpy().flatten()
T_raw_flat = out[:, 1].numpy().flatten()
x_flat = X_grid.flatten()
y_flat = Y_grid.flatten()

# 應用 Hard Constraint (現在大家都是 1D，相乘絕對安全)
dist_flat = (1 - x_flat**2) * (1 - y_flat**2)
T_pred_flat = T_raw_flat * dist_flat

# 重塑回 2D 網格 (400, 400)
phi_grid = phi_pred_flat.reshape(n_grid, n_grid)
T_grid = T_pred_flat.reshape(n_grid, n_grid)

# 計算梯度
dy = dx = 2.0 / (n_grid - 1)
grad_phi_y, grad_phi_x = np.gradient(phi_grid, dy, dx)
grad_T_y, grad_T_x = np.gradient(T_grid, dy, dx)

# 計算電流與熱源
Jx_grid = -SIGMA_ELEC * grad_phi_x
Jy_grid = -SIGMA_ELEC * grad_phi_y
J_mag_grid = np.sqrt(Jx_grid**2 + Jy_grid**2)
Q_joule_grid = (1.0 / SIGMA_ELEC) * J_mag_grid**2

# 計算邏輯熱 (直接用 meshgrid 計算，避免維度問題)
Q_logic_tf = get_chip_layout_heat(X_grid, Y_grid)
Q_logic_grid = Q_logic_tf.numpy()
Q_total_grid = Q_joule_grid + Q_logic_grid

# 熱通量
qx_grid = -K_THERM * grad_T_x
qy_grid = -K_THERM * grad_T_y

# --- 3. 開始繪圖 ---

# Figure 1: Training Diagnostics
# 檢查 history 是否為空
if len(history['total']) > 0:
    fig1, ax1 = plt.subplots(1, 2, figsize=(14, 5))
    steps_per_record = 10 
    epochs_idx = np.arange(len(history['total'])) * steps_per_record

    ax1[0].semilogy(epochs_idx, history['total'], 'k-', label='Total')
    ax1[0].semilogy(epochs_idx, history['elec'], 'b--', label='Elec')
    ax1[0].semilogy(epochs_idx, history['therm'], 'r--', label='Therm')
    ax1[0].semilogy(epochs_idx, history['glob'], 'g:', label='Global')
    ax1[0].axvline(x=EPOCHS_PHASE_1, color='gray', linestyle='--')
    ax1[0].axvline(x=EPOCHS_PHASE_1+EPOCHS_PHASE_2, color='gray', linestyle='--')
    ax1[0].set_title('Loss History')
    ax1[0].legend()
    ax1[0].grid(True, linestyle='--', alpha=0.5)

    ax1[1].plot(epochs_idx, history['gen_val'], 'r-', label='Gen (Input)')
    ax1[1].plot(epochs_idx, history['flux_val'], 'b--', label='Flux (Output)')
    ax1[1].axvline(x=EPOCHS_PHASE_1, color='gray', linestyle='--')
    ax1[1].axvline(x=EPOCHS_PHASE_1+EPOCHS_PHASE_2, color='gray', linestyle='--')
    ax1[1].set_title('Global Energy Conservation')
    ax1[1].legend()
    ax1[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('1_training_diagnostics.png', dpi=300)
    plt.close()

# Figure 2: Multi-Physics Analysis
fig2, ax2 = plt.subplots(1, 3, figsize=(20, 5))

# Potential & Current
c1 = ax2[0].contourf(X_grid, Y_grid, phi_grid, 50, cmap='plasma')
plt.colorbar(c1, ax=ax2[0], label='Potential (V)')
ax2[0].streamplot(X_grid, Y_grid, Jx_grid, Jy_grid, color='white', linewidth=0.8, density=1.0)
ax2[0].set_title('Potential & Current Flow')

# Heat Source
c2 = ax2[1].contourf(X_grid, Y_grid, Q_total_grid, 50, cmap='inferno')
plt.colorbar(c2, ax=ax2[1], label='Heat Source')
ax2[1].set_title('Total Heat Source')

# Temp & Flux
c3 = ax2[2].contourf(X_grid, Y_grid, T_grid, 50, cmap='turbo')
plt.colorbar(c3, ax=ax2[2], label='Temp (K)')
skip = 25
ax2[2].quiver(X_grid[::skip, ::skip], Y_grid[::skip, ::skip], 
              qx_grid[::skip, ::skip], qy_grid[::skip, ::skip], 
              color='black', scale=500, width=0.005)
ax2[2].set_title('Temperature & Flux')

plt.tight_layout()
plt.savefig('2_multiphysics_analysis.png', dpi=300)
plt.close()

# Figure 3: 1D Slices
mid = n_grid // 2
fig3, ax3 = plt.subplots(1, 2, figsize=(14, 5))

# Elec Slice
ax3[0].plot(x_vals, phi_grid[mid, :], 'b-', label='Phi')
ax3[0].set_ylabel('Potential')
ax3_twin = ax3[0].twinx()
ax3_twin.plot(x_vals, J_mag_grid[mid, :], 'r--', label='|J|')
ax3_twin.set_ylabel('|J|', color='r')
ax3[0].set_title('1D Slice: Electrical (y=0)')
ax3[0].grid(True)

# Therm Slice
ax3[1].plot(x_vals, T_grid[mid, :], 'k-', label='Temp')
ax3[1].set_ylabel('Temp')
ax3_twin2 = ax3[1].twinx()
ax3_twin2.plot(x_vals, Q_total_grid[mid, :], 'm:', label='Q')
ax3_twin2.set_ylabel('Heat Source', color='m')
ax3[1].set_title('1D Slice: Thermal (y=0)')
ax3[1].grid(True)

plt.tight_layout()
plt.savefig('3_quantitative_slices.png', dpi=300)
plt.close()

print("All plots saved successfully.")