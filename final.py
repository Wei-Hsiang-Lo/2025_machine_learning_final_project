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
# 8. Advanced Visualization
# =================================
print("Generating Advanced Visualization Plots...")

# --- 準備繪圖資料 ---
n_grid = 400
x_vals = np.linspace(-1, 1, n_grid)
y_vals = np.linspace(-1, 1, n_grid)
# 產生 2D 網格 (400, 400)
X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

# 準備給神經網路預測用的扁平輸入 (160000, 1)
x_flat_np = X_grid.flatten()[:, None]
y_flat_np = Y_grid.flatten()[:, None]
x_tf = tf.cast(x_flat_np, DTYPE)
y_tf = tf.cast(y_flat_np, DTYPE)

# 1. 預測基礎物理量 (phi, T)
# model 需要扁平輸入 (N, 2)
out = model(tf.concat([x_tf, y_tf], axis=1))
phi_pred_flat = out[:, 0].numpy()
T_raw_flat = out[:, 1].numpy()

# 應用 Hard Constraint
dist_flat = (1 - x_flat_np**2) * (1 - y_flat_np**2)
T_pred_flat = T_raw_flat * dist_flat

# 重塑回 (400, 400)
phi_grid = phi_pred_flat.reshape(n_grid, n_grid)
T_grid = T_pred_flat.reshape(n_grid, n_grid)

# 2. 計算衍生物理量
# 計算梯度
dy = dx = 2.0 / (n_grid - 1)
grad_phi_y, grad_phi_x = np.gradient(phi_grid, dy, dx)
grad_T_y, grad_T_x = np.gradient(T_grid, dy, dx)

# 電流密度 J
Jx_grid = -SIGMA_ELEC * grad_phi_x
Jy_grid = -SIGMA_ELEC * grad_phi_y
J_mag_grid = np.sqrt(Jx_grid**2 + Jy_grid**2)

# 焦耳熱 Q_joule
Q_joule_grid = (1.0 / SIGMA_ELEC) * J_mag_grid**2

# --- [關鍵修正] 計算邏輯熱 Q_logic ---
# 直接傳入 (400, 400) 的網格，不要傳入扁平向量
# 這樣 TF 會直接輸出 (400, 400)，不需要 reshape，也不會發生廣播錯誤
Q_logic_tf = get_chip_layout_heat(X_grid, Y_grid) 
Q_logic_grid = Q_logic_tf.numpy() # 確保轉回 numpy

# 總熱源
Q_total_grid = Q_joule_grid + Q_logic_grid

# 熱通量 q
qx_grid = -K_THERM * grad_T_x
qy_grid = -K_THERM * grad_T_y

# -------------------------------------------------
# Figure 1: Training Diagnostics
# -------------------------------------------------
fig1, ax1 = plt.subplots(1, 2, figsize=(14, 5))
epochs_idx = np.arange(len(history['total'])) * 10 # 假設每 10 epoch 存一次

# Loss History
ax1[0].semilogy(epochs_idx, history['total'], 'k-', label='Total', linewidth=2)
ax1[0].semilogy(epochs_idx, history['elec'], 'b--', label='Elec')
ax1[0].semilogy(epochs_idx, history['therm'], 'r--', label='Therm')
ax1[0].semilogy(epochs_idx, history['glob'], 'g:', label='Global')
ax1[0].semilogy(epochs_idx, history['bc'], 'c:', label='BC')
ax1[0].axvline(x=EPOCHS_PHASE_1, color='gray', linestyle='--')
ax1[0].axvline(x=EPOCHS_PHASE_1+EPOCHS_PHASE_2, color='gray', linestyle='--')
ax1[0].set_xlabel('Epochs')
ax1[0].set_ylabel('Loss (Log Scale)')
ax1[0].set_title('Training Diagnostics: Loss History')
ax1[0].legend()
ax1[0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Conservation Check
ax1[1].plot(epochs_idx, history['gen_val'], 'r-', label='Total Heat Gen')
ax1[1].plot(epochs_idx, history['flux_val'], 'b--', label='Total Flux Out')
ax1[1].axvline(x=EPOCHS_PHASE_1, color='gray', linestyle='--')
ax1[1].axvline(x=EPOCHS_PHASE_1+EPOCHS_PHASE_2, color='gray', linestyle='--')
ax1[1].set_xlabel('Epochs')
ax1[1].set_ylabel('Power (W)')
ax1[1].set_title('Global Energy Balance Check')
ax1[1].legend()
ax1[1].grid(True)

plt.tight_layout()
plt.savefig('1_training_diagnostics.png', dpi=300)
plt.show()
plt.close() # 釋放記憶體

# -------------------------------------------------
# Figure 2: Multi-Physics Analysis
# -------------------------------------------------
fig2, ax2 = plt.subplots(1, 3, figsize=(20, 5))

# C. Potential & Current
c1 = ax2[0].contourf(X_grid, Y_grid, phi_grid, 50, cmap='plasma')
plt.colorbar(c1, ax=ax2[0], label='Potential (V)')
# Streamplot 密度調整
ax2[0].streamplot(X_grid, Y_grid, Jx_grid, Jy_grid, color='white', linewidth=0.8, density=1.0, arrowsize=1.0)
ax2[0].set_title('C. Electric Potential & Current Flow (J)')
ax2[0].set_aspect('equal')

# D. Total Heat Source
c2 = ax2[1].contourf(X_grid, Y_grid, Q_total_grid, 50, cmap='inferno')
plt.colorbar(c2, ax=ax2[1], label='W/m^3')
ax2[1].set_title('D. Total Heat Source (Logic + Joule)')
ax2[1].set_aspect('equal')

# E. Temp & Heat Flux
c3 = ax2[2].contourf(X_grid, Y_grid, T_grid, 50, cmap='turbo')
plt.colorbar(c3, ax=ax2[2], label='Temperature (K)')
# Quiver 採樣調整
skip = 25
ax2[2].quiver(X_grid[::skip, ::skip], Y_grid[::skip, ::skip], 
              qx_grid[::skip, ::skip], qy_grid[::skip, ::skip], 
              color='black', scale=500, width=0.005)
ax2[2].set_title('E. Temperature & Heat Flux Vectors (q)')
ax2[2].set_aspect('equal')

plt.tight_layout()
plt.savefig('2_multiphysics_analysis.png', dpi=300)
plt.show()
plt.close()

# -------------------------------------------------
# Figure 3: 1D Slice Analysis
# -------------------------------------------------
mid_idx = n_grid // 2
x_slice = X_grid[mid_idx, :]
phi_slice = phi_grid[mid_idx, :]
T_slice = T_grid[mid_idx, :]
J_mag_slice = J_mag_grid[mid_idx, :]
Q_total_slice = Q_total_grid[mid_idx, :]

fig3, ax3 = plt.subplots(1, 2, figsize=(14, 5))

# F. Electrical Slice
ax3[0].plot(x_slice, phi_slice, 'b-', label='Potential', linewidth=2)
ax3[0].set_xlabel('x (at y=0)')
ax3[0].set_ylabel('Potential (V)')
ax3[0].set_title('F. 1D Slice: Electrical (y=0)')
ax3[0].grid(True)
ax3_twin = ax3[0].twinx()
ax3_twin.plot(x_slice, J_mag_slice, 'r--', label='|J|')
ax3_twin.set_ylabel('|J|', color='r')
ax3_twin.tick_params(axis='y', labelcolor='r')
# 合併圖例
l1, lab1 = ax3[0].get_legend_handles_labels()
l2, lab2 = ax3_twin.get_legend_handles_labels()
ax3[0].legend(l1+l2, lab1+lab2, loc='upper right')

# G. Thermal Slice
ax3[1].plot(x_slice, T_slice, 'k-', label='Temp', linewidth=2)
ax3[1].set_xlabel('x (at y=0)')
ax3[1].set_ylabel('Temperature (K)')
ax3[1].set_title('G. 1D Slice: Thermal (y=0)')
ax3[1].grid(True)
ax3_twin2 = ax3[1].twinx()
ax3_twin2.plot(x_slice, Q_total_slice, 'm:', label='Q Source')
ax3_twin2.set_ylabel('Heat Source Q', color='m')
ax3_twin2.tick_params(axis='y', labelcolor='m')
# 合併圖例
l3, lab3 = ax3[1].get_legend_handles_labels()
l4, lab4 = ax3_twin2.get_legend_handles_labels()
ax3[1].legend(l3+l4, lab3+lab4, loc='upper right')

plt.tight_layout()
plt.savefig('3_quantitative_slices.png', dpi=300)
plt.show()
plt.close()

print("All plots saved successfully.")