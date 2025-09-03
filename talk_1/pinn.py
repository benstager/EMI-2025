import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import lhs
import numpy as np
from scipy.interpolate import griddata

# check that boundary and initial are defined on given domain
# physical loss should be defined on sampled collocation domain
# the 'collocation' for an ODE is just the temporal mesh, the given data loss can just be the initial condition
# the collocation for a PDE could be a space-time mesh, the given data loss is the spatial mesh evaluated at the initial condition, and the boundary condition are the endpoints evaluated across the temporal mesh

class PINN1DHeatLoss(nn.Module):
    def __init__(self, ic_str, bc_str, *args, **kwargs):
        self.ic_str = ic_str
        self.bc_str = bc_str
        super().__init__(*args, **kwargs)

    def ic(self, x):
        if self.ic_str == 'sin':
            return np.sin(np.pi*x)
        
    # need to generalize to all BCs
        
    def du_dt_grad(self, outputs, inputs):
        return torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True, 
            retain_graph=True
        )[0][:,1]
    
    def d2u_dx2_grad(self, outputs, inputs):
        du_dx = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True, 
            retain_graph=True
        )[0][:,0]

        d2u_dx2 = torch.autograd.grad(
            outputs=du_dx,
            inputs=inputs,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True, 
            retain_graph=True
        )[0][:,0]

        return d2u_dx2
    
    # unused arguments here
    def physical_loss(self, model, ts, xs, mesh_points):
        inputs = torch.tensor(mesh_points, requires_grad=True).float()
        outputs = model(inputs)
        d2u_dx2 = self.d2u_dx2_grad(outputs=outputs, inputs=inputs)
        du_dt = self.du_dt_grad(outputs=outputs, inputs=inputs)
        loss_term = torch.mean((du_dt - d2u_dx2)**2)
        return loss_term
    
    def ic_loss(self, model, xs, ts):
        xs = torch.tensor(xs).float()
        ts = torch.tensor(np.array([0 for _ in range(len(xs))])).float()
        ic = torch.tensor(np.array([self.ic(x) for x in xs])).float()
        xt = torch.stack((xs, ts), dim=1)
        ic_pred = model(xt)
        return torch.mean((ic_pred - ic)**2)

    # needs to be generalized much better
    def bc_loss(self, model, ts): 
        xs_0 = torch.tensor(np.array([0 for _ in range(len(ts))])).float()
        xs_L = torch.tensor(np.array([1 for _ in range(len(ts))])).float()
        xt_0 = torch.stack([xs_0, torch.tensor(ts).float()], dim=1)
        xt_L = torch.stack([xs_L, torch.tensor(ts).float()], dim=1)
        bc_pred = torch.mean((model(xt_0)**2 + model(xt_L)**2))
        return bc_pred

    # needs to tune to account for the weights of each loss term
    def forward(self, model, xs, ts, mesh_points):
        loss = .01*self.physical_loss(model=model, ts=ts, xs=xs, mesh_points=mesh_points) + self.ic_loss(model=model, xs=xs, ts=ts) + self.bc_loss(model=model, ts=ts)
        return loss

# check that boundary and initial are defined on given domain
# physical loss should be defined on sampled collocation domain
# the 'collocation' for an ODE is just the temporal mesh, the given data loss can just be the initial condition
# the collocation for a PDE could be a space-time mesh, the given data loss is the spatial mesh evaluated at the initial condition, and the boundary condition are the endpoints evaluated across the temporal mesh

class PINN1DHeatLoss(nn.Module):
    def __init__(self, ic_str, bc_str='dirichlet', bc_points=[0,1], *args, **kwargs):
        self.ic_str = ic_str
        self.bc_points = bc_points
        self.bc_str = bc_str
        super().__init__(*args, **kwargs)

    def ic(self, x):
        if self.ic_str == 'sin':
            return np.sin(np.pi*x)
    
    def bc(self, t):
        if self.bc_str == 'dirichlet':
            return [0, 0]
        
    # need to generalize to all BCs
        
    def du_dt_grad(self, outputs, inputs):
        return torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True, 
            retain_graph=True
        )[0][:,1]
    
    def d2u_dx2_grad(self, outputs, inputs):
        du_dx = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True, 
            retain_graph=True
        )[0][:,0]

        d2u_dx2 = torch.autograd.grad(
            outputs=du_dx,
            inputs=inputs,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True, 
            retain_graph=True
        )[0][:,0]

        return d2u_dx2
    
    def physical_loss(self, model, ts, xs, mesh_points):
        inputs = torch.tensor(mesh_points, requires_grad=True).float()
        outputs = model(inputs)
        d2u_dx2 = self.d2u_dx2_grad(outputs=outputs, inputs=inputs)
        du_dt = self.du_dt_grad(outputs=outputs, inputs=inputs)
        loss_term = torch.mean((du_dt - d2u_dx2)**2)
        return loss_term
    
    def ic_loss(self, model, xs, ts):
        xs = torch.tensor(xs).float()
        ts = torch.tensor(np.array([0 for _ in range(len(xs))])).float()
        ic = torch.tensor(np.array([self.ic(x) for x in xs])).float()
        xt = torch.stack((xs, ts), dim=1)
        ic_pred = model(xt)
        return torch.mean((ic_pred - ic)**2)

    def bc_loss(self, model, ts): 
        xs_0 = torch.tensor(np.array([self.bc_points[0] for _ in range(len(ts))])).float()
        xs_L = torch.tensor(np.array([self.bc_points[1] for _ in range(len(ts))])).float()
        xt_0 = torch.stack([xs_0, torch.tensor(ts).float()], dim=1)
        xt_L = torch.stack([xs_L, torch.tensor(ts).float()], dim=1)
        bc_pred = torch.mean(((model(xt_0) - self.bc(ts)[0])**2 + (model(xt_L) - self.bc(ts)[1])**2))
        return bc_pred

    def forward(self, model, xs, ts, mesh_points):
        loss = .01*self.physical_loss(model=model, ts=ts, xs=xs, mesh_points=mesh_points) + self.ic_loss(model=model, xs=xs, ts=ts) + self.bc_loss(model=model, ts=ts)
        return loss

class PINN1DHeat:
    def __init__(self, ic='sin', bcs='dirichlet', bc_points=[0,1], pinn_arch='default', lr=.01, epochs=3000, lhs_samples=500):
        self.ic = ic
        self.bcs = bcs
        self.bc_points = bc_points
        self.pinn_arch = pinn_arch
        self.lr = lr
        self.epochs = epochs
        self.lhs_samples = lhs_samples

    def solve(self):
        lhs_points = lhs(2, samples=self.lhs_samples)

        xs = lhs_points[:,0]
        ts = lhs_points[:,1]

        # need to run some sort of hyperparameter tuning here 
        if self.pinn_arch == 'default':
            pinn = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1)
            )

        criterion = PINN1DHeatLoss(self.ic, self.bcs, self.bc_points)
        optimizer = torch.optim.Adam(pinn.parameters(), lr=self.lr)

        print("------SOLVING-------\n")
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
            torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=1.0)
            loss = criterion(pinn, xs, ts, mesh_points=lhs_points)
            if epoch % 100 == 0:
                print(f"Loss at epoch {epoch}: {loss}")
            loss.backward()
            optimizer.step()
        print()
        print("------SOLVE COMPLEETE------")

        self.pinn = pinn

    def plot_soln(self):

        # Generate fine mesh
        xs = np.linspace(0, 1, 1000)
        ts = np.linspace(0, 1, 1000)
        X, T = np.meshgrid(xs, ts)

        # Flatten and prepare for model input
        XT = np.stack([X.flatten(), T.flatten()], axis=-1)
        XT_tensor = torch.tensor(XT, dtype=torch.float32)

        U_pred = self.pinn((XT_tensor))

        # Reshape to match grid
        U = U_pred.reshape(X.shape)

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(T, X, U.detach().numpy(), cmap='coolwarm', edgecolor='none')
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_zlabel("u(x,t)")
        plt.title("Numerical solution to 1-D heat, using PINN Loss")
        plt.show()