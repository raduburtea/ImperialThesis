import sys
sys.path.append("./or-gym")
sys.path.append(".")
sys.path.append("./OMLT/src")
from tqdm import tqdm
import time
import random
import numpy as np
import torch
import pyomo.environ as pyo
import omlt
from omlt.block import OmltBlock
from omlt.neuralnet import FullSpaceNNFormulation
import torch.onnx
from omlt.io import write_onnx_model_with_bounds, load_onnx_neural_network
import tempfile


def make_deterministic(seed: int = 1234):
    """Makes PyTorch deterministic for reproducibility."""
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def optimise_actor_with_pyomo(actor, critic, state_batch, action_batch, device, node_batch, \
    BATCH_SIZE = 1, mode='train', solver_name = 'ipopt', max_iter=1000):    
    """
    Inputs:
    - actor: neural network
    - critic: neural_network
    - state_batch: torch tensor
    - action_batch: torch tensor
    - device: string, indicates which device PyTorch should use
    - node_batch: list of dictionaries containing information for each node - specific to the 
                  InvenotryManagement Environment
    - BATCH_SIZE: int, number of samples in batch
    - mode: string, indicate whether the function should operate in training or testing mode
    - solver_name: string, indicate which solver should be used
    - max_iter: int, maximum number of iterations allowed for the solver to run
    
    Function that takes as inputs the actor and the critic network along with data samples and outputs the optimal
    actions obtained by maximizing the constrained optimization problem in Pyomo using the desired solver. The function
    also runs the optimization step for the actor.
    """

    #Provide sample data shape to be used to write the neural network to ONNX
    x = torch.cat((state_batch.to(device), action_batch.reshape(BATCH_SIZE,3).to(device)), dim=-1).detach().to(device)
    pytorch_model = None
    
    #Write the neural network to ONNX
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        torch.onnx.export(
            critic.model,
            x,
            f,
            input_names=['inputs'],
            output_names=['outputs'],
            dynamic_axes={
                'inputs': {0: 'batch_size'},
                'outputs': {0: 'batch_size'}
            }
        )
        write_onnx_model_with_bounds(f.name, None, input_bounds = None)
        pytorch_model = f.name
    
    #Load network definition in OMLT
    network_definition = load_onnx_neural_network(pytorch_model)

    optimal_actions = []

    #Iterate over the samples in the batch
    for i in range(BATCH_SIZE):

        #Initialize a Pyomo model
        m = pyo.ConcreteModel()
        #Initialize OMLT block
        m.neural_net = OmltBlock()

        #Translate a PyTorch model which was first expressed in an ONNX formulation to a Pyomo readable format
        formulation = FullSpaceNNFormulation(network_definition)

        #Neural network formulation is built using OMLT
        m.neural_net.build_formulation(formulation)

        #Objective function of the Pyomo formulation is set
        m.obj = pyo.Objective(expr=m.neural_net.outputs[0], sense=pyo.maximize)
        input_state_action = torch.cat((state_batch[i].to(device), action_batch.reshape(BATCH_SIZE,3)[i].to(device)), dim=-1).detach().to(device)
        
        #Constraint list is initialized
        m.cuts = pyo.ConstraintList()
        #Constraints are added to make sure that the optimization problem does not optimize with respect to the 
        #variables associated to the environment state
        for k in range(33):
            m.cuts.add(expr=m.neural_net.inputs[k] == input_state_action[k].cpu().item())
        
        #Constraints are added for each action, such that they would satisfy the constraints imposed by the environment
        if mode == 'train':
            #Training mode is used and current constraints are used
            m.cuts.add(expr=m.neural_net.inputs[33] <= min(node_batch[i]['Node1/Inventory constraint'], node_batch[i]['Node1/Capacity constraint']))
            m.cuts.add(expr=m.neural_net.inputs[34] <= min(node_batch[i]['Node2/Inventory constraint'], node_batch[i]['Node2/Capacity constraint']))
            m.cuts.add(expr=m.neural_net.inputs[35] <= node_batch[i]['Node3/Capacity constraint'])
        else:
            #Testing mode is used the future constraints are used
            m.cuts.add(expr=m.neural_net.inputs[33] <= min(node_batch['Node1/Inventory constraint next'], node_batch['Node1/Capacity constraint']))
            m.cuts.add(expr=m.neural_net.inputs[34] <= min(node_batch['Node2/Inventory constraint next'], node_batch['Node2/Capacity constraint']))
            m.cuts.add(expr=m.neural_net.inputs[35] <= node_batch['Node3/Capacity constraint'])
        m.cuts.add(expr=m.neural_net.inputs[33] >= 0)
        m.cuts.add(expr=m.neural_net.inputs[34] >= 0)
        m.cuts.add(expr=m.neural_net.inputs[35] >= 0)
        
        #Solver is initialized
        solver = pyo.SolverFactory(solver_name)
        solver.options['max_iter']= max_iter
        status = solver.solve(m, tee=False)

        #Optimal actions are collected
        action = [pyo.value(m.neural_net.inputs[33]), pyo.value(m.neural_net.inputs[34]), \
                  pyo.value(m.neural_net.inputs[35])]
        optimal_actions.append(torch.tensor(action).to(device))
    
    optimal_actions = torch.cat(optimal_actions).reshape(BATCH_SIZE, 3).to(device)
    
    if mode == 'train':
        #Optimization step is run for the actor
        actor_actions = actor.model(state_batch.to(device))
        loss = ((actor_actions - optimal_actions)**2).mean()   
        loss_val = loss
        actor.optim.zero_grad()
        loss.backward()

        actor.optim.step()
        return optimal_actions, loss_val
    return optimal_actions