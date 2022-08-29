from sandbox.cpo.safety_constraints.base import *
import numpy as np

class InvManagmentSafetyConstraint(SafetyConstraint):
    """Class implementing the safety constraints for the Invenotry Management Environment"""
    def __init__(self, env, max_value=0.0001, **kwargs):
        super(InvManagmentSafetyConstraint,self).__init__(max_value, **kwargs)
        self.env = env
    
    def evaluate(self, path):
        """
        Returns a binary value of the state of the constraint. If any of the constraint is violated
        for any of the nodes, the function will return the value 1. If no constraint is violated the 
        function returns the value 0.
        """
        c1 = np.less_equal(path["env_infos"]["Node1/Replenishment order"], path["env_infos"]["Node1/Capacity constraint"])
        c2 = np.less_equal(path["env_infos"]["Node2/Replenishment order"], path["env_infos"]["Node2/Capacity constraint"]) 
        c3 = np.less_equal(path["env_infos"]["Node3/Replenishment order"], path["env_infos"]["Node3/Capacity constraint"])
        c4 = np.less_equal(path["env_infos"]["Node1/Replenishment order"], path["env_infos"]["Node1/Inventory constraint"])
        c5 = np.less_equal(path["env_infos"]["Node2/Replenishment order"], path["env_infos"]["Node2/Inventory constraint"])
        truth_array = []
        for i in range(len(c1)):
            truth = c1[i] and c2[i] and c3[i] and c4[i] and c5[i]
            truth_array.append(1-int(truth))

        return truth_array


        