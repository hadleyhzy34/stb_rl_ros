import torch
import numpy as np
import matplotlib.pyplot as plt
import unittest
import pdb

def state2costmap(state):
    """
    description:
    args:
        state: torch.tensor, (scan.ranges, rel_goal_pos), (b,362)
    return:
        costmap: torch.tensor, (b,3,360,256)
    """
    # pdb.set_trace()
    b = state.shape[0]
    state[state > 8.] = 0.

    dist_increment =  (4.+1e-4) / 256
    angle_increment = (2 * np.pi + 1e-4) / 360
    # angle_increment = 0.0175

    # pdb.set_trace()
    costmap = torch.zeros((b,360,256,3)).to(state.device)
    idx = (state[:,:360]/dist_increment).to(torch.long)  #(b,360)
    idx = torch.roll(idx,180,1)  #(b,360)
    costmap[:,:,:,0] = torch.zeros((b,360,256),device=state.device).scatter_(2,idx[:,:,None],1.)
    costmap[:,:,0,0] = 0.

    # assign waypoint
    deg = torch.atan2(state[:,-1],state[:,-2])  #(b,)
    deg = torch.clamp(deg, min=-torch.pi+(2*np.pi+2e-4)/360, max=torch.pi-(2*np.pi+2e-4)/360)
    deg = ((deg + torch.pi)/ angle_increment).to(torch.long)  #(b,)
    # deg = int(deg / angle_increment)
    cur_dist=torch.linalg.norm(state[:,-2:],dim=-1)  #(b,)
    # make sure cur_dist is withing range of 4 meters
    cur_dist = torch.clamp(cur_dist, max=4-4./256)
    dist = (cur_dist / dist_increment).to(torch.long)

    dist_range=torch.cat([(dist-1)[:,None],dist[:,None],(dist+1)[:,None]],dim=-1)  #(b,3)
    deg_range=torch.cat([(deg-1)[:,None],deg[:,None],(deg+1)[:,None]],dim=-1)  #(b,3)
    costmap[torch.arange(b)[:,None].expand(b,3), deg[:,None].expand(b,3), dist_range, :] = torch.ones(3, device=state.device)
    costmap[torch.arange(b)[:,None].expand(b,3), deg_range, dist[:,None].expand(b,3), :] = torch.ones(3, device=state.device)

    # for i in range(b):
    #     pdb.set_trace()
    #     plt.axis('equal')
    #     plt.imshow(costmap[i])
    #     plt.show()
    # image = preprocess(state)

    return costmap.permute(0,3,1,2)

class CostmapTestCase(unittest.TestCase):
    def test_goal_bound_0(self):
        state = torch.rand(1,362) * 3.5
        state[0,-2] = -4.
        state[0,-1] = 0.
        res = state2costmap(state)
        dist_increment = (4.+1e-4)/256

        for i in range(360):
            if state[0,i] <= dist_increment:
                continue
            # pdb.set_trace()
            assert torch.eq(res[0,:,(i+180)%360,int(state[0,i]/dist_increment)],torch.tensor([1.,0.,0.]))[0],f'i:{i},v:{res[0,:,(i+180)%360,int(state[0,i]/dist_increment)]}'
            assert torch.eq(res[0,:,(i+180)%360,int(state[0,i]/dist_increment)],torch.tensor([1.,0.,0.]))[1]
            assert torch.eq(res[0,:,(i+180)%360,int(state[0,i]/dist_increment)],torch.tensor([1.,0.,0.]))[2]

    def test_goal_bound_1(self):
        state = torch.rand(1,362) * 4.
        state[0,-2] = -4.
        state[0,-1] = 1e-4
        res = state2costmap(state)

        dist_increment = (4.+1e-4)/256
        for i in range(360):
            if state[0,i] <= dist_increment:
                continue
            # pdb.set_trace()
            assert torch.eq(res[0,:,(i+180)%360,int(state[0,i]/dist_increment)],torch.tensor([1.,0.,0.]))[0],f'i:{i},v:{res[0,:,(i+180)%360,int(state[0,i]/dist_increment)]}'
            assert torch.eq(res[0,:,(i+180)%360,int(state[0,i]/dist_increment)],torch.tensor([1.,0.,0.]))[1]
            assert torch.eq(res[0,:,(i+180)%360,int(state[0,i]/dist_increment)],torch.tensor([1.,0.,0.]))[2]

    def test_goal_bound_2(self):
        state = torch.rand(1,362) * 4.
        state[0,-2] = -4.
        state[0,-1] = -1e-4
        res = state2costmap(state)

        dist_increment = (4.+1e-4)/256
        for i in range(360):
            if state[0,i] <= dist_increment:
                continue
            # pdb.set_trace()
            assert torch.eq(res[0,:,(i+180)%360,int(state[0,i]/dist_increment)],torch.tensor([1.,0.,0.]))[0]
            assert torch.eq(res[0,:,(i+180)%360,int(state[0,i]/dist_increment)],torch.tensor([1.,0.,0.]))[1]
            assert torch.eq(res[0,:,(i+180)%360,int(state[0,i]/dist_increment)],torch.tensor([1.,0.,0.]))[2]

    def test_goal_bound_3(self):
        state = torch.rand(1,362) * 4.
        state[0,-2] = 0.
        state[0,-1] = -4.
        res = state2costmap(state)

        dist_increment = (4.+1e-4)/256
        for i in range(360):
            if state[0,i] <= dist_increment:
                continue
            # pdb.set_trace()
            assert torch.eq(res[0,:,(i+180)%360,int(state[0,i]/dist_increment)],torch.tensor([1.,0.,0.]))[0]
            assert torch.eq(res[0,:,(i+180)%360,int(state[0,i]/dist_increment)],torch.tensor([1.,0.,0.]))[1]
            assert torch.eq(res[0,:,(i+180)%360,int(state[0,i]/dist_increment)],torch.tensor([1.,0.,0.]))[2]

    def test_goal_bound_4(self):
        state = torch.rand(1,362) * 4.
        state[0,-2] = 0.
        state[0,-1] = 4.
        res = state2costmap(state)

        dist_increment = (4.+1e-4)/256

        for i in range(360):
            if state[0,i] <= dist_increment:
                continue
            # pdb.set_trace()
            assert torch.eq(res[0,:,(i+180)%360,int(state[0,i]/dist_increment)],torch.tensor([1.,0.,0.]))[0]
            assert torch.eq(res[0,:,(i+180)%360,int(state[0,i]/dist_increment)],torch.tensor([1.,0.,0.]))[1]
            assert torch.eq(res[0,:,(i+180)%360,int(state[0,i]/dist_increment)],torch.tensor([1.,0.,0.]))[2]
