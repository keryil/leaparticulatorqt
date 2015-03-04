'''
Created on Apr 14, 2013

@author: kerem
'''

from numpy.random import randint, random, normal
from numpy.linalg import norm
import numpy as np
from matplotlib import pyplot as plt
from copy import copy, deepcopy
from scipy.spatial.distance import euclidean

np.seterr(all='raise')

class Trajectory(object):
    '''
    This class represents a trajectory in an n-dimensional space consisting of a number of 
    points joined by lines. It is generated as a collection of points in space. All points are 
    generated using the previous point, and all are equidistant. The direction of change 
    is random (positive vs. negative). With a certain probability prob_c, the bearing 
    also changes with a normal distribution of (0, step_size_per_dim).
    '''
    data = None
    duration, prob_c, cost_c = None, None, None
    step_size = None

    def __init__(self, duration, step_size, prob_c, dim_size, ndim, origin=None,
        from_arr=None,plot=False, leading_silence=False, trailing_silence=False, *args, **kwargs):
        '''
        Constructor
        Creates a new trajectory.
        Args:
            duration: number of points in the signal. 
            prob_c: probability of change in direction at every point.
            ndim: number of dimensions in the space
            dim_size: a list containing the sizes of dimensions
            step_size: eucledian distance that can be traveled between two time steps
        '''
        self.duration, self.prob_c, self.dim_size = duration, prob_c, dim_size
        self.ndim = ndim
        assert len(dim_size) == ndim
        self.step_size = float(step_size)
        self.step_size_per_dim = np.sqrt((step_size ** 2) / ndim)
        if origin is None:
            self.origin = tuple([0]*ndim)
        else:
            assert len(origin) == ndim
            self.origin = origin
#         print self.duration,  self.prob_c, self.dim_size, self.step_size, self.step_size_per_dim
#         print norm((3,4))
        if from_arr == None:
            self.__create(leading_silence, trailing_silence)
        else:
            assert ndim == len(from_arr[0])
            assert duration == len(from_arr)
            self.__createFromArr(from_arr)
        if plot:
            self.plot2d(*args, show=False, **kwargs)
    
    def __createFromArr(self, arr):
        self.data = arr

    def randomPoint(self):
        return tuple([randint(z, z + x) for z, x in zip(self.origin, self.dim_size)])

    def __create(self, leading_silence=False, trailing_silence=False):
        self.data = []
        plt.clf()
        current_duration = 0
        first = self.randomPoint()
        change_dim = True
        last_dim = None
        while current_duration < self.duration:
#             print "\n=============================\n"
#             print "Beginning: %s" % self.data
            last_point = None
            new_point = None
            try:
                if current_duration == 0:
                    raise IndexError
                last_point = self.data[current_duration - 1]
            except IndexError:
                # if this is the first point, initialize and add it
                if not leading_silence:
                    p = self.randomPoint()
                    self.data.append(p)
                    current_duration += 1
                # if there is supposed to be leading silence, add it (0,0,0,...,0 is silence)
                else:
                    p = tuple(self.origin)
                    self.data.append(p)
                    self.data.append(p)
                    current_duration += 2
                continue
            
            # will calculate degree of change between each dimension
#             change = [[0. for y in xrange(self.ndim-1)] for x in xrange(self.ndim)]
            
            
            # print "Last point: %s, New point: %s" % (last_point, new_point)
            # print "Step size: %s, Euclidean: %s"% (self.step_size, distance)
            ratio = None
            try:# first, get a new candidate point
                new_point = list(self.origin)
                for dim in range(self.ndim):
                    alpha = random()
                    if alpha <= self.prob_c:
                        if randint(2):
                            new_point[dim] = last_point[dim] + abs(normal(0., self.step_size_per_dim))
                        else:
                            new_point[dim] = last_point[dim] - abs(normal(0., self.step_size_per_dim))
                    # else:
                        # if randint(2):
                        #     new_point[dim] = last_point[dim] + self.step_size_per_dim
                        # else:
                        #     new_point[dim] = last_point[dim] - self.step_size_per_dim
                new_point = self.boundary_check(new_point)

                distance = euclidean(last_point, new_point)
                ratio = float(self.step_size) / distance
            except FloatingPointError, err:
                self.__create(leading_silence, trailing_silence)
                return
#             print "Before scaling: %f" % norm([x2-x1 for x1,x2 in zip(last_point, new_point)])
            while abs(ratio - 1) > 0.00001:
#                 print "Ratio is %f" % ratio
                for x in range(self.ndim):
                    if randint(2):
                        new_point[x] = last_point[x] + (new_point[x] - last_point[x]) * ratio
                    else:
                        new_point[x] = last_point[x] - (new_point[x] - last_point[x]) * ratio
                new_point = self.boundary_check(new_point)
                distance = euclidean(last_point, new_point)
                # print "Last point: %s, New point: %s" % (last_point, new_point)
                # print "Step size: %s, Euclidean: %s"% (self.step_size, distance)
                ratio = None
                try:
                    ratio = float(self.step_size) / distance
                except FloatingPointError, err:
                    self.__create(leading_silence, trailing_silence)
                    return
#             print "Final length of segment is %f (%f)" %  (norm([x2-x1 for x1,x2 in zip(last_point, new_point)]),self.step_size)
            self.data.append(tuple(new_point))
            current_duration += 1
        self.current_duration = current_duration
        # self.boundary_check()
            
    def plot2d(self, show=True, width=0.002, path=None, *args, **kwargs):
        x, y = zip(*self.data)
        x, y = np.asarray(x), np.asarray(y)
        q = plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1, width=width, *args, **kwargs)
        print q
        # print x[:1], y[:1], [x[1]-x[2]], [y[1]-y[2]]
        # plt.arrow(x[:1], y[:1], [x[1]-x[0]], [y[1]-y[0]], edgecolor="red", scale_units='xy', angles='xy', scale=1)
        # plt.plot(xdif, ydif)
        # plt.plot(x, y, *args, **kwargs)
        if show:
            # plt.xlim((-1, self.dim_size))
            # plt.ylim((-1, self.dim_size))
            plt.show()
        return q
    
    def noise(self, spread, in_place=True):
        """
        Applies shape-preserving noise to the trajectory as described in 
        Boer & Zuidema 2010 p.8, "spread" parameter being the spread 
        of the normal distribution used for the noise.
        """
        obj = None
        if in_place:
            obj = self
        else:
            obj = deepcopy(self)

        # the change vector for point 0
        sx_1 = normal(scale=spread, size=(obj.ndim,))
        sx_n = normal(scale=spread, size=(obj.ndim,))
        
        for i in range(len(obj.data)):
            n_i = normal(scale=float(spread) / len(obj.data), size=(obj.ndim,))
            last_state = obj.data[i]
            
            # we use i instead of i-1 because index starts at 0, unlike 
            # the paper where it starts at 1
            alpha = float(i) / (len(obj.data) - 1)
            new_state = last_state + alpha * sx_1 + (1 - alpha) * sx_n + n_i
            
            # check for and correct boundary issues\
            new_state = obj.boundary_check(new_state)
#             for d in range(len(new_state)):
#                 x = new_state[d]
#                 if x > obj.dim_size:
#                     x = obj.dim_size
#                 elif x < 0:
#                     x = 0
#                 new_state[d] = x
            # commit
            obj.data[i] = new_state
        return obj

    def boundary_check(self, point=None):
        """
        Checks if a point (or any point) violates the boundaries of the acoustic space, 
        and applies corrections as necessary. If it is given a point, it returns the modified 
        version
        
        Here is how it works for a point:
        >>> t = Trajectory(duration=50, step_size=7, prob_c=1, dim_size=200, ndim=2)
        >>> p = (-1,1,251)
        >>> t.boundary_check(p)
        (0, 1, 200)
        
        And now for the trajectory data:
        >>> t.data[0] = (-50,500,10)
        >>> t.boundary_check()
        >>> t.data[0]
        (0, 200, 10)
        """
        if point == None:
            for i in range(len(self.data)):
                point = list(self.data[i])
                for d in range(len(point)):
                    dim = point[d]
                    dim = min(max(self.origin[d], dim), self.origin[d] + self.dim_size[d] - 1)
                    point[d] = dim
                self.data[i] = tuple(point)
        else:
            is_tuple = (type(point) == tuple)
            if is_tuple:
                point = list(point) 
            for d in range(len(point)):
                dim = point[d]
                dim = min(max(self.origin[d], dim), self.origin[d] + self.dim_size[d] - 1)
                point[d] = dim
            if is_tuple:
                point = tuple(point)
            return point
                
    def mutate(self, shift, plot=False):
        """
        Mutates the trajectory by adding a normal random vector (spread=shift) to 
        a random point on the trajectory, as in Boer & Zuidema 2010 p.9
        """
        # add a random vector to a random point
        point = randint(len(self.data))
        p = list(self.data[point])
        print(p)
        dif = normal(scale=float(shift), size=(self.ndim,))
        print(dif)
        p += dif
        print(p)
        self.data[point] = tuple(p)
        if plot:
            plt.plot(p[0], p[1], "go")
        
        # check for boundary conditions and correct them
        self.boundary_check()
        
        # check for max. segment lengths and correct them
        # first, from j to the end - 1
        for j in range(point, len(self.data) - 1):
            this_point = np.array(self.data[j + 1])
            last_point = np.array(self.data[j])
            distance = euclidean(this_point, last_point)
            if distance > self.step_size:
                print("Old this_point (%f): %s\nLast point: %s" % (distance, this_point, last_point))
                this_point = last_point + (self.step_size / distance) * (this_point - last_point)
                print("New this_point (%f): %s\nLast point: %s" % (euclidean(this_point, last_point), this_point, last_point))
            self.data[j + 1] = tuple(this_point)
            
        # then, from j to 1
        for j in range(point, 0, -1):
            this_point = np.array(self.data[j - 1])
            last_point = np.array(self.data[j])
            distance = euclidean(this_point, last_point)
            if distance > self.step_size:
                print("Old this_point (%f): %s\nLast point: %s" % (distance, this_point, last_point))
                this_point = last_point + self.step_size / distance * (this_point - last_point)
                print("New this_point (%f): %s\nLast point: %s" % (euclidean(this_point, last_point), this_point, last_point))
            self.data[j - 1] = tuple(this_point)
            
            
            
        
        
        
    def __iter__(self):
        return iter(self.data)
    
    if __name__ == "__main__":
        import sys,os
        sys.path.append(os.path.expanduser("~/Dropbox/ABACUS/Workspace/Abacus"))
        from abacus.experiments.artificial.trajectory import Trajectory
        from copy import deepcopy as copy
#         t = [Trajectory(duration=50, step_size=10, prob_c=.5, dim_size=200, ndim=2) for tr in range(300)]
        t = [Trajectory(duration=2040, step_size=50, prob_c=.05, dim_size=(4000, 4000), ndim=2, origin=(-2000,-2000))]
        t[0].plot2d(False, label="Original")
        for gen in range(0):
            n = copy(t[gen])
            n.noise(3)
            n.plot2d(False, label="%d" % (gen + 1), color="red", alpha="0.4")
            t.append(n)
        for tt in t:
            print(t)
        plt.legend()
        plt.show()
        
