from matplotlib.pylab import scatter, annotate, gca, quiver, gcf
import numpy as np

from leaparticulator.data.trajectory import Trajectory


annotations = []
means = []


def plot_quiver2d(data, alpha=.75, C=[], *args, **kwargs):
    X, Y = zip(*tuple(data))
    U = [x1 - x0 for x0, x1 in zip(X[:-1], X[1:])]
    V = [y1 - y0 for y0, y1 in zip(Y[:-1], Y[1:])]
    if C == []:
        color_delta = 1. / (len(X) - 1)
        C = [(color_delta * i, color_delta * i, color_delta * i) for i in range(len(X) - 1)]
    # print C
    X, Y = X[:-1], Y[:-1]
    # print X, Y, U, V
    return quiver(X, Y, U, V, C, *args, scale_units='xy', angles='xy', scale=1, width=0.005, alpha=alpha, **kwargs)


def find_bounding_box(trajectories):
    xmin = ymin = 1000
    xmax = ymax = -1000
    delta = 1
    for signal in trajectories:
        # print signal
        for frame in signal:
            # print frame
            x, y, z = frame.get_stabilized_position()
            xmax = max(x + delta, xmax)
            xmin = min(x - delta, xmin)
            ymax = max(y + delta, ymax)
            ymin = min(y - delta, ymin)
    return xmin, xmax, ymin, ymax


def to_trajectory_object(trajectories, step_size=10, units="xy"):
    tt = []
    xmin, xmax, ymin, ymax = find_bounding_box(trajectories)
    for trajectory in trajectories:
        data = [frame.get_stabilized_position() for frame in trajectory]
        func = None
        if units == "xy":
            func = lambda x: x[:2]
        elif units == "amp_and_freq":
            from LeapTheremin import palmToAmpAndFreq, freqToMel

            func = lambda x: (palmToAmpAndFreq(x)[1], palmToAmpAndFreq(x)[0])
        elif units == "amp_and_mel":
            from LeapTheremin import palmToAmpAndFreq, freqToMel

            func = lambda x: (freqToMel(palmToAmpAndFreq(x)[1]), palmToAmpAndFreq(x)[0])

        arr = [func(frame) for frame in data]
        t = Trajectory(from_arr=arr, duration=len(arr), step_size=step_size, origin=(xmin, ymin),
                       ndim=2, dim_size=(xmax - xmin, ymax - ymin), prob_c=1)
        tt.append(t)
    return tt


def to_trajectory_file(trajectories, filename):
    xmin, xmax, ymin, ymax = find_bounding_box(trajectories)
    start = 0
    end = 1
    import os

    if os.path.isfile(filename):
        os.remove(filename)
    with open(filename, "w") as f:
        print(xmin, xmax, ymin, ymax, start, end)
        f.write("%d %d %d %d %d %d\n" % (xmin, xmax, ymin, ymax, start, end))
        for signal in trajectories:
            for frame in signal:
                x, y, z = frame.get_stabilized_position()
                time = frame.timestamp
                f.write("%f %f %f\n" % (x, y, time))
            f.write("0.0 0.0 0.0\n")


def responses_to_trajectories(responses):
    counter = 0
    trajectories = []
    for phase in r:
        for image in r[phase]:
            # if image == u'./img/meanings/5_1.png':
            counter += 1
            trajectory = r[phase][image]
            trajectories.append(trajectory)
            data = []
            for frame in trajectory[:-1]:
                x, y, z = frame.get_stabilized_position()
                data.append((x, y))
    # print frame.timestamp
    #             plot_quiver2d(data)
    #             break
    #             plot(X,Y,label="%s-%s" % (phase, image))
    return trajectories


from matplotlib.colors import colorConverter
from matplotlib.patches import Ellipse
# figure()
# means = []
# annotations = []

def on_pick(event):
    print event
    print annotations
    if event.artist in annotations:
        on_pick_annotation(event)
    elif event.artist in means:
        on_pick_means(event)
    draw()
    time.sleep(1)


def on_pick_annotation(event):
    print "Annotation:", event.artist
    event.artist.set_visible(False)


def on_pick_means(event):
    print "Mean:", means.index(event.artist)
    annotations[means.index(event.artist)].set_visible(True)
    print annotations[means.index(event.artist)]


colors = ['red', 'green', 'yellow', 'magenta', 'orange', 'black', 'cyan', 'white']
colors = [(x / 10., y / 20., z / 30.) for x, y, z in zip(range(10), range(10), range(10))]
print colors


def plot_trajectories(trajectories):
    from collections import defaultdict

    for trajectory in trajectories:
        trajectory_in_state_space = []
        state_counts = defaultdict(int)
        for x, y in trajectory.data:
            state = hidden_states[counter]
            state_counts[state] += 1
            trajectory_in_state_space.append(tuple(model.means_[state]))
            counter += 1
        state = max(state_counts.iteritems(), key=operator.itemgetter(1))[0]
        X, Y = zip(*tuple(trajectory_in_state_space))
        # print X
        color = colorConverter.to_rgb(colors[state])
        # print color
        for i, (x, y) in enumerate(zip(X[:-1], Y[:-1])):
            y1 = Y[i + 1]
            x1 = X[i + 1]
            if (x != x1) or (y != y1):
                rotation = degrees(arctan((y1 - y) / (x1 - x))) + 270
                print "Rotation is", rotation, " (%f,%f to %f,%f)" % (x, y, x1, y1)
                arc = Arc(xy=((x1 + x) / 2., (y1 + y) / 2.), width=x1 - x, height=sqrt((y1 - y) ** 2 + (x1 - x) ** 2),
                          angle=rotation, theta1=90, theta2=270, alpha=.05)
                gca().add_patch(arc)


def plot_hmm(means_, transmat, covars, axes=None):
    if axes != None:
        axes(axes)
        # f, axes = subplots(2)#,sharex=True, sharey=True)
    #     sca(axes[0])
    global annotations
    annotations = []
    global means
    means = []
    color_map = colors  # [colorConverter.to_rgb(colors[i]) for i in range(len(means_))]
    if isinstance(means_[0], float):
        for i in range(len(means_)):
            raise Exception("Cannot plot a univariate HMM because it is stupid.")
    else:
        for i in range(len(means_)):
            means.append(scatter(*tuple(means_[i]), color=colors[i], picker=10, label="State%i" % i))
            print "Means of the state: %s" % (list(means_[i]))
            annotate(s="%d" % i, xy=means[i], xytext=(1, -10), xycoords="data", textcoords="offset points",
                     alpha=1, bbox=dict(boxstyle='round,pad=0.2', fc=colors[i], alpha=0.3))

            # add the 95% error ellipse
            eig_val, eig_vec = np.linalg.eig(covars[i])
            w = 2 * np.sqrt(5.991 * eig_val[0])
            h = 2 * np.sqrt(5.991 * eig_val[1])
            eig_vec = eig_vec[:, np.where(eig_val == max(eig_val))]
            angle = np.arctan(eig_vec[1] / eig_vec[0]) / (np.pi) * 180
            # error = Ellipse(xy = means_[i], width = np.diag(covars[i])[0], height = np.diag(covars[i])[1],
            # alpha=.15, color=colors[i])
            print "Angle:", angle
            error = Ellipse(xy=means_[i], width=w, height=h, angle=angle,
                            alpha=.15, color=colors[i])
            gca().add_patch(error)
            #         x0, y0 = means_[i]
            #         prob_string = ""
            #         for j in range(len(means)):
            #             xdif = 10
            #             ydif = 5
            #             s = "P(%d->%d)=%f" % (i,j,transmat[i][j])
            #             prob_string = "%s\n%s" % (prob_string,s)
            #             if i != j:
            #                 x1, y1 = means_[j]
            #                 print transmat[i][j]
            #                 # if transmat[i][j] is too low, we get an underflow here
            # #                 q = quiver([x0], [y0], [x1-x0], [y1-y0], alpha = 10000 * (transmat[i][j]**2),
            #                 alpha = 10 ** -300
            #                 if transmat[i][j] > 10 ** -100:
            #                     alpha = (100 * transmat[i][j])**2
            #                 q = quiver([x0], [y0], [x1-x0], [y1-y0], alpha = alpha,
            #                        scale_units='xy',angles='xy', scale=1, width=0.005, label="P(%d->%d)=%f" % (i,j,transmat[i][j]))
            #                 legend()

            #         annotations.append(annotate(s=prob_string, xy=means_[i], xytext=(0, 10), xycoords="data",textcoords="offset points",
            #                          alpha=1,bbox=dict(boxstyle='round,pad=0.2', fc=colors[i], alpha=0.3), picker=True,
            #                          visible=True))


            print "State%i is %s" % (i, colors[i])
    cid = gcf().canvas.mpl_connect('pick_event', on_pick)
